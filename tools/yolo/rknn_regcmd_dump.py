#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rknn_regcmd_dump.py — 从 Rockchip 官方 .rknn 中提取内嵌 regcmd 并解码，
作为 RK3588 原生卷积的寄存器 golden oracle（K-YOLO-001 §9 Phase-0）。

背景：官方 librknnrt 在 RK3588 上能正确执行原生卷积，但 regcmd 生成在闭源
userspace 内（RKNN_DUMP_DIR 只 dump 张量，不 dump regcmd）。实测发现官方
`.rknn` 文件**内嵌完整 regcmd**（yolov8.rknn：`subgraph0 regcmdbuffer size:
2229696`），因此可直接从文件解析出权威的 CNA/CORE/DPU 寄存器值。

regcmd 条目格式（8 字节，小端，实测确认）:
    offset 0  u16  reg   寄存器地址
    offset 2  u32  val   寄存器值
    offset 6  u16  op    块码: 0x1001=DPU, 0x2001=DPU_RDMA,
                         0x0201=CNA, 0x0801=CORE, 0x0101=PC, 0x0000=空

用法:
    python rknn_regcmd_dump.py <model.rknn> [--csv out.csv]
    python rknn_regcmd_dump.py <model.rknn> --block CNA --index 1
"""
import argparse
import struct
import sys

OP = {0x1001: "DPU", 0x2001: "DPU_RDMA", 0x0201: "CNA", 0x0801: "CORE",
      0x0101: "PC", 0x0000: "NOP"}

REGNAME = {
    0x1004: "CNA_S_PTR", 0x100c: "CONV_CON1", 0x1010: "CONV_CON2",
    0x1014: "CONV_CON3", 0x1018: "CONV_CON4", 0x1020: "DATA_SIZE0",
    0x1024: "DATA_SIZE1", 0x1028: "DATA_SIZE2", 0x102c: "DATA_SIZE3",
    0x1030: "WEIGHT_SIZE0", 0x1034: "WEIGHT_SIZE1", 0x1038: "WEIGHT_SIZE2",
    0x1040: "CBUF_CON0", 0x1044: "CBUF_CON1", 0x104c: "CVT_CON0",
    0x1068: "PAD_CON0", 0x1070: "FEAT_ADDR", 0x1078: "DMA_CON0",
    0x107c: "DMA_CON1", 0x1080: "DMA_CON2", 0x1084: "FC_SIZE0",
    0x1088: "FC_SIZE1",
    0x3010: "CORE_MISC", 0x3014: "CORE_DATAOUT_SIZE_0",
    0x3018: "CORE_DATAOUT_SIZE_1", 0x301c: "CORE_CLIP_TRUNC",
    0x4004: "DPU_S_PTR", 0x400c: "FEAT_MODE", 0x4010: "DATA_FORMAT",
    0x4020: "DST_BASE", 0x4024: "DST_SURF_STRIDE", 0x4030: "CUBE_W",
    0x4034: "CUBE_H", 0x4038: "CUBE_NOTCH", 0x403c: "CUBE_CH",
    0x4040: "BS_CFG", 0x4058: "WDMA_SIZE_0", 0x405c: "WDMA_SIZE_1",
    0x4060: "BN_CFG", 0x4070: "EW_CFG", 0x40c0: "SURFACE_ADD",
    0x5004: "RDMA_S_PTR", 0x500c: "RDMA_CUBE_W", 0x5010: "RDMA_CUBE_H",
    0x5014: "RDMA_CUBE_CH", 0x5034: "ERDMA", 0x5040: "EW_SURF_STRIDE",
    0x5044: "FMC",
}


def bits(v, lo, hi=None):
    if hi is None:
        return (v >> lo) & 0xffff
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def iter_entries(data, start=0):
    """Yield (offset, reg, val, op) for every 8-byte-aligned entry."""
    n = len(data)
    o = start - (start % 8)
    while o + 8 <= n:
        reg, val, op = struct.unpack_from("<HIH", data, o)
        yield o, reg, val, op
        o += 8


def find_blocks(data, want_op, min_len=8):
    """Return [(offset, {reg: val}), ...] for maximal runs of want_op."""
    out = []
    cur_off, cur = None, {}
    for o, reg, val, op in iter_entries(data):
        if op == want_op:
            if cur_off is None:
                cur_off, cur = o, {}
            cur[reg] = val
        else:
            if cur_off is not None and len(cur) >= min_len:
                out.append((cur_off, cur))
            cur_off, cur = None, {}
    if cur_off is not None and len(cur) >= min_len:
        out.append((cur_off, cur))
    return out


def cna_row(off, b):
    ds0, ds1, ds2, ds3 = (b.get(0x1020, 0), b.get(0x1024, 0),
                          b.get(0x1028, 0), b.get(0x102c, 0))
    ws2, c0, c1, cc3, pad = (b.get(0x1038, 0), b.get(0x1040, 0),
                             b.get(0x1044, 0), b.get(0x1014, 0),
                             b.get(0x1068, 0))
    return dict(off=off,
                W=bits(ds0, 16, 26), H=bits(ds0, 0, 10),
                Cin=bits(ds1, 0, 15), Cin_real=bits(ds1, 16, 29),
                KW=bits(ws2, 24, 28), KH=bits(ws2, 16, 20),
                KK=bits(ws2, 0, 13),
                OW=bits(ds2, 0, 10), atomics=bits(ds3, 0, 21),
                wt_bank=bits(c0, 4, 7), dt_bank=bits(c0, 0, 3),
                wt_bytes=b.get(0x1030, 0), wt_k=b.get(0x1034, 0),
                sx=bits(cc3, 0, 2), sy=bits(cc3, 3, 5),
                cbuf1=c1, pad=pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--block", choices=["CNA", "CORE", "DPU", "DPU_RDMA"],
                    help="dump full register list of one block type")
    ap.add_argument("--index", type=int, default=0, help="which block")
    ap.add_argument("--csv", help="write CNA geometry table to CSV")
    a = ap.parse_args()

    data = open(a.model, "rb").read()
    print(f"[regcmd] {a.model}: {len(data)} bytes", file=sys.stderr)

    if a.block:
        want = {v: k for k, v in OP.items()}[a.block]
        blks = find_blocks(data, want, min_len=4)
        print(f"[regcmd] {a.block}: {len(blks)} blocks", file=sys.stderr)
        if not blks:
            return 1
        off, b = blks[a.index]
        print(f"--- {a.block} block #{a.index} @0x{off:x} ---")
        for reg in sorted(b):
            print(f"  0x{reg:04x} {REGNAME.get(reg, ''):20s} 0x{b[reg]:08x}")
        return 0

    cn = find_blocks(data, 0x0201, min_len=8)
    print(f"[regcmd] CNA blocks: {len(cn)}", file=sys.stderr)
    hdr = ("idx    W    H  Cin Creal KW KH  KK   OW atomics wt_bk dt_bk"
           "  wt_bytes  wt_k  s  cbuf1     pad")
    print(hdr)
    rows = []
    for i, (off, b) in enumerate(cn):
        r = cna_row(off, b)
        rows.append(r)
        print(f"{i:3d} {r['W']:5d} {r['H']:4d} {r['Cin']:4d} {r['Cin_real']:5d}"
              f" {r['KW']:2d} {r['KH']:2d} {r['KK']:3d} {r['OW']:5d}"
              f" {r['atomics']:7d} {r['wt_bank']:5d} {r['dt_bank']:5d}"
              f" {r['wt_bytes']:9d} {r['wt_k']:5d} {r['sx']:2d}"
              f" 0x{r['cbuf1']:06x} 0x{r['pad']:08x}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[regcmd] wrote {a.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
