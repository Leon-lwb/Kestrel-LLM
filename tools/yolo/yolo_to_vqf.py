#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_to_vqf.py — Ultralytics YOLO (.pt) → 引擎自有格式 VQF v4（检测）。

设计红线（用户约束）：**部署格式 = 引擎自有 VQF 家族**，不引入 ONNX / .rknn。
本工具是**离线**解析器：torch/ultralytics 只用于读取 .pt 的模块图与权重，
并把 BN 折进卷积、把图展平成原语算子，直接写出 VQF v4。产物由引擎加载。

格式规格 = docs/立项-K-YOLO-001-G0-NPU卷积检测适配.md 附录 A（单一真源）。

用法:
  # 转换（默认 F32 权重；--f16 半精度；--q8 per-channel int8）
  python yolo_to_vqf.py --pt yolov8n.pt --out yolov8n.vqf
  # 转换 + 读回自检（对比 torch golden）
  python yolo_to_vqf.py --pt yolov8n.pt --out yolov8n.vqf --check
  # 仅打印展平后的图摘要
  python yolo_to_vqf.py --pt yolov8n.pt --dump-graph
"""
import argparse
import os
import struct
import sys

import numpy as np

# ================================================================
# VQF v4 格式常量（镜像 include/model/vqf_format.h — 不漂移）
# ================================================================
VQF_MAGIC        = 0x57465146          # "VQFW"
VQF_VERSION_DET  = 4
VQF_HEADER_BYTES = 4096
VQF_ALIGN        = 64
VQF_FLAG_DET     = (1 << 10)

QT_F32, QT_Q8, QT_Q4, QT_F16, QT_BLOB = 0x01, 0x02, 0x03, 0x05, 0x06

OP_CONV, OP_ADD, OP_CONCAT, OP_MAXPOOL, OP_UPSAMPLE = 1, 2, 3, 4, 5
OP_SIGMOID, OP_MUL, OP_SLICE, OP_SOFTMAX, OP_ATTN_PSA, OP_DETECT = 6, 7, 8, 9, 10, 11
ACT_NONE, ACT_SILU = 0, 1

FAM_V5, FAM_V8, FAM_V11 = 0, 1, 2
GRAPH_MAGIC, GRAPH_VERSION = 0x50524759, 1   # 'YGRP'

# VQFHeader 实测布局（含 uint64 8 字节对齐填充）：
#   0 magic,4 version,8 flags,12 n_tensors,16 arch[180],196 pad[4],
#   200 data_offset,208 file_len,216 enc_iv[16],232 enc_rsvd[16],248 sig[200] → 448
HDR_ARCH_OFF, HDR_ARCH_LEN = 16, 180
HDR_DATAOFF, HDR_FILELEN = 200, 208
HDR_BYTES = 448


# ================================================================
# 图展平：ultralytics 模块 → 原语算子 + 权重
# ================================================================
class Builder:
    def __init__(self, wmode="f32", f16_blob=False):
        self.ops = []          # (opcode, [in_slots], out_slot, [params])
        self.weights = []      # (name, np.ndarray)
        self.windex = {}       # name -> idx
        self.nslots = 0
        self.convs = 0         # 卷积序号（= 权重名 c%04d）
        self.wmode = wmode     # "f32" | "f16" | "q8"

    # ---- slot ----
    def new_slot(self):
        s = self.nslots
        self.nslots += 1
        return s

    def emit(self, op, ins, out, params):
        self.ops.append((op, list(ins), out, list(params)))

    # ---- 权重 ----
    def add_weight(self, base, arr):
        name = base
        if name in self.windex:
            raise RuntimeError(f"duplicate weight {name}")
        self.windex[name] = len(self.weights)
        self.weights.append((name, np.ascontiguousarray(arr, dtype=np.float32)))
        return name

    def add_conv_weight(self, w, b):
        """w [oc,ic,kh,kw] fp32, b [oc] or None -> conv ordinal"""
        cid = self.convs
        self.convs += 1
        self.add_weight(f"c{cid:04d}_w", w)
        if b is not None:
            self.add_weight(f"c{cid:04d}_b", b)
        return cid

    # ---- 卷积（含 BN 折叠 / 激活 / groups） ----
    def fold_conv(self, conv2d, bn=None):
        """返回 (w [oc,ic,kh,kw], b [oc] or None)，BN 折进 weight/bias。"""
        w = conv2d.weight.detach().cpu().numpy().astype(np.float32)
        b = conv2d.bias.detach().cpu().numpy().astype(np.float32) if conv2d.bias is not None else None
        if bn is not None:
            g = bn.weight.detach().cpu().numpy().astype(np.float32)
            be = bn.bias.detach().cpu().numpy().astype(np.float32)
            mu = bn.running_mean.detach().cpu().numpy().astype(np.float32)
            var = bn.running_var.detach().cpu().numpy().astype(np.float32)
            scale = g / np.sqrt(var + float(bn.eps))         # per-out-channel
            w = w * scale.reshape(-1, 1, 1, 1)
            # y=(x-mu)/std*g+be = x*scale + (be-mu*scale)
            b = (be - mu * scale) if b is None else (b * scale + (be - mu * scale))
        return w, b

    def conv_op(self, x, conv2d, bn=None, act=ACT_NONE, tag=""):
        w, b = self.fold_conv(conv2d, bn)
        sh, sw = int(conv2d.stride[0]), int(conv2d.stride[1])
        ph, pw = int(conv2d.padding[0]), int(conv2d.padding[1])
        kh, kw = int(conv2d.kernel_size[0]), int(conv2d.kernel_size[1])
        gp = int(conv2d.groups)
        cid = self.add_conv_weight(w, b)
        out = self.new_slot()
        # params: w_id, has_bias, sh, sw, ph, pw, groups, act, kh, kw
        self.emit(OP_CONV, [x], out,
                  [cid, 1 if b is not None else 0, sh, sw, ph, pw, gp, act, kh, kw])
        return out

    def ultralytics_conv(self, x, mod, tag=""):
        """ultralytics Conv 块（conv2d + BN + act）或裸 nn.Conv2d。"""
        import torch.nn as nn
        if isinstance(mod, nn.Conv2d):
            return self.conv_op(x, mod, None, ACT_NONE, tag)
        c2d = getattr(mod, "conv", None)
        bn = getattr(mod, "bn", None)
        if c2d is None:
            raise RuntimeError(f"unsupported conv-like {type(mod).__name__} {tag}")
        act = getattr(mod, "act", None)
        a = ACT_NONE
        if act is not None and not isinstance(act, nn.Identity) and act is not False:
            if isinstance(act, nn.SiLU):
                a = ACT_SILU
            else:
                raise RuntimeError(f"unsupported act {type(act).__name__} {tag}")
        return self.conv_op(x, c2d, bn, a, tag)

    # ---- 复合模块 ----
    def bottleneck(self, x, m, tag=""):
        y = self.ultralytics_conv(x, m.cv1, tag + ".cv1")
        y = self.ultralytics_conv(y, m.cv2, tag + ".cv2")
        if getattr(m, "add", False):
            o = self.new_slot()
            self.emit(OP_ADD, [x, y], o, [])
            return o
        return y

    def cat(self, slots, dim=1):
        o = self.new_slot()
        self.emit(OP_CONCAT, slots, o, [dim])
        return o

    def slice_ch(self, x, a, b):
        o = self.new_slot()
        self.emit(OP_SLICE, [x], o, [1, a, b])   # axis=1(channel), [a,b)
        return o

    def c2f_like(self, x, m, tag=""):
        """C2f / C3k2：y=chunk(cv1(x),2); y+=[mi(y[-1])]; cv2(cat(y))"""
        y0 = self.ultralytics_conv(x, m.cv1, tag + ".cv1")
        c = m.c            # 半通道
        parts = [self.slice_ch(y0, 0, c), self.slice_ch(y0, c, 2 * c)]
        cur = parts[1]
        for i, mi in enumerate(m.m):
            cur = self.block(cur, mi, f"{tag}.m{i}")
            parts.append(cur)
        cat = self.cat(parts, 1)
        return self.ultralytics_conv(cat, m.cv2, tag + ".cv2")

    def c3_like(self, x, m, tag=""):
        """C3 / C3k：cv3(cat((m(cv1(x)), cv2(x))))"""
        a = self.ultralytics_conv(x, m.cv1, tag + ".cv1")
        a = self.block_seq(a, m.m, tag + ".m")
        b = self.ultralytics_conv(x, m.cv2, tag + ".cv2")
        cat = self.cat([a, b], 1)
        return self.ultralytics_conv(cat, m.cv3, tag + ".cv3")

    def block_seq(self, x, seq, tag=""):
        cur = x
        for i, mi in enumerate(seq):
            cur = self.block(cur, mi, f"{tag}.{i}")
        return cur

    def block(self, x, m, tag=""):
        import torch.nn as nn
        n = type(m).__name__
        if isinstance(m, nn.Conv2d):
            return self.ultralytics_conv(x, m, tag)
        if n == "Bottleneck":
            return self.bottleneck(x, m, tag)
        if n in ("C2f", "C3k2"):
            return self.c2f_like(x, m, tag)
        if n in ("C3", "C3k"):
            return self.c3_like(x, m, tag)
        if n in ("Sequential",):
            return self.block_seq(x, m, tag)
        if n == "PSABlock":
            return self.psa_block(x, m, tag)
        if n == "DWConv" or hasattr(m, "conv"):
            return self.ultralytics_conv(x, m, tag)
        raise RuntimeError(f"unsupported block {n} @{tag}")

    def sppf(self, x, m, tag=""):
        y0 = self.ultralytics_conv(x, m.cv1, tag + ".cv1")
        parts = [y0]
        cur = y0
        for i in range(getattr(m, "n", 3)):
            o = self.new_slot()
            k = int(m.m.kernel_size if isinstance(m.m.kernel_size, int) else m.m.kernel_size[0])
            s = int(m.m.stride if isinstance(m.m.stride, int) else m.m.stride[0])
            p = int(m.m.padding if isinstance(m.m.padding, int) else m.m.padding[0])
            self.emit(OP_MAXPOOL, [cur], o, [k, s, p])
            parts.append(o)
            cur = o
        cat = self.cat(parts, 1)
        y = self.ultralytics_conv(cat, m.cv2, tag + ".cv2")
        if getattr(m, "add", False):
            o = self.new_slot()
            self.emit(OP_ADD, [x, y], o, [])
            return o
        return y

    def attn_psa(self, x, at, tag=""):
        """ultralytics Attention：qkv/proj/pe 三卷积（BN 已折叠）+ 多头注意力。"""
        nh = int(at.num_heads)
        hd = int(at.head_dim)
        kd = int(at.key_dim)
        wq, bq = self.fold_conv(at.qkv.conv, at.qkv.bn)
        wp, bp = self.fold_conv(at.proj.conv, at.proj.bn)
        wpe, bpe = self.fold_conv(at.pe.conv, at.pe.bn)
        i_qkv = self.add_conv_weight(wq, bq)
        i_proj = self.add_conv_weight(wp, bp)
        i_pe = self.add_conv_weight(wpe, bpe)
        o = self.new_slot()
        # params: nh, hd, kd, w_qkv, w_proj, w_pe（bias 由各自 c%04d_b 张量携带）
        self.emit(OP_ATTN_PSA, [x], o, [nh, hd, kd, i_qkv, i_proj, i_pe])
        return o

    def psa_block(self, x, m, tag=""):
        a = self.attn_psa(x, m.attn, tag + ".attn")
        a = self.new_slot() if False else a
        if getattr(m, "add", True):
            o = self.new_slot(); self.emit(OP_ADD, [x, a], o, []); a = o
        f = self.ultralytics_conv(a, m.ffn[0], tag + ".ffn0")
        f = self.ultralytics_conv(f, m.ffn[1], tag + ".ffn1")
        if getattr(m, "add", True):
            o = self.new_slot(); self.emit(OP_ADD, [a, f], o, []); f = o
        return f

    def c2psa(self, x, m, tag=""):
        y = self.ultralytics_conv(x, m.cv1, tag + ".cv1")
        c = m.c
        a = self.slice_ch(y, 0, c)
        b = self.slice_ch(y, c, 2 * c)
        b = self.block_seq(b, m.m, tag + ".m")
        cat = self.cat([a, b], 1)
        return self.ultralytics_conv(cat, m.cv2, tag + ".cv2")

    # ---- Detect head ----
    def detect(self, x_slots, det, tag=""):
        import torch.nn as nn
        assert getattr(det, "one2one_cv2", None) is None, "end2end/one2one 头不支持"
        nc = int(det.nc)
        reg_max = int(det.reg_max)
        strides = [int(s) for s in det.stride.detach().cpu().numpy().tolist()]
        box_slots, cls_slots = [], []
        for i, xi in enumerate(x_slots):
            b = self.block_seq(xi, det.cv2[i], f"{tag}.cv2[{i}]")
            box_slots.append(b)
            c = self.block_seq(xi, det.cv3[i], f"{tag}.cv3[{i}]")
            cls_slots.append(c)
        dfl_w = det.dfl.conv.weight.detach().cpu().numpy().astype(np.float32)
        w_dfl = self.add_conv_weight(dfl_w, None)
        o = self.new_slot()
        params = [nc, reg_max, strides[0], strides[1], strides[2], w_dfl]
        self.emit(OP_DETECT, box_slots + cls_slots, o, params)
        return o


def flatten(net):
    """net = ultralytics DetectionModel（.model 即顶层 Sequential）。"""
    import torch.nn as nn
    b = Builder()
    seq = net.model
    slots = {}          # 顶层索引 -> slot
    in_slot = b.new_slot()          # slot 0 = 网络输入（图像 [3,H,W]）
    b.input_slot = in_slot
    fmap = getattr(net, "f", None)   # 顶层 from 列表（若缺，用模块自身 .f）
    def resolve(j):
        jj = int(j)
        k = (i + jj) if jj < 0 else jj       # ultralytics 约定：负=相对，正=绝对索引
        return slots[k] if k >= 0 else in_slot
    for i, mod in enumerate(seq):
        f = getattr(mod, "f", None)
        if f is None and fmap is not None:
            f = fmap[i]
        if f is None:
            f = -1
        if isinstance(f, (list, tuple)):
            ins = [resolve(j) for j in f]
        else:
            ins = [resolve(f)]
        n = type(mod).__name__
        if n in ("Conv", "DWConv") or (hasattr(mod, "conv") and n not in
                ("C2f", "C3", "C3k2", "C3k", "SPPF", "C2PSA", "RepConv", "GhostConv")):
            out = b.ultralytics_conv(ins[0], mod, f"[{i}]{n}")
        elif n in ("C2f", "C3k2"):
            out = b.c2f_like(ins[0], mod, f"[{i}]{n}")
        elif n in ("C3", "C3k"):
            out = b.c3_like(ins[0], mod, f"[{i}]{n}")
        elif n == "SPPF":
            out = b.sppf(ins[0], mod, f"[{i}]SPPF")
        elif n == "C2PSA":
            out = b.c2psa(ins[0], mod, f"[{i}]C2PSA")
        elif n == "Upsample":
            sc = int(mod.scale_factor)
            o = b.new_slot(); b.emit(OP_UPSAMPLE, [ins[0]], o, [sc]); out = o
        elif n == "Concat":
            out = b.cat(ins, int(mod.d))
        elif n == "MaxPool2d":
            k = int(mod.kernel_size); s = int(mod.stride); p = int(mod.padding)
            o = b.new_slot(); b.emit(OP_MAXPOOL, [ins[0]], o, [k, s, p]); out = o
        elif n == "Detect":
            out = b.detect(ins, mod, f"[{i}]Detect")
        else:
            raise RuntimeError(f"unsupported top-level module {n} @[{i}]")
        slots[i] = out
    return b, slots


# ================================================================
# graph blob 编码
# ================================================================
def encode_graph(ops, n_slots):
    out = bytearray()
    out += struct.pack("<III", GRAPH_MAGIC, GRAPH_VERSION, len(ops))
    for op, ins, o, params in ops:
        out += struct.pack("<IIII", op, len(ins), len(params), o)
        for s in ins:
            out += struct.pack("<I", s)
        for p in params:
            out += struct.pack("<I", int(p))
    return bytes(out), n_slots


# ================================================================
# 量化（int8 per-channel 对称；与 collapse_priority_rule 一致）
# ================================================================
def quant_q8(w):
    """w [oc, ic, kh, kw] -> (int8 [oc,ic,kh,kw], scale[oc])"""
    amax = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
    amax = np.maximum(amax, 1e-10)
    s = amax / 127.0
    q = np.clip(np.round(w / s.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)


# ================================================================
# VQF v4 写盘
# ================================================================
def write_vqf(path, arch, tensors, family, graph_bytes):
    """tensors: list of (name, qtype, rows, cols, payload_bytes)"""
    dir_off = (HDR_BYTES + VQF_ALIGN - 1) & ~(VQF_ALIGN - 1)
    n = len(tensors)
    data_off = (dir_off + n * 64 + VQF_HEADER_BYTES - 1) & ~(VQF_HEADER_BYTES - 1)
    offs, off = [], data_off
    for t in tensors:
        offs.append(off)
        off = (off + len(t[4]) + VQF_ALIGN - 1) & ~(VQF_ALIGN - 1)
    data_end = off

    # ---- 头 ----
    hdr = bytearray(HDR_BYTES)
    struct.pack_into("<IIII", hdr, 0, VQF_MAGIC, VQF_VERSION_DET, VQF_FLAG_DET, n)
    arch_pad = arch + b"\x00" * (HDR_ARCH_LEN - len(arch))
    hdr[HDR_ARCH_OFF:HDR_ARCH_OFF + HDR_ARCH_LEN] = arch_pad
    struct.pack_into("<QQ", hdr, HDR_DATAOFF, data_off, data_end)

    # ---- 目录 ----
    directory = bytearray(n * 64)
    for i, (name, qt, rows, cols, payload) in enumerate(tensors):
        nb = name.encode("utf-8")[:31]
        directory[i * 64:i * 64 + len(nb)] = nb
        struct.pack_into("<IIIIQQ", directory, i * 64 + 32, qt, rows, cols, 0, offs[i], len(payload))

    # ---- 数据区 ----
    dat = bytearray(data_end - data_off)
    for i, t in enumerate(tensors):
        p = t[4]
        dat[offs[i] - data_off: offs[i] - data_off + len(p)] = p

    # ---- FNV-1a（dir 续 data） ----
    s = 0xCBF29CE484222325
    for buf in (bytes(directory), bytes(dat)):
        for byte in buf:
            s ^= byte
            s = (s * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    with open(path, "wb") as f:
        f.write(hdr)
        f.write(directory)
        f.write(b"\x00" * (data_off - dir_off - n * 64))
        f.write(dat)
        f.write(struct.pack("<Q", s))
    return data_end, s


def build_arch(family, input_hw, nc, strides, reg_max, n_ops, graph_bytes,
               conf=0.25, iou=0.7):
    # 12 u32 + 2 f32 + 1 u32 + 8 u32 = 92B
    return struct.pack("<12I", 0, family, input_hw[1], input_hw[0], nc,
                       strides[0], strides[1], strides[2],
                       1, reg_max, n_ops, 0) + \
           struct.pack("<ffI", conf, iou, graph_bytes) + \
           struct.pack("<8I", *([0] * 8))


# ================================================================
# 主流程
# ================================================================
def family_of(net):
    names = {type(m).__name__ for m in net.model}
    if "C3k2" in names or "C2PSA" in names:
        return FAM_V11, "v11"
    if "C2f" in names:
        return FAM_V8, "v8"
    if "C3" in names:
        return FAM_V5, "v5"
    return None, "?"


def to_bytes(np_arr):
    return np.ascontiguousarray(np_arr, dtype=np.float32).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out")
    ap.add_argument("--f16", action="store_true", help="权重存 F16")
    ap.add_argument("--q8", action="store_true", help="权重 per-channel int8")
    ap.add_argument("--dump-graph", action="store_true")
    ap.add_argument("--check", action="store_true", help="写盘后读回并对比 torch golden")
    args = ap.parse_args()

    import torch

    # 直接 torch.load 读 checkpoint（避免 YOLO() 包装触发 yolov5nu 自动下载）
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    net = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.float().eval()          # checkpoint 为 FP16，统一到 F32
    fam, famn = family_of(net)
    if fam is None:
        sys.exit("无法识别 YOLO 家族")
    b, slots = flatten(net)
    nm = net.names
    names = [nm[i] for i in range(len(nm))] if isinstance(nm, dict) else list(nm)

    gb, nslots = encode_graph(b.ops, b.nslots)
    strides = [int(s) for s in net.stride.detach().cpu().numpy().tolist()]
    det = net.model[-1]
    arch = build_arch(fam, (640, 640), int(det.nc), strides, int(det.reg_max),
                      len(b.ops), len(gb))

    print(f"[yolo2vqf] family={famn} ops={len(b.ops)} slots={nslots} "
          f"convs={b.convs} graph={len(gb)}B nc={det.nc} reg_max={det.reg_max} strides={strides}")

    if args.dump_graph:
        for i, (op, ins, o, params) in enumerate(b.ops):
            print(f"  {i:3d} op={op:2d} out=s{o:<3d} in={ins} params={params}")
        return

    # ---- 组装张量表 ----
    tensors = []
    tensors.append(("graph", QT_BLOB, len(b.ops), 0, gb))
    tensors.append(("names", QT_BLOB, len(names), 0, ("\n".join(names)).encode("utf-8")))
    for name, arr in b.weights:
        if arr.ndim == 4:                       # conv 权重 [oc,ic,kh,kw]
            rows, cols = int(arr.shape[0]), int(np.prod(arr.shape[1:]))
        else:                                   # bias / norm [n]
            rows, cols = int(arr.size), 1
        if args.q8 and arr.ndim == 4:
            q, s = quant_q8(arr)
            tensors.append((name, QT_Q8, rows, cols, q.tobytes()))
            tensors.append((name[:-2] + "_s", QT_F32, rows, 1, s.tobytes()))
        elif args.f16 and arr.ndim == 4:
            tensors.append((name, QT_F16, rows, cols,
                            arr.astype(np.float16).tobytes()))
        else:
            tensors.append((name, QT_F32, rows, cols, to_bytes(arr)))

    if not args.out:
        sys.exit("需要 --out 或 --dump-graph")
    data_end, fsum = write_vqf(args.out, arch, tensors, fam, len(gb))
    print(f"[yolo2vqf] wrote {args.out}: {len(tensors)} tensors, "
          f"{data_end/1048576:.2f} MB, FNV={fsum:016x}")

    if args.check:
        check(args.pt, args.out)


# ================================================================
# 读回自检（numpy 执行 VQF v4 graph，对比 torch golden）
# ================================================================
def read_vqf(path):
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, flags, n = struct.unpack_from("<IIII", blob, 0)
    assert magic == VQF_MAGIC and version == VQF_VERSION_DET, "not a VQF v4 file"
    data_off, file_len = struct.unpack_from("<QQ", blob, HDR_DATAOFF)
    arch = blob[HDR_ARCH_OFF:HDR_ARCH_OFF + HDR_ARCH_LEN]
    (task, family, iw, ih, nc, s0, s1, s2, head, reg_max, n_ops, n_anchors) = \
        struct.unpack_from("<12I", arch, 0)
    conf, iou, graph_bytes = struct.unpack_from("<ffI", arch, 48)
    tensors = {}
    for i in range(n):
        base = 448 + i * 64
        nm = blob[base:base + 32].split(b"\x00")[0].decode("utf-8")
        qt, rows, cols, _rsv, off, byts = struct.unpack_from("<IIIIQQ", blob, base + 32)
        tensors[nm] = (qt, rows, cols, blob[off:off + byts])
    return dict(magic=magic, version=version, flags=flags, n=n, data_off=data_off,
                file_len=file_len, family=family, iw=iw, ih=ih, nc=nc,
                strides=[s0, s1, s2], reg_max=reg_max, n_ops=n_ops, conf=conf,
                iou=iou, tensors=tensors)


def check(pt, vqf):
    """读回自检：委托 yolo_vqf_run.py（numpy 执行 VQF v4 graph 对比 torch）。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import yolo_vqf_run
    return yolo_vqf_run.run_check(pt, vqf)


if __name__ == "__main__":
    main()
