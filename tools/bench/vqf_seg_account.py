#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vqf_seg_account.py — VQF 权重驻留「静态字节账」测算（G0 立项门专用）

用途
    不加载模型、不跑推理，只解析 VQF 的 header + 目录，按引擎的 seg 判定规则
    把数据区切成「可驱逐段」与「常驻段」，给出：
      - 每层可驱逐字节（per-layer）
      - 常驻（不可驱逐）字节
      - keep=1/2/4/8 下的常驻量上界与相对全驻留的倍数
      - 三个候选下界（lm 可驱逐 / lm+embed 可驱逐）

为什么是「上界」
    measured RSS <= 静态常驻量，因为未被触碰的张量（如仅末位需要的 lm_head，
    仅首 token 需要的 embed）不会缺页进来。故本工具给的是**保守(偏悲观)上界**，
    用于「否掉方向」而非「预测收益」——
    见公理 perf_axiom_candidate_root_cause_upper_bound_v1。

判定规则镜像（必须与源码一致）
    src/model/vqf.c  L744-L755
      可驱逐名: attn_norm / ffn_norm / q_norm / k_norm
                q8_*（除 q8_lm）/ q4_*（除 q4_lm）/ x8_*
      几何门  : rows % nl == 0 且 rows/nl > 0 且 bytes % nl == 0
    src/model/vqf.c  L1501-L1502
      常驻段  : 0..keep 层 + token_embed / final_norm / lm / vision 等

format 参考
    include/model/vqf_format.h —— VQFHeader 448B(+dir 到 4096) / VQFTensor 64B
    data_offset 新头@200 file_len@208；旧(41 字段)头 @184/@192

用法
    python vqf_seg_account.py                     # 扫描 <repo>/modl 下全部 *.vqf
    python vqf_seg_account.py --json out.json     # 另存机器可读结果
    python vqf_seg_account.py path1.vqf path2.vqf
    python vqf_seg_account.py --dir <某个目录>
"""

import argparse
import glob
import json
import os
import struct
import sys

VQF_MAGIC = 0x57465146  # "VQFW"
DIR_OFF = 448
TENSOR_SZ = 64
EVICT_EXACT = {"attn_norm", "ffn_norm", "q_norm", "k_norm"}
RESIDENT_LM = {"q8_lm", "q4_lm"}
RESIDENT_OTHER = {"token_embed", "final_norm", "moe_router"}
MB = 1048576.0


def _model_dirs():
    """默认扫描目录候选：<repo>/modl、<repo>/../modl、<repo>/models。<repo> = 本脚本上两级。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))          # .../GitHupSRC
    parent = os.path.dirname(root)                          # .../NewVLLM
    return [os.path.join(root, "modl"), os.path.join(parent, "modl"),
            os.path.join(root, "models")]


def parse_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4096)
    if len(head) < 4096:
        raise ValueError("file smaller than one 4K header block")
    magic, version, flags, n_tensors = struct.unpack_from("<IIII", head, 0)
    if magic != VQF_MAGIC:
        raise ValueError("bad magic 0x%08x (not a VQF)" % magic)
    # 逐个候选布局验证自洽性（data_offset 必须 4096 对齐且落在文件内）
    resolved = None
    for doff_off, flen_off, layout in ((200, 208, "new"), (184, 192, "legacy")):
        doff, flen = struct.unpack_from("<QQ", head, doff_off)
        if doff % 4096 == 0 and 0 < doff < size and doff < flen <= size:
            resolved = (doff, flen, layout)
            break
    if resolved is None:
        raise ValueError("cannot resolve data_offset/file_len (unknown header layout)")
    doff, flen, layout = resolved

    (dim, nl, n_heads, n_kv_heads, head_dim, ffn_dim, vocab_size) = \
        struct.unpack_from("<IIIIIII", head, 16)
    has_q_norm = struct.unpack_from("<I", head, 64)[0]
    # MoE 尾部只在「新头 + v3/FLAG_MOE」存在；legacy 头视为 0（引擎归一化同此）
    if layout == "new" and (version == 3 or (flags & (1 << 9))):
        n_experts, moe_ffn = struct.unpack_from("<II", head, 180)
    else:
        n_experts, moe_ffn = 0, 0
    moe_geom = bool(n_experts > 0 and moe_ffn > 0 and n_experts * moe_ffn == ffn_dim)

    ents = []
    with open(path, "rb") as f:
        f.seek(DIR_OFF)
        raw = f.read(n_tensors * TENSOR_SZ)
    for i in range(min(n_tensors, len(raw) // TENSOR_SZ)):
        e = raw[i * TENSOR_SZ:(i + 1) * TENSOR_SZ]
        name = e[0:32].split(b"\x00", 1)[0].decode("ascii", "replace")
        qtype, rows, cols = struct.unpack_from("<III", e, 32)
        toff, tbytes = struct.unpack_from("<QQ", e, 48)
        ents.append(dict(name=name, qtype=qtype, rows=rows, cols=cols,
                         offset=toff, bytes=tbytes))
    return dict(path=path, size=size, version=version, flags=flags,
                n_tensors=n_tensors, dim=dim, nl=nl, ffn=ffn_dim,
                vocab=vocab_size, has_q_norm=has_q_norm,
                n_experts=n_experts, moe_ffn=moe_ffn, moe_geom=moe_geom,
                data_offset=doff, file_len=flen, hdr_layout=layout, ents=ents)


def seg_eligible(te, nl, moe_geom):
    """返回段字典（可驱逐）或 None（常驻）。镜像 vqf.c L744-L755。"""
    nm, rows, tbytes = te["name"], te["rows"], te["bytes"]
    evict = (nm in EVICT_EXACT
             or (nm.startswith("q8_") and nm != "q8_lm")
             or (nm.startswith("q4_") and nm != "q4_lm")
             or nm.startswith("x8_"))
    if not evict:
        return None
    if nl <= 0 or rows % nl != 0 or rows // nl == 0 or tbytes % nl != 0:
        return None
    kind = 0
    if moe_geom:
        if nm in ("q4_gate", "q4_up", "q8_gate", "q8_up"):
            kind = 1
        elif nm in ("q4_down", "q8_down"):
            kind = 2
    return dict(name=nm, step=tbytes // nl, kind=kind, is_expert=int(bool(kind)),
                off=te["offset"])


def account(r):
    nl = r["nl"]
    segs, resident = [], []
    for te in r["ents"]:
        s = seg_eligible(te, nl, r["moe_geom"])
        (segs if s else resident).append(s if s else te)
    per_layer = sum(s["step"] for s in segs)
    total = r["file_len"] - r["data_offset"]
    res_bytes = total - per_layer * nl

    lm = sum(t["bytes"] for t in resident if t["name"] in RESIDENT_LM)
    emb = sum(t["bytes"] for t in resident if t["name"] == "token_embed")
    other = res_bytes - lm - emb

    def ratio(floor):
        return (total / floor) if floor > 0 else 0.0

    floors = {
        "full": total,
        "keep1_baseline": res_bytes + per_layer,
        "keep1_lm_evict": res_bytes - lm + per_layer,
        "keep1_lm_embed_evict": res_bytes - lm - emb + per_layer,
    }
    return dict(
        name=os.path.basename(os.path.dirname(r["path"])) + "/" + os.path.basename(r["path"]),
        path=r["path"], file_mb=r["size"] / MB, version=r["version"],
        flags=r["flags"], hdr_layout=r["hdr_layout"],
        dim=r["dim"], n_layers=nl, ffn=r["ffn"], vocab=r["vocab"],
        n_tensors=r["n_tensors"], moe_geom=r["moe_geom"], n_experts=r["n_experts"],
        data_mb=total / MB, per_layer_mb=per_layer / MB, n_seg=len(segs),
        n_expert_seg=sum(s["is_expert"] for s in segs),
        resident_mb=res_bytes / MB, resident_pct=100.0 * res_bytes / total if total else 0.0,
        resident_lm_mb=lm / MB, resident_embed_mb=emb / MB, resident_other_mb=other / MB,
        floors_mb={k: v / MB for k, v in floors.items()},
        ratios={k: ratio(v) for k, v in floors.items()},
        keep_curve={k: (res_bytes + per_layer * min(k, nl)) / MB for k in (1, 2, 4, 8)},
        keep_ratio={k: ratio(res_bytes + per_layer * min(k, nl)) for k in (1, 2, 4, 8)},
        segs=sorted(segs, key=lambda x: -x["step"]),
        segs_by_off=sorted(segs, key=lambda x: x["off"]),
        resident_top=sorted((dict(name=t["name"], mb=t["bytes"] / MB) for t in resident),
                            key=lambda x: -x["mb"])[:8],
    )


def layer_locality(a, layer):
    """第 layer 层需要触碰的读取区间 vs 其跨度 —— 量化当前布局的「散」程度。

    局部性 = 有效字节 / 首尾跨度。layer-major 连续布局下该值 = 1.0。
    """
    nl = a["n_layers"]
    if not (0 <= layer < nl):
        return None
    rngs = []
    for s in a["segs_by_off"]:
        lo = s["off"] + layer * s["step"]
        rngs.append((lo, s["step"], s["name"]))
    rngs.sort()
    lo0 = rngs[0][0]
    hi1 = max(r[0] + r[1] for r in rngs)
    useful = sum(r[1] for r in rngs)
    span = hi1 - lo0
    return dict(layer=layer, n_ranges=len(rngs), useful_mb=useful / MB,
                span_mb=span / MB, locality=(useful / span) if span else 0.0,
                ranges=[(r[2], r[0], r[1]) for r in rngs])


def render(a):
    L = []
    W = L.append
    W("=" * 76)
    W("MODEL   %s" % a["name"])
    W("  file    %s" % a["path"])
    W("  size    %.1f MB  version=%d flags=0x%x hdr=%s"
      % (a["file_mb"], a["version"], a["flags"], a["hdr_layout"]))
    W("  arch    dim=%d nl=%d ffn=%d vocab=%d  moe_geom=%s(ne=%d)"
      % (a["dim"], a["n_layers"], a["ffn"], a["vocab"],
         a["moe_geom"], a["n_experts"]))
    W("  tensors %d total / %d evictable segs (%d expert)"
      % (a["n_tensors"], a["n_seg"], a["n_expert_seg"]))
    W("  data    %.2f MB   per-layer %.2f MB (x%d = %.2f MB)"
      % (a["data_mb"], a["per_layer_mb"], a["n_layers"],
         a["per_layer_mb"] * a["n_layers"]))
    W("  resident %.2f MB (%.1f%% of data) = embed %.1f + lm %.1f + 其它 %.1f"
      % (a["resident_mb"], a["resident_pct"], a["resident_embed_mb"],
         a["resident_lm_mb"], a["resident_other_mb"]))
    W("  -- keep 曲线（常驻量 / 相对全驻留倍数）--")
    for k in (1, 2, 4, 8):
        W("     keep=%-2d  %9.2f MB   %.2fx" % (k, a["keep_curve"][k], a["keep_ratio"][k]))
    W("  -- 候选下界（G0 上界用）--")
    W("     keep=1 现状(仅层可驱逐)      %9.2f MB   %.2fx"
      % (a["floors_mb"]["keep1_baseline"], a["ratios"]["keep1_baseline"]))
    W("     keep=1 若 lm 段可驱逐        %9.2f MB   %.2fx"
      % (a["floors_mb"]["keep1_lm_evict"], a["ratios"]["keep1_lm_evict"]))
    W("     keep=1 若 lm+embed 均可驱逐  %9.2f MB   %.2fx"
      % (a["floors_mb"]["keep1_lm_embed_evict"], a["ratios"]["keep1_lm_embed_evict"]))
    W("  -- 可驱逐段 --")
    for s in a["segs"]:
        W("     %-12s %8.2f MB/层 kind=%d%s"
          % (s["name"], s["step"] / MB, s["kind"], " EXPERT" if s["is_expert"] else ""))
    W("  -- 常驻明细 top --")
    for t in a["resident_top"]:
        W("     %-14s %8.2f MB" % (t["name"], t["mb"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="VQF 文件路径")
    ap.add_argument("--dir", help="扫描该目录下所有 *.vqf")
    ap.add_argument("--json", help="把结果另存为 JSON")
    ap.add_argument("--out", help="把报告另存为 UTF-8 文本（Windows 控制台代码页会乱码，取证用这个）")
    ap.add_argument("--locality", type=int, metavar="L",
                    help="打印第 L 层需触碰的读取区间与局部性指标（评估「改布局」收益用）")
    args = ap.parse_args()

    targets = list(args.paths)
    if args.dir:
        targets += sorted(glob.glob(os.path.join(args.dir, "**", "*.vqf"), recursive=True))
    if not targets:
        for d in _model_dirs():
            targets += sorted(glob.glob(os.path.join(d, "**", "*.vqf"), recursive=True))
        # 去重（同一文件可能被多个候选目录覆盖）
        seen, uniq = set(), []
        for t in targets:
            k = os.path.normcase(os.path.abspath(t))
            if k not in seen:
                seen.add(k)
                uniq.append(t)
        targets = uniq
    if not targets:
        print("no .vqf found", file=sys.stderr)
        return 2

    results, texts = [], []
    for p in targets:
        try:
            results.append(account(parse_header(p)))
            texts.append(render(results[-1]))
        except Exception as ex:  # noqa: BLE001 — 工具：逐文件隔离失败
            texts.append("SKIP %s : %s" % (p, ex))
    out = "\n\n".join(texts)
    if args.locality is not None:
        L = []
        for a in results:
            loc = layer_locality(a, args.locality)
            if loc:
                L.append("LAYER-LOCALITY  %s  layer=%d" % (a["name"], loc["layer"]))
                L.append("  ranges=%d  useful=%.2f MB  span=%.2f MB  locality=%.3f"
                         % (loc["n_ranges"], loc["useful_mb"], loc["span_mb"], loc["locality"]))
                L.append("  (locality=1.0 表示该层数据完全连续；越小越散)")
        out += "\n\n" + "\n".join(L)
    print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print("\n[report] %s (%d models)" % (args.out, len(results)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n[json] %s (%d models)" % (args.json, len(results)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
