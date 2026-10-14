#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_vqf_run.py — 读回 VQF v4（检测）并执行，对比 torch golden。

这是 M0 的"数值门"：证明**自有格式**里编码的图 + 权重能完整复现
ultralytics torch 前向（同权重、同输入）。C 引擎实现为同一套图/权重的
另一执行器，本脚本给出它应当匹配的 golden。

用法:
  python yolo_vqf_run.py --pt yolov8n.pt --vqf yolov8n.vqf [--seed 0] [--h 640 --w 640]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_to_vqf import (  # noqa: E402
    read_vqf, OP_CONV, OP_ADD, OP_CONCAT, OP_MAXPOOL, OP_UPSAMPLE,
    OP_SIGMOID, OP_MUL, OP_SLICE, OP_SOFTMAX, OP_ATTN_PSA, OP_DETECT,
    ACT_SILU, QT_F32, QT_F16, QT_Q8, GRAPH_MAGIC)


# ---------------- 权重 ----------------
def load_conv_w(T, cid, kh, kw):
    qt, rows, cols, payload = T[f"c{cid:04d}_w"]
    oc = rows
    ic = cols // (kh * kw)
    if qt == QT_F32:
        w = np.frombuffer(payload, np.float32).reshape(oc, ic, kh, kw)
    elif qt == QT_F16:
        w = np.frombuffer(payload, np.float16).reshape(oc, ic, kh, kw).astype(np.float32)
    elif qt == QT_Q8:
        q = np.frombuffer(payload, np.int8).reshape(oc, ic, kh, kw).astype(np.float32)
        s = np.frombuffer(T[f"c{cid:04d}_s"][3], np.float32).reshape(oc, 1, 1, 1)
        w = q * s
    else:
        raise RuntimeError(f"bad qtype {qt} for {cid}")
    return np.ascontiguousarray(w, dtype=np.float32)


def load_bias(T, cid):
    n = f"c{cid:04d}_b"
    if n not in T:
        return None
    return np.frombuffer(T[n][3], np.float32).astype(np.float32)


# ---------------- 算子 ----------------
def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def conv2d(x, w, b, sh, sw, ph, pw, groups, act):
    C, H, W = x.shape
    oc, icg, kh, kw = w.shape
    oh = (H + 2 * ph - kh) // sh + 1
    ow = (W + 2 * pw - kw) // sw + 1
    xp = np.zeros((C, H + 2 * ph, W + 2 * pw), dtype=np.float32)
    xp[:, ph:ph + H, pw:pw + W] = x
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(C, oh, ow, kh, kw),
        strides=(xp.strides[0], xp.strides[1] * sh, xp.strides[2] * sw,
                 xp.strides[1], xp.strides[2]))
    if groups == 1:
        out = np.einsum("cxykl,ockl->oxy", cols, w, optimize=True)
    else:
        ocg = oc // groups
        icpg = C // groups
        out = np.empty((oc, oh, ow), np.float32)
        for g in range(groups):
            cg = cols[g * icpg:(g + 1) * icpg]
            wg = w[g * ocg:(g + 1) * ocg]
            out[g * ocg:(g + 1) * ocg] = np.einsum("cxykl,ockl->oxy", cg, wg, optimize=True)
    if b is not None:
        out = out + b.reshape(-1, 1, 1)
    if act == ACT_SILU:
        out = silu(out)
    return out


def maxpool2d(x, k, s, p):
    C, H, W = x.shape
    oh = (H + 2 * p - k) // s + 1
    ow = (W + 2 * p - k) // s + 1
    xp = np.full((C, H + 2 * p, W + 2 * p), -np.inf, np.float32)
    xp[:, p:p + H, p:p + W] = x
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(C, oh, ow, k, k),
        strides=(xp.strides[0], xp.strides[1] * s, xp.strides[2] * s,
                 xp.strides[1], xp.strides[2]))
    return cols.max(axis=(3, 4))


def upsample2(x, sc):
    return np.repeat(np.repeat(x, sc, axis=1), sc, axis=2)


def softmax(x, axis):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def attn_psa(x, nh, hd, kd, w_qkv, b_qkv, w_proj, b_proj, w_pe, b_pe):
    C, H, W = x.shape
    N = H * W
    qkv = conv2d(x, w_qkv, b_qkv, 1, 1, 0, 0, 1, 0)         # [nh*(2kd+hd), H, W]
    qkv = qkv.reshape(nh, 2 * kd + hd, N)
    q = qkv[:, :kd]
    k = qkv[:, kd:2 * kd]
    v = qkv[:, 2 * kd:]
    scale = kd ** -0.5
    att = softmax(np.einsum("hdq,hdk->hqk", q * scale, k, optimize=True), axis=-1)
    o = np.einsum("hdj,hij->hdi", v, att, optimize=True).reshape(C, H, W)
    o = o + conv2d(v.reshape(C, H, W), w_pe, b_pe, 1, 1, 1, 1, C, 0)
    return conv2d(o, w_proj, b_proj, 1, 1, 0, 0, 1, 0)


def detect(boxes, clss, shapes, strides, reg_max, nc):
    """复刻 Detect._inference：dfl -> dist2bbox(xywh=True) -> *stride -> cat(sigmoid(scores))。
    注意 decode_bboxes 默认 xywh=True（中心+宽高），非 xyxy。"""
    A = sum(h * w for (h, w) in shapes)
    box = np.concatenate([b.reshape(4 * reg_max, -1) for b in boxes], axis=1)
    sco = np.concatenate([c.reshape(nc, -1) for c in clss], axis=1)
    an, st = [], []
    for i, (h, w) in enumerate(shapes):
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        an.append(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5]))   # [2, h*w]
        st.append(np.full(h * w, strides[i], np.float32))
    anchor = np.concatenate(an, axis=1)
    stride = np.concatenate(st)
    d = softmax(box.reshape(4, reg_max, A), axis=1)
    d = (d * np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)).sum(axis=1)  # [4,A]
    lt, rb = d[:2], d[2:]
    x1y1 = anchor - lt
    x2y2 = anchor + rb
    c_xy = (x1y1 + x2y2) * 0.5          # dist2bbox xywh=True
    wh = x2y2 - x1y1
    dbox = np.concatenate([c_xy, wh], axis=0) * stride
    return np.concatenate([dbox, 1.0 / (1.0 + np.exp(-sco))], axis=0)


# ---------------- 执行 ----------------
def execute(v, x0):
    T = v["tensors"]
    gb = T["graph"][3]
    magic, gver, nops = struct.unpack_from("<III", gb, 0)
    assert magic == GRAPH_MAGIC, "bad graph magic"
    p = 12
    S = {}
    S[0] = x0
    order = []
    for _ in range(nops):
        op, nin, nparam, o = struct.unpack_from("<IIII", gb, p); p += 16
        ins = list(struct.unpack_from("<%dI" % nin, gb, p)); p += 4 * nin
        pr = list(struct.unpack_from("<%dI" % nparam, gb, p)); p += 4 * nparam
        order.append((op, ins, o, pr))
        if op == OP_CONV:
            cid, hb, sh, sw, ph, pw, gp, act, kh, kw = pr
            w = load_conv_w(T, cid, kh, kw)
            b = load_bias(T, cid) if hb else None
            S[o] = conv2d(S[ins[0]], w, b, sh, sw, ph, pw, gp, act)
        elif op == OP_ADD:
            S[o] = S[ins[0]] + S[ins[1]]
        elif op == OP_CONCAT:
            S[o] = np.concatenate([S[i] for i in ins], axis=0)
        elif op == OP_MAXPOOL:
            S[o] = maxpool2d(S[ins[0]], pr[0], pr[1], pr[2])
        elif op == OP_UPSAMPLE:
            S[o] = upsample2(S[ins[0]], pr[0])
        elif op == OP_SIGMOID:
            S[o] = 1.0 / (1.0 + np.exp(-S[ins[0]]))
        elif op == OP_MUL:
            S[o] = S[ins[0]] * S[ins[1]]
        elif op == OP_SLICE:
            _ax, a, bnd = pr
            S[o] = S[ins[0]][a:bnd]
        elif op == OP_SOFTMAX:
            S[o] = softmax(S[ins[0]], axis=pr[0])
        elif op == OP_ATTN_PSA:
            nh, hd, kd, wq, wp, wpe = pr
            S[o] = attn_psa(S[ins[0]], nh, hd, kd,
                            load_conv_w(T, wq, 1, 1), load_bias(T, wq),
                            load_conv_w(T, wp, 1, 1), load_bias(T, wp),
                            load_conv_w(T, wpe, 3, 3), load_bias(T, wpe))
        elif op == OP_DETECT:
            nc, reg_max, s0, s1, s2, _wdfl = pr
            bx = [S[i] for i in ins[:3]]
            cl = [S[i] for i in ins[3:6]]
            shapes = [(b.shape[1], b.shape[2]) for b in bx]
            S[o] = detect(bx, cl, shapes, [s0, s1, s2], reg_max, nc)
        else:
            raise RuntimeError(f"unknown op {op}")
    return S, order


def load_net(pt):
    """直接 torch.load（避免 YOLO() 触发 yolov5nu 自动下载）；checkpoint 为 FP16，转 F32。"""
    import torch
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    net = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.float().eval()
    return net


def run_check(pt, vqf, seed=0, hw=(640, 640)):
    import torch

    v = read_vqf(vqf)
    print(f"[run] VQF v4: family={v['family']} nc={v['nc']} reg_max={v['reg_max']} "
          f"strides={v['strides']} tensors={v['n']}")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3, hw[0], hw[1])).astype(np.float32)

    net = load_net(pt)
    with torch.no_grad():
        out = net(torch.from_numpy(x)[None])
    golden = out[0].numpy()[0] if isinstance(out, (tuple, list)) else out.numpy()[0]
    print(f"[run] torch golden shape={golden.shape}")

    S, order = execute(v, x)
    final_slot = order[-1][2]
    mine = S[final_slot]
    print(f"[run] vqf    output shape={mine.shape}")

    d = np.abs(mine - golden)
    scale = float(np.abs(golden).max()) + 1e-9
    print(f"[run] max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  rel={d.max()/scale:.3e}")
    ok = (d.max() / scale) < 1e-4
    print(f"[run] NUMERIC GATE: {'PASS' if ok else 'FAIL'} (rel tol 1e-4)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--vqf", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=640)
    a = ap.parse_args()
    sys.exit(run_check(a.pt, a.vqf, a.seed, (a.h, a.w)))


if __name__ == "__main__":
    main()
