#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_j2_check.py — J2 检测等价门：C 引擎 vs ultralytics（同权重、同图）。

对每张图：C 引擎 CLI（--yolo-image）输出检测框 JSON；ultralytics predict 同图。
判据：按类别 + IoU≥0.5 配对，统计配对率与 score/框偏差。

用法:
  python yolo_j2_check.py --exe <exe> --vqf <x.vqf> --pt <x.pt> --images a.jpg b.jpg
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--vqf", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    a = ap.parse_args()

    from yolo_vqf_run import load_net
    import torch
    net = load_net(a.pt)
    # ultralytics 参考：直接 torch.load 的模型不支持 predict 包装，用 YOLO
    from ultralytics import YOLO
    y = YOLO(a.pt)

    total = matched = 0
    for img in a.images:
        r = subprocess.run([a.exe, "--yolo-model", a.vqf, "--yolo-image", img,
                            "--yolo-conf", str(a.conf), "--yolo-iou", str(a.iou)],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(f"[{img}] C FAILED: {r.stderr[-200:]}"); continue
        C = json.loads(line[-1])["detections"]
        ref = y.predict(img, conf=a.conf, iou=a.iou, verbose=False)[0]
        R = []
        if ref.boxes is not None and len(ref.boxes):
            for i in range(len(ref.boxes)):
                R.append((int(ref.boxes.cls[i]), float(ref.boxes.conf[i]),
                          [float(v) for v in ref.boxes.xyxy[i]]))
        print(f"==== {img.split(chr(92))[-1]}  C={len(C)} ref={len(R)}")
        used = set()
        for c in C:
            total += 1
            best, bi = 0.0, -1
            for i, rr in enumerate(R):
                if i in used or rr[0] != c["cls"]:
                    continue
                v = iou(c["box"], rr[2])
                if v > best: best, bi = v, i
            if bi >= 0 and best >= 0.5:
                matched += 1; used.add(bi)
                print(f"   MATCH cls={c['cls']:3d} iou={best:.3f} "
                      f"score C={c['score']:.3f} ref={R[bi][1]:.3f} "
                      f"boxΔ={max(abs(c['box'][k]-R[bi][2][k]) for k in range(4)):.1f}px")
            else:
                print(f"   MISS  cls={c['cls']:3d} score={c['score']:.3f} "
                      f"box={[round(v,1) for v in c['box']]}")
        for i, rr in enumerate(R):
            if i not in used:
                print(f"   REF-ONLY cls={rr[0]} score={rr[1]:.3f}")
    print(f"[J2] matched {matched}/{total} (C detections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
