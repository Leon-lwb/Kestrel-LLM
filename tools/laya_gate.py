"""laya_gate.py - Engine vs Python golden for the Laya decision model.

Runs `vllm_kestrel --laya-pack <pack> --laya-cases <cases>` and compares the
per-option scorer logits to the golden npz produced by tools/laya_ref.py.

Frozen criteria (K-LAYA-001):
  - per-option logits: max|Δ| <= 5e-3  (absolute)  AND  <= 1e-3 (relative to |golden|+1)
  - argmax (after official per-(qtype,#options) temperature + softmax): identical
  - no NaN/Inf in engine logits

Usage:
  python tools/laya_gate.py [engine_exe] [pack_dir] [golden.npz]
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

ENGINE = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\Administrator\vllm_x64_build\vllm_kestrel_x64.exe"
PACK = sys.argv[2] if len(sys.argv) > 2 else r"d:\项目\NewVLLM\build_laya"
GOLD = sys.argv[3] if len(sys.argv) > 3 else r"d:\项目\NewVLLM\build_laya_golden.npz"
CASES = GOLD + ".cases.txt"

QT = {0: "choice", 1: "score", 2: "noul"}
ABS_TOL = 5e-3
REL_TOL = 1e-3


def temp_bucket(qt, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QT[qt], size)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    cfg = json.load(open(os.path.join(PACK, "rl_agent_config.json"), encoding="utf-8"))
    gold = np.load(GOLD)

    print("[gate] running engine ...", flush=True)
    r = subprocess.run([ENGINE, "--laya-pack", PACK, "--laya-cases", CASES],
                       capture_output=True, timeout=3600)
    out = r.stdout.decode("utf-8", "replace")
    eng = {}
    for m in re.finditer(r"\[LAYA\] q=(\d+) n=(\d+) nm=(\d+) logits=\[([^\]]*)\]", out):
        q = int(m.group(1))
        eng[q] = np.array([float(x) for x in m.group(4).split(",")], dtype=np.float64)
    if not eng:
        print("[gate] FAIL: no engine output (rc=%d)" % r.returncode)
        print(out[-2000:])
        return 1

    order = [k[:-len("_logits")] for k in gold.files if k.endswith("_logits")]
    # establish engine-q -> golden-key mapping by matching the ids in the cases file
    cases = []
    for ln in open(CASES).read().split("\n"):
        t = ln.split()
        if not t:
            continue
        qt = int(t[0]); n = int(t[1]); ids = [int(x) for x in t[2:2 + n]]
        cases.append((qt, np.array(ids, dtype=np.int64)))
    by_ids = {}
    for key in order:
        by_ids[gold[key + "_ids"].tobytes()] = key

    npass = 0
    for i in range(len(cases)):
        key = by_ids.get(cases[i][1].tobytes())
        if key is None:
            print("FAIL[q=%d] no golden key for these ids" % i)
            continue
        gl = gold[key + "_logits"].astype(np.float64)
        qtype = int(gold[key + "_qtype"][0])
        k = len(gl)
        el = eng.get(i)
        if el is None or len(el) != k:
            print("FAIL[%s] engine missing/mismatch (got %s)" % (key, None if el is None else len(el)))
            continue
        ad = float(np.max(np.abs(el - gl)))
        rd = float(np.max(np.abs(el - gl) / (np.abs(gl) + 1.0)))
        t = cfg.get("temperature_by_options", {}).get(
            temp_bucket(qtype, k), cfg.get("temperature", [1, 1, 1])[qtype])
        pa, pb = softmax(el / t), softmax(gl / t)
        aa, ab = int(pa.argmax()), int(pb.argmax())
        bad = (not np.all(np.isfinite(el))) or ad > ABS_TOL or rd > REL_TOL or aa != ab
        npass += (not bad)
        print("%s[%-14s] type=%-6s k=%d  maxabs=%.2e maxrel=%.2e  argmax eng=%d gold=%d"
              % ("FAIL" if bad else "PASS", key, QT[qtype], k, ad, rd, aa, ab))
        if bad:
            print("   eng =%s" % np.round(el, 5).tolist())
            print("   gold=%s" % np.round(gl, 5).tolist())
    print("laya_gate: %d/%d PASS" % (npass, len(cases)))
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
