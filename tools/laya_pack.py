"""laya_pack.py - Pack the Laya (English root) safetensors into a flat F16 blob
+ a text manifest the C engine can mmap/read.

Layout:
  <out>/laya_f16.bin     : concatenated raw tensor bytes (fp16, row-major as in safetensors)
  <out>/laya_manifest.txt: one line per tensor:
                             <name> <ndim> <d0> <d1> ... <dtype> <offset> <nbytes>

Usage: python tools/laya_pack.py [<model_dir>] [<out_dir>]
"""
import json
import os
import shutil
import sys

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "d:/项目/NewVLLM/modl/convaiinnovations--laya/snapshots/master"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "d:/项目/NewVLLM/build_laya"


def main():
    from safetensors import safe_open
    os.makedirs(OUT_DIR, exist_ok=True)
    src = os.path.join(MODEL_DIR, "model.safetensors")
    bin_path = os.path.join(OUT_DIR, "laya_f16.bin")
    man_path = os.path.join(OUT_DIR, "laya_manifest.txt")

    lines = []
    off = 0
    with safe_open(src, framework="pt") as f, open(bin_path, "wb") as out:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            shp = list(t.shape)
            dt = str(t.dtype).replace("torch.", "").upper()
            t = t.contiguous()
            raw = t.view(__import__("torch").uint8).numpy().tobytes()
            nbytes = len(raw)
            lines.append("%s %d %s %s %d %d" %
                         (k, len(shp), " ".join(str(s) for s in shp), dt, off, nbytes))
            out.write(raw)
            off += nbytes

    with open(man_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # copy configs + tokenizer sources for the engine side
    for name in ("rl_agent_config.json", "config.json", "configuration.json"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT_DIR, name))
    shutil.copy(os.path.join(MODEL_DIR, "encoder", "config.json"),
                os.path.join(OUT_DIR, "encoder_config.json"))

    print("wrote %s (%d tensors, %.1f MB)" % (bin_path, len(lines), off / 1e6))
    print("wrote %s" % man_path)
    with open(man_path) as f:
        for i, ln in enumerate(f):
            if i < 6:
                print("  " + ln.rstrip())


if __name__ == "__main__":
    main()
