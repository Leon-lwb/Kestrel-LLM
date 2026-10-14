"""gen_modernbert_bin.py - ModernBERT tokenizer.json -> engine-domain vocab.bin + merges.bin.

The engine's BPE (vllm_tokenizer_qwen.c) consumes:
  - vocab.bin : u32 vocab_size, then per entry { u32 tid; u16 len; bytes[len] }
                byte strings are in the *engine domain*: raw bytes, except bytes
                <= 0x20 which use the GPT2 bytes_to_unicode 2-byte escape
                (space -> C4 A0 = 'Ġ', newline -> C4 8A = 'Ċ').
  - merges.bin: see tools/gen_merges_bin.py layout (same engine domain for merges;
                added-token contents stored RAW and matched against raw input text).

Usage: python tools/gen_modernbert_bin.py <tokenizer.json> <out_dir>
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_merges_bin import MAGIC, to_engine  # noqa: E402


def raw_to_engine(bs: bytes) -> bytes:
    """raw bytes -> engine domain (<=0x20 escaped like bytes_to_unicode)."""
    out = bytearray()
    for b in bs:
        if b <= 0x20:
            out += chr(256 + b).encode("utf-8")
        else:
            out.append(b)
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    d = json.load(open(src, encoding="utf-8"))
    vocab = d["model"]["vocab"]                 # token(str, repr domain) -> id
    merges = d["model"]["merges"]
    added = d.get("added_tokens", [])
    addset = {int(a["id"]): a["content"] for a in added}

    maxid = max(list(vocab.values()) + list(addset.keys()))
    n = maxid + 1
    inv = {v: k for k, v in vocab.items()}

    # ---- vocab.bin ----
    buf = bytearray(struct.pack("<I", n))
    missing = 0
    for i in range(n):
        if i in addset:                         # added-token content is RAW text
            s = raw_to_engine(addset[i].encode("utf-8"))
        elif i in inv:
            s = to_engine(inv[i])
        else:
            s = b""
            missing += 1
        buf += struct.pack("<IH", i, len(s)) + s
    with open(os.path.join(outdir, "vocab.bin"), "wb") as f:
        f.write(buf)

    # ---- merges.bin (same layout as gen_merges_bin.py) ----
    mb = bytearray()
    mb += struct.pack("<IIII", MAGIC, 1, len(merges), len(added))
    for m in merges:
        left, right = (m.split(" ", 1) if isinstance(m, str) else m)
        lb, rb = to_engine(left), to_engine(right)
        mb += struct.pack("<HH", len(lb), len(rb)) + lb + rb
    for a in added:
        c = a["content"].encode("utf-8")
        mb += struct.pack("<IH", int(a["id"]), len(c)) + c
    with open(os.path.join(outdir, "merges.bin"), "wb") as f:
        f.write(mb)

    print("wrote %s/vocab.bin (%d ids, %d empty) + merges.bin (%d merges, %d added)"
          % (outdir, n, missing, len(merges), len(added)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
