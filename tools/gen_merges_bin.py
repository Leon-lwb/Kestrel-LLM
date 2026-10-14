#!/usr/bin/env python3
"""gen_merges_bin.py - 由 HF tokenizer.json 生成引擎用的 merges.bin。

引擎（vllm_tokenizer_qwen.c / tok_bpe_load_bin）需要一份**二进制**合并表：
 - merges：tokenizer.json 的 model.merges（已是 repr 域，如 "Ġ" / "plain"），
   列表顺序即 BPE rank；引擎按 (left,right) 查 rank 做最小合并。
 - added_tokens：内容为**原始文本**（如 "<|im_end|>" / "<think>"），引擎在
   文本里优先整串匹配它们，避免被 BPE 拆碎。

二进制布局（小端，自描述）：
  u32 magic='QBM1'(0x314D4251); u32 version=1; u32 n_merges; u32 n_added;
  n_merges × { u16 llen; u16 rlen; bytes left[llen]; bytes right[rlen]; }
  n_added  × { u32 id; u16 len; bytes content[len]; }

用法： python tools/gen_merges_bin.py <tokenizer.json> <out.merges.bin>
"""
import json
import struct
import sys

MAGIC = 0x314D4251  # 'QBM1'


def build_u2b():
    """bytes_to_unicode 的逆表：repr 字符 → 原始字节。"""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = build_u2b()


def to_engine(s):
    """把 tokenizer.json 的 repr 串转成**引擎词表域**：
    引擎 vocab.bin 存的是「原始字节」，仅把 <=0x20 的字节按 bytes_to_unicode
    编码（空格→'Ġ'=C4A0、换行→'Ċ'=C48A），其余字节原样（CJK 等多字节原样）。
    实测该变换与 vocab.bin 151643/151643 基表逐字节一致。"""
    out = bytearray()
    for ch in s:
        r = U2B[ch]
        if r <= 0x20:
            out += chr(256 + r).encode("utf-8")
        else:
            out.append(r)
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        d = json.load(f)

    merges = d["model"]["merges"]
    added = d.get("added_tokens", [])

    buf = bytearray()
    buf += struct.pack("<IIII", MAGIC, 1, len(merges), len(added))

    for m in merges:
        if isinstance(m, str):            # 旧格式 "left right"
            parts = m.split(" ", 1)
            left, right = (parts[0], parts[1]) if len(parts) == 2 else (m, "")
        else:                             # 新格式 [left, right]
            left, right = m
        # merges 键在 tokenizer.json 里是 repr 域 → 转到引擎词表域。
        lb = to_engine(left)
        rb = to_engine(right)
        buf += struct.pack("<HH", len(lb), len(rb)) + lb + rb

    for a in added:
        # added_tokens 是**原始文本**（引擎先在文本里整串匹配）→ 原样存。
        c = a["content"].encode("utf-8")
        buf += struct.pack("<IH", int(a["id"]), len(c)) + c

    with open(out, "wb") as f:
        f.write(buf)
    print("wrote %s: %d merges, %d added, %d bytes"
          % (out, len(merges), len(added), len(buf)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
