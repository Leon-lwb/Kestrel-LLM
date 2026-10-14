#!/usr/bin/env python3
"""tok_gate.py - 引擎分词器 vs HF/tokenizers 的位级质量门。

golden = tokenizers 库（tokenizer.json），encode(text, add_special_tokens=False)。
engine = 引擎 --embed 的 [EMB-IDS] 输出（VLLM_EMB_IDS=1），去掉末尾自动追加的
EOS（<|endoftext|> 151643）后逐位对比。

用法：
  set VLLM_TOK_ENGINE=<exe>  VLLM_TOK_MODEL=<model dir with vocab.bin/merges.bin>
  python tools/tok_gate.py <tokenizer.json>
默认读取与环境变量；可用 argv[2]=engine argv[3]=model 覆盖。
"""
import json
import os
import re
import subprocess
import sys

TOKJSON = sys.argv[1] if len(sys.argv) > 1 else \
    r'd:\项目\NewVLLM\modl\Qwen--Qwen3-Embedding-0.6B\snapshots\master\tokenizer.json'
ENGINE = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get('VLLM_TOK_ENGINE')) or \
    r'C:\Users\Administrator\vllm_x64_build\vllm_kestrel_x64.exe'
MODEL = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get('VLLM_TOK_MODEL')) or \
    r'C:\Users\Administrator\emb_model2'

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKJSON)

TEXT = "<|im_start|>system\n直接给出最终答案，简洁作答，不要展开逐步解释。<|im_end|>\n" \
       "<|im_start|>user\nWhat is 12*8?<|im_end|>\n<|im_start|>assistant\n"

cases = [
    "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?",
    "hello world",
    "Explain gravity",
    "Query:Explain gravity",
    "Instruct: test",
    "I can't believe it's already 3:30 PM!",
    "你好，世界！12×8=96。",
    "“引号”和（括号）；全角，标点。",
    TEXT,
    "The quick brown fox jumps over the lazy dog.",
    "  \n\nhello",
    "a1b2c3",
    "don't stop",
    "world\n\nnext",
    "hello   ",
    "Numbers: 1234567890, 3.14159, -42.",
    "email a.b-c@example.com or https://x.io/p?q=1",
    "def add(a, b):\n    return a + b",
    "温度是 25.5°C，概率 3/4。",
    "Q: What is the capital of China?\nA: Beijing.",
]

env = dict(os.environ)
env['VLLM_EMB_IDS'] = '1'
env['PATH'] = r'C:\Users\Administrator\tools\winlibs-gcc\mingw64\bin;' + env.get('PATH', '')

npass = 0
for i, t in enumerate(cases):
    gold = tok.encode(t, add_special_tokens=False).ids
    r = subprocess.run([ENGINE, '--model', MODEL, '--embed-stdin'],
                       input=t.encode('utf-8'), capture_output=True, env=env)
    sout = r.stdout.decode('utf-8', 'replace')
    m = re.search(r'\[EMB-IDS\] n=\d+:(.*)', sout)
    if not m:
        print('FAIL[%2d] no ids (rc=%d): %r' % (i, r.returncode, t[:40]))
        continue
    eng = [int(x) for x in m.group(1).split()]
    if eng and eng[-1] == 151643:
        eng = eng[:-1]
    ok = (eng == gold)
    npass += ok
    tag = 'PASS' if ok else 'FAIL'
    print('%s[%2d] %r' % (tag, i, t[:48]))
    if not ok:
        print('   gold(%d)=%s' % (len(gold), gold))
        print('   eng (%d)=%s' % (len(eng), eng))
print('tok_gate: %d/%d PASS' % (npass, len(cases)))
