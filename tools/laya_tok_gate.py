"""laya_tok_gate.py - Engine ModernBERT tokenizer vs HF `tokenizers` (bit-level).

golden = tokenizers.Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False)
engine = `vllm_kestrel --laya-tok <tokenizer_dir> --laya-cases <tmp>` -> "[LTOK] i=.. n=..: ids"

Usage: python tools/laya_tok_gate.py [tokenizer.json] [tok_dir] [engine_exe]
"""
import os
import re
import subprocess
import sys

import numpy as np

TOKJSON = sys.argv[1] if len(sys.argv) > 1 else \
    "d:/项目/NewVLLM/modl/convaiinnovations--laya/snapshots/master/tokenizer/tokenizer.json"
TOKDIR = sys.argv[2] if len(sys.argv) > 2 else "d:/项目/NewVLLM/build_laya"
ENGINE = sys.argv[3] if len(sys.argv) > 3 else \
    r"C:\Users\Administrator\vllm_x64_build\vllm_kestrel_x64.exe"

FILLER = ("Our team has been using the platform daily for the last several months. "
          "We rely on the nightly invoice export, the audit log, and the scheduled "
          "reports to reconcile charges across four departments. ")

STRINGS = [
    "choice question: Which department should handle this?",
    " score question: How urgent is this?",
    " noul question: Does the user threaten to cancel or leave?",
    " billing: invoices, payments, refunds",
    " technical: bugs, outages, system errors",
    " other: everything else",
    " level 0: not urgent",
    " level 2: blocking",
    " false: no, the statement does not hold",
    " true: yes, the statement holds",
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    FILLER * 3,
    "Hello world",
    "It's 3:30 PM, isn't it?",
    "user@acme.com",
    "https://x.io/p?q=1",
    "12 boxes x 8 apples = 96 apples",
    "March 3rd and March 7th, duplicate of $ 480",
    "  leading spaces and   inner   runs  ",
    "Numbers: 1234567890, 3.14159, -42.",
    "Temperature 25.5°C, probability 3/4.",
    "你好，世界！12×8=96。",
    "\u201cquoted\u201d and (parentheses); full-width punctuation.",
    "def add(a, b):\n    return a + b",
    "A train travels at 60 miles per hour for 2.5 hours.",
    "|||EMAIL_ADDRESS||| blocked and |||PHONE_NUMBER||| too",
]

from tokenizers import Tokenizer  # noqa: E402
hf = Tokenizer.from_file(TOKJSON)

tmp = os.environ.get("TEMP", "/tmp") + "/laya_tok_cases.txt"
# the engine reads one case per line -> compare on the newline-folded string on BOTH sides
CASES = [s.replace("\n", " ") for s in STRINGS]
with open(tmp, "w", encoding="utf-8", newline="\n") as f:
    for s in CASES:
        f.write(s + "\n")

r = subprocess.run([ENGINE, "--laya-tok", TOKDIR, "--laya-cases", tmp],
                   capture_output=True, timeout=300)
out = r.stdout.decode("utf-8", "replace")
eng = {}
for m in re.finditer(r"\[LTOK\] i=(\d+) n=(\d+):(.*)", out):
    ids = [int(x) for x in m.group(3).split()]
    eng[int(m.group(1))] = ids

npass = 0
for i, s in enumerate(CASES):
    gold = hf.encode(s, add_special_tokens=False).ids
    e = eng.get(i)
    ok = (e == gold)
    npass += ok
    print("%s[%2d] %r" % ("PASS" if ok else "FAIL", i, s[:56]))
    if not ok:
        print("   gold(%d)=%s" % (len(gold), gold))
        print("   eng (%d)=%s" % (len(e) if e is not None else -1, e))
print("laya_tok_gate: %d/%d PASS" % (npass, len(STRINGS)))
sys.exit(0 if npass == len(STRINGS) else 1)
