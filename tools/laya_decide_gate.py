"""laya_decide_gate.py - engine `--laya-decide` (text -> answers) vs official Python RLAgent.

golden = rl_agent_api.RLAgent(MODEL_DIR).system_one(state, questions)   (official path)
engine = vllm_kestrel --laya-decide --laya-pack <pack> --laya-cases <req.json>

Frozen criteria (K-LAYA-001 §L5):
  - choice key identical; score/noul within 5e-3 of golden
  - per-option probabilities max|Δ| <= 5e-3
Usage: python tools/laya_decide_gate.py
"""
import json
import os
import subprocess
import sys

MODEL_DIR = "d:/项目/NewVLLM/modl/convaiinnovations--laya/snapshots/master"
PACK = r"d:\项目\NewVLLM\build_laya"
ENGINE = r"C:\Users\Administrator\vllm_x64_build\vllm_kestrel_x64.exe"
TOL = 5e-3

sys.path.insert(0, MODEL_DIR)
os.environ.setdefault("USE_TF", "0")
from rl_agent_api import RLAgent  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
STATE_SHORT = ("Hi, we were billed twice for March. Please refund the duplicate today "
               "or we will cancel our plan.")
FILLER = ("Our team has been using the platform daily for the last several months. "
          "We rely on the nightly invoice export, the audit log, and the scheduled "
          "reports to reconcile charges across four departments. ")
STATE_LONG = FILLER * 5 + ("To summarize: on March 3rd and again on March 7th the same "
                           "subscription was charged twice, leaving a duplicate of $ 480. "
                           "We need the duplicate reversed before the next renewal, otherwise "
                           "we will move the account to another provider.")


def main():
    agent = RLAgent(MODEL_DIR, device="cpu")
    scenarios = (("short", STATE_SHORT), ("long", STATE_LONG))
    # compute all goldens first, then release the torch model so the engine child
    # process has RAM headroom for its F32 arena.
    golds = {tag: agent.system_one(state, QUESTIONS)["answers"] for tag, state in scenarios}
    del agent
    import gc
    gc.collect()

    npass = 0
    total = 0
    for tag, state in scenarios:
        req = {"state": state, "questions": QUESTIONS}
        rp = os.path.join(os.environ.get("TEMP", "/tmp"), "laya_req_%s.json" % tag)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(req, f, ensure_ascii=False)
        gold = golds[tag]

        r = subprocess.run([ENGINE, "--laya-decide", "--laya-pack", PACK, "--laya-cases", rp],
                           capture_output=True, timeout=1800)
        out = r.stdout.decode("utf-8", "replace")
        line = [l for l in out.splitlines() if l.startswith('{"model"')]
        if not line:
            print("[%s] FAIL: no engine JSON (rc=%d)" % (tag, r.returncode))
            print("STDOUT:", out[-800:])
            print("STDERR:", r.stderr.decode("utf-8", "replace")[-800:])
            return 1
        eng = json.loads(line[-1])["answers"]

        for qid, q in QUESTIONS.items():
            total += 1
            g, e = gold[qid], eng.get(qid, {})
            ok = True
            detail = []
            if q["type"] == "choice":
                ok &= (g["choice"] == e.get("choice"))
                detail.append("choice %s/%s" % (e.get("choice"), g["choice"]))
                for kk, gv in g["probabilities"].items():
                    d = abs(e["probabilities"][kk] - gv)
                    ok &= (d <= TOL)
                    detail.append("%s d=%.1e" % (kk, d))
            elif q["type"] == "score":
                d = abs(e["score"] - g["score"])
                ok &= (d <= TOL)
                detail.append("score %.4f/%.4f d=%.1e" % (e["score"], g["score"], d))
            else:
                d = abs(e["noul"] - g["noul"])
                ok &= (d <= TOL)
                detail.append("noul %.4f/%.4f d=%.1e" % (e["noul"], g["noul"], d))
            npass += ok
            print("%s[%s/%-11s] %s" % ("PASS" if ok else "FAIL", tag, qid, "; ".join(detail)))
    print("laya_decide_gate: %d/%d PASS" % (npass, total))
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())
