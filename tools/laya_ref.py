"""laya_ref.py - Laya (English root) Python golden reference.

Loads the tokenizer + DecisionModel from the local HF snapshot (no network),
reproduces the official `RLAgent.system_one` path, and dumps golden tensors
for engine cross-checking:
  - input ids / option marker positions (bit-level target)
  - encoder last_hidden_state
  - marker hidden states (pre-scorer)
  - scorer logits (per option, pre-temperature)
  - calibrated probabilities

Two scenarios are emitted (keys prefixed):
  "s_"  short state  (exercises full attention; seq < sliding window)
  "L_"  long state   (~400 tokens; exercises sliding-window layers)

Usage:
  python tools/laya_ref.py                 # run + print
  python tools/laya_ref.py --dump out.npz  # also dump golden + out.npz.cases.txt
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_DIR = os.environ.get("LAYA_MODEL",
                           "d:/项目/NewVLLM/modl/convaiinnovations--laya/snapshots/master")
sys.path.insert(0, MODEL_DIR)
import rl_common as R  # noqa: E402


def build():
    cfg = R.load_cfg(os.path.join(MODEL_DIR, "rl_agent_config.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    model = R.build_model(cfg, encoder_dir=os.path.join(MODEL_DIR, "encoder"))
    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return cfg, tok, model, missing, unexpected


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

_FILLER = ("Our team has been using the platform daily for the last several months. "
           "We rely on the nightly invoice export, the audit log, and the scheduled "
           "reports to reconcile charges across four departments. Last quarter we "
           "opened two tickets about delayed exports and one about a missing receipt, "
           "and all of them were resolved by your support team within a couple of days. ")
STATE_LONG = _FILLER * 5 + ("To summarize the current situation: on March 3rd and again on "
                            "March 7th the same monthly subscription was charged to our card, "
                            "leaving a duplicate of $ 480 on the account. The invoice "
                            "downloaded from the portal lists only one charge, so the second "
                            "line item appears to be a billing system error. We need the "
                            "duplicate reversed before the next renewal, otherwise we will "
                            "have to move the whole account to another provider. Please "
                            "confirm the refund and the timeline today.")


def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


def main():
    dump = None
    if "--dump" in sys.argv:
        dump = sys.argv[sys.argv.index("--dump") + 1]
    cfg, tok, model, missing, unexpected = build()
    print("[ref] missing keys:", list(missing), " unexpected:", list(unexpected))
    print("[ref] tokenizer cls=%s sep=%s pad=%s mask=%s vocab=%d" % (
        tok.cls_token_id, tok.sep_token_id, tok.pad_token_id,
        tok.mask_token_id, tok.vocab_size))

    ids_list = list(QUESTIONS.keys())

    def encode_one(state, qid):
        q = to_internal(QUESTIONS[qid])
        seq, markers = R.build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        return q, {"ids": seq, "markers": markers, "qtype": R.QTYPES[q["t"]],
                   "target": [0.0] * len(markers), "label": -1,
                   "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"}

    def fwd(items):
        b = R.collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                b["marker_mask"], b["qtype"])
            h = model.encoder(input_ids=b["input_ids"],
                              attention_mask=b["attention_mask"]).last_hidden_state
            hh = h + model.type_emb(b["qtype"])[:, None, :]
            for layer in model.head.layers:
                hh = layer(hh, src_key_padding_mask=~b["attention_mask"].bool())
            idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, hh.size(-1))
            m = torch.gather(hh, 1, idx)
        return b, logits.float().numpy(), act.float().numpy(), h.numpy(), m.numpy()

    scenarios = [("s", STATE_SHORT), ("L", STATE_LONG)]
    gold, cases_lines, maxdiffs = {}, [], []
    for stag, state in scenarios:
        all_items = [encode_one(state, q)[1] for q in ids_list]
        _, log_all, _, _, _ = fwd(all_items)
        for r, qid in enumerate(ids_list):
            q, it = encode_one(state, qid)
            b1, lg1, act1, h1, m1 = fwd([it])
            k = len(it["markers"])
            d = float(np.max(np.abs(lg1[0, :k] - log_all[r, :k])))
            maxdiffs.append(d)
            tk = R.temp_bucket(R.QTYPES[q["t"]], k)
            temp = cfg.get("temperature_by_options", {}).get(
                tk, cfg.get("temperature", [1, 1, 1])[R.QTYPES[q["t"]]])
            z = lg1[0, :k] / temp
            p = np.exp(z - z.max())
            p = p / p.sum()
            key = "%s_%s" % (stag, qid)
            print("[ref] %-14s type=%-6s nopts=%d seq=%4d markers=%s |Δbatch|=%.2e argmax=%d" %
                  (key, q["t"], k, len(it["ids"]), it["markers"], d, int(p.argmax())))
            gold["%s_ids" % key] = np.array(it["ids"], dtype=np.int64)
            gold["%s_markers" % key] = np.array(it["markers"], dtype=np.int64)
            gold["%s_qtype" % key] = np.array([it["qtype"]], dtype=np.int64)
            gold["%s_enc" % key] = h1[0]
            gold["%s_marker" % key] = m1[0, :k]
            gold["%s_logits" % key] = lg1[0, :k]
            gold["%s_probs" % key] = p
            cases_lines.append("%d %d %s %d %s" %
                               (it["qtype"], len(it["ids"]), " ".join(map(str, it["ids"])),
                                k, " ".join(map(str, it["markers"]))))

    print("[ref] batch-invariance max|Δ| = %.3e" % max(maxdiffs))
    if dump:
        np.savez(dump, **gold)
        with open(dump + ".cases.txt", "w") as cf:
            cf.write("\n".join(cases_lines) + "\n")
        print("[ref] dumped golden -> %s (+ .cases.txt)" % dump)


if __name__ == "__main__":
    main()
