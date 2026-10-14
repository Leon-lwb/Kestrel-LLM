/**
 * vllm_laya.h - Laya (ModernBERT-large encoder + decision head) decision path.
 *
 * Laya is a NON-autoregressive System-1 decision model: a bidirectional
 * ModernBERT encoder + a from-scratch decision head that scores one [MASK]
 * marker per option and softmaxes over a question's options. It never
 * generates text. This module is an independent inference path, decoupled
 * from the causal-LM engine.
 *
 * Weights are read from a packed F16 blob produced by tools/laya_pack.py:
 *   <pack_dir>/laya_f16.bin        raw fp16 tensors
 *   <pack_dir>/laya_manifest.txt   "<name> <ndim> <d..> <dtype> <offset> <nbytes>"
 */
#ifndef VLLM_LAYA_H
#define VLLM_LAYA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayaModel LayaModel;

/** Load a packed model. Returns NULL on failure. */
LayaModel *laya_load(const char *pack_dir);
void laya_free(LayaModel *m);

/**
 * Run one question (single sequence, no padding; batch=1).
 *   ids[n]          token ids
 *   markers[nm]     positions of the per-option [MASK] markers
 *   qtype           0=choice 1=score 2=noul
 *   logits_out[nm]  per-option scorer logits (pre-temperature)
 * Returns 0 on success.
 */
int laya_forward(LayaModel *m, const int *ids, int n,
                 const int *markers, int nm, int qtype,
                 float *logits_out);

/**
 * CLI test entry: read cases from a text file and run them.
 *   cases file line: "<qtype> <n_ids> <id0..> <n_markers> <m0..>"
 * Prints "[LAYA] q=<i> logits=[..]" per line.
 * Returns 0 on success, -1 on load/parse error.
 */
int laya_run_cases(const char *pack_dir, const char *cases_path);

/**
 * Tokenizer gate entry: load the engine tokenizer from <dir> (vocab.bin +
 * merges.bin in engine domain, ModernBERT GPT-2/ByteLevel pre-tokenizer) and
 * print "[LTOK] i=<i> n=<N>: id id ..." for each line of <path>.
 */
int laya_tok_dump(const char *dir, const char *path);

/**
 * End-to-end decision: load the model + tokenizer from <pack_dir>, read a Jev/OpenAI-shaped
 * JSON request at <req_path> ({"state":..., "questions":{qid:{type,instructions,criteria}}}),
 * render -> tokenize -> forward -> temperature -> softmax, and print the answers JSON.
 */
int laya_decide(const char *pack_dir, const char *req_path);

/**
 * Persistent runner: model + tokenizer + temperature config loaded once and
 * reused across requests. Backs both the CLI and HTTP /v1/systemone (the HTTP
 * server must not reload the ~1.7 GB model per request). Opaque.
 */
typedef struct LayaRunner LayaRunner;

/** Load model + tokenizer + rl_agent_config.json from <pack_dir>. NULL on failure. */
LayaRunner *laya_runner_new(const char *pack_dir);
void laya_runner_free(LayaRunner *r);

/**
 * Run one Jev-shaped JSON request (req_text) through the runner and write the
 * answers JSON ({"model":"laya","answers":{...}}) into out (NUL-terminated,
 * capped at cap). Returns 0 on success, -1 on parse/forward error or if the
 * output would not fit.
 */
int laya_runner_decide(LayaRunner *r, const char *req_text, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* VLLM_LAYA_H */
