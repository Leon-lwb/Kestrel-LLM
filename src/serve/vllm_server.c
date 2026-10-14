/* ================================================================
 * vllm_server.c - OpenAI-compatible API server for vllm_kestrel
 *
 * Routes /v1/models, /health and /v1/chat/completions (with SSE
 * streaming) against the engine's Qwen inference state.
 *
 * The engine keeps a single inference state, so inference-critical
 * sections are serialized: clients queue on an internal mutex (bounded,
 * overflow -> 429) and are served one at a time. Metadata endpoints
 * (/v1/models, /health) run concurrently and never wait on inference.
 * Robustness guards: prompt-length check (400), free-RAM check (503),
 * slow-client recv timeout, bounded connection queue (vllm_http).
 * ================================================================ */
#include "vllm_server.h"
#include "vllm_batch.h"      /* continuous batching (--batch-max N) */
#include "vllm_superpos.h"   /* sample_token(): argmax/top-p sampler */
#include "vllm_l3.h"         /* STL3State / l3_state_free (per-user L3 files) */
#include "vllm_attest.h"     /* verifiable-inference attestation (方案 2) */
#include "vqf.h"             /* vqf_stream_* 运行时权重逐层驻留控制 (res policy) */
#include "embedded_web.h"
#include "vllm_rulebook.h"   /* 规则包注册中心：预置命名上下文（--rulebook-dir） */
#include "vllm_session.h"    /* 服务端会话存储：会话分区 + 历史审查（--session-dir） */
#include "vllm_laya.h"       /* K-LAYA-001：Laya 决策模型独立路径（POST /v1/systemone） */
#include "vllm_yolo.h"       /* K-YOLO-001：YOLO 检测独立路径（POST /v1/detect） */
#include "media/vllm_vsrc.h" /* K-E2E-005：视频源抽象（MP4/目录/MJPEG/RTSP） */
#include "vllm_media.h"      /* 图像解码（media_load_image_memory / base64） */
#include "vllm_kvstore.h"    /* K-COLLAB-001 P0：向量库（/v1/rag 检索） */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Portable manual-load mode: /admin/api/model/load may supply a model
 * directory that was not present at startup (the packaged exe can be run
 * from any directory; the model path is chosen on the admin page). */
void vllm_serve_set_model_dir(VLLMServerCtx *ctx, const char *dir) {
    if (!ctx || !dir || !dir[0]) return;
    snprintf(ctx->model_dir_buf, sizeof(ctx->model_dir_buf), "%s", dir);
    ctx->model_dir = ctx->model_dir_buf;
}

#include <unistd.h>  /* unlink */
#include <sys/stat.h>
#include <dirent.h>

/* ---------- small growable string buffer ---------- */

typedef struct { char *s; size_t len, cap; } StrBuf;

static void sb_init(StrBuf *b) { b->s = NULL; b->len = 0; b->cap = 0; }
static void sb_free(StrBuf *b) { free(b->s); b->s = NULL; b->len = b->cap = 0; }

static void sb_put(StrBuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (nc < b->len + n + 1) nc *= 2;
        char *ns = (char *)realloc(b->s, nc);
        if (!ns) return;
        b->s = ns; b->cap = nc;
    }
    memcpy(b->s + b->len, s, n);
    b->len += n;
    b->s[b->len] = '\0';
}
static void sb_str(StrBuf *b, const char *s) { sb_put(b, s, strlen(s)); }

/* ---------- robustness helpers ---------- */

/* Free physical RAM in MB; -1 if unknown. Used to refuse new inference
 * when the machine is about to run out of memory (503 instead of a
 * hard-to-diagnose allocation crash mid-prefill).
 * POSIX 优先 MemAvailable（/proc/meminfo，含可回收 page cache），回退
 * sysconf(_SC_AVPHYS_PAGES)（仅自由页——满载盒会低估触发假 503）；
 * Windows 走 GlobalMemoryStatusEx。统一由 vllm_platform.h 提供。 */
static long mem_avail_mb(void) {
    return st_avail_mem_mb();
}

/* Enter the inference critical section. Applies the admission policy:
 *   -1 -> reject with 429 (queue full)
 *   -2 -> reject with 503 (not enough free RAM)
 *    0 -> inference may proceed (queued + lock held; busy=1)
 * Rejection happens before taking the inference lock so a full/stressed
 * server answers fast instead of accumulating blocked workers. */
static int infer_gate(VLLMServerCtx *ctx) {
    if (ctx->min_free_mb > 0) {
        long avail = mem_avail_mb();
        if (avail >= 0 && avail < ctx->min_free_mb) {
            fprintf(stderr, "[SRV] 503: free RAM %ld MB < min_free_mb=%d MB "
                    "(refusing inference; lower --min-free-mb or use a smaller "
                    "model on this machine)\n", avail, ctx->min_free_mb);
            fflush(stderr);
            return -2;
        }
    }
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) {      /* 模型卸载中：拒绝新推理（503） */
        vhttp_mutex_unlock(ctx->stat_lock);
        return -3;
    }
    if (ctx->max_queued > 0 && ctx->queued_requests >= ctx->max_queued) {
        vhttp_mutex_unlock(ctx->stat_lock);
        return -1;
    }
    ctx->total_requests++;
    ctx->active_requests++;
    ctx->queued_requests++;
    vhttp_mutex_unlock(ctx->stat_lock);

    vhttp_mutex_lock(ctx->inf_lock);   /* wait for the running request */
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->queued_requests--;
    ctx->busy = 1;
    ctx->last_infer_s = (long)time(NULL);   /* 请求开始推理即刷新空闲计时 */
    vhttp_mutex_unlock(ctx->stat_lock);
    return 0;
}

static void infer_done(VLLMServerCtx *ctx) {
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->busy = 0;
    ctx->active_requests--;
    ctx->last_infer_s = (long)time(NULL);
    vhttp_mutex_unlock(ctx->stat_lock);
    vhttp_mutex_unlock(ctx->inf_lock);
}

/* Batch-mode admission: identical 429/503 policy but WITHOUT blocking on the
 * inference lock (the batch scheduler owns engine serialization). */
static int infer_gate_batch(VLLMServerCtx *ctx) {
    if (ctx->min_free_mb > 0) {
        long avail = mem_avail_mb();
        if (avail >= 0 && avail < ctx->min_free_mb) {
            fprintf(stderr, "[SRV] 503: free RAM %ld MB < min_free_mb=%d MB "
                    "(refusing inference; lower --min-free-mb or use a smaller "
                    "model on this machine)\n", avail, ctx->min_free_mb);
            fflush(stderr);
            return -2;
        }
    }
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) {      /* 模型卸载中：拒绝新推理（503） */
        vhttp_mutex_unlock(ctx->stat_lock);
        return -3;
    }
    if (ctx->max_queued > 0 && ctx->queued_requests >= ctx->max_queued) {
        vhttp_mutex_unlock(ctx->stat_lock);
        return -1;
    }
    ctx->total_requests++;
    ctx->active_requests++;
    ctx->queued_requests++;
    ctx->busy = 1;
    ctx->last_infer_s = (long)time(NULL);   /* 请求开始推理即刷新空闲计时 */
    vhttp_mutex_unlock(ctx->stat_lock);
    return 0;
}

static void infer_done_batch(VLLMServerCtx *ctx) {
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->queued_requests--;
    if (ctx->queued_requests < 0) ctx->queued_requests = 0;
    ctx->active_requests--;
    ctx->busy = ctx->active_requests > 0;
    ctx->last_infer_s = (long)time(NULL);
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* Lazy scheduler creation once the model is loaded (works for both
 * --auto-load and admin manual-load). No-op when batching is disabled.
 * Skipped while a model unload is in progress (the unload destroys the
 * scheduler; creating a fresh one against a freed ist would race). */
static void batch_ensure(VLLMServerCtx *ctx) {
    if (ctx->batch_max < 2 || ctx->batch || !ctx->ist) return;
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) { vhttp_mutex_unlock(ctx->stat_lock); return; }
    if (!ctx->batch && ctx->ist) {
        ctx->batch = vllm_batch_create(&ctx->ist->weights, ctx->tok,
                                       ctx->model_id, ctx->batch_max);
        fprintf(stderr, "[BATCH] continuous batching on (batch_max=%d)\n",
                ctx->batch_max);
        fflush(stderr);
    }
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* ---------- Qwen chat template ---------- */

/* Build the Qwen im_chat prompt from OpenAI messages.
 * content may be a plain string or a part array ([{type:"text",...}, ...]);
 * the chat client always sends arrays, so text parts are extracted here.
 * Media parts never reach this path (detect_media routes them to the
 * multimodal builder). Returns malloc'd NUL-terminated string. */
/* Qwen3「非 thinking」渲染后缀。HF 官方模板在 enable_thinking=False 时，于
 * `<|im_start|>assistant\n` 之后追加 `<think>\n\n</think>\n\n`（强制空思考段），
 * 使模型直接作答；其 id 序列为 151667,271,151668,271，与 HF 一致。 */
#define QWEN_NO_THINK_SUFFIX "<think>\n\n</think>\n\n"

/* Qwen3 thinking 开关。HF apply_chat_template 默认 enable_thinking=True，
 * 本引擎默认与之一致。优先级：
 *   请求体 chat_template_kwargs.enable_thinking > 顶层 enable_thinking
 *   > 环境变量 VLLM_ENABLE_THINKING（"0" 表示关） > 默认 1（开）。 */
static int resolve_thinking(const VJson *body) {
    const VJson *kw = vjson_obj_get(body, "chat_template_kwargs");
    if (kw && kw->type == VJ_OBJECT) {
        const VJson *e = vjson_obj_get(kw, "enable_thinking");
        if (e && e->type == VJ_BOOL) return e->u.boolean;
        if (e && e->type == VJ_NUMBER) return e->u.number != 0.0;
    }
    const VJson *e2 = vjson_obj_get(body, "enable_thinking");
    if (e2 && e2->type == VJ_BOOL) return e2->u.boolean;
    if (e2 && e2->type == VJ_NUMBER) return e2->u.number != 0.0;
    const char *env = getenv("VLLM_ENABLE_THINKING");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static char *build_chat_prompt(VLLMServerCtx *ctx, const VJson *messages,
                               int thinking) {
    StrBuf b; sb_init(&b);
    size_t n = vjson_array_len(messages);
    for (size_t i = 0; i < n; i++) {
        const VJson *m = vjson_array_get(messages, i);
        if (!m) continue;
        const char *role = vjson_str(vjson_obj_get(m, "role"));
        const VJson *content = vjson_obj_get(m, "content");
        if (!role || !content) continue;
        if (strcmp(role, "system") == 0) {
            sb_str(&b, "<|im_start|>system\n");
        } else if (strcmp(role, "user") == 0) {
            sb_str(&b, "<|im_start|>user\n");
        } else if (strcmp(role, "assistant") == 0) {
            sb_str(&b, "<|im_start|>assistant\n");
        } else {
            sb_str(&b, "<|im_start|>"); sb_str(&b, role); sb_str(&b, "\n");
        }
        if (content->type == VJ_STRING) {
            sb_str(&b, vjson_str(content));
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&b, t);
                }
                /* image_url / video_frames parts are handled by the
                 * multimodal path (detect_media) and never reach here. */
            }
        } else {
            continue;
        }
        sb_str(&b, "<|im_end|>\n");
    }
    sb_str(&b, "<|im_start|>assistant\n");
    if (!thinking) sb_str(&b, QWEN_NO_THINK_SUFFIX);
    if (!b.s) b.s = strdup("");
    return b.s;
}

/* Legacy /v1/completions: wrap prompt in a plain user turn. */
static char *build_completion_prompt(VLLMServerCtx *ctx, const char *prompt,
                                     int thinking) {
    (void)ctx;
    StrBuf b; sb_init(&b);
    sb_str(&b, "<|im_start|>user\n"); sb_str(&b, prompt); sb_str(&b, "<|im_end|>\n");
    sb_str(&b, "<|im_start|>assistant\n");
    if (!thinking) sb_str(&b, QWEN_NO_THINK_SUFFIX);
    if (!b.s) b.s = strdup("");
    return b.s;
}

/* ---------- response helpers ---------- */

static long unix_now(void) { return (long)time(NULL); }

/* 标准 OpenAI usage 对象（prompt/completion/total tokens），供非流式响应
 * 的 "usage" 字段使用（此前为空对象，标准客户端拿不到 token 计数）。 */
static VJson *usage_json(const VLLMMetrics *met) {
    VJson *u = vjson_new_object();
    int pt = met ? met->prompt_tokens : 0;
    int ct = met ? met->n_tokens : 0;
    vjson_obj_set(u, "prompt_tokens", vjson_new_number((double)pt));
    vjson_obj_set(u, "completion_tokens", vjson_new_number((double)ct));
    vjson_obj_set(u, "total_tokens", vjson_new_number((double)(pt + ct)));
    return u;
}

/* Fill resp with a heap-allocated JSON error (thread-safe; the server
 * frees resp->body_owned after sending). */
static void json_error(VHttpResponse *resp, int status, const char *message) {
    char *body = (char *)malloc(512);
    if (body) {
        vhttp_json_error(message, body, 512);
        resp->body = body;
        resp->body_len = strlen(body);
        resp->body_owned = body;
    } else {
        static const char oom[] =
            "{\"error\":{\"message\":\"Out of memory\",\"type\":\"server_error\"}}";
        resp->body = oom;
        resp->body_len = sizeof(oom) - 1;
    }
    resp->status = status;
    resp->content_type = "application/json";
}

/* ---------- per-user L3 eviction isolation ----------
 *
 * --l3-evict + --l3-user-ttl: every user (the request JSON "user" field,
 * "anon" fallback) gets its own L3 cache file. The file is derived from the
 * base --l3-path by inserting "_u<hex8>" (FNV-1a hash of the user id, no
 * path chars) before the extension, e.g. /mnt/emmc/kv_l3.bin ->
 * /mnt/emmc/kv_l3_u1a2b3c4d.bin. A user whose last request is older than
 * --l3-user-ttl minutes is considered offline and its file is deleted on the
 * next request (lazy cleanup; st->l3 is closed first if still bound to it).
 * The session table is touched only while the inference lock is held
 * (infer_gate passed), so no locking is needed. */

/* FNV-1a 64-bit -> low 32 bits (deterministic, hex-safe). */
static unsigned long l3_user_hash(const char *s) {
    unsigned long long h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return (unsigned long)(h & 0xFFFFFFFFul);
}

/* Per-user L3 file path from the base path (g_l3_path or "kv_l3.bin"). */
static void l3_user_fname(const char *base, const char *user,
                          char *out, size_t cap) {
    const char *b = (base && base[0]) ? base : "kv_l3.bin";
    const char *slash = strrchr(b, '/');
    const char *dot = strrchr(b, '.');
    if (dot && (!slash || dot > slash)) {
        snprintf(out, cap, "%.*s_u%08lx%s", (int)(dot - b), b,
                 l3_user_hash(user), dot);
    } else {
        snprintf(out, cap, "%s_u%08lx", b, l3_user_hash(user));
    }
}

/* Update (or create) the user's session entry and set the current request's
 * L3 file. Falls back to the base file when the table cannot grow. */
static void l3_user_touch(VLLMServerCtx *ctx, const char *user) {
    long now = (long)unix_now();
    for (int i = 0; i < ctx->l3_users_n; i++) {
        if (strcmp(ctx->l3_users[i].user, user) == 0) {
            ctx->l3_users[i].last_active = now;
            snprintf(ctx->l3_cur_fname, sizeof(ctx->l3_cur_fname), "%s",
                     ctx->l3_users[i].fname);
            return;
        }
    }
    if (ctx->l3_users_n >= ctx->l3_users_cap) {
        int nc = ctx->l3_users_cap ? ctx->l3_users_cap * 2 : 8;
        L3UserEntry *ne = (L3UserEntry *)realloc(
            ctx->l3_users, (size_t)nc * sizeof(*ne));
        if (!ne) {
            l3_user_fname(g_l3_path, user, ctx->l3_cur_fname,
                          sizeof(ctx->l3_cur_fname));
            return;
        }
        ctx->l3_users = ne;
        ctx->l3_users_cap = nc;
    }
    L3UserEntry *e = &ctx->l3_users[ctx->l3_users_n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->user, sizeof(e->user), "%s", user);
    l3_user_fname(g_l3_path, user, e->fname, sizeof(e->fname));
    e->last_active = now;
    snprintf(ctx->l3_cur_fname, sizeof(ctx->l3_cur_fname), "%s", e->fname);
}

/* Delete the L3 files of users idle longer than --l3-user-ttl minutes.
 * Runs with the inference lock held (single-threaded vs st->l3). */
static void l3_user_cleanup(VLLMServerCtx *ctx) {
    STQwenInferenceState *st = ctx->ist;
    long now = (long)unix_now();
    long ttl = g_l3_user_ttl * 60L;
    if (ttl <= 0 || ctx->l3_users_n == 0) return;
    for (int i = 0; i < ctx->l3_users_n; i++) {
        L3UserEntry *e = &ctx->l3_users[i];
        if (now - e->last_active < ttl) continue;
        if (st && st->l3.fp && strcmp(st->l3.path, e->fname) == 0) {
            fprintf(stderr, "[L3] user idle: close bound file %s\n", e->fname);
            l3_state_free(&st->l3);
        }
        if (unlink(e->fname) == 0)
            fprintf(stderr, "[L3] user '%s' offline: removed %s\n",
                    e->user, e->fname);
        else
            fprintf(stderr, "[L3] user '%s' offline: %s absent\n",
                    e->user, e->fname);
        ctx->l3_users[i] = ctx->l3_users[ctx->l3_users_n - 1];
        ctx->l3_users_n--;
        i--;
    }
}

/* Per-request entry point (called with the inference lock held): resolve the
 * user, reap offline files, set the current user's L3 file. */
static void l3_user_apply(VLLMServerCtx *ctx, const char *user) {
    if (!g_l3_evict) {
        ctx->l3_cur_fname[0] = '\0';
        g_l3_cur_path = NULL;
        return;
    }
    l3_user_cleanup(ctx);
    l3_user_touch(ctx, (user && user[0]) ? user : "anon");
    g_l3_cur_path = ctx->l3_cur_fname;
}

/* KV prefix reuse: minimum shared-prefix length (tokens) worth skipping. */
#define PREFIX_KV_MIN_LCP 16

/* 预置命名上下文（规则包）注入时从上下文窗口里保留的余量：规则前缀之外还要
 * 放得下对话与生成本身，否则 prompt 会超窗（400）。 */
#define RB_BUDGET_RESERVE 1024

/* P3 前缀复用门（VLLM_L3_PREFIX_REUSE，默认关）。
 * 语义：L3 关 → 恒允许（与改动前逐位相同）；L3 开 → 只有门开才允许。
 * 注意 g_l3_evict 可被 admin 在运行时切（vllm_server.c 的 L3 on/off 分支），
 * 所以这里只缓存 env，不缓存 g_l3_evict。 */
static int l3_prefix_reuse_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_L3_PREFIX_REUSE");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* 文本路径前缀复用是否放行。替代原先散落的 `!g_l3_evict`：
 * 那个硬门存在的原因是 L3 会把被驱逐块置 NULL，prefill 打包路径读不了。
 * 步骤 3 由 ist_reset 先调 l3_restore_prefix() 把前缀行重建回 RAM，
 * 门开才成立。
 * **危险死区**：步骤 3 接线完成前，门开会让 prefill 读 NULL 块 → 崩溃。
 * 故默认关，且步骤 3 之前不得开启。 */
static int prefix_kv_allowed(void) {
    return (!g_l3_evict) || l3_prefix_reuse_env();
}

/* Longest common prefix of two token sequences (token-exact, so a mismatch
 * anywhere safely degrades to a full prefill). */
static int kv_lcp(const int *a, int na, const int *b, int nb) {
    int n = na < nb ? na : nb;
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return n;
}

/* P4 multimodal prefix reuse: longest common prefix over token ids AND
 * per-row media marks. Text rows have mark 0 on both sides so equality is
 * decided by the token id; media rows carry the image/video byte hash, so a
 * same-pad-count different-content image stops the reuse exactly at the row
 * whose KV content would differ. */
static int mm_lcp(const int *a, const unsigned long long *ma, int na,
                  const int *b, const unsigned long long *mb, int nb) {
    int n = na < nb ? na : nb;
    for (int i = 0; i < n; i++)
        if (a[i] != b[i] || ma[i] != mb[i]) return i;
    return n;
}

/* Reset inference state for a fresh request. keep_lcp > 0 asks to KEEP the
 * KV of the first keep_lcp tokens (the request's common prefix with the
 * previous one - either left in RAM by the previous request or restored from
 * a disk KV checkpoint by st_kv_disk_load) so prefill only computes the
 * remainder. The reuse is only honored when those KV rows are actually
 * present (cache_len[0] >= keep_lcp) and prefix reuse is allowed
 * (prefix_kv_allowed(): L3 off, or L3 on with VLLM_L3_PREFIX_REUSE=1 after the
 * prefix rows have been restored from the Q4 payload). */
static void ist_reset(VLLMServerCtx *ctx, int keep_lcp) {
    STQwenInferenceState *st = ctx->ist;
    int nl = ctx->w->n_layers_allocated;
    int reuse = (ctx->prefix_kv || ctx->disk_kv) &&
                keep_lcp >= PREFIX_KV_MIN_LCP && prefix_kv_allowed() &&
                keep_lcp <= st->cache_len[0];
    if (getenv("VLLM_DEBUG_PREFIX"))
        fprintf(stderr,
                "[KV-PREFIX-DBG] ist_reset keep_lcp=%d cache_len0=%d "
                "prefix_kv=%d disk_kv=%d allowed=%d l3_evict=%d reuse=%d\n",
                keep_lcp, st->cache_len[0], ctx->prefix_kv, ctx->disk_kv,
                prefix_kv_allowed(), g_l3_evict, reuse);
    /* P3：L3 开时前缀块可能已被驱逐（k_cache[l][b] == NULL），而 prefill 的
     * 打包路径读不了 NULL 块 —— 这正是硬门存在的原因。复用前先把
     * [0, keep_lcp) 的行从 Q4 载荷重建回 RAM（纯解压，无重算）。
     * 返回 -1 = 前缀不完整，此时**绝不复用**，退回全量 prefill（正确性优先）；
     * 返回 0 = 本来就没有缺失块（L3 未驱逐 / 已全部驻留），照常复用。
     * L3 关时无需重建（prefix_kv_allowed() 已放行，块从未被释放）。 */
    if (reuse && g_l3_evict) {
        /* VLLM_L3_PROF=1：回填是一个独立阶段，墙钟 + 子项分开统计。
         * 先开 scope（fetch/fill 的子项只在回填窗口内计入），再清零计数。 */
        int prof = l3_prof_on();
        double t_r0 = 0.0;
        if (prof) { l3_prof_reset(); l3_prof_scope(1); t_r0 = l3_prof_now(); }
        int rs = l3_restore_prefix(st, keep_lcp);
        if (prof) { l3_prof_scope(0); l3_prof_report("l3-restore", l3_prof_now() - t_r0); }
        if (rs < 0) {
            fprintf(stderr, "[KV-PREFIX] L3 restore incomplete -> full prefill "
                            "(prefix=%d, evicted=%d)\n", keep_lcp, st->l3.evicted);
            fflush(stderr);
            reuse = 0;
        } else {
            fprintf(stderr, "[KV-PREFIX] L3 restored %d block(s) for %d-token "
                            "prefix\n", rs, keep_lcp);
            fflush(stderr);
        }
    }
    if (reuse) {
        /* P3 修复（2026-09-17）：l3_restore_prefix() 只把**前缀范围内**的块从 Q4
         * 载荷重建回 RAM（其 n_blocks = ceil(prefix_len/bs)），而本请求**余量**
         * （keep_lcp..n）要写入的块可能同样被 Phase-2 驱逐置成了 NULL ——
         * st_qwen_kv_rebuild_freed() 原先只在全量 prefill 分支调过。
         * 症状：部分前缀复用（针位不同的长提示）时，续算 prefill 写 NULL 块 →
         * 进程直接消失，日志停在 "reuse N-token KV prefix, prefill rest" 之后，
         * 没有任何 [PREFILL] 进度行（板端 8B / 7.2K / ROUND1 复现）。
         * 注意 vllm_safetensors.c 里该函数的注释已记录过同类事故：
         * "the previous version crashed with a NULL deref on the second prefill
         * after an eviction" —— 那次只修了全量路径。 */
        size_t rb = st_qwen_kv_rebuild_freed(st);
        if (rb > 0)
            fprintf(stderr, "[KV] rebuilt %.1f MB of L3-freed KV blocks "
                            "(reuse path)\n", (double)rb / 1048576.0);
        for (int l = 0; l < nl; l++) st->cache_len[l] = keep_lcp;
        st->seq_len = keep_lcp;
        st->mrope_pos = 0;   /* text-only; prefill_ex sets it for multimodal */
        fprintf(stderr, "[KV-PREFIX] reuse %d-token KV prefix, prefill rest\n",
                keep_lcp);
        fflush(stderr);
        return;
    }
    for (int l = 0; l < nl; l++) st->cache_len[l] = 0;
    st->seq_len = 0;
    st->mrope_pos = 0;   /* text-only default; prefill_ex sets it for multimodal */
    /* Phase-2 L3 eviction physically frees the cold KV blocks (pointers set
     * NULL). Re-back them so this request's prefill can write the new KV. */
    size_t rb = st_qwen_kv_rebuild_freed(st);
    if (rb > 0)
        fprintf(stderr, "[KV] rebuilt %.1f MB of L3-freed KV blocks\n",
                (double)rb / 1048576.0);
    /* Per-user L3 files: if st->l3 is still bound to another user's file
     * (previous request), close it so this request's eviction opens the
     * right file. Same-user requests reuse (overwrite) their own file. */
    if (st->l3.fp && ctx->l3_cur_fname[0] &&
        strcmp(st->l3.path, ctx->l3_cur_fname) != 0) {
        fprintf(stderr, "[L3] user switch: closed %s (now %s)\n",
                st->l3.path, ctx->l3_cur_fname);
        l3_state_free(&st->l3);
    }
}

/* ================================================================
 * Disk KV persistence (--disk-kv DIR)
 *
 * After each successful text request its F32 KV snapshot (token ids + K/V
 * rows, see st_kv_disk_save) is written to DIR/kv_<fnv64>.kv. On a later
 * request - including after a process restart - the checkpoint whose tokens
 * share the longest common prefix is loaded into the KV cache and only the
 * remainder is prefilled. F32 storage keeps the restore bit-exact with a
 * fresh prefill (same guarantee as the in-RAM prefix reuse); files whose
 * model geometry does not match the loaded model are skipped.
 * ================================================================ */
#define DISKKV_MAX_TOKENS 8192  /* snapshot cap raised to max_seq window
                                 * (4K/8K conversations) so a process restart
                                 * can restore long-context KV. F32 snapshot
                                 * ~224 KB/token => up to ~1.8 GB per file. */
#define DISKKV_MAX_FILES  8     /* LRU cap on checkpoint files */
#define DISKKV_SEP '/'

static uint64_t dkv_hash(const int *tokens, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t x = (uint32_t)tokens[i];
        for (int b = 0; b < 4; b++) {
            h ^= (x & 0xFFu);
            h *= 1099511628211ULL;
            x >>= 8;
        }
    }
    return h;
}

/* mkdir -p: mkdir only creates one level. */
static void dkv_mkdirs(char *dir) {
    if (!dir || !dir[0]) return;
    for (char *p = dir + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char save = *p;
            *p = '\0';
            st_mkdir(dir, 0755);
            *p = save;
        }
    }
    st_mkdir(dir, 0755);
}

/* Read one checkpoint header + tokens. Geometry must match the loaded model
 * or the file is ignored (stale model = invalid cache). Returns 0 on success
 * with e filled (e->tokens malloc'd). */
static int dkv_read_entry(VLLMServerCtx *ctx, const char *path, DiskKVEntry *e) {
    memset(e, 0, sizeof(*e));
    if (!ctx->ist || !ctx->cfg) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint8_t hdr[64];
    int rc = -1;
    if (fread(hdr, 1, 64, fp) == 64) {
        uint32_t magic, ver, nl, nkv, hd, bs, ftok, fdim;
        memcpy(&magic, hdr + 0, 4); memcpy(&ver, hdr + 4, 4);
        memcpy(&nl, hdr + 8, 4); memcpy(&nkv, hdr + 12, 4);
        memcpy(&hd, hdr + 16, 4); memcpy(&bs, hdr + 20, 4);
        memcpy(&ftok, hdr + 24, 4); memcpy(&fdim, hdr + 28, 4);
        int want_nl = ctx->ist->weights.n_layers_allocated;
        if (want_nl <= 0) want_nl = ctx->cfg->n_layers;
        /* P1b 起默认（use_kv_q8）写 v2（q8+scale）快照；v1/v2 头与 token
         * 段布局一致，scan 只需做 LCP，两者都收。载荷差异由 st_kv_disk_load
         * 按版本处理（nof32 下 v1 会在 load 处拒绝，安全退化为全量 prefill）。 */
        if (magic == 0x564C4B56u && (ver == 1 || ver == 2) &&
            (int)nl == want_nl && (int)nkv == ctx->cfg->n_kv_heads &&
            (int)hd == ctx->cfg->head_dim && (int)bs == ctx->ist->kv_bs &&
            (int)fdim == ctx->cfg->n_kv_heads * ctx->cfg->head_dim &&
            (int)ftok >= PREFIX_KV_MIN_LCP && (int)ftok <= DISKKV_MAX_TOKENS) {
            int *toks = (int *)malloc((size_t)ftok * sizeof(int));
            if (toks && fread(toks, sizeof(int), (size_t)ftok, fp) == (size_t)ftok) {
                snprintf(e->path, sizeof(e->path), "%s", path);
                e->n_tokens = (int)ftok;
                e->tokens = toks;
                struct stat sb;
                if (stat(path, &sb) == 0) e->mtime = (long)sb.st_mtime;
                rc = 0;
            } else {
                free(toks);
            }
        }
    }
    fclose(fp);
    return rc;
}

static void dkv_push(VLLMServerCtx *ctx, const DiskKVEntry *e) {
    if (ctx->dkv_n >= ctx->dkv_cap) {
        int nc = ctx->dkv_cap ? ctx->dkv_cap * 2 : 8;
        DiskKVEntry *n = (DiskKVEntry *)realloc(ctx->dkv, (size_t)nc * sizeof(DiskKVEntry));
        if (!n) return;
        ctx->dkv = n;
        ctx->dkv_cap = nc;
    }
    ctx->dkv[ctx->dkv_n++] = *e;
}

static void dkv_free_list(VLLMServerCtx *ctx) {
    for (int i = 0; i < ctx->dkv_n; i++) free(ctx->dkv[i].tokens);
    free(ctx->dkv);
    ctx->dkv = NULL;
    ctx->dkv_n = 0;
    ctx->dkv_cap = 0;
}

void vllm_server_diskkv_scan(VLLMServerCtx *ctx) {
    dkv_free_list(ctx);
    if (!ctx->disk_kv || !ctx->kvdir[0]) return;
    DIR *d = opendir(ctx->kvdir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            size_t ln = strlen(de->d_name);
            if (ln < 4 || strcmp(de->d_name + ln - 3, ".kv") != 0) continue;
            char p[576];
            snprintf(p, sizeof(p), "%s/%s", ctx->kvdir, de->d_name);
            DiskKVEntry e;
            if (dkv_read_entry(ctx, p, &e) == 0) dkv_push(ctx, &e);
        }
        closedir(d);
    }
    fprintf(stderr, "[KV-DISK] index: %d checkpoint(s) in %s\n",
            ctx->dkv_n, ctx->kvdir[0] ? ctx->kvdir : "(none)");
    fflush(stderr);
}

/* Free the server-side session state tied to the loaded model: continuous
 * batching scheduler, the in-RAM prefix-KV sequence (last_ids) and the disk
 * KV index. Called on unload so no stale context (KV/prefix/batch) survives
 * a reload — the "上下文状态清零" half of the load-on-use lifecycle. */
void vllm_server_reset_session(VLLMServerCtx *ctx) {
    if (ctx->batch) {
        vllm_batch_destroy(ctx->batch);
        ctx->batch = NULL;
    }
    free(ctx->last_ids);
    ctx->last_ids = NULL;
    ctx->last_n = 0;
    ctx->last_cap = 0;
    free(ctx->last_mark);
    ctx->last_mark = NULL;
    ctx->last_mark_cap = 0;
    ctx->last_was_mm = 0;
    dkv_free_list(ctx);
}

/* ================================================================
 * 规则包预处理（工业边缘：预置命名上下文）
 *
 * 声明见 vllm_server.h。流程：渲染 system 前导块 → ist_reset(全量) →
 * st_qwen_model_prefill_batch → st_kv_disk_save 落盘 KV 快照 → 写元数据 →
 * 复位上下文。**不新增数值路径**：prefill 与落盘全部复用既有原语。
 *
 * 超出窗口的手册（n > max_kv_slots 或 DISKKV_MAX_TOKENS）不在此落快照：运行
 * 时由 vllm_rb_select_text 分块检索注入，并在首次成功使用后懒预热快照。
 * ================================================================ */
int vllm_rb_build(VLLMServerCtx *ctx, const char *text,
                  const struct VLLMRulebook *meta, char *err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!ctx || !ctx->rulebook_on || !ctx->rulebook_dir[0]) {
        if (err) snprintf(err, errcap, "rulebook dir not configured (--rulebook-dir)");
        return -1;
    }
    if (ctx->load_state != 2 || !ctx->ist || !ctx->tok || !ctx->w) {
        if (err) snprintf(err, errcap, "model not loaded");
        return -1;
    }
    if (!text || !text[0]) {
        if (err) snprintf(err, errcap, "empty rulebook text");
        return -1;
    }
    if (strlen(text) > VLLM_RB_MAX_TEXT) {
        if (err) snprintf(err, errcap, "rulebook text too large (>%d bytes)",
                          VLLM_RB_MAX_TEXT);
        return -1;
    }
    if (meta) {
        char probe[VLLM_RB_ID_MAX];
        snprintf(probe, sizeof(probe), "%s", meta->id);
        if (!probe[0]) { if (err) snprintf(err, errcap, "rulebook id required"); return -1; }
    } else {
        if (err) snprintf(err, errcap, "rulebook metadata required");
        return -1;
    }

    VLLMRulebook rb;
    memset(&rb, 0, sizeof(rb));
    rb = *meta;
    rb.text = NULL;                     /* 源文本由调用方持有，此处不用 */
    rb.created_s = (long)time(NULL);
    rb.updated_s = rb.created_s;
    vllm_rb_hash_hex(text, strlen(text), rb.hash);
    vllm_rb_model_fp(ctx, rb.model_fp);
    rb.n_chunks = vllm_rb_chunk_count(text);

    STQwenInferenceState *st = ctx->ist;
    int max_ids = st->max_kv_slots;
    if (max_ids < 1024) max_ids = 1024;

    int *ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    int *scratch = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!ids || !scratch) {
        free(ids); free(scratch);
        if (err) snprintf(err, errcap, "out of memory");
        return -1;
    }
    /* n_tokens 必须报**真实**长度：源文本可能远超窗口，若沿用窗口大小的缓冲
     * 编码会被截断低报（超长手册的分块检索与审查视图都依赖这个数字）。
     * 最坏情况 1 token/字节，文本上限 VLLM_RB_MAX_TEXT → 至多约 1MB 缓冲，
     * 只在离线预处理路径分配，不影响服务路径。 */
    {
        size_t tcap = strlen(text) + 16;
        int *tcnt = (int *)malloc(tcap * sizeof(int));
        if (tcnt) {
            rb.n_tokens = qwen_tokenizer_encode(ctx->tok, text, tcnt, (int)tcap);
            free(tcnt);
        }
    }

    char *block = vllm_rb_render_block(text);
    if (!block) {
        free(ids); free(scratch);
        if (err) snprintf(err, errcap, "out of memory");
        return -1;
    }
    int n = qwen_tokenizer_encode(ctx->tok, block, ids, max_ids + 16);
    free(block);
    free(scratch);
    if (n <= 0) {
        free(ids);
        if (err) snprintf(err, errcap, "tokenization failed");
        return -1;
    }

    /* 元数据先落盘（同时建目录），随后才写 KV 快照。 */
    if (vllm_rb_write_meta(ctx, &rb, text, err, errcap) != 0) {
        free(ids);
        return -1;
    }

    int snap_ok = 0;
    if (n <= max_ids && n <= DISKKV_MAX_TOKENS) {
        ist_reset(ctx, 0);               /* 独立于既有上下文的全量复位 */
        if (st_qwen_model_prefill_batch(st, ids, n) != 0) {
            if (err) snprintf(err, errcap, "prefill failed");
            ist_reset(ctx, 0);
            free(ids);
            return -1;
        }
        char path[1800];
        vllm_rb_kv_path(ctx, &rb, ids, n, path, sizeof(path));
        if (st_kv_disk_save(st, path, ids, n) == 0) snap_ok = 1;
    }

    /* 复位：预处理用的 prefill 不作为后续请求的复用前缀。 */
    ist_reset(ctx, 0);
    ctx->last_n = 0;
    ctx->last_was_mm = 0;

    fprintf(stderr, "[RB] built '%s' v%s: text=%d tok, block=%d tok, chunks=%d, "
                    "model_fp=%s, kv_snapshot=%s\n",
            rb.id, rb.version[0] ? rb.version : "-", rb.n_tokens, n, rb.n_chunks,
            rb.model_fp[0] ? rb.model_fp : "-", snap_ok ? "yes" : "no (runtime lazy)");
    fflush(stderr);
    if (!snap_ok) {
        if (err) snprintf(err, errcap,
                          "metadata written; text exceeds prefill window "
                          "(block=%d > %d/%d) -> chunked injection at runtime",
                          n, max_ids, DISKKV_MAX_TOKENS);
        /* 元数据已落盘，手册可用；仅无预处理快照，不算失败。 */
    }
    free(ids);
    return 0;
}

/* Longest shared prefix of the request tokens with any checkpoint; returns
 * the best LCP (>= PREFIX_KV_MIN_LCP) and copies its file path to `path`. */
static int dkv_best_lcp(VLLMServerCtx *ctx, const int *ids, int n_ids,
                        char *path, int path_cap) {
    int best = 0;
    const char *bp = NULL;
    for (int i = 0; i < ctx->dkv_n; i++) {
        const DiskKVEntry *e = &ctx->dkv[i];
        int l = kv_lcp(e->tokens, e->n_tokens, ids, n_ids);
        /* Skip checkpoints that fully cover the request (l == n_ids): keep
         * = n_ids would prefill 0 tokens, which cannot refresh logits and
         * decode would sample stale logits (predicts eos) -> empty response.
         * Prefer the longest proper prefix strictly below n_ids so the
         * caller's `l < n_ids` guard never starves the whole disk-kv path
         * (falling back to the full prefill). */
        if (l > best && l < n_ids) {
            best = l;
            bp = e->path;
        }
    }
    if (best >= PREFIX_KV_MIN_LCP && bp) {
        snprintf(path, path_cap, "%s", bp);
        return best;
    }
    return 0;
}

/* Snapshot the finished conversation (token ids + KV for those tokens) into
 * kv_<fnv64>.kv, enforce the LRU file cap, and refresh the index. */
static void dkv_save(VLLMServerCtx *ctx, const int *ids, int n_ids) {
    if (!ctx->disk_kv || !ctx->kvdir[0] || !ctx->ist) return;
    if (n_ids <= 0 || n_ids > DISKKV_MAX_TOKENS) return;
    if (n_ids > ctx->ist->cache_len[0]) return;   /* KV not computed yet */
    char dirbuf[512];
    snprintf(dirbuf, sizeof(dirbuf), "%s", ctx->kvdir);
    dkv_mkdirs(dirbuf);
    char path[576];
    snprintf(path, sizeof(path), "%s%c%s%016llx%s", ctx->kvdir, DISKKV_SEP,
             "kv_", (unsigned long long)dkv_hash(ids, n_ids), ".kv");
    if (st_kv_disk_save(ctx->ist, path, ids, n_ids) != 0) return;
    fprintf(stderr, "[KV-DISK] saved %d-token KV -> %s\n", n_ids, path);
    fflush(stderr);
    /* LRU: drop the oldest file (never the one just written) over the cap. */
    char del_path[576] = "";
    long oldest = 0;
    int cnt = 0;
    DIR *d = opendir(ctx->kvdir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            size_t ln = strlen(de->d_name);
            if (ln < 4 || strcmp(de->d_name + ln - 3, ".kv") != 0) continue;
            cnt++;
            char p2[576];
            snprintf(p2, sizeof(p2), "%s/%s", ctx->kvdir, de->d_name);
            if (strcmp(p2, path) != 0) {
                struct stat sb;
                if (stat(p2, &sb) == 0 && (oldest == 0 || sb.st_mtime < oldest)) {
                    oldest = (long)sb.st_mtime;
                    snprintf(del_path, sizeof(del_path), "%s", p2);
                }
            }
        }
        closedir(d);
    }
    if (cnt > DISKKV_MAX_FILES && del_path[0]) {
        remove(del_path);
        fprintf(stderr, "[KV-DISK] LRU evicted %s\n", del_path);
        fflush(stderr);
    }
    vllm_server_diskkv_scan(ctx);
}

/* ---------- /v1/models ---------- */

static void handle_models(VLLMServerCtx *ctx, VHttpResponse *resp) {
    char *body = (char *)malloc(2048);
    if (!body) { json_error(resp, 500, "Out of memory"); return; }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "object", vjson_new_string("list"));
    VJson *data = vjson_new_array();
    VJson *m = vjson_new_object();
    vjson_obj_set(m, "id", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    vjson_obj_set(m, "object", vjson_new_string("model"));
    vjson_obj_set(m, "created", vjson_new_number((double)unix_now()));
    vjson_obj_set(m, "owned_by", vjson_new_string("vllm_kestrel"));
    vjson_array_push(data, m);
    vjson_obj_set(root, "data", data);
    size_t n = vjson_serialize(root, body, 2048);
    vjson_free(root);
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body = body;
    resp->body_len = n;
    resp->body_owned = body;
}

/* ---------- inference core ---------- */

/* ================================================================
 * Incremental UTF-8 decoder.
 *
 * vocab.bin stores RAW bytes (bytes_to_unicode inverse). Fragment tokens
 * (mid-BPE emoji states) are NOT valid UTF-8 on their own — e.g. the 3
 * bytes F0 9F 98 are the first 3 bytes of a 4-byte emoji. Decoding each
 * token in isolation is what produced "ðŁĺ" garbage. This decoder keeps an
 * incomplete multi-byte sequence in `pending` and splices it with the next
 * token's bytes, so fragments reassemble into a correct emoji.
 * ================================================================ */
typedef struct {
    unsigned char pending[4];
    int expect;   /* 0 = none; 2/3/4 = expected total sequence length */
    int npend;    /* bytes currently held in pending */
} Utf8Incr;

static void utf8_incr_init(Utf8Incr *d) {
    d->expect = 0;
    d->npend = 0;
}

/* Feed raw bytes; append decoded (complete) chars to out; returns out len. */
static int utf8_incr_feed(Utf8Incr *d, const unsigned char *buf, int len,
                          char *out, int out_cap) {
    int w = 0;
    for (int i = 0; i < len; i++) {
        unsigned char b = buf[i];
        if (d->expect == 0) {
            if (b < 0x80) {
                if (w < out_cap - 1) out[w++] = (char)b;
            } else if (b >= 0xC2 && b <= 0xDF) {
                d->expect = 2; d->pending[0] = b; d->npend = 1;
            } else if (b >= 0xE0 && b <= 0xEF) {
                d->expect = 3; d->pending[0] = b; d->npend = 1;
            } else if (b >= 0xF0 && b <= 0xF4) {
                d->expect = 4; d->pending[0] = b; d->npend = 1;
            } else {
                /* stray continuation byte or invalid lead -> U+FFFD */
                if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; }
            }
        } else {
            d->pending[d->npend++] = b;
            if (d->npend == d->expect) {
                int ok = 1;
                for (int k = 1; k < d->expect; k++)
                    if ((d->pending[k] & 0xC0) != 0x80) { ok = 0; break; }
                if (ok) {
                    for (int k = 0; k < d->expect; k++)
                        if (w < out_cap - 1) out[w++] = (char)d->pending[k];
                } else {
                    if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; }
                }
                d->expect = 0; d->npend = 0;
            }
        }
    }
    out[w] = '\0';
    return w;
}

/* Replace byte-level BPE markers Ġ(U+0120, C4 A0) -> space and Ċ(U+010A,
 * C4 8A) -> newline inside an already-decoded UTF-8 string. A piece may
 * contain several markers (e.g. "ĠĊ" from multi-token BPE output). */
static const char *gj_replace(const char *s, char *tmp, size_t tmplen) {
    if (!s || !s[0]) return s;
    const unsigned char *u = (const unsigned char *)s;
    int len = (int)strlen(s);
    int has_marker = 0;
    for (int i = 0; i + 1 < len; i++)
        if (u[i] == 0xC4 && u[i + 1] >= 0x80) {
            has_marker = 1;
            break;
        }
    if (!has_marker) return s;
    int w = 0;
    for (int i = 0; i < len && w < (int)tmplen - 1; i++) {
        if (u[i] == 0xC4 && i + 1 < len && u[i + 1] >= 0x80) {
            unsigned char b = (unsigned char)(u[i + 1] - 0x80);
            /* byte-level BPE 伪字符 U+0100+ -> 真实字节。\n(0x0A)、空格(0x20)
             * 与可打印/多字节字节还原为真实字符（修复含英文/数字/标点的回复
             * 在客户端被丢弃显示的问题）；其他控制字符(0x00-0x1F 除 \n)保留
             * 伪字符形式由客户端兜底丢弃，避免裸控制字符破坏 JSON。 */
            if (b == 0x0A || b == 0x20 || b >= 0x21) { tmp[w++] = (char)b; i++; continue; }
            i++;
            continue;
        }
        tmp[w++] = s[i];
    }
    tmp[w] = '\0';
    return tmp;
}

/* Shared autoregressive decode loop (sample + forward + optional SSE).
 * Collects text into out (must be non-NULL). When spec is enabled the loop
 * first tries an n-gram speculative decode round (--spec): a draft of K
 * tokens is generated from an earlier repetition of the current context
 * suffix and verified in ONE batched prefill (weights read once for all K
 * positions); accepted drafts are emitted together and only the rejected /
 * bonus token is forwarded. Greedy-only, text-only, KV prefix/L3 disabled. */

/* The current request's full token sequence = prompt_ids + gids. */
static int spec_seq_at(const int *prompt, int n_prompt,
                       const int *gids, int gn, int i) {
    return i < n_prompt ? prompt[i] : gids[i - n_prompt];
}

/* n-gram draft: find an earlier occurrence of the current context's tail
 * (window size longest-first, 8 down to 3 tokens) and take up to `k` tokens
 * that followed that occurrence. Longer windows win (more reliable
 * continuation); exact-repetition output (refrains, fixed-form lists, code
 * templates) drafts fully, while counter/varies-text patterns systematically
 * diverge and degrade to normal decode. Returns the draft length. */
static int spec_build_draft(const int *prompt, int n_prompt,
                            const int *gids, int gn, int k, int *draft) {
    int len = n_prompt + gn;
    if (len < 6 || k < 1 || k > SPEC_DRAFT_MAX) return 0;
    for (int w = 8; w >= 3; w--) {
        if (len < w + 2) continue;
        int lo = len - w;              /* window = [lo, len) */
        for (int s = 0; s + w + 1 <= lo; s++) {   /* candidate window start */
            int match = 1;
            for (int j = 0; j < w; j++) {
                if (spec_seq_at(prompt, n_prompt, gids, gn, s + j) !=
                    spec_seq_at(prompt, n_prompt, gids, gn, lo + j)) { match = 0; break; }
            }
            if (!match) continue;
            int avail = len - (s + w);
            int n = avail < k ? avail : k;
            for (int i = 0; i < n; i++)
                draft[i] = spec_seq_at(prompt, n_prompt, gids, gn, s + w + i);
            return n;
        }
    }
    return 0;
}

/* VLLM_SPEC_VEXACT=1：声明「验证路径已与 decode 位级等价」，因而可直接采用
 * `st_qwen_model_prefill_batch` 写入的 KV，免去 KV 回滚 + 逐 token 重放。
 * 依据：L1 对拍实测 PASS（2026-09-24，板端 8B-q4，`VLLM_SPEC_VNORM=1` +
 * VEXACT + `spec_k<=3` ⇒ 逐层 KV 行与逐 token decode 全等，见
 * `src/model/vllm_safetensors.c` 的 spec_verify_attention 注释）。
 * 前置条件（缺一不可，否则应回退到重放路径）：
 *   ① VLLM_SPEC_VNORM=1（批式 RMSNorm 归约序对齐）
 *   ② VLLM_SPEC_VEXACT=1（验证注意力走 decode 同核 + 小批强制 C tile 序）
 *   ③ spec_k <= 8（VEXACT 下 `q4x4_gemm_batched` 对 n_batch<=8 强制逐 token
 *      C tile 序且 tile 拉满；>8 会回到非等价的 M4h 主段）
 * 默认 0 = 原「回滚 + 重放」路径（零回归）。 */
static int spec_vexact_env(void) {
    const char *e = getenv("VLLM_SPEC_VEXACT");
    return (e && e[0] == '1') ? 1 : 0;
}

/* VLLM_SPEC_FAST=1：放开「位级一致」硬约束，改求「输出 token 几乎一致」。
 * 验证走 st_qwen_model_prefill_batch 的默认路径（n_batch=K>3 自动走 asm 快路
 * GEMM + batched attention），免去 KV 回滚 + 逐 token 重放，直接采用验证 KV
 * 与验证 logits。代价：验证路径（asm 快路 f16 scale + 32 元素一次）与 decode
 * 路径（C tile f32 scale + 16 元素一次）位级不同，logits 差 ULP + 系统性 f16
 * 偏差，near-tie 位置可能翻转 → 输出与无 spec 几乎一致但非逐 token 位级一致
 * （重复/模板文本更明显，历史既证 repeat_same 90 vs 86）。
 * 默认 0 = 原「回滚 + 重放」路径（位级一致，零回归）。 */
static int spec_fast_env(void) {
    const char *e = getenv("VLLM_SPEC_FAST");
    return (e && e[0] == '1') ? 1 : 0;
}

/* Emit one generated token: record id, decode to text, append to the output
 * (SSE chunk or buffer), update gen / first-token time. Returns 1 if the
 * token is an end token (stop decoding), -1 if the streaming client went
 * away (abort), 0 on success. */
static int emit_token(VLLMServerCtx *ctx, QwenTokenizer *tok, Utf8Incr *dec,
                      int id, int stream, VHttpConn *conn, StrBuf *out,
                      VLLMMetrics *met, int *gen, double *t_first,
                      const char *chunk_id, long created, const char *model,
                      char *dbuf, int dbuf_cap, int *gids, int gcap, int *gn) {
    if (id == tok->eos_id || id == tok->im_end_id) return 1;
    if (gids && *gn < gcap) gids[(*gn)++] = id;
    const char *raw = qwen_tokenizer_decode(tok, id);
    int rawlen = tok->str_lens[id];
    char piece[2048];
    int plen = utf8_incr_feed(dec, (const unsigned char *)raw, rawlen,
                              piece, (int)sizeof(piece));
    char pbuf[2048];
    const char *text = gj_replace(piece, pbuf, sizeof(pbuf));
    if (plen > 0 && text && text[0]) {
        if (stream) {
            VJson *root = vjson_new_object();
            vjson_obj_set(root, "id", vjson_new_string(chunk_id));
            vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
            vjson_obj_set(root, "created", vjson_new_number((double)created));
            vjson_obj_set(root, "model", vjson_new_string(model));
            VJson *choices = vjson_new_array();
            VJson *ch = vjson_new_object();
            vjson_obj_set(ch, "index", vjson_new_number(0));
            VJson *delta = vjson_new_object();
            vjson_obj_set(delta, "content", vjson_new_string(text));
            vjson_obj_set(ch, "delta", delta);
            vjson_obj_set(ch, "finish_reason", vjson_new_null());
            vjson_array_push(choices, ch);
            vjson_obj_set(root, "choices", choices);
            size_t n = vjson_serialize(root, dbuf, (size_t)dbuf_cap);
            vjson_free(root);
            char line[4300];
            int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
            if (ln > 0 && vhttp_stream_write(conn, line, (size_t)ln) < 0)
                return -1;
            /* Stream 模式也累计输出文本：流式出证需要全量 text。 */
            if (out) sb_str(out, text);
        } else {
            sb_str(out, text);
        }
    }
    (*gen)++;
    if (met && *gen == 1) *t_first = st_now_sec();
    return 0;
}

static int decode_loop(VLLMServerCtx *ctx, double temperature, double top_p,
                       double min_p, int top_k, int thinking, int max_tokens,
                       int stream,
                       VHttpConn *conn, int emit_meta, StrBuf *out,
                       VLLMMetrics *met, int **gen_ids_out, int *gen_n_out,
                       const int *prompt_ids, int n_prompt,
                       const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    QwenTokenizer *tok = ctx->tok;
    int vc = ctx->cfg->vocab_size;

    /* Optional collector of the generated token ids (for KV prefix reuse:
     * they extend the previous request's KV sequence). */
    int *gids = NULL;
    int gn = 0, gcap = 0;
    if (gen_ids_out) {
        *gen_ids_out = NULL;
        *gen_n_out = 0;
        if (max_tokens > 0) {
            gids = (int *)malloc((size_t)max_tokens * sizeof(int));
            if (gids) gcap = max_tokens;
        }
    }

    /* SSE chunk id (thread-safe: workers may stream concurrently). */
    vhttp_mutex_lock(ctx->stat_lock);
    long my_id = ctx->next_chunk_id++;
    vhttp_mutex_unlock(ctx->stat_lock);
    char chunk_id[64];
    snprintf(chunk_id, sizeof(chunk_id), "chatcmpl-%08lx", my_id);

    long created = unix_now();
    const char *model = ctx->model_id ? ctx->model_id : "qwen3-vl-8b";
    char dbuf[4096];
    int done = 0;
    const char *finish = "stop";
    double t_first = 0.0;
    int gen = 0;
    Utf8Incr dec;
    utf8_incr_init(&dec);

    /* Speculative decode setup (--spec). Greedy-only (temp<=0/top_p>=1/
     * min_p<=0), text-only (prompt ids known), and off under L3 eviction or
     * continuous batching (they share the same inference state). */
    int spec_on = ctx->spec && prompt_ids && n_prompt > 0 &&
                  temperature <= 0.0 && top_p >= 1.0 && min_p <= 0.0 &&
                  !g_l3_evict && !g_sparse_attn && ctx->batch_max < 2;
    int spec_k = ctx->spec_k;
    if (spec_k < 1) spec_k = 1;
    if (spec_k > SPEC_DRAFT_MAX) spec_k = SPEC_DRAFT_MAX;
    /* VEXACT 前置条件③：spec_k<=8。q4 GEMM 在 n_batch>3 时会切到「M4h 舍入序」
     * 主段；VEXACT 下 `q4x4_gemm_batched` 对 n_batch<=8 强制走逐 token C tile 序
     * （与 decode 同一 worker ⇒ 位级等价）并把 tile 拉满到整批，故上限取该区间 8。 */
    int spec_vexact = spec_vexact_env();
    if (spec_vexact && spec_k > 8) spec_k = 8;
    if (spec_vexact) {
        static int vexact_logged = 0;
        if (!vexact_logged) {
            vexact_logged = 1;
            fprintf(stderr, "[SPEC] VLLM_SPEC_VEXACT=1：spec_k<=%d、验证 KV 直接采用"
                            "（免重放）；前置 VLLM_SPEC_VNORM=1 须同时开启\n", spec_k);
            fflush(stderr);
        }
    }
    /* VLLM_SPEC_FAST：放开位级一致（免重放 + asm 快路验证，输出几乎一致）。 */
    int spec_fast = spec_fast_env();
    if (spec_fast) {
        static int fast_logged = 0;
        if (!fast_logged) {
            fast_logged = 1;
            fprintf(stderr, "[SPEC] VLLM_SPEC_FAST=1：验证走 asm 快路 + 免重放，"
                            "输出与无 spec 几乎一致（near-tie 可能翻转，非位级一致）\n");
            fflush(stderr);
        }
    }
    float *vlog = NULL;
    if (spec_on) {
        /* +1 extra batch slot: slot K holds the pre-verify context logits
         * backup for the lossless decode-path replay (see spec block). */
        vlog = (float *)malloc((size_t)(SPEC_DRAFT_MAX + 1) * vc * sizeof(float));
        if (!vlog) spec_on = 0;
    }
    /* 神经草稿头（P2）：VLLM_SPEC_DRAFT=<f32 权重文件> 时用 1 层草稿头替换 n-gram
     * 草稿生成。草稿头前向吃主模型最后一层 hidden（st->hidden）。 */
    STDraftHead draft_head;
    int draft_head_on = 0;
    memset(&draft_head, 0, sizeof(draft_head));
    if (spec_on) {
        const char *dp = getenv("VLLM_SPEC_DRAFT");
        if (dp && dp[0]) {
            if (st_draft_head_load(&draft_head, dp, st->cfg.dim, st->cfg.ffn_dim,
                                   st->cfg.n_kv_heads, st->cfg.head_dim) == 0) {
                draft_head_on = 1;
                fprintf(stderr, "[SPEC] VLLM_SPEC_DRAFT=%s：神经草稿头（f32 1层）替换 n-gram\n", dp);
                fflush(stderr);
            } else {
                fprintf(stderr, "[SPEC] VLLM_SPEC_DRAFT 加载失败：%s\n", dp);
                fflush(stderr);
            }
        }
    }
    int spec_tries = 0, spec_hits = 0, spec_kv_rolls = 0;

    while (gen < max_tokens) {
        int id = -1;
        int emit_n = 0;         /* draft tokens to emit this round */
        int emit_extra = -1;    /* rejected/bonus token after the drafts */
        int draft[SPEC_DRAFT_MAX];
        int nl = st->weights.n_layers_allocated;

        if (spec_on && gn < gcap) {
            int K;
            if (draft_head_on)
                K = st_draft_head_forward(st, &draft_head, st->hidden, spec_k, draft, vlog);
            else
                K = spec_build_draft(prompt_ids, n_prompt, gids, gn, spec_k, draft);
            if (K > 0) {
                spec_tries++;
                /* Draft starts at the current KV length (multimodal prompts
                 * include visual tokens, so cache_len may exceed the text
                 * token count n_prompt + gn). */
                int L = st->cache_len[0];
                /* Context logits predict the next token; save the greedy pick
                 * before the verify clobbers st->logits. */
                int q_top = 0;
                float qb = -1.0e30f;
                for (int t = 0; t < vc; t++)
                    if (st->logits[t] > qb) { qb = st->logits[t]; q_top = t; }
                /* Backup the pre-verify context logits (decode path) BEFORE
                 * the batched verify clobbers st->logits; the lossless replay
                 * below restores it to re-derive the draft picks. */
                float *saved_logits = vlog + (size_t)K * vc;
                memcpy(saved_logits, st->logits, (size_t)vc * sizeof(float));
                for (int i = 0; i < K; i++) g_verify_draft[i] = draft[i];
                g_verify_invalid = 0;
                /* Verify positions the draft at the decode position base
                 * (mrope_pos for multimodal, seq_len for text-only). */
                int gv_pos = st->mrope_pos > 0 ? st->mrope_pos : st->seq_len;
                g_verify_pos = gv_pos;
                g_verify_logits = vlog;
                int pf = st_qwen_model_prefill_batch(st, draft, K);
                g_verify_logits = NULL;
                g_verify_pos = -1;
                /* Acceptance probe: draft[0] must match the context pick,
                 * then each position's prediction must match the next draft
                 * token. This ONLY probes the upper bound - the batched
                 * verify runs a different numeric path than single-token
                 * decode (fused-batched GEMM vs single-row GEMM), so its KV
                 * and top-1 can drift from decode near stop boundaries
                 * (2026-08-31: repeat_same produced 90 vs 86 tokens). The
                 * accepted prefix is therefore replayed through the decode
                 * path below for a lossless, KV-exact confirmation. */
                int acc_probe = (pf == 0 && !g_verify_invalid &&
                                 q_top == draft[0]) ? 1 : 0;
                while (acc_probe < K &&
                       g_verify_pred[acc_probe - 1] == draft[acc_probe]) acc_probe++;
                /* Lossless replay: roll KV back to L, restore the context
                 * logits, then confirm each draft token through the decode
                 * path (st_qwen_model_forward). This keeps KV bit-exact with
                 * normal decoding; the batched verify above only caps how
                 * far the replay may go. */
                for (int l = 0; l < nl; l++) st->cache_len[l] = L;
                st->seq_len = L;
                if (st->mrope_pos > 0) st->mrope_pos = gv_pos;
                memcpy(st->logits, saved_logits, (size_t)vc * sizeof(float));
                int real_acc = 0;
                int fast = spec_fast || spec_vexact;
                if (fast) {
                    /* FAST/VEXACT 免重放：直接采用验证 KV——把长度计数设回
                     * 「已接受前缀」即可（被拒位置的行落在 cache_len 之外，
                     * 稍后被正常 forward 覆盖），logits 取该前缀末位的验证输出。
                     *   VEXACT = 验证 KV 与 decode 位级等价（L1 实测 PASS）
                     *   FAST   = 放开位级一致，验证走 asm 快路（输出几乎一致） */
                    real_acc = acc_probe;
                    for (int l = 0; l < nl; l++) st->cache_len[l] = L + real_acc;
                    st->seq_len = L + real_acc;
                    if (st->mrope_pos > 0) st->mrope_pos = gv_pos + real_acc;
                    if (real_acc > 0)
                        memcpy(st->logits, vlog + (size_t)(real_acc - 1) * vc,
                               (size_t)vc * sizeof(float));
                } else {
                    for (int i = 0; i < acc_probe; i++) {
                        int top = 0;
                        float tb = -1.0e30f;
                        for (int t = 0; t < vc; t++)
                            if (st->logits[t] > tb) { tb = st->logits[t]; top = t; }
                        if (top == tok->eos_id || top == tok->im_end_id) break;
                        if (top != draft[i]) break;
                        st_qwen_model_forward(st, draft[i]);
                        real_acc++;
                    }
                }
                if (getenv("VLLM_SPEC_DBG")) {
                    int q2 = 0;
                    float qv2 = -1.0e30f;
                    for (int t = 0; t < vc; t++)
                        if (t != q_top && st->logits[t] > qv2) { qv2 = st->logits[t]; q2 = t; }
                    fprintf(stderr, "[SPEC-DBG] L=%d q_top=%d(q2=%d,m=%.3f) K=%d draft=[",
                            L, q_top, q2, qb - qv2, K);
                    for (int i = 0; i < K; i++) fprintf(stderr, "%d%s", draft[i], i + 1 < K ? "," : "");
                    fprintf(stderr, "] gvp=[");
                    for (int i = 0; i < K; i++) fprintf(stderr, "%d(%.2f)%s", g_verify_pred[i],
                                                        g_verify_margin[i], i + 1 < K ? "," : "");
                    fprintf(stderr, "] probe=%d replay=%d\n", acc_probe, real_acc);
                    fflush(stderr);
                }
                if (real_acc >= 1) {
                    /* KV now holds the decode-path-accepted prefix (bit-exact
                     * with normal decode). The bonus token is the decode
                     * path's own prediction at L+real_acc. */
                    spec_kv_rolls += K - real_acc;
                    spec_hits += real_acc;
                    emit_n = real_acc;
                    emit_extra = 0;
                    float eb = -1.0e30f;
                    for (int t = 0; t < vc; t++)
                        if (st->logits[t] > eb) { eb = st->logits[t]; emit_extra = t; }
                } else {
                    /* First draft token mismatches the decode path: keep the
                     * context's own pick (KV already rolled back to L). */
                    spec_kv_rolls += K;
                    id = q_top;
                }
            }
        }

        if (emit_n > 0) {
            /* Emit the accepted draft tokens, then the rejected-position or
             * bonus token (all greedy-verified). */
            int stop = 0;
            int extra_emitted = 0;
            for (int i = 0; i < emit_n && gen < max_tokens && stop == 0; i++) {
                int r = emit_token(ctx, tok, &dec, draft[i], stream, conn, out,
                                   met, &gen, &t_first, chunk_id, created, model,
                                   dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
                if (r != 0) stop = r;
            }
            if (emit_extra >= 0 && gen < max_tokens && stop == 0) {
                int r = emit_token(ctx, tok, &dec, emit_extra, stream, conn, out,
                                   met, &gen, &t_first, chunk_id, created, model,
                                   dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
                if (r != 0) stop = r;
                else extra_emitted = 1;
            }
            if (stop < 0) { done = -1; break; }   /* client went away */
            if (stop == 1) { done = 1; break; }   /* eos/im_end */
            /* Restore the model state after the accepted prefix: the extra
             * token (rejected-position pick or bonus) is the real next token
             * and must be forwarded to rebuild KV + logits at L+acc (same
             * contract as a normal decode step: even the max_tokens-th token
             * is forwarded so KV stays consistent with the token history). */
            if (extra_emitted) st_qwen_model_forward(st, emit_extra);
        } else if (id >= 0) {
            /* Spec round fully rejected: emit the context pick and forward. */
            int r = emit_token(ctx, tok, &dec, id, stream, conn, out,
                               met, &gen, &t_first, chunk_id, created, model,
                               dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
            if (r < 0) { done = -1; break; }
            if (r == 1) { done = 1; break; }
            st_qwen_model_forward(st, id);
        } else {
            /* Normal single-token decode step. */
            /* 插桩（VLLM_DEBUG_LOGITS=1）：把本步 logits 打成一行指纹，用来判
             * 「同一输入 → logits 是否逐位相同」。指纹 = FNV-1a over raw bytes
             * （能分辨 -0.0/+0.0 这类位级差异），另附 top-2 id 与差值：
             *   指纹相同而选中 token 不同 → 采样侧问题；
             *   指纹不同 → 数值侧问题，margin 量级说明是否 near-tie 翻转。
             * 第一个 decode 步的 logits 就是 prefill 的产物，故首行即 prefill 指纹。 */
            if (getenv("VLLM_DEBUG_LOGITS")) {
                const unsigned char *lb = (const unsigned char *)st->logits;
                unsigned long long h = 1469598103934665603ULL;
                for (size_t bi = 0; bi < (size_t)vc * sizeof(float); bi++) {
                    h ^= lb[bi];
                    h *= 1099511628211ULL;
                }
                int b1 = 0, b2 = -1;
                for (int i = 1; i < vc; i++)
                    if (st->logits[i] > st->logits[b1]) b1 = i;
                for (int i = 0; i < vc; i++)
                    if (i != b1 && (b2 < 0 || st->logits[i] > st->logits[b2])) b2 = i;
                fprintf(stderr,
                        "[LOGITS-DBG] seq=%d gen=%d fnv=%016llx top1=%d top2=%d "
                        "margin=%.9g\n",
                        st->seq_len, gen, h, b1, (b2 < 0 ? -1 : b2),
                        (b2 < 0 ? 0.0 : (double)(st->logits[b1] - st->logits[b2])));
                fflush(stderr);
            }
            id = sample_token_pk(st->logits, vc, (float)temperature,
                                 (float)top_p, (float)min_p, top_k);
            if (id == tok->eos_id || id == tok->im_end_id) { finish = "stop"; done = 1; break; }
            if (gids && gn < gcap) gids[gn++] = id;
            const char *raw = qwen_tokenizer_decode(tok, id);
            int rawlen = tok->str_lens[id];
            char piece[2048];
            int plen = utf8_incr_feed(&dec, (const unsigned char *)raw, rawlen,
                                       piece, (int)sizeof(piece));
            char pbuf[2048];
            const char *text = gj_replace(piece, pbuf, sizeof(pbuf));
            if (plen > 0 && text && text[0]) {
                if (stream) {
                    VJson *root = vjson_new_object();
                    vjson_obj_set(root, "id", vjson_new_string(chunk_id));
                    vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
                    vjson_obj_set(root, "created", vjson_new_number((double)created));
                    vjson_obj_set(root, "model", vjson_new_string(model));
                    VJson *choices = vjson_new_array();
                    VJson *ch = vjson_new_object();
                    vjson_obj_set(ch, "index", vjson_new_number(0));
                    VJson *delta = vjson_new_object();
                    vjson_obj_set(delta, "content", vjson_new_string(text));
                    vjson_obj_set(ch, "delta", delta);
                    vjson_obj_set(ch, "finish_reason", vjson_new_null());
                    vjson_array_push(choices, ch);
                    vjson_obj_set(root, "choices", choices);
                    size_t n = vjson_serialize(root, dbuf, sizeof(dbuf));
                    vjson_free(root);
                    char line[4300];
                    int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
                    /* Client went away: stop decoding now so the inference
                     * lock is released for the next queued request. */
                    if (ln > 0 && vhttp_stream_write(conn, line, (size_t)ln) < 0) {
                        done = -1;
                        break;
                    }
                    /* Stream 累计输出文本（出证需要全量 text）。 */
                    if (out) sb_str(out, text);
                } else {
                    sb_str(out, text);
                }
            }
            gen++;
            if (met && gen == 1) t_first = st_now_sec();   /* first token produced */
            st_qwen_model_forward(st, id);
        }
    }
    if (done != 1) finish = "length";
    if (spec_on && spec_tries > 0) {
        fprintf(stderr, "[SPEC] tries=%d hits=%d kv_rolls=%d (%.0f%% draft hit)\n",
                spec_tries, spec_hits, spec_kv_rolls,
                spec_tries ? 100.0 * (double)spec_hits / (double)(spec_tries * spec_k) : 0.0);
        fflush(stderr);
    }
    free(vlog);
    if (draft_head_on) st_draft_head_free(&draft_head);

    if (met) {
        double t_end = st_now_sec();
        met->n_tokens = gen;
        if (t_first > 0) met->ttft_ms = (t_first - met->start_s) * 1000.0;
        met->total_ms = (t_end - met->start_s) * 1000.0;
        if (gen > 1) met->tpot_ms = (t_end - t_first) / (double)(gen - 1) * 1000.0;
        else if (t_first > 0) met->tpot_ms = met->ttft_ms;
    }

    if (stream && done != -1) {
        VJson *root = vjson_new_object();
        vjson_obj_set(root, "id", vjson_new_string(chunk_id));
        vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
        vjson_obj_set(root, "created", vjson_new_number((double)created));
        vjson_obj_set(root, "model", vjson_new_string(model));
        VJson *choices = vjson_new_array();
        VJson *ch = vjson_new_object();
        vjson_obj_set(ch, "index", vjson_new_number(0));
        vjson_obj_set(ch, "delta", vjson_new_object());
        vjson_obj_set(ch, "finish_reason", vjson_new_string(finish));
        vjson_array_push(choices, ch);
        vjson_obj_set(root, "choices", choices);
        size_t n = vjson_serialize(root, dbuf, sizeof(dbuf));
        vjson_free(root);
        char line[4300];
        int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
        if (ln > 0) vhttp_stream_write(conn, line, (size_t)ln);
        /* Final event: per-request metrics for the chat client.
         * 标准 OpenAI 客户端只认 choices 事件，故自定义事件仅在客户端显式
         * opt-in（请求头 X-Kestrel-Meta）时下发，默认流严格标准。 */
        if (emit_meta && met) {
            char mline[768];
            double tok_s = met->total_ms > 0.0
                           ? (double)met->n_tokens * 1000.0 / met->total_ms : 0.0;
            int ln2 = snprintf(mline, sizeof(mline),
                     "data: {\"object\":\"chat.completion.metrics\",\"metrics\":"
                     "{\"prompt_tokens\":%d,\"last_user_tokens\":%d,\"n_tokens\":%d,"
                     "\"ttft_ms\":%.1f,\"tpot_ms\":%.1f,\"total_ms\":%.1f,"
                     "\"prefill_ms\":%.1f,\"tok_s\":%.2f,"
                     "\"trace_id\":\"%s\",\"rulebook_id\":\"%s\","
                     "\"rulebook_version\":\"%s\",\"rulebook_tokens\":%d,"
                     "\"reused_prefix\":%s}}\n\n",
                     met->prompt_tokens, met->last_user_tokens, met->n_tokens,
                     met->ttft_ms, met->tpot_ms, met->total_ms, met->prefill_ms, tok_s,
                     met->trace_id, met->rulebook_id, met->rulebook_version,
                     met->rulebook_tokens, met->reused_prefix ? "true" : "false");
            if (ln2 > 0 && ln2 < (int)sizeof(mline))
                vhttp_stream_write(conn, mline, (size_t)ln2);
        }
        /* history_tokens 事件：完整 token 序列（prompt + generated），客户端
         * 保存供下轮 context_tokens 精确复用。 */
        if (emit_meta) {
            int total = n_prompt + gn;
            size_t cap = (size_t)total * 8 + 128;
            char *tbuf = (char *)malloc(cap);
            if (tbuf) {
                int off = 0;
                off += snprintf(tbuf + off, cap - (size_t)off,
                                "data: {\"object\":\"chat.completion.tokens\",\"tokens\":[");
                int idx = 0;
                for (int i = 0; i < n_prompt; i++) {
                    int w = snprintf(tbuf + off, cap - (size_t)off, "%s%d",
                                     idx ? "," : "", prompt_ids[i]);
                    if (w <= 0 || off + w >= (int)cap - 2) break;
                    off += w; idx++;
                }
                for (int i = 0; i < gn; i++) {
                    int w = snprintf(tbuf + off, cap - (size_t)off, "%s%d",
                                     idx ? "," : "", gids[i]);
                    if (w <= 0 || off + w >= (int)cap - 2) break;
                    off += w; idx++;
                }
                off += snprintf(tbuf + off, cap - (size_t)off, "]}\n\n");
                vhttp_stream_write(conn, tbuf, (size_t)off);
                free(tbuf);
            }
        }
        /* 方案 2：流式出证。对所有文本已累计（out）的流式响应附一个
         * chat.completion.attest 事件（含 SM2 签名凭证）。 */
        if (emit_meta && vatt_active() && out && out->s) {
            VAttestReq ar;
            memset(&ar, 0, sizeof(ar));
            ar.user = att_user;
            ar.body_sha = att_bodyhex;   /* SM3(原始请求体) 64-hex */
            ar.text = out->s;
            ar.text_len = out->len;
            ar.n_prompt_tokens = (met && met->prompt_tokens > 0)
                                     ? met->prompt_tokens : n_prompt;
            ar.n_gen_tokens = gen;
            ar.finish = finish;
            ar.temperature = temperature;
            ar.top_p = top_p;
            ar.min_p = min_p;
            ar.top_k = top_k;
            ar.thinking = thinking;
            ar.max_tokens = max_tokens;
            ar.ts = created;
            char *aj = vatt_seal_json(&ar);
            if (aj) {
                char aline[4500];
                int ln = snprintf(aline, sizeof(aline),
                                  "data: {\"object\":\"chat.completion.attest\","
                                  "\"attest\":%s}\n\n", aj);
                if (ln > 0 && ln < (int)sizeof(aline))
                    vhttp_stream_write(conn, aline, (size_t)ln);
                free(aj);
            }
        }
        vhttp_stream_done(conn);
    }
    if (gen_ids_out) { *gen_ids_out = gids; *gen_n_out = gn; }
    return 0;
}

/* Text-only completion: encode prompt, prefill, decode.
 * ctx_tokens/n_ctx: 客户端回传的完整历史 token 序列（上一响应 history_tokens）。
 *   若与上一请求序列完全一致，则只重新编码 tail_prompt（本轮新增 user 段）并
 *   prefill 增量，历史 KV 完整复用；任何不一致回退到完整 prompt 全量 prefill。
 * hist_ids/hist_n (非流式): 输出本轮完整序列（prompt+generated），供响应
 *   history_tokens 字段；流式由 decode_loop 内单独 data 事件发送。
 * Returns 0 on success, -1 on engine failure, -2 if the prompt exceeds
 * the KV cache span (client must get 400). */
static int run_completion(VLLMServerCtx *ctx, const char *prompt,
                          const char *tail_prompt,
                          const int *ctx_tokens, int n_ctx,
                          const int *pfx_tokens, int pfx_n,
                          const char *pfx_kv_path,
                          double temperature, double top_p, double min_p,
                          int top_k, int thinking,
                          int max_tokens, int stream, VHttpConn *conn,
                          int emit_meta,
                          StrBuf *out, VLLMMetrics *met,
                          int **hist_ids, int *hist_n,
                          const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    QwenTokenizer *tok = ctx->tok;

    if (met) met->start_s = st_now_sec();

    /* Cap prompt length to the KV cache span. ids is heap-allocated so a
     * huge prompt cannot overflow a stack array. */
    int max_ids = st->max_kv_slots;
    if (max_ids < 1024) max_ids = 1024;
    int *ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!ids) return -1;
    int n_ids = 0;

    /* KV prefix reuse: the shared prefix with the previous request is kept
     * in the KV cache and only the remainder is prefilled. Two sources:
     * 1) the in-RAM KV of the previous text request (ctx->last_ids);
     * 2) a disk KV checkpoint (--disk-kv) whose tokens share the longest
     *    prefix - this survives process restarts. L3 eviction (which frees
     *    prefix blocks) and multimodal prompts (visual KV) both disable reuse:
     *    L3 侧由 prefix_kv_allowed() 决定（默认关；VLLM_L3_PREFIX_REUSE=1 且
     *    步骤 3 接线后才放行），多模态侧仍是硬门（一期不做，见 R5）。
     *    注意 disk_kv 的两条分支（1306/1344）保持原 `!g_l3_evict` 不动 ——
     *    它恢复的是原始 f32 KV 快照，不经 Q4、无 NULL 块问题，与 L3 重建是
     *    两套机制，混进同一门会让归因困难。
     * context_tokens 是其上层的精确变体：客户端回传的序列与 last_ids 完全
     * 一致（LCP==n_ctx）时 KV 行真实存在，可直接 keep。
     * pfx_tokens/pfx_n/pfx_kv_path 是"预置命名上下文"（规则包）的变体：prompt
     * 前缀与规则包渲染块逐 token 一致（pfx_n 个）时，用 pfx_kv_path 指向的
     * KV 快照恢复该段前缀（st_kv_disk_load），只 prefill 余量。读不到快照或
     * 前缀不匹配即静默回退全量 prefill（正确性优先）。 */
    int keep = 0;
    if (ctx_tokens && n_ctx > 0 && ctx->prefix_kv && prefix_kv_allowed() &&
        !ctx->last_was_mm &&
        tail_prompt && tail_prompt[0] && n_ctx < max_ids) {
        int matched = 0;
        /* 1) 进程内 last_ids 精确匹配（正常轮次复用）。 */
        if (ctx->last_n > 0 && n_ctx <= st->cache_len[0] &&
            kv_lcp(ctx->last_ids, ctx->last_n, ctx_tokens, n_ctx) == n_ctx)
            matched = 1;
        /* 2) 模型卸载/引擎重启后 last_ids 失效：disk_kv 精确恢复历史 KV，
         *    使多轮对话上下文跨卸载/重启延续（页面只需重发 context_tokens）。 */
        if (!matched && ctx->disk_kv && !g_l3_evict) {
            for (int i = 0; i < ctx->dkv_n; i++) {
                const DiskKVEntry *e = &ctx->dkv[i];
                if (e->n_tokens >= n_ctx &&
                    kv_lcp(e->tokens, e->n_tokens, ctx_tokens, n_ctx) == n_ctx) {
                    if (st_kv_disk_load(st, e->path, n_ctx) == 0) {
                        fprintf(stderr, "[KV-DISK] context restore %d-token KV prefix\n",
                                n_ctx);
                        fflush(stderr);
                        matched = 1;
                    }
                    break;
                }
            }
        }
        if (matched) {
            memcpy(ids, ctx_tokens, (size_t)n_ctx * sizeof(int));
            int tn = qwen_tokenizer_encode(tok, tail_prompt, ids + n_ctx,
                                           max_ids + 16 - n_ctx);
            if (tn > 0) {
                n_ids = n_ctx + tn;
                keep = n_ctx;
            }
        }
    }
    if (n_ids == 0) {
        n_ids = qwen_tokenizer_encode(tok, prompt, ids, max_ids + 16);
        if (n_ids <= 0) { free(ids); return -1; }
        if (n_ids > max_ids) { free(ids); return -2; }
        /* 预置命名上下文（规则包）KV 快照复用：优先级最高（前缀最长且最稳
         * 定）。门与既有前缀复用一致（prefix_kv / prefix_kv_allowed /
         * !last_was_mm）；`--no-prefix-kv` 或 reuse_prefix_cache=false 时调用
         * 方不传 pfx_kv_path，自然回退全量 prefill。 */
        if (ctx->prefix_kv && prefix_kv_allowed() && !ctx->last_was_mm &&
            pfx_tokens && pfx_kv_path && pfx_kv_path[0] &&
            pfx_n >= PREFIX_KV_MIN_LCP && pfx_n < n_ids &&
            kv_lcp(pfx_tokens, pfx_n, ids, n_ids) == pfx_n) {
            if (st_kv_disk_load(st, pfx_kv_path, pfx_n) == 0) {
                keep = pfx_n;
                ctx->rulebook_hits++;
                if (met) met->reused_prefix = 1;
                fprintf(stderr, "[RB] reuse %d-token preset-context KV prefix, "
                                "prefill rest\n", pfx_n);
                fflush(stderr);
            } else {
                fprintf(stderr, "[RB] preset-context KV snapshot unusable: %s "
                                "(full prefill)\n", pfx_kv_path);
                fflush(stderr);
            }
        }
        if (keep == 0 && ctx->prefix_kv && prefix_kv_allowed() &&
            !ctx->last_was_mm && ctx->last_n > 0) {
            int l = kv_lcp(ctx->last_ids, ctx->last_n, ids, n_ids);
            /* l == n_ids（新 prompt 与上次完全相同）不能直接 keep = n_ids：
             * prefill rest=0 不刷新 logits，decode 首步会采样到上次请求末尾的
             * 陈旧 logits（预测 eos）→ 空响应。改为保留到 n_ids-1，只重算最后
             * 一个 token —— 前缀 KV 全部复用，同时刷新 logits。 */
            if (l >= PREFIX_KV_MIN_LCP && l <= st->cache_len[0]) {
                keep = (l >= n_ids) ? (n_ids - 1) : l;
                if (getenv("VLLM_DEBUG_PREFIX"))
                    fprintf(stderr,
                            "[KV-PREFIX-DBG] last_n=%d n_ids=%d lcp=%d "
                            "cache_len0=%d keep=%d exact_repeat=%d\n",
                            ctx->last_n, n_ids, l, st->cache_len[0], keep,
                            (l >= n_ids) ? 1 : 0);
            }
        }
        if (keep == 0 && ctx->disk_kv && !g_l3_evict && !ctx->last_was_mm) {
            char dk_path[576];
            int l = dkv_best_lcp(ctx, ids, n_ids, dk_path, sizeof(dk_path));
            if (l >= PREFIX_KV_MIN_LCP && l < n_ids &&
                st_kv_disk_load(st, dk_path, l) == 0) {
                keep = l;
                fprintf(stderr, "[KV-DISK] loaded %d-token KV prefix from %s\n",
                        l, dk_path);
                fflush(stderr);
            }
        }
    }
    ist_reset(ctx, keep);
    if (st_qwen_model_prefill_batch(st, ids + keep, n_ids - keep) != 0) {
        free(ids); return -1;
    }
    /* Phase 2a: L3 cold-block Q4 disk eviction (--l3-evict).
     * P3：keep > 0 时**也要驱逐** —— 否则 L3 会退化成"只对首轮生效"（R4：
     * 复用轮不驱逐 -> KV 全驻留）。incremental 只在门开时为 1：跨轮保留 L3
     * 状态、跳过已打包的前缀块（消 R1 重复量化）。门关时 keep 恒为 0（复用
     * 被 prefix_kv_allowed() 挡住），行为与加参数前逐位一致。 */
    l3_evict_after_prefill(st, keep, l3_prefix_reuse_env());
    if (met) {
        met->prompt_tokens = n_ids;
        met->prefill_ms = (st_now_sec() - met->start_s) * 1000.0;
    }

    int *gids = NULL; int gn = 0;
    int rc = decode_loop(ctx, temperature, top_p, min_p, top_k, thinking,
                         max_tokens, stream,
                         conn, emit_meta, out, met, &gids, &gn, ids, n_ids,
                         att_user, att_bodyhex);
    if (rc == 0) {
        /* Remember this request's full KV token sequence (prompt + generated)
         * so the next request can reuse the shared prefix. */
        int total = n_ids + gn;
        if (total > ctx->last_cap) {
            int nc = ctx->last_cap ? ctx->last_cap * 2 : 512;
            while (nc < total) nc *= 2;
            int *nbuf = (int *)realloc(ctx->last_ids, (size_t)nc * sizeof(int));
            if (nbuf) { ctx->last_ids = nbuf; ctx->last_cap = nc; }
        }
        if (ctx->last_cap >= total) {
            memcpy(ctx->last_ids, ids, (size_t)n_ids * sizeof(int));
            memcpy(ctx->last_ids + n_ids, gids, (size_t)gn * sizeof(int));
            ctx->last_n = total;
            /* The last saved request is text-only: multimodal bookkeeping
             * (media marks) must not leak across request types. */
            ctx->last_was_mm = 0;
            /* Disk KV persistence: snapshot the finished conversation (token
             * ids + KV rows) so a later process can restore this prefix. */
            if (ctx->disk_kv) dkv_save(ctx, ctx->last_ids, ctx->last_n);
        }
        /* 客户端历史 token 序列（prompt + generated）回传，供下轮 context_tokens
         * 精确复用（流式经 decode_loop 内 data 事件，非流式经响应字段）。 */
        if (hist_ids && hist_n) {
            int total = n_ids + gn;
            int *h = (int *)malloc((size_t)total * sizeof(int));
            if (h) {
                memcpy(h, ids, (size_t)n_ids * sizeof(int));
                if (gn > 0) memcpy(h + n_ids, gids, (size_t)gn * sizeof(int));
                *hist_ids = h;
                *hist_n = total;
            } else {
                *hist_ids = NULL;
                *hist_n = 0;
            }
        }
    } else if (rc != 0) {
        /* prefill 失败：KV 缓存状态未定义，历史序列失效，下轮回退全量 prefill。 */
        ctx->last_n = 0;
        ctx->last_was_mm = 0;
    }
    free(gids);
    free(ids);
    return rc;
}

/* Multimodal completion: prompt tokens + concatenated visual tokens already
 * built and length-checked by the route layer (build_multimodal_prompt).
 * mm_marks: per-row media-content marks aligned with prompt_tokens (built by
 * build_multimodal_prompt; NULL disables multimodal prefix reuse). */
static int run_completion_mm(VLLMServerCtx *ctx,
                             const int *prompt_tokens, int n_prompt,
                             const float *vis_tokens, int n_vis,
                             const int *grids, int n_regions,
                             const float *const *ds_features, int n_ds,
                             double temperature, double top_p, double min_p,
                             int top_k, int thinking,
                             int max_tokens, int stream, VHttpConn *conn,
                             int emit_meta,
                             StrBuf *out, VLLMMetrics *met,
                             const unsigned long long *mm_marks,
                             const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    if (met) met->start_s = st_now_sec();

    /* P4 multimodal KV prefix reuse: the previous multimodal request's KV
     * still occupies the cache; compare token ids AND per-row media hashes
     * (vision-token hash alignment) so the shared prefix rows are reused
     * only when the same image/video bytes sit at the same rows (pad-count
     * equality alone would be fooled by equal-geometry different-content
     * images). Serial path only; L3 eviction frees prefix rows -> off. */
    int keep = 0;
    if (ctx->prefix_kv && !g_l3_evict && ctx->last_was_mm &&
        ctx->last_n > 0 && ctx->last_mark && mm_marks) {
        int l = mm_lcp(ctx->last_ids, ctx->last_mark, ctx->last_n,
                       prompt_tokens, mm_marks, n_prompt);
        /* l < n_prompt: a full coverage would prefill 0 tokens and decode
         * would sample the stale tail logits of the previous request. */
        if (l >= PREFIX_KV_MIN_LCP && l < n_prompt && l <= st->cache_len[0])
            keep = l;
    }
    if (keep > 0) {
        fprintf(stderr, "[MM-PREFIX] reuse %d-token KV prefix, prefill %d tokens\n",
                keep, n_prompt - keep);
        fflush(stderr);
    }
    ist_reset(ctx, keep);
    if (st_qwen_model_multimodal_prefill_ex(st, prompt_tokens, n_prompt,
                                            vis_tokens, n_vis,
                                            grids, n_regions,
                                            ds_features, n_ds, keep) != 0)
        return -1;
    if (keep == 0) {
        /* Phase 2a: L3 cold-block Q4 disk eviction (--l3-evict).
         * 多模态前缀复用一期不开（硬门，见 R5），L3 下 keep 恒为 0；这里显式传
         * incremental=0 —— 多模态路径永不启用增量语义。 */
        l3_evict_after_prefill(st, keep, 0);
    }
    if (met) {
        met->prompt_tokens = n_prompt + n_vis;
        met->prefill_ms = (st_now_sec() - met->start_s) * 1000.0;
    }
    /* Decode with speculative decoding: the multimodal prompt's TEXT token
     * sequence (visual tokens are positional, not part of the n-gram history)
     * plus the generated ids feed the draft builder; verification runs on the
     * decode stage where only text tokens are produced. */
    int *gids = NULL; int gn = 0;
    int rc = decode_loop(ctx, temperature, top_p, min_p, top_k, thinking,
                         max_tokens, stream,
                         conn, emit_meta, out, met, &gids, &gn,
                         prompt_tokens, n_prompt,
                         att_user, att_bodyhex);
    if (rc == 0 && ctx->prefix_kv && mm_marks) {
        /* Remember this multimodal request's full sequence (prompt tokens +
         * media marks + generated ids) so the next request can reuse the
         * shared prefix (KV rows + mrope positions stay resident). */
        int total = n_prompt + gn;
        if (total > ctx->last_cap) {
            int nc = ctx->last_cap ? ctx->last_cap * 2 : 512;
            while (nc < total) nc *= 2;
            int *nbuf = (int *)realloc(ctx->last_ids, (size_t)nc * sizeof(int));
            if (nbuf) { ctx->last_ids = nbuf; ctx->last_cap = nc; }
        }
        if (total > ctx->last_mark_cap) {
            int nc = ctx->last_mark_cap ? ctx->last_mark_cap * 2 : 512;
            while (nc < total) nc *= 2;
            unsigned long long *nb =
                (unsigned long long *)realloc(ctx->last_mark,
                                              (size_t)nc * sizeof(unsigned long long));
            if (nb) { ctx->last_mark = nb; ctx->last_mark_cap = nc; }
        }
        if (ctx->last_cap >= total && ctx->last_mark_cap >= total) {
            memcpy(ctx->last_ids, prompt_tokens, (size_t)n_prompt * sizeof(int));
            if (gn > 0)
                memcpy(ctx->last_ids + n_prompt, gids, (size_t)gn * sizeof(int));
            memset(ctx->last_mark, 0, (size_t)total * sizeof(unsigned long long));
            if (n_prompt > 0)
                memcpy(ctx->last_mark, mm_marks,
                       (size_t)n_prompt * sizeof(unsigned long long));
            ctx->last_n = total;
            ctx->last_was_mm = 1;
        }
    } else if (rc != 0) {
        /* prefill/decode 失败：KV 缓存状态未定义，历史序列失效，下轮全量 prefill。 */
        ctx->last_n = 0;
        ctx->last_was_mm = 0;
    }
    free(gids);
    return rc;
}

/* ---------- multimodal (image/video) input ---------- */

#define MM_MAX_REGIONS 16
#define MM_MAX_FRAMES  32

/* DeepStack accumulation across media regions. rows[d] = [written[d], dim],
 * aligned with the concatenated visual-token rows in vis_all. */
typedef struct {
    float *rows[3];
    int    written[3];
    int    cap[3];
} DSAccum;

static void ds_accum_init(DSAccum *a) { memset(a, 0, sizeof(*a)); }
static void ds_accum_free(DSAccum *a) {
    for (int i = 0; i < 3; i++) free(a->rows[i]);
    memset(a, 0, sizeof(*a));
}
/* Append the vision state's DeepStack features (n rows of dim) for all
 * deepstack layers. Returns 0 ok, -1 on OOM. */
static int ds_accum_append(DSAccum *a, STVisionState *vis, int n, int dim) {
    for (int d = 0; d < vis->cfg->vis_ds_count; d++) {
        int need = a->written[d] + n;
        if (need > a->cap[d]) {
            int nc = a->cap[d] ? a->cap[d] : 1024;
            while (nc < need) nc *= 2;
            float *nb = (float *)realloc(a->rows[d], (size_t)nc * dim * sizeof(float));
            if (!nb) return -1;
            a->rows[d] = nb;
            a->cap[d] = nc;
        }
        memcpy(a->rows[d] + (size_t)a->written[d] * dim,
               vis->ds_features[d], (size_t)n * dim * sizeof(float));
        a->written[d] += n;
    }
    return 0;
}

/* ---- 图片视觉 token 缓存（重复图跳过 ViT 编码） ----
 * 键：原始图像字节的 128-bit FNV-1a（两次不同初值的 64-bit）。命中时回放
 * 与 st_vision_encode_image 完全一致的视觉 token/网格/DeepStack 行，输出与
 * 重新编码逐字节相同。容量 VVIS_CACHE_CAP 条目，每条 512 图 ≈ 17MB（vis
 * 4.2MB + ds 3×4.2MB），LRU 淘汰。请求已在 inf_lock 内串行，无需加锁。 */
#define VVIS_CACHE_CAP 4
typedef struct {
    unsigned long long h1, h2;   /* 128-bit FNV-1a 键 */
    long last_use;               /* LRU 时间戳 */
    int n;                       /* n_vis */
    int dim;                     /* vis_out_dim == LLM dim */
    float *vis;                  /* [n][dim] */
    int grid_t, grid_h, grid_w;  /* 合并后网格（region） */
    int n_ds;                    /* deepstack 行数 */
    float *ds[3];                /* [n][dim] × n_ds */
} VisCacheEntry;

static VisCacheEntry g_vis_cache[VVIS_CACHE_CAP];
static int g_vis_cache_n = 0;

/* FNV-1a 64-bit over bytes（见 l3_user_hash 同族）。 */
static unsigned long long fnv1a64_buf(const void *p, size_t n,
                                      unsigned long long h) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* 把任意 DeepStack 行追加进累积器（缓存回放用，与 ds_accum_append 逐字节一致）。 */
static int ds_accum_append_rows(DSAccum *a, float *const *rows, int n_ds,
                                int n, int dim) {
    for (int d = 0; d < n_ds; d++) {
        int need = a->written[d] + n;
        if (need > a->cap[d]) {
            int nc = a->cap[d] ? a->cap[d] : 1024;
            while (nc < need) nc *= 2;
            float *nb = (float *)realloc(a->rows[d], (size_t)nc * dim * sizeof(float));
            if (!nb) return -1;
            a->rows[d] = nb;
            a->cap[d] = nc;
        }
        memcpy(a->rows[d] + (size_t)a->written[d] * dim,
               rows[d], (size_t)n * dim * sizeof(float));
        a->written[d] += n;
    }
    return 0;
}

/* 查找并回放缓存；返回 1 命中（vis_all/grids/dsa 已填充）、0 未命中。 */
static int vis_cache_apply(VLLMServerCtx *ctx, unsigned long long h1,
                           unsigned long long h2,
                           float **vis_all, int *vis_total, int *vis_cap,
                           int *grids, int *n_regions, DSAccum *dsa,
                           int *out_n_pads) {
    const int d = ctx->cfg->dim;
    for (int ci = 0; ci < g_vis_cache_n; ci++) {
        VisCacheEntry *e = &g_vis_cache[ci];
        if (e->h1 != h1 || e->h2 != h2) continue;
        e->last_use = unix_now();
        if (e->dim != d || e->n < 1) return 0;   /* 维度不匹配：当作未命中 */
        if (*vis_total + e->n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + e->n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return 0;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, e->vis,
               (size_t)e->n * d * sizeof(float));
        *vis_total += e->n;
        grids[(*n_regions) * 3 + 0] = e->grid_t;
        grids[(*n_regions) * 3 + 1] = e->grid_h;
        grids[(*n_regions) * 3 + 2] = e->grid_w;
        (*n_regions)++;
        if (dsa && ds_accum_append_rows(dsa, e->ds, e->n_ds, e->n, d) != 0)
            return 0;
        *out_n_pads = e->n;
        return 1;
    }
    return 0;
}

/* 编码成功后写入缓存（LRU 淘汰最久未用）。 */
static void vis_cache_store(unsigned long long h1, unsigned long long h2,
                            STVisionState *vis, int n, int d,
                            int grid_t, int grid_h, int grid_w) {
    int slot;
    if (g_vis_cache_n < VVIS_CACHE_CAP) {
        slot = g_vis_cache_n;
    } else {
        long oldest = g_vis_cache[0].last_use;
        slot = 0;
        for (int ci = 1; ci < VVIS_CACHE_CAP; ci++)
            if (g_vis_cache[ci].last_use < oldest) {
                oldest = g_vis_cache[ci].last_use;
                slot = ci;
            }
    }
    VisCacheEntry *e = &g_vis_cache[slot];
    if (e->vis) free(e->vis);
    for (int k = 0; k < 3; k++) if (e->ds[k]) free(e->ds[k]);
    memset(e, 0, sizeof(*e));
    e->h1 = h1; e->h2 = h2;
    e->last_use = unix_now();
    e->n = n; e->dim = d;
    e->grid_t = grid_t; e->grid_h = grid_h; e->grid_w = grid_w;
    e->n_ds = vis->cfg->vis_ds_count;
    if (e->n_ds > 3) e->n_ds = 3;
    e->vis = (float *)malloc((size_t)n * d * sizeof(float));
    if (!e->vis) { e->n = 0; return; }
    memcpy(e->vis, st_vision_get_tokens(vis), (size_t)n * d * sizeof(float));
    for (int k = 0; k < e->n_ds; k++) {
        e->ds[k] = (float *)malloc((size_t)n * d * sizeof(float));
        if (e->ds[k])
            memcpy(e->ds[k], vis->ds_features[k], (size_t)n * d * sizeof(float));
    }
    if (g_vis_cache_n < VVIS_CACHE_CAP) g_vis_cache_n++;
}

/* True if any message content carries image_url / video_frames parts. */
static int detect_media(const VJson *messages) {
    size_t n = vjson_array_len(messages);
    for (size_t m = 0; m < n; m++) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content || content->type != VJ_ARRAY) continue;
        size_t np = vjson_array_len(content);
        for (size_t p = 0; p < np; p++) {
            const VJson *part = vjson_array_get(content, p);
            if (!part) continue;
            const char *ptype = vjson_str(vjson_obj_get(part, "type"));
            if (ptype && (strcmp(ptype, "image_url") == 0 ||
                          strcmp(ptype, "video_frames") == 0 ||
                          strcmp(ptype, "vision_tokens") == 0))
                return 1;
        }
    }
    return 0;
}

/* Decode a "data:<mime>;base64,...." URL into raw bytes (malloc'd). */
static uint8_t *decode_data_url(const char *url, size_t *out_len) {
    const char *p = strstr(url, "base64,");
    if (!p) return NULL;
    p += 7;   /* skip "base64," */
    return media_base64_decode(p, out_len);
}

/* Encode one media part (image_url / video_frames) with the vision encoder,
 * append its visual tokens to *vis_all and its grid to grids[].
 * out_kind: 1=image, 2=video; out_n_pads: visual tokens emitted. */
static int encode_media_item(VLLMServerCtx *ctx, const VJson *part,
                             float **vis_all, int *vis_total, int *vis_cap,
                             int *grids, int *n_regions,
                             int *out_kind, int *out_n_pads, DSAccum *dsa,
                             unsigned long long *ohash) {
    STVisionState *vis = ctx->vis;
    const int d = ctx->cfg->dim;
    if (ohash) *ohash = 0;   /* media-content mark for P4 prefix reuse */

    /* 客户端预处理模式（vision_tokens part）：浏览器在本机算好的视觉特征，
     * 直接接收（跳过 ViT 编码）。协议 v1：tokens_b64 = base64(fp32 LE, n*d)，
     * ds_b64（可选）= base64 数组，每条 n*d；grid = [gt, gh, gw] 合并后网格。 */
    const VJson *vt = vjson_obj_get(part, "vision_tokens");
    if (vt) {
        const VJson *g = vjson_obj_get(vt, "grid");
        if (!g || g->type != VJ_ARRAY || vjson_array_len(g) < 3) return -1;
        long gt = (long)vjson_array_get(g, 0)->u.number;
        long gh = (long)vjson_array_get(g, 1)->u.number;
        long gw = (long)vjson_array_get(g, 2)->u.number;
        if (gt < 1 || gh < 1 || gw < 1 || gt * gh * gw > 4096) return -1;
        long n = gt * gh * gw;
        const char *tb64 = vjson_str(vjson_obj_get(vt, "tokens_b64"));
        if (!tb64) return -1;
        size_t tlen = 0;
        uint8_t *traw = media_base64_decode(tb64, &tlen);
        if (!traw) return -1;
        size_t need = (size_t)n * d * sizeof(float);
        if (tlen != need) { free(traw); return -1; }
        if (*n_regions >= MM_MAX_REGIONS) { free(traw); return -1; }
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) { free(traw); return -1; }
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, traw, need);
        *vis_total += n;
        if (ohash) *ohash = fnv1a64_buf(traw, tlen, 1469598103934665603ull);
        free(traw);
        grids[(*n_regions) * 3 + 0] = (int)gt;
        grids[(*n_regions) * 3 + 1] = (int)gh;
        grids[(*n_regions) * 3 + 2] = (int)gw;
        (*n_regions)++;
        /* DeepStack 特征（可选）：ds_b64 数组，每条 n*d 个 fp32。缺失则 n_ds=0
         * （prefill 跳过注入，输出与带 ds 路径不一致，客户端须自行负责）。 */
        float *drows[3] = { NULL, NULL, NULL };
        int n_ds = 0;
        const VJson *ds = vjson_obj_get(vt, "ds_b64");
        if (ds && ds->type == VJ_ARRAY) {
            size_t nds = vjson_array_len(ds);
            if (nds > 3) nds = 3;
            int bad = 0;
            for (size_t k = 0; k < nds; k++) {
                const char *db64 = vjson_str(vjson_array_get(ds, k));
                if (!db64) { bad = 1; break; }
                size_t dlen = 0;
                uint8_t *draw = media_base64_decode(db64, &dlen);
                if (!draw || dlen != need) { free(draw); bad = 1; break; }
                drows[k] = (float *)draw;
            }
            if (bad) {
                for (int k = 0; k < 3; k++) free(drows[k]);
                return -1;
            }
            n_ds = (int)nds;
        }
        if (dsa && n_ds > 0) {
            if (ds_accum_append_rows(dsa, drows, n_ds, (int)n, d) != 0) {
                for (int k = 0; k < n_ds; k++) free(drows[k]);
                return -1;
            }
        }
        for (int k = 0; k < n_ds; k++) free(drows[k]);
        fprintf(stderr, "[VIS-TOKENS] client-preprocess n=%ld d=%d ds=%d grid=[%ld,%ld,%ld]\n",
                n, d, n_ds, gt, gh, gw);
        fflush(stderr);
        *out_kind = 1;
        *out_n_pads = (int)n;
        return 0;
    }

    const VJson *iu = vjson_obj_get(part, "image_url");
    if (iu) {
        const char *url = vjson_str(vjson_obj_get(iu, "url"));
        if (!url) return -1;
        size_t blen = 0;
        uint8_t *raw = decode_data_url(url, &blen);
        if (!raw) return -1;
        /* 128-bit FNV-1a 图片键：相同上传字节命中缓存，跳过 ViT 编码。 */
        unsigned long long h1 = fnv1a64_buf(raw, blen, 1469598103934665603ull);
        unsigned long long h2 = fnv1a64_buf(raw, blen,
                                            1469598103934665603ull ^ 0x9E3779B97F4A7C15ull);
        /* P4 prefix-reuse mark: the KV rows this image occupies are only
         * reusable when the same upload bytes reappear at the same rows. */
        if (ohash) *ohash = h1 ^ h2;
        int w = 0, h = 0;
        uint8_t *rgb = media_load_image_memory(raw, blen, &w, &h);
        free(raw);
        if (!rgb) return -1;
        if (vis_cache_apply(ctx, h1, h2, vis_all, vis_total, vis_cap,
                            grids, n_regions, dsa, out_n_pads)) {
            fprintf(stderr, "[VIS-CACHE] hit (%dx%d, %d vis tokens, skip ViT)\n",
                    w, h, *out_n_pads);
            fflush(stderr);
            free(rgb);
            *out_kind = 1;
            return 0;
        }
        int n = st_vision_encode_image(vis, rgb, w, h);
        free(rgb);
        if (n < 0) return -1;
        if (*n_regions >= MM_MAX_REGIONS) return -1;
        int gt = 1;
        int gh = vis->grid_h / ctx->cfg->vis_merge;
        int gw = vis->grid_w / ctx->cfg->vis_merge;
        grids[(*n_regions) * 3 + 0] = gt;
        grids[(*n_regions) * 3 + 1] = gh;
        grids[(*n_regions) * 3 + 2] = gw;
        (*n_regions)++;
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return -1;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, st_vision_get_tokens(vis),
               (size_t)n * d * sizeof(float));
        *vis_total += n;
        if (dsa && ds_accum_append(dsa, vis, n, d) != 0) return -1;
        vis_cache_store(h1, h2, vis, n, d, gt, gh, gw);
        *out_kind = 1;
        *out_n_pads = n;
        return 0;
    }

    const VJson *vf = vjson_obj_get(part, "video_frames");
    if (vf && vf->type == VJ_ARRAY) {
        size_t nf = vjson_array_len(vf);
        if (nf == 0) return -1;
        if (nf > MM_MAX_FRAMES) nf = MM_MAX_FRAMES;
        uint8_t **frames = (uint8_t **)calloc(nf, sizeof(uint8_t *));
        if (!frames) return -1;
        int fw = 0, fh = 0, loaded = 0;
        unsigned long long vh = 0;   /* P4 prefix-reuse mark: hash of all frames */
        for (size_t i = 0; i < nf; i++) {
            const char *url = vjson_str(vjson_array_get(vf, i));
            if (!url) continue;
            size_t blen = 0;
            uint8_t *raw = decode_data_url(url, &blen);
            if (!raw) continue;
            int w = 0, h = 0;
            uint8_t *rgb = media_load_image_memory(raw, blen, &w, &h);
            if (rgb && (loaded == 0 || (w == fw && h == fh))) {
                vh = fnv1a64_buf(raw, blen,
                                 vh ? vh : 1469598103934665603ull);
            }
            free(raw);
            if (!rgb) continue;
            if (loaded == 0) { fw = w; fh = h; }
            else if (w != fw || h != fh) { free(rgb); continue; }
            frames[loaded++] = rgb;
        }
        if (loaded == 0) { free(frames); return -1; }
        if (ohash) *ohash = vh;
        int n = st_vision_encode_video(vis, (const uint8_t **)frames, fw, fh, loaded);
        for (int i = 0; i < loaded; i++) free(frames[i]);
        free(frames);
        if (n < 0) return -1;
        if (*n_regions >= MM_MAX_REGIONS) return -1;
        grids[(*n_regions) * 3 + 0] = vis->n_frames_eff;
        grids[(*n_regions) * 3 + 1] = vis->grid_h / ctx->cfg->vis_merge;
        grids[(*n_regions) * 3 + 2] = vis->grid_w / ctx->cfg->vis_merge;
        (*n_regions)++;
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return -1;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, st_vision_get_tokens(vis),
               (size_t)n * d * sizeof(float));
        *vis_total += n;
        if (dsa && ds_accum_append(dsa, vis, n, d) != 0) return -1;
        *out_kind = 2;
        *out_n_pads = n;
        return 0;
    }
    return -1;
}

/* Append helpers (bounded). Return -1 if the prompt buffer is full. */
static int app_tok(int *buf, int cap, int *n, int tid) {
    if (*n >= cap) return -1;
    buf[(*n)++] = tid;
    return 0;
}
static int app_text_tok(int *buf, int cap, int *n, QwenTokenizer *tok, const char *text) {
    if (!text || !*text) return 0;
    int used = qwen_tokenizer_encode(tok, text, buf + *n, cap - *n);
    if (used <= 0) return -1;
    *n += used;
    return 0;
}

/* Build the multimodal chat prompt (im_chat template) and encode all media.
 * prompt_tokens/max_tok: output buffer (must hold the whole sequence).
 * vis_all/vis_total: concatenated visual tokens (realloc'd as needed).
 * grids/n_regions: per-region {grid_t, grid_h, grid_w}.
 * marks/mark_cap: optional per-token media-content marks aligned 1:1 with
 *   prompt_tokens (0 = text row; non-zero = image/video byte hash on the KV
 *   rows replaced by that media's visual tokens). Caller zero-fills it; only
 *   media pad rows are written. NULL disables (batch/legacy callers).
 * last_user_vis: optional out — visual tokens emitted for the LAST user
 *   message (0 when it carries no media). Lets the metric report the full
 *   "上一问长度" (text + image/video tokens) instead of text only.
 * Returns 0 ok, -1 media/decode failure, -2 prompt exceeds max_tok. */
static int build_multimodal_prompt(VLLMServerCtx *ctx, const VJson *messages,
                                   int *prompt_tokens, int max_tok, int *pn,
                                   float **vis_all, int *vis_total, int *vis_cap,
                                   int *grids, int *n_regions, DSAccum *dsa,
                                   unsigned long long *marks, int mark_cap,
                                   int thinking, int *last_user_vis) {
    QwenTokenizer *tok = ctx->tok;
    STModelConfig *cfg = ctx->cfg;
    size_t n = vjson_array_len(messages);
    if (last_user_vis) *last_user_vis = 0;

    for (size_t m = 0; m < n; m++) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) continue;
        if (!role) role = "user";
        int msg_vis = 0;   /* 本条消息的视觉 token 数 */

        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_start|>") < 0) return -2;
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, role) < 0) return -2;
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "\n") < 0) return -2;

        if (content->type == VJ_STRING) {
            if (app_text_tok(prompt_tokens, max_tok, pn, tok, vjson_str(content)) < 0) return -2;
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    if (app_text_tok(prompt_tokens, max_tok, pn, tok,
                                     vjson_str(vjson_obj_get(part, "text"))) < 0) return -2;
                } else if (ptype && (strcmp(ptype, "image_url") == 0 ||
                                     strcmp(ptype, "video_frames") == 0 ||
                                     strcmp(ptype, "vision_tokens") == 0)) {
                    int kind = 0, npads = 0;
                    unsigned long long mh = 0;
                    if (encode_media_item(ctx, part, vis_all, vis_total, vis_cap,
                                          grids, n_regions, &kind, &npads, dsa,
                                          &mh) != 0)
                        return -1;
                    /* <|vision_start|> <|image_pad|>/<|video_pad|> xN <|vision_end|> */
                    if (app_tok(prompt_tokens, max_tok, pn, cfg->vision_start_id) < 0) return -2;
                    int pad_id = (kind == 2) ? cfg->video_token_id : cfg->image_token_id;
                    int ps = *pn;   /* first pad row (marks aligned to rows) */
                    for (int i = 0; i < npads; i++) {
                        if (app_tok(prompt_tokens, max_tok, pn, pad_id) < 0) return -2;
                        if (marks && ps + i < mark_cap) marks[ps + i] = mh;
                    }
                    if (app_tok(prompt_tokens, max_tok, pn, cfg->vision_end_id) < 0) return -2;
                    msg_vis += npads;
                } else {
                    /* Unknown part type: ignore. */
                }
            }
        }
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_end|>\n") < 0) return -2;
        if (last_user_vis && strcmp(role, "user") == 0) *last_user_vis = msg_vis;
    }
    if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_start|>assistant\n") < 0) return -2;
    if (!thinking &&
        app_text_tok(prompt_tokens, max_tok, pn, tok, QWEN_NO_THINK_SUFFIX) < 0)
        return -2;
    return 0;
}

/* Token count of the LAST user message's text content (the question just
 * asked). Used for the "上一问长度" metric. 图文轮的视觉 token 不在此处计入
 * （本函数只见文本），由多模态路径把该消息的视觉 token 数（last_user_vis）
 * 补加到调用点，使指标反映"上一问"的真实输入量。 */
static int last_user_text_tokens(VLLMServerCtx *ctx, const VJson *messages) {
    QwenTokenizer *tok = ctx->tok;
    size_t n = vjson_array_len(messages);
    for (size_t m = n; m-- > 0; ) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role || strcmp(role, "user") != 0) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) return 0;

        StrBuf sb; sb_init(&sb);
        if (content->type == VJ_STRING) {
            const char *t = vjson_str(content);
            if (t) sb_str(&sb, t);
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&sb, t);
                }
            }
        }
        int cnt = 0;
        if (sb.s && sb.s[0]) {
            int cap = ctx->ist->max_kv_slots;
            if (cap < 1024) cap = 1024;
            int *ids = (int *)malloc(((size_t)cap + 16) * sizeof(int));
            if (ids) {
                cnt = qwen_tokenizer_encode(tok, sb.s, ids, cap + 16);
                free(ids);
            }
        }
        sb_free(&sb);
        return cnt;
    }
    return 0;
}

/* 返回最后一条 user 消息的文本（malloc，调用方 free）；无则 NULL。
 * 供 context_tokens 模式拼接本轮新增的 user 模板段。 */
static char *last_user_text_str(const VJson *messages) {
    size_t n = vjson_array_len(messages);
    for (size_t m = n; m-- > 0; ) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role || strcmp(role, "user") != 0) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) return NULL;
        StrBuf sb; sb_init(&sb);
        if (content->type == VJ_STRING) {
            const char *t = vjson_str(content);
            if (t) sb_str(&sb, t);
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&sb, t);
                }
            }
        }
        char *r = sb.s;   /* transfer ownership to caller */
        return r;
    }
    return NULL;
}

/* ---------- /v1/chat/completions ---------- */

/* Ensure a model is loaded before serving inference. With --auto-load the
 * first request triggers a load-on-use (用时加载) and waits for it; without
 * it an unloaded model returns 503 (use /admin/api/model/load). */
static int ensure_model_ready(VLLMServerCtx *ctx, VHttpResponse *resp) {
    if (ctx->load_state == 2) return 0;
    if (ctx->auto_load && ctx->load_state != 1) {
        if (vllm_serve_load_on_use(ctx) != 0) {
            json_error(resp, 503, "model load failed (see engine log)");
            return -1;
        }
    }
    if (ctx->load_state != 2) {
        json_error(resp, 503, ctx->load_state == 1
                   ? "model is loading" : "model not loaded (use /admin/api/model/load)");
        return -1;
    }
    return 0;
}

/* ---------- 云边协同：云端上游回退（--cloud-endpoint） ----------
 * 触发由 ctx->cloud_fallback 决定：0=never（默认，零回归）/ 1=on-unready / 2=always。
 * 命中时把请求转发到云端上游，并**显式**标记 `x-kestrel-source: cloud` + 记 [UPLINK]
 * 日志；转发失败返回**显式 502**，绝不静默退回本地（公理 silent_fallback_false_pass）。
 * 标准 OpenAI 契约：客户端 stream=true 时透传上游 SSE（vhttp_stream_*），
 * 非流式则回 JSON；上游错误码一律回 JSON 错误体（与 OpenAI 一致）。
 * 返回 0 = 已应答（resp 填好，调用方立即 return）；-1 = 未触发（调用方走本地路径）。
 * 边界：仅明文 http://（https 显式 502，前置 TLS 网关）。 */
static int cloud_fallback_serve(VLLMServerCtx *ctx, const VHttpRequest *req,
                                VHttpResponse *resp, VHttpConn *conn) {
    if (!ctx->cloud_endpoint || ctx->cloud_endpoint[0] == 0) return -1;
    if (ctx->cloud_fallback == 0) return -1;                         /* never */
    if (ctx->cloud_fallback == 1 && ctx->load_state == 2) return -1; /* on-unready: 本地就绪则本地服务 */

    char url[1200];
    snprintf(url, sizeof(url), "%s%s", ctx->cloud_endpoint, req->path);

    /* 请求体：解析后可选覆盖 model；stream 由客户端原样透传（标准契约）。
     * 解析失败则原样转发。 */
    char *fwd = NULL;
    size_t fwd_len = 0;
    int want_stream = 0;
    VJson *b = req->body ? vjson_parse(req->body) : NULL;
    if (b && b->type == VJ_OBJECT) {
        VJson *sv = vjson_obj_get(b, "stream");
        if (sv && sv->type == VJ_BOOL) want_stream = sv->u.boolean;
        if (ctx->cloud_model && ctx->cloud_model[0])
            vjson_obj_set(b, "model", vjson_new_string(ctx->cloud_model));
        size_t cap = req->body_len + 4096;
        fwd = (char *)malloc(cap);
        if (fwd) {
            fwd_len = vjson_serialize(b, fwd, cap);
            if (fwd_len == 0) { free(fwd); fwd = NULL; }
        }
    }
    vjson_free(b);
    if (!fwd) {
        fwd = (char *)malloc(req->body_len + 1);
        if (fwd) {
            memcpy(fwd, req->body ? req->body : "", req->body_len);
            fwd[req->body_len] = '\0';
            fwd_len = req->body_len;
        }
    }

    char auth[512];
    auth[0] = '\0';
    if (ctx->cloud_key && ctx->cloud_key[0])
        snprintf(auth, sizeof(auth), "Authorization: Bearer %s\r\n", ctx->cloud_key);

    char *out = NULL;
    int st = 0;
    char ctype[128] = "";
    int rc = vhttp_post_json(url, auth, fwd ? fwd : "", fwd_len,
                             &out, &st, ctype, sizeof(ctype));
    free(fwd);

    if (rc != 0 || !out) {
        free(out);
        fprintf(stderr, "[UPLINK] forward FAILED rc=%d url=%s (explicit 502, no silent local fallback)\n",
                rc, url);
        resp->extra_header = "x-kestrel-source: cloud\r\n";
        json_error(resp, 502, rc == -2
                   ? "cloud uplink https unsupported (front a TLS-terminating gateway)"
                   : "cloud uplink failed");
        return 0;
    }
    fprintf(stderr, "[UPLINK] forwarded -> %s status=%d bytes=%zu stream=%d\n",
            url, st, strlen(out), want_stream);

    /* 流式透传：客户端 stream=true 且上游 2xx → 原样回放上游 SSE。
     * 上游 SSE 以 "data: [DONE]\n\n" 收尾；统一由 vhttp_stream_done 追加尾标，
     * 故先截掉上游自带尾标避免重复。 */
    if (want_stream && st >= 200 && st < 300) {
        const char *done = strstr(out, "data: [DONE]");
        size_t cplen = done ? (size_t)(done - out) : strlen(out);
        if (vhttp_stream_begin_ex(conn, st,
                                  ctype[0] ? ctype : "text/event-stream",
                                  "x-kestrel-source: cloud\r\n") == 0 && cplen > 0)
            vhttp_stream_write(conn, out, cplen);
        free(out);
        resp->stream = 1;                 /* handler owns the connection */
        resp->content_type = "text/event-stream";
        vhttp_stream_done(conn);          /* 发送 data: [DONE] + close */
        return 0;
    }

    /* 非流式（或流式请求命中上游错误码 → 按 OpenAI 惯例回 JSON 错误体）。 */
    resp->extra_header = "x-kestrel-source: cloud\r\n";
    resp->status = st > 0 ? st : 502;
    resp->content_type = "application/json";
    resp->body = out;
    resp->body_len = 0;        /* 0 = strlen(body) */
    resp->body_owned = out;
    return 0;
}

/* 方案 2：给非流式响应对象附加 "attest" 出证字段。
 * body_sha = SM3(原始请求体) 的 64-hex（schema=3，绑定"收到什么证明什么"）；
 * text 为输出文本（绑定进摘要）。params 为本次请求实际采样参数。 */
static void attest_attach(VJson *root, const char *user, const char *body_sha,
                          const char *text, int npt, int ngt,
                          const char *finish, double temperature, double top_p,
                          double min_p, int top_k, int thinking,
                          int max_tokens) {
    if (!vatt_active() || !root || !text) return;
    VAttestReq ar;
    memset(&ar, 0, sizeof(ar));
    ar.user = user;
    ar.body_sha = body_sha;
    ar.text = text;
    ar.text_len = strlen(text);
    ar.n_prompt_tokens = npt;
    ar.n_gen_tokens = ngt;
    ar.finish = finish;
    ar.temperature = temperature;
    ar.top_p = top_p;
    ar.min_p = min_p;
    ar.top_k = top_k;
    ar.thinking = thinking;
    ar.max_tokens = max_tokens;
    ar.ts = unix_now();
    char *aj = vatt_seal_json(&ar);
    if (aj) {
        VJson *av = vjson_parse(aj);
        if (av) {
            vjson_obj_set(root, "attest", av);
        }
        free(aj);
    }
}

/* ---------- 工业边缘：预置上下文 / 会话的落盘收尾 ---------- */

/* 推理成功后把规则包前缀 KV 落盘（首次使用时懒预热；此后同规则+同分块选择
 * 即可直接复用）。门与既有 disk-kv 一致：L3 驱逐时冷块可能已是 NULL，跳过。 */
static void rb_lazy_snapshot(VLLMServerCtx *ctx, const char *rb_id,
                             const char *rb_ver, const int *ids, int n) {
    if (!ctx->rulebook_on || !ctx->ist || !rb_id || !rb_id[0]) return;
    if (!ids || n < PREFIX_KV_MIN_LCP) return;
    if (g_l3_evict) return;                       /* L3 可能已把冷块置 NULL */
    if (n > DISKKV_MAX_TOKENS) return;            /* 与既有快照上限同源 */
    if (n > ctx->ist->cache_len[0]) return;       /* 该前缀 KV 尚未算好 */
    VLLMRulebook tag;
    memset(&tag, 0, sizeof(tag));
    snprintf(tag.id, sizeof(tag.id), "%s", rb_id);
    snprintf(tag.version, sizeof(tag.version), "%s", rb_ver ? rb_ver : "");
    char path[1800];
    vllm_rb_kv_path(ctx, &tag, ids, n, path, sizeof(path));
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return; }                 /* 快照已存在 */
    if (st_kv_disk_save(ctx->ist, path, ids, n) != 0) return;
    fprintf(stderr, "[RB] cached %d-token preset-context KV -> %s\n", n, path);
    fflush(stderr);
}

/* 推理成功后把会话时间线（含本轮 assistant）落盘；写失败只告警，不影响响应。
 * 目录 LRU 清理按 1/32 概率触发（会话目录有上限，无需每请求遍历）。 */
static void sess_commit(VLLMServerCtx *ctx, const VJson *body, const char *sid,
                        const char *tenant, const char *user,
                        const char *rb_id, const char *answer, int n_tok,
                        double last_ms) {
    if (!ctx->session_on || !sid || !sid[0]) return;
    const VJson *tl = vjson_obj_get(body, "__sess_timeline");
    if (!tl) return;
    VJson *fin = vllm_sess_append_turns(tl, NULL, answer);
    char err[256];
    if (vllm_sess_write(ctx, tenant, user, sid, rb_id, fin, n_tok, last_ms, err,
                        sizeof(err)) != 0) {
        fprintf(stderr, "[SESS] write failed: %s\n", err);
        fflush(stderr);
    } else if (getenv("VLLM_DEBUG_SESSION")) {
        fprintf(stderr, "[SESS] saved %s (%d msgs)\n", sid,
                (int)vjson_array_len(fin));
        fflush(stderr);
    }
    vjson_free(fin);
    static int gc_tick = 0;
    if ((++gc_tick & 31) == 0) vllm_sess_gc(ctx);
}

/* 把请求级标识填进 metrics（随响应/SSE 返回，供审计与排障对齐）。 */
static void met_fill_ids(VLLMMetrics *met, const char *trace, const char *rbid,
                         const char *rbver, int rbtok) {
    if (!met) return;
    snprintf(met->trace_id, sizeof(met->trace_id), "%s", trace ? trace : "");
    snprintf(met->rulebook_id, sizeof(met->rulebook_id), "%s", rbid ? rbid : "");
    snprintf(met->rulebook_version, sizeof(met->rulebook_version), "%s",
             rbver ? rbver : "");
    met->rulebook_tokens = rbtok;
}

static void handle_chat(VLLMServerCtx *ctx, const VHttpRequest *req,
                        VHttpResponse *resp, VHttpConn *conn) {
    /* 云边协同：仅在配置了云端点时标记来源（未配置 = 响应逐字节不变，零回归）。 */
    if (ctx->cloud_endpoint && ctx->cloud_endpoint[0])
        resp->extra_header = "x-kestrel-source: edge\r\n";
    if (cloud_fallback_serve(ctx, req, resp, conn) == 0) return;
    /* Inference endpoints are unavailable until the model is loaded
     * (manual-load mode; see /admin/api/model/load). */
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) {
        fprintf(stderr, "[SRV] JSON parse failed, body[0..128]: %.*s\n",
                (int)(req->body_len < 128 ? req->body_len : 128), req->body);
        json_error(resp, 400, "Invalid JSON body");
        return;
    }

    const VJson *messages = vjson_obj_get(body, "messages");
    int have_msgs = messages && messages->type == VJ_ARRAY &&
                    vjson_array_len(messages) > 0;
    /* 工业边缘：会话模式下允许"只发本轮 query + session_id"（历史由服务端从
     * 会话文件重建，客户端上行更小）。其余情况仍要求 messages 非空。 */
    const char *q_probe = vjson_str(vjson_obj_get(body, "query"));
    const char *sid_probe = vjson_str(vjson_obj_get(body, "session_id"));
    int query_only = q_probe && q_probe[0] && sid_probe && sid_probe[0];
    if (!have_msgs && !query_only) {
        json_error(resp, 400,
                   "'messages' array is required (or session_id + query)");
        vjson_free(body);
        return;
    }

    /* 方案 2：原始请求体 SM3（逐字节绑定，出证/验证均基于它）。 */
    char bodyhash[65];
    vatt_body_digest(req->body, req->body_len, bodyhash);

    double temperature = 0.7;
    double top_p = 0.9;
    double min_p = ctx->min_p;
    int top_k = ctx->top_k;
    int thinking = resolve_thinking(body);
    int max_tokens = 64;
    int stream = 0;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER)
        temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER)
        top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "min_p")) && jv->type == VJ_NUMBER)
        min_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_k")) && jv->type == VJ_NUMBER) {
        top_k = (int)jv->u.number;
        if (top_k < 0) top_k = 0;
    }
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number;
        if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }
    if ((jv = vjson_obj_get(body, "stream")) && jv->type == VJ_BOOL)
        stream = jv->u.boolean;
    /* 自定义 SSE 事件（metrics/tokens/attest）为 Kestrel 扩展，会破坏标准
     * OpenAI 客户端的逐 chunk 解析；仅在客户端显式 opt-in（请求头
     * X-Kestrel-Meta 非 "0"/空）时下发，默认流严格标准。 */
    const char *kmeta = vhttp_req_header(req, "X-Kestrel-Meta");
    int emit_meta = (kmeta && kmeta[0] && kmeta[0] != '0') ? 1 : 0;
    const char *user = vjson_str(vjson_obj_get(body, "user"));

    /* ---- 工业边缘：预置命名上下文 + 会话分区（可选，缺省不改行为） ----
     * session_id / tenant_id / user_id 三元组对会话分区；rulebook_id 选中一个
     * 预置上下文（规则包）；rulebook_version 由客户端钉住版本（不匹配即拒绝，
     * 使规则变更必然失效旧缓存）；query 是"客户端只发本轮提问"的简写（服务端
     * 从会话文件重建历史）；reuse_prefix_cache=false 显式关闭规则前缀复用。 */
    const char *session_id = vjson_str(vjson_obj_get(body, "session_id"));
    const char *tenant_id  = vjson_str(vjson_obj_get(body, "tenant_id"));
    const char *user_id    = vjson_str(vjson_obj_get(body, "user_id"));
    const char *query      = vjson_str(vjson_obj_get(body, "query"));
    int reuse_prefix = 1;
    if ((jv = vjson_obj_get(body, "reuse_prefix_cache")) && jv->type == VJ_BOOL)
        reuse_prefix = jv->u.boolean;
    char rb_id_buf[VLLM_RB_ID_MAX];
    char rb_ver_buf[VLLM_RB_VER_MAX];
    rb_id_buf[0] = '\0';
    rb_ver_buf[0] = '\0';
    int rb_tokens_val = 0;   /* 规则前缀实际注入的 token 数（观测/响应） */
    {
        const char *s = vjson_str(vjson_obj_get(body, "rulebook_id"));
        if (s && s[0]) snprintf(rb_id_buf, sizeof(rb_id_buf), "%s", s);
        s = vjson_str(vjson_obj_get(body, "rulebook_version"));
        if (s && s[0]) snprintf(rb_ver_buf, sizeof(rb_ver_buf), "%s", s);
    }
    /* 会话分区用 user_id（未给则退回 user 字段），anon 为兜底。 */
    const char *part_user = (user_id && user_id[0]) ? user_id
                            : ((user && user[0]) ? user : "anon");
    /* trace_id：本请求追溯标识（响应/SSE 的 metrics 里返回，供审计对齐）。 */
    char trace_id[48];
    {
        char bh[VLLM_RB_HASH_HEX];
        vllm_rb_hash_hex(req->body, req->body_len, bh);
        snprintf(trace_id, sizeof(trace_id), "%.8s%08lx",
                 bh, (unsigned long)unix_now() & 0xFFFFFFFFul);
    }

    /* context_mode (per-request context budget):
     *   "full"  (default) : whole conversation history every turn
     *   "last"  / "none"  : keep system + the final user turn only, so every
     *                       prefill stays short and TTFT does not grow across
     *                       a multi-turn chat
     *   "lastN" (e.g. "last10") : keep system + the last N messages (a
     *                       sliding window of ~N/2 turns)
     * The clipped array is a deep copy owned by `body`, so the existing
     * vjson_free(body) paths stay correct. */
    if ((jv = vjson_obj_get(body, "context_mode")) && jv->type == VJ_STRING) {
        const char *cm = vjson_str(jv);
        size_t keep = 0;   /* 0 = full history */
        if (strcmp(cm, "last") == 0 || strcmp(cm, "none") == 0) {
            keep = 1;
        } else if (strncmp(cm, "last", 4) == 0) {
            long n = atol(cm + 4);
            if (n > 0) keep = (size_t)n;
        }
        if (keep > 0) {
            VJson *clipped = vjson_new_array();
            size_t nm = vjson_array_len(messages);
            /* system (if any) always survives at the front. */
            for (size_t i = 0; i < nm; i++) {
                const VJson *m = vjson_array_get(messages, i);
                const char *role = vjson_str(vjson_obj_get(m, "role"));
                if (role && strcmp(role, "system") == 0) {
                    vjson_array_push(clipped, vjson_clone(m));
                    break;
                }
            }
            size_t start = (nm >= keep) ? (nm - keep) : 0;
            for (size_t i = start; i < nm; i++) {
                const VJson *m = vjson_array_get(messages, i);
                const char *role = vjson_str(vjson_obj_get(m, "role"));
                if (role && strcmp(role, "system") == 0) continue; /* at front */
                vjson_array_push(clipped, vjson_clone(m));
            }
            vjson_obj_set(body, "messages", clipped);
            messages = vjson_obj_get(body, "messages");
        }
    }

    /* ---- 服务端会话历史重建（--session-dir）----
     * 带 session_id 且带 query 时，客户端只发本轮提问；服务端从会话文件取回
     * 时间线并拼上本轮 user，历史只在本会话（tenant/user/session 分区）内使用。
     * 只带 session_id 不带 query 时以客户端 messages 为准，推理后整条时间线
     * 落盘，供管理页审查。 */
    int sess_active = 0;
    if (ctx->session_on && session_id && session_id[0]) {
        char *raw = vllm_sess_read(ctx, tenant_id, part_user, session_id);
        VJson *stored = raw ? vllm_sess_parse(raw) : NULL;
        const VJson *stored_msgs = stored ? vjson_obj_get(stored, "messages") : NULL;
        if (stored_msgs && stored_msgs->type != VJ_ARRAY) stored_msgs = NULL;
        /* 会话绑定的规则包：本次未显式指定时继承（会话→规则的一致性）。 */
        if (!rb_id_buf[0] && stored) {
            const char *s = vjson_str(vjson_obj_get(stored, "rulebook_id"));
            if (s && s[0]) snprintf(rb_id_buf, sizeof(rb_id_buf), "%s", s);
        }
        int server_hist = (query && query[0]);
        VJson *tl = server_hist ? vllm_sess_append_turns(stored_msgs, query, NULL)
                                : vjson_clone(messages);
        if (server_hist) ctx->session_turns++;
        /* 逻辑时间线（不含规则前导块）挂在 body 上：body 在 handle_chat 的所有
         * 退出路径都会被 vjson_free，故无需在错误分支逐个释放。 */
        vjson_obj_set(body, "__sess_timeline", tl);
        if (server_hist) {
            vjson_obj_set(body, "messages", vjson_clone(tl));
            messages = vjson_obj_get(body, "messages");
        }
        sess_active = 1;
        {
            char sp[1400];
            vllm_sess_path(ctx, tenant_id, part_user, session_id, sp, sizeof(sp));
            fprintf(stderr, "[SESS] %s%s tenant=%s user=%s msgs=%d\n", sp,
                    server_hist ? " (server-history)" : " (client-array)",
                    tenant_id && tenant_id[0] ? tenant_id : "anon", part_user,
                    (int)vjson_array_len(messages));
            fflush(stderr);
        }
        free(raw);
        vjson_free(stored);
    }
    /* "只发 query" 依赖会话存储：未开 --session-dir 时无历史可重建，明确拒绝。 */
    if (!have_msgs && !sess_active) {
        json_error(resp, 400, "session_id + query requires --session-dir on the "
                              "server (or send a 'messages' array)");
        vjson_free(body);
        return;
    }

    int max_ids = ctx->ist->max_kv_slots;
    if (max_ids < 1024) max_ids = 1024;

    /* ---- 预置命名上下文（规则包）注入（--rulebook-dir）----
     * 规则文本作为**最前面的 system 前导块**注入（客户端自己的 system 消息随
     * 后排），使其成为 prompt 的稳定前缀 —— 与预处理时落盘的 KV 快照同源，
     * 于是该段 KV 可被逐 token 复用（见 run_completion 的 pfx_* 参数）。 */
    if (ctx->rulebook_on && rb_id_buf[0]) {
        VLLMRulebook rb;
        if (vllm_rb_load(ctx, rb_id_buf, &rb) != 0) {
            char msg[256];
            snprintf(msg, sizeof(msg), "rulebook '%s' not found", rb_id_buf);
            json_error(resp, 404, msg);
            vjson_free(body);
            return;
        }
        if (rb_ver_buf[0] && strcmp(rb_ver_buf, rb.version) != 0) {
            char msg[384];
            snprintf(msg, sizeof(msg),
                     "rulebook '%s' version mismatch: requested '%s', available '%s'",
                     rb_id_buf, rb_ver_buf, rb.version);
            json_error(resp, 409, msg);
            vllm_rb_free(&rb);
            vjson_free(body);
            return;
        }
        /* 模型指纹一致才敢复用 KV 快照（防"重建引擎后混用旧检查点"这一项目
         * 自陈的唯一真实风险）；不一致仍注入文本，只是回退全量 prefill。 */
        char cur_fp[VLLM_RB_HASH_HEX];
        vllm_rb_model_fp(ctx, cur_fp);
        int fp_ok = rb.model_fp[0] && cur_fp[0] &&
                    strcmp(rb.model_fp, cur_fp) == 0;
        if (!fp_ok) {
            fprintf(stderr, "[RB] model fingerprint mismatch (rulebook=%s "
                            "current=%s): KV snapshot invalidated -> full prefill\n",
                    rb.model_fp[0] ? rb.model_fp : "-", cur_fp[0] ? cur_fp : "-");
            fflush(stderr);
        }
        /* 规则前缀之外必须留出对话与生成余量，否则 prompt 超窗（400）。 */
        int budget = max_ids - RB_BUDGET_RESERVE;
        if (budget < 256) budget = 256;
        char *luq = last_user_text_str(messages);
        const char *qsel = (query && query[0]) ? query : luq;
        int sel_tok = 0;
        char *sel = vllm_rb_select_text(ctx, &rb, qsel, budget, &sel_tok);
        const char *inject = sel ? sel : (rb.text ? rb.text : "");
        VJson *arr2 = vjson_new_array();
        VJson *sm = vjson_new_object();
        vjson_obj_set(sm, "role", vjson_new_string("system"));
        vjson_obj_set(sm, "content", vjson_new_string(inject));
        vjson_array_push(arr2, sm);
        size_t nm = vjson_array_len(messages);
        for (size_t i = 0; i < nm; i++)
            vjson_array_push(arr2, vjson_clone(vjson_array_get(messages, i)));
        vjson_obj_set(body, "messages", arr2);
        messages = vjson_obj_get(body, "messages");

        snprintf(rb_ver_buf, sizeof(rb_ver_buf), "%s", rb.version);
        rb_tokens_val = sel_tok;
        vjson_obj_set(body, "__rb_inject", vjson_new_string(inject));
        vjson_obj_set(body, "__rb_reuse", vjson_new_bool(fp_ok));
        fprintf(stderr, "[RB] inject rulebook '%s' v%s: %d tokens%s%s\n",
                rb.id, rb.version[0] ? rb.version : "-", sel_tok,
                sel ? " (partial)" : " (full)",
                fp_ok ? "" : " [KV reuse disabled]");
        fflush(stderr);
        free(sel);
        free(luq);
        vllm_rb_free(&rb);
    }

    /* "上一问长度" metric: token count of the last user message (text). */
    int last_user_tokens = last_user_text_tokens(ctx, messages);

    /* Continuous batching mode (--batch-max >= 2): concurrent requests step
     * together via the batch scheduler instead of serializing on inf_lock. */
    const int batch_mode = (ctx->batch_max >= 2);
    if (batch_mode) batch_ensure(ctx);

    /* context_tokens：客户端回传的完整历史 token 序列（上轮响应 history_tokens
     * 提供）。仅串行文本路径支持；batch 模式与多模态请求忽略（各自回退全量）。 */
    const VJson *ct = vjson_obj_get(body, "context_tokens");
    int *ctx_tokens = NULL; int n_ctx = 0;
    char *tail_prompt = NULL;
    if (ct && ct->type == VJ_ARRAY && !batch_mode && !detect_media(messages)) {
        size_t cn = vjson_array_len(ct);
        if (cn > 0 && cn <= (size_t)max_ids) {
            ctx_tokens = (int *)malloc(cn * sizeof(int));
            if (ctx_tokens) {
                int bad = 0;
                for (size_t i = 0; i < cn; i++) {
                    const VJson *e = vjson_array_get(ct, i);
                    if (!e || e->type != VJ_NUMBER) { bad = 1; break; }
                    ctx_tokens[i] = (int)e->u.number;
                }
                if (!bad) n_ctx = (int)cn;
                else { free(ctx_tokens); ctx_tokens = NULL; }
            }
        }
        if (n_ctx > 0) {
            char *lt = last_user_text_str(messages);
            if (lt && lt[0]) {
                size_t tl = strlen(lt) + 64;
                tail_prompt = (char *)malloc(tl);
                if (tail_prompt)
                    snprintf(tail_prompt, tl,
                             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
                             lt, thinking ? "" : QWEN_NO_THINK_SUFFIX);
            }
            free(lt);
        }
    }

    /* ---- multimodal path (image_url / video_frames) ---- */
    if (detect_media(messages)) {
        if (!ctx->vis || !ctx->cfg->has_vision) {
            json_error(resp, 400, "Vision encoder not loaded on this server");
            vjson_free(body);
            return;
        }
        int *pt = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
        if (!pt) {
            json_error(resp, 500, "Out of memory");
            vjson_free(body);
            return;
        }
        /* Vision encoding + multimodal prefill share per-request state and
         * the single inference state: hold the inference lock throughout. */
        int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
        if (g == -1) {
            json_error(resp, 429, "Too many queued requests, retry later");
            free(pt);
            vjson_free(body);
            return;
        }
        if (g == -2) {
            json_error(resp, 503, "Insufficient memory, retry later");
            free(pt);
            vjson_free(body);
            return;
        }
        if (g == -3) {
            json_error(resp, 503, "Model is unloading, retry later");
            free(pt);
            vjson_free(body);
            return;
        }

        int pn = 0;
        float *vis_all = NULL;
        int vis_total = 0, vis_cap = 0;
        int grids[MM_MAX_REGIONS * 3];
        int n_regions = 0;
        DSAccum dsa; ds_accum_init(&dsa);
        /* Vision encoding (st_vision_encode_*) uses the engine thread pool:
         * serialize it against batch forwards in batch mode (the serial path
         * relies on inf_lock for the same). */
        /* P4 multimodal prefix reuse bookkeeping: per-row media marks aligned
         * with pt[] (only the serial path reuses prefix KV; batch ignores). */
        unsigned long long *mm_marks = NULL;
        if (!batch_mode)
            mm_marks = (unsigned long long *)calloc((size_t)max_ids + 16,
                                                    sizeof(unsigned long long));
        int rc;
        int last_user_vis = 0;
        if (batch_mode) vllm_batch_engine_lock(ctx->batch);
        rc = build_multimodal_prompt(ctx, messages, pt, max_ids + 8, &pn,
                                     &vis_all, &vis_total, &vis_cap,
                                     grids, &n_regions, &dsa,
                                     mm_marks, (int)(max_ids + 16), thinking,
                                     &last_user_vis);
        if (batch_mode) vllm_batch_engine_unlock(ctx->batch);
        /* "上一问长度"补视觉 token：图文轮只报文本会显著偏小 */
        if (rc == 0) last_user_tokens += last_user_vis;

        if (batch_mode) {
            /* ---- continuous-batching multimodal path ---- */
            VLLMMetrics met; memset(&met, 0, sizeof(met));
            met.last_user_tokens = last_user_tokens;
            VBatchReq r; memset(&r, 0, sizeof(r));
            r.is_mm = 1;
            r.pt = pt; r.pn = pn;
            r.vis_all = vis_all; r.vis_total = vis_total;
            r.grids = grids; r.n_regions = n_regions;
            const float *ds_rows[3] = { NULL, NULL, NULL };
            int n_ds = ctx->vis->cfg->vis_ds_count;
            for (int i = 0; i < n_ds && i < 3; i++) ds_rows[i] = dsa.rows[i];
            const float *const *ds_arg = (dsa.written[0] > 0) ? ds_rows : NULL;
            r.ds_features = ds_arg; r.n_ds = n_ds;
            r.temperature = temperature;
            r.top_p = top_p;
            r.min_p = min_p;
            r.top_k = top_k;
            r.thinking = thinking;
            r.max_tokens = max_tokens;
            r.stream = stream;
            r.conn = conn;
            r.met = &met;
            /* 流式也累积输出文本：batch 流式出证需要全量 text（vb 末尾 SSE
             * attest 事件前读取）。非流式本就写入 out_txt。 */
            char *out_txt = NULL; int out_len = 0;
            r.out_text = &out_txt;
            r.out_len = &out_len;
            r.body_sha = bodyhash;
            r.emit_meta = emit_meta;
            int brc = 0;
            if (rc != 0 || pn > max_ids) {
                brc = rc == -2 ? -2 : -1;
            } else if (stream) {
                resp->status = 200;
                resp->content_type = "text/event-stream";
                resp->stream = 1;
                vhttp_stream_begin(conn, 200, "text/event-stream");
                static const char role_preamble[] =
                    "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                    "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
                vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
                vllm_batch_run(ctx->batch, &r);
            } else {
                brc = vllm_batch_run(ctx->batch, &r);
                if (brc == -2) {
                    json_error(resp, 400, "Prompt exceeds the context limit");
                } else if (brc != 0) {
                    json_error(resp, 500, "Inference failed");
                } else {
                    char *body = (char *)malloc(1 << 20);
                    if (!body) {
                        json_error(resp, 500, "Out of memory");
                    } else {
                        VJson *root = vjson_new_object();
                        vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                        vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                        vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                        vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                        VJson *choices = vjson_new_array();
                        VJson *ch = vjson_new_object();
                        vjson_obj_set(ch, "index", vjson_new_number(0));
                        VJson *msg = vjson_new_object();
                        vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                        vjson_obj_set(msg, "content", vjson_new_string(out_txt ? out_txt : ""));
                        vjson_obj_set(ch, "message", msg);
                        vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                        vjson_array_push(choices, ch);
                        vjson_obj_set(root, "choices", choices);
                        vjson_obj_set(root, "usage", usage_json(&met));
                        VJson *mj = vjson_new_object();
                        vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                        vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                        vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                        vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                        vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                        vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                        vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                        vjson_obj_set(mj, "tok_s", vjson_new_number(
                            met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                        vjson_obj_set(root, "metrics", mj);
                        /* 方案 2：多模态批处理非流式出证。 */
                        attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                                      met.prompt_tokens, met.n_tokens, "stop",
                                      temperature, top_p, min_p, top_k, thinking,
                                      max_tokens);
                        size_t n = vjson_serialize(root, body, 1 << 20);
                        vjson_free(root);
                        resp->status = 200;
                        resp->content_type = "application/json";
                        resp->body = body;
                        resp->body_len = n;
                        resp->body_owned = body;
                    }
                }
            }
            free(out_txt);
            ds_accum_free(&dsa);
            infer_done_batch(ctx);
            free(pt);
            free(vis_all);
            free(mm_marks);   /* NULL in batch mode: no-op */
            vjson_free(body);
            return;
        }

        if (rc != 0 || pn > max_ids) {
            json_error(resp, 400,
                       rc == -2 ? "Prompt exceeds the context limit"
                                : "Media decode/encode failed (base64 data URLs required)");
            ds_accum_free(&dsa);
            infer_done(ctx);
        } else {
            /* Serial path only: per-user L3 isolation (batch mode skips L3). */
            l3_user_apply(ctx, user);
            /* DeepStack features 鈫?[n_ds][vis_total, dim] (NULL if no media) */
            const float *ds_rows[3] = { NULL, NULL, NULL };
            int n_ds = ctx->vis->cfg->vis_ds_count;
            for (int i = 0; i < n_ds && i < 3; i++) ds_rows[i] = dsa.rows[i];
            const float *const *ds_arg = (dsa.written[0] > 0) ? ds_rows : NULL;
            if (stream) {
                resp->status = 200;
                resp->content_type = "text/event-stream";
                resp->stream = 1;
                vhttp_stream_begin(conn, 200, "text/event-stream");
                static const char role_preamble[] =
                    "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                    "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
                vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
                StrBuf sink; sb_init(&sink);
                VLLMMetrics met; memset(&met, 0, sizeof(met));
                met.last_user_tokens = last_user_tokens;
                run_completion_mm(ctx, pt, pn, vis_all, vis_total, grids, n_regions,
                                  ds_arg, n_ds, temperature, top_p, min_p, top_k,
                                  thinking, max_tokens, 1, conn, emit_meta, &sink, &met,
                                  mm_marks, user, bodyhash);
                sb_free(&sink);
                ds_accum_free(&dsa);
                infer_done(ctx);
            } else {
                StrBuf out; sb_init(&out);
                VLLMMetrics met; memset(&met, 0, sizeof(met));
                met.last_user_tokens = last_user_tokens;
                int rc2 = run_completion_mm(ctx, pt, pn, vis_all, vis_total, grids,
                                            n_regions, ds_arg, n_ds, temperature,
                                            top_p, min_p, top_k, thinking,
                                            max_tokens, 0, NULL, 0, &out, &met,
                                            mm_marks, user, bodyhash);
                ds_accum_free(&dsa);
                infer_done(ctx);
                if (rc2 != 0) {
                    json_error(resp, 500, "Inference failed");
                } else {
                    char *body = (char *)malloc(1 << 20);
                    if (!body) {
                        json_error(resp, 500, "Out of memory");
                    } else {
                        VJson *root = vjson_new_object();
                        vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                        vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                        vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                        vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                        VJson *choices = vjson_new_array();
                        VJson *ch = vjson_new_object();
                        vjson_obj_set(ch, "index", vjson_new_number(0));
                        VJson *msg = vjson_new_object();
                        vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                        vjson_obj_set(msg, "content", vjson_new_string(out.s ? out.s : ""));
                        vjson_obj_set(ch, "message", msg);
                        vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                        vjson_array_push(choices, ch);
                        vjson_obj_set(root, "choices", choices);
                        vjson_obj_set(root, "usage", usage_json(&met));
                        /* Per-request inference metrics. */
                        VJson *mj = vjson_new_object();
                        vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                        vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                        vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                        vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                        vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                        vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                        vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                        vjson_obj_set(mj, "tok_s", vjson_new_number(
                            met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                        vjson_obj_set(root, "metrics", mj);
                        /* 方案 2：多模态串行非流式出证。 */
                         attest_attach(root, user, bodyhash, out.s ? out.s : "",
                                      met.prompt_tokens, met.n_tokens, "stop",
                                      temperature, top_p, min_p, top_k, thinking,
                                      max_tokens);
                        size_t n = vjson_serialize(root, body, 1 << 20);
                        vjson_free(root);
                        resp->status = 200;
                        resp->content_type = "application/json";
                        resp->body = body;
                        resp->body_len = n;
                        resp->body_owned = body;
                    }
                }
                sb_free(&out);
            }
        }
        free(pt);
        free(vis_all);
        free(mm_marks);
        vjson_free(body);
        return;
    }

    /* ---- text-only path ---- */
    char *prompt = build_chat_prompt(ctx, messages, thinking);
    if (!prompt) {
        json_error(resp, 500, "Prompt build failed");
        vjson_free(body);
        return;
    }

    /* Prompt-length pre-check BEFORE any 200/SSE header goes out, so an
     * over-long context gets a clean 400 instead of a mid-prefill crash. */
    int *ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!ids) {
        json_error(resp, 500, "Out of memory");
        free(prompt);
        vjson_free(body);
        return;
    }
    int n_ids = qwen_tokenizer_encode(ctx->tok, prompt, ids, max_ids + 16);
    free(ids);
    if (n_ids <= 0) {
        json_error(resp, 400, "Prompt tokenization failed");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (n_ids > max_ids) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "Prompt of %d tokens exceeds the %d-token context limit",
                 n_ids, max_ids);
        json_error(resp, 400, msg);
        free(prompt);
        vjson_free(body);
        return;
    }

    /* Admission control: bounded queue (429) + free-RAM check (503).
     * Serial: infer_gate blocks until the running request finishes, then
     * holds the inference lock for this request (queuing semantics).
     * Batch: infer_gate_batch never blocks; the batch scheduler owns the
     * engine critical section (continuous batching). */
    int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
    if (g == -1) {
        json_error(resp, 429, "Too many queued requests, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (g == -2) {
        json_error(resp, 503, "Insufficient memory, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (g == -3) {
        json_error(resp, 503, "Model is unloading, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    /* 规则包未启用（无 --rulebook-dir）时不得把请求里的 id 回显成"已生效的
     * 规则包"：metrics / 审计把 rulebook_id 当作"实际使用的规则"的证据，
     * 回显未生效的 id 会让排障与审计误判。降级模式下留空（配合
     * rulebook_tokens=0 / reused_prefix=false，调用方能量化"未生效"）。 */
    const char *met_rb_id  = ctx->rulebook_on ? rb_id_buf  : "";
    const char *met_rb_ver = ctx->rulebook_on ? rb_ver_buf : "";
    if (batch_mode) {
        /* ---- continuous-batching path (shared weights, private KV) ---- */
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        met_fill_ids(&met, trace_id, met_rb_id, met_rb_ver, rb_tokens_val);
        VBatchReq r; memset(&r, 0, sizeof(r));
        r.prompt = prompt;
        r.temperature = temperature;
        r.top_p = top_p;
        r.min_p = min_p;
        r.top_k = top_k;
        r.thinking = thinking;
        r.max_tokens = max_tokens;
        r.stream = stream;
        r.conn = conn;
        r.met = &met;
        /* 流式也累积输出文本：batch 流式出证需要全量 text。 */
        char *out_txt = NULL; int out_len = 0;
        r.out_text = &out_txt;
        r.out_len = &out_len;
        r.body_sha = bodyhash;
        r.emit_meta = emit_meta;
        if (stream) {
            resp->status = 200;
            resp->content_type = "text/event-stream";
            resp->stream = 1;
            vhttp_stream_begin(conn, 200, "text/event-stream");
            static const char role_preamble[] =
                "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
            vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
            vllm_batch_run(ctx->batch, &r);
        } else {
            int rc = vllm_batch_run(ctx->batch, &r);
            if (rc == -2) {
                json_error(resp, 400, "Prompt exceeds the context limit");
            } else if (rc != 0) {
                json_error(resp, 500, "Inference failed");
            } else {
                char *body = (char *)malloc(1 << 20);
                if (!body) {
                    json_error(resp, 500, "Out of memory");
                } else {
                    VJson *root = vjson_new_object();
                    vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                    vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                    vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                    VJson *choices = vjson_new_array();
                    VJson *ch = vjson_new_object();
                    vjson_obj_set(ch, "index", vjson_new_number(0));
                    VJson *msg = vjson_new_object();
                    vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                    vjson_obj_set(msg, "content", vjson_new_string(out_txt ? out_txt : ""));
                    vjson_obj_set(ch, "message", msg);
                    vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                    vjson_array_push(choices, ch);
                    vjson_obj_set(root, "choices", choices);
                    vjson_obj_set(root, "usage", usage_json(&met));
                    VJson *mj = vjson_new_object();
                    vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                    vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                    vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                    vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                    vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                    vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                    vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                    vjson_obj_set(mj, "tok_s", vjson_new_number(
                        met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                    vjson_obj_set(root, "metrics", mj);
                    /* 方案 2：文本批处理非流式出证。 */
                     attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                                  met.prompt_tokens, met.n_tokens, "stop",
                                  temperature, top_p, min_p, top_k, thinking,
                                  max_tokens);
                    size_t n = vjson_serialize(root, body, 1 << 20);
                    vjson_free(root);
                    resp->status = 200;
                    resp->content_type = "application/json";
                    resp->body = body;
                    resp->body_len = n;
                    resp->body_owned = body;
                }
            }
        }
        free(out_txt);
        infer_done_batch(ctx);
        free(prompt);
        vjson_free(body);
        return;
    }
    /* Per-user L3 isolation (reap offline users, bind this request's file). */
    l3_user_apply(ctx, user);

    /* ---- 预置上下文 KV 快照复用（串行文本路径）----
     * 渲染规则前导块并编码，得到"本次实际注入前缀"的 token 序列与该选择对应
     * 的快照路径；run_completion 校验 prompt 前缀与之逐 token 一致后，从快照
     * 恢复整段 KV，只 prefill 余量。batch 路径不支持（与既有前缀复用同一
     * 边界）；reuse_prefix_cache=false 或模型指纹不符时不传路径 → 全量 prefill。 */
    int *rb_ids = NULL; int n_rb = 0;
    char rb_kv_path[1800];
    rb_kv_path[0] = '\0';
    {
        const char *rx = vjson_str(vjson_obj_get(body, "__rb_inject"));
        int rb_reuse_ok = vjson_bool(vjson_obj_get(body, "__rb_reuse"));
        if (rx && rx[0] && reuse_prefix && rb_reuse_ok) {
            char *blk = vllm_rb_render_block(rx);
            if (blk) {
                rb_ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
                if (rb_ids)
                    n_rb = qwen_tokenizer_encode(ctx->tok, blk, rb_ids,
                                                 max_ids + 16);
                free(blk);
            }
            if (rb_ids && n_rb > 0) {
                VLLMRulebook tag;
                memset(&tag, 0, sizeof(tag));
                snprintf(tag.id, sizeof(tag.id), "%s", rb_id_buf);
                snprintf(tag.version, sizeof(tag.version), "%s", rb_ver_buf);
                vllm_rb_kv_path(ctx, &tag, rb_ids, n_rb, rb_kv_path,
                                sizeof(rb_kv_path));
            }
        }
    }

    if (stream) {
        resp->status = 200;
        resp->content_type = "text/event-stream";
        resp->stream = 1;
        /* Must send the HTTP response header before any SSE payload. */
        vhttp_stream_begin(conn, 200, "text/event-stream");
        /* First chunk: role delta. */
        static const char role_preamble[] =
            "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
            "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
        vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
        StrBuf sink; sb_init(&sink);
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        met_fill_ids(&met, trace_id, met_rb_id, met_rb_ver, rb_tokens_val);
        int rc = run_completion(ctx, prompt, tail_prompt, ctx_tokens, n_ctx,
                                rb_ids, n_rb, rb_kv_path,
                                temperature, top_p, min_p, top_k, thinking,
                                max_tokens, 1, conn, emit_meta, &sink, &met, NULL, NULL,
                                user, bodyhash);
        if (rc == 0) {
            sess_commit(ctx, body, session_id, tenant_id, part_user,
                        rb_id_buf[0] ? rb_id_buf : NULL,
                        sink.s ? sink.s : "", met.n_tokens, met.total_ms);
            rb_lazy_snapshot(ctx, rb_id_buf, rb_ver_buf, rb_ids, n_rb);
        }
        sb_free(&sink);
        (void)rc;   /* streaming errors surface as a dropped connection */
        infer_done(ctx);
    } else {
        StrBuf out; sb_init(&out);
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        met_fill_ids(&met, trace_id, met_rb_id, met_rb_ver, rb_tokens_val);
        int *hist_ids = NULL; int hist_n = 0;
        int rc = run_completion(ctx, prompt, tail_prompt, ctx_tokens, n_ctx,
                                rb_ids, n_rb, rb_kv_path,
                                temperature, top_p, min_p, top_k, thinking,
                                max_tokens, 0, NULL, 0, &out, &met,
                                &hist_ids, &hist_n, user, bodyhash);
        infer_done(ctx);
        if (rc == 0) {
            sess_commit(ctx, body, session_id, tenant_id, part_user,
                        rb_id_buf[0] ? rb_id_buf : NULL,
                        out.s ? out.s : "", met.n_tokens, met.total_ms);
            rb_lazy_snapshot(ctx, rb_id_buf, rb_ver_buf, rb_ids, n_rb);
        }
        if (rc == -2) {
            json_error(resp, 400, "Prompt exceeds the context limit");
        } else if (rc != 0) {
            json_error(resp, 500, "Inference failed");
        } else {
            char *body = (char *)malloc(1 << 20);
            if (!body) {
                json_error(resp, 500, "Out of memory");
            } else {
                VJson *root = vjson_new_object();
                vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                VJson *choices = vjson_new_array();
                VJson *ch = vjson_new_object();
                vjson_obj_set(ch, "index", vjson_new_number(0));
                VJson *msg = vjson_new_object();
                vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                vjson_obj_set(msg, "content", vjson_new_string(out.s ? out.s : ""));
                vjson_obj_set(ch, "message", msg);
                vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                vjson_array_push(choices, ch);
                vjson_obj_set(root, "choices", choices);
                vjson_obj_set(root, "usage", usage_json(&met));
                if (hist_ids && hist_n) {
                    VJson *ht = vjson_new_array();
                    for (int i = 0; i < hist_n; i++)
                        vjson_array_push(ht, vjson_new_number((double)hist_ids[i]));
                    vjson_obj_set(root, "history_tokens", ht);
                }
                /* Per-request inference metrics for the chat client. */
                VJson *mj = vjson_new_object();
                vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                vjson_obj_set(mj, "tok_s", vjson_new_number(
                    met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                /* 工业边缘：请求级标识（追溯 / 规则包版本 / 前缀复用命中）。 */
                vjson_obj_set(mj, "trace_id", vjson_new_string(trace_id));
                vjson_obj_set(mj, "rulebook_id", vjson_new_string(met_rb_id));
                vjson_obj_set(mj, "rulebook_version", vjson_new_string(met_rb_ver));
                vjson_obj_set(mj, "rulebook_tokens",
                              vjson_new_number((double)rb_tokens_val));
                vjson_obj_set(mj, "reused_prefix",
                              vjson_new_bool(met.reused_prefix));
                vjson_obj_set(root, "metrics", mj);
                vjson_obj_set(root, "trace_id", vjson_new_string(trace_id));
                /* 方案 2：文本串行非流式出证。 */
                 attest_attach(root, user, bodyhash, out.s ? out.s : "",
                              met.prompt_tokens, met.n_tokens, "stop",
                              temperature, top_p, min_p, top_k, thinking,
                              max_tokens);
                size_t n = vjson_serialize(root, body, 1 << 20);
                vjson_free(root);
                resp->status = 200;
                resp->content_type = "application/json";
                resp->body = body;
                resp->body_len = n;
                resp->body_owned = body;
            }
        }
        sb_free(&out);
        if (hist_ids) free(hist_ids);
    }
    free(prompt);
    free(ctx_tokens);
    free(tail_prompt);
    free(rb_ids);
    vjson_free(body);
}

/* ---- K-COLLAB-001 P2：专用 embedding 模型懒加载（独立 tok/ist/w/cfg 槽位）----
 * 仅加载 causal-LM 文本塔（无 vision）。调用方须已持 inf_lock（与 chat 串行），
 * 本函数内部用 embed_lock 保护加载互斥。返回 0 成功，-1 失败（embed_error 记因）。 */
static int vllm_embed_load(VLLMServerCtx *ctx) {
    const char *dir = ctx->embed_model_dir;
    if (!dir || !dir[0]) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "embed_model_dir is empty");
        return -1;
    }
    char vqf_path[768];
    snprintf(vqf_path, sizeof(vqf_path), "%s/model.vqf", dir);
    if (!vqf_is_file(vqf_path)) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "no model.vqf in %s", dir);
        return -1;
    }
    STModelWeights *w = (STModelWeights *)calloc(1, sizeof(STModelWeights));
    if (!w) { snprintf(ctx->embed_error, sizeof(ctx->embed_error), "OOM (weights)"); return -1; }
    if (vqf_load(w, vqf_path) != 0) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "vqf_load failed: %s", vqf_path);
        free(w); return -1;
    }
    QwenTokenizer *tok = (QwenTokenizer *)calloc(1, sizeof(QwenTokenizer));
    if (!tok) { snprintf(ctx->embed_error, sizeof(ctx->embed_error), "OOM (tokenizer)"); st_weights_free(w); free(w); return -1; }
    if (qwen_tokenizer_load(tok, dir) != 0) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "tokenizer load failed: %s", dir);
        qwen_tokenizer_free(tok); free(tok); st_weights_free(w); free(w); return -1;
    }
    qwen_tokenizer_load_special(tok, dir);
    STQwenInferenceState *ist = (STQwenInferenceState *)calloc(1, sizeof(STQwenInferenceState));
    if (!ist) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "OOM (inference state)");
        qwen_tokenizer_free(tok); free(tok); st_weights_free(w); free(w); return -1;
    }
    if (st_qwen_inference_init(ist, w) != 0) {
        snprintf(ctx->embed_error, sizeof(ctx->embed_error), "inference init failed: %s", dir);
        st_qwen_inference_free(ist); free(ist);
        qwen_tokenizer_free(tok); free(tok); st_weights_free(w); free(w); return -1;
    }
    ctx->embed_w = w;
    ctx->embed_cfg = &w->cfg;
    ctx->embed_tok = tok;
    ctx->embed_ist = ist;
    ctx->embed_loaded = 1;
    fprintf(stderr, "[EMB-MODEL] dedicated embedding model ready: %s (dim=%d layers=%d)\n",
            dir, ctx->embed_cfg->dim, ctx->embed_cfg->n_layers);
    fflush(stderr);
    return 0;
}

static void vllm_embed_free(VLLMServerCtx *ctx) {
    if (ctx->embed_ist) { st_qwen_inference_free(ctx->embed_ist); free(ctx->embed_ist); ctx->embed_ist = NULL; }
    if (ctx->embed_tok) { qwen_tokenizer_free(ctx->embed_tok); free(ctx->embed_tok); ctx->embed_tok = NULL; }
    if (ctx->embed_w)   { st_weights_free(ctx->embed_w); free(ctx->embed_w); ctx->embed_w = NULL; }
    ctx->embed_cfg = NULL;
    ctx->embed_loaded = 0;
}

/* ---------- /v1/embeddings (OpenAI-compatible) ----------
 * K-EMB-001：Qwen3-Embedding 系（last-token pooling + L2 normalize）。
 * 输入按**裸文本**编码（不套 chat 模板），逐条全量复位后单次 prefill，
 * 取末位 final-norm 隐状态（引擎 prefill 收尾的 g_embed_out 钩子，已 L2
 * 归一化）。仅串行路径；--batch-max 调度器持独立 engine 锁且无 pooling
 * 分支，故批处理开启时显式拒绝以免与调度器并发抢引擎。 */
static void handle_embeddings(VLLMServerCtx *ctx, const VHttpRequest *req,
                              VHttpResponse *resp) {
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }
    VJson *jin = vjson_obj_get(body, "input");
    if (!jin || (jin->type != VJ_STRING && jin->type != VJ_ARRAY)) {
        json_error(resp, 400, "'input' must be a string or an array of strings");
        vjson_free(body);
        return;
    }
    int n_in = (jin->type == VJ_ARRAY) ? (int)vjson_array_len(jin) : 1;
    if (n_in <= 0) {
        json_error(resp, 400, "'input' is empty");
        vjson_free(body);
        return;
    }
    if (ctx->batch) {
        json_error(resp, 400, "Embeddings are unavailable when --batch-max is on");
        vjson_free(body);
        return;
    }
    int g = infer_gate(ctx);
    if (g == -1) { json_error(resp, 429, "Too many queued requests, retry later"); vjson_free(body); return; }
    if (g == -2) { json_error(resp, 503, "Insufficient memory, retry later"); vjson_free(body); return; }
    if (g == -3) { json_error(resp, 503, "Model is unloading, retry later"); vjson_free(body); return; }

    /* K-COLLAB-001 P2：embedding 用主模型（默认）或专用 --embed-model（独立槽位）。
     * 二者共用 inf_lock 串行；专用模型在首次请求时懒加载（embed_lock 保护）。 */
    STQwenInferenceState *st;
    STModelConfig *cfg;
    QwenTokenizer *tok;
    int nl;
    if (ctx->embed_model_dir) {
        pthread_mutex_lock(&ctx->embed_lock);
        if (ctx->embed_loaded == 0 && vllm_embed_load(ctx) != 0) {
            pthread_mutex_unlock(&ctx->embed_lock);
            infer_done(ctx);
            json_error(resp, 503, ctx->embed_error);
            vjson_free(body);
            return;
        }
        pthread_mutex_unlock(&ctx->embed_lock);
        st = ctx->embed_ist;
        cfg = ctx->embed_cfg;
        tok = ctx->embed_tok;
        nl = ctx->embed_w->n_layers_allocated;
    } else {
        st = ctx->ist;
        cfg = ctx->cfg;
        tok = ctx->tok;
        nl = ctx->w->n_layers_allocated;
    }
    int dim = cfg->dim;
    int max_seq = cfg->max_seq_len;
    float *vec = (float *)malloc((size_t)dim * sizeof(float));
    int   *ids = (int *)malloc(((size_t)max_seq + 16) * sizeof(int));

    VJson *root = NULL;
    if (!vec || !ids) {
        json_error(resp, 500, "Out of memory");
    } else {
        root = vjson_new_object();
        vjson_obj_set(root, "object", vjson_new_string("list"));
        vjson_obj_set(root, "model",
                      vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-embedding-0.6b"));
        VJson *data = vjson_new_array();
        int total_tokens = 0;
        for (int i = 0; i < n_in; i++) {
            const char *txt = (jin->type == VJ_ARRAY)
                ? vjson_str(vjson_array_get(jin, (size_t)i)) : vjson_str(jin);
            if (!txt) {
                json_error(resp, 400, "each 'input' element must be a string");
                vjson_free(data); vjson_free(root); root = NULL;
                break;
            }
            /* 每条输入 = 全新序列：复刻 ist_reset 全量复位分支（无前缀复用）。 */
            for (int l = 0; l < nl; l++) st->cache_len[l] = 0;
            st->seq_len = 0; st->mrope_pos = 0;
            st_qwen_kv_rebuild_freed(st);
            int n = qwen_tokenizer_encode(tok, txt, ids, max_seq - 1);
            if (n <= 0) {
                json_error(resp, 400, "input tokenization failed");
                vjson_free(data); vjson_free(root); root = NULL;
                break;
            }
            if (n > max_seq - 8) n = max_seq - 8;
            /* 官方 recipe：追加 EOS（add_special_tokens=True），last-token pooling
             * 取 EOS 位置（见 main.c run_embed 同段注释）。 */
            if (n < max_seq && ids[n-1] != tok->eos_id) ids[n++] = tok->eos_id;
            st_embed_request(vec, dim);              /* 登记钩子（一次性） */
            if (st_qwen_model_prefill_batch(st, ids, n) != 0) {
                json_error(resp, 500, "Inference failed");
                vjson_free(data); vjson_free(root); root = NULL;
                break;
            }
            total_tokens += n;
            VJson *item = vjson_new_object();
            vjson_obj_set(item, "object", vjson_new_string("embedding"));
            vjson_obj_set(item, "index", vjson_new_number((double)i));
            VJson *arr = vjson_new_array();
            for (int k = 0; k < dim; k++)
                vjson_array_push(arr, vjson_new_number((double)vec[k]));
            vjson_obj_set(item, "embedding", arr);
            vjson_array_push(data, item);
        }
        if (root) {
            vjson_obj_set(root, "data", data);
            VJson *usage = vjson_new_object();
            vjson_obj_set(usage, "prompt_tokens", vjson_new_number((double)total_tokens));
            vjson_obj_set(usage, "total_tokens", vjson_new_number((double)total_tokens));
            vjson_obj_set(root, "usage", usage);
        }
    }

    infer_done(ctx);
    free(vec);
    free(ids);
    vjson_free(body);

    if (root) {
        size_t cap = (size_t)dim * 24 * (size_t)n_in + 4096;
        char *buf = (char *)malloc(cap);
        if (!buf) {
            json_error(resp, 500, "Out of memory");
        } else {
            size_t n = vjson_serialize(root, buf, cap);
            if (n > 0) {
                resp->status = 200;
                resp->content_type = "application/json";
                resp->body = buf;
                resp->body_len = n;
                resp->body_owned = buf;
            } else {
                free(buf);
                json_error(resp, 500, "Response too large");
            }
        }
        vjson_free(root);
    }
}

/* ---------- /v1/completions (legacy) ---------- */

static void handle_completions(VLLMServerCtx *ctx, const VHttpRequest *req,
                               VHttpResponse *resp, VHttpConn *conn) {
    /* 云边协同：仅在配置了云端点时标记来源（未配置 = 响应逐字节不变，零回归）。 */
    if (ctx->cloud_endpoint && ctx->cloud_endpoint[0])
        resp->extra_header = "x-kestrel-source: edge\r\n";
    if (cloud_fallback_serve(ctx, req, resp, conn) == 0) return;
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) {
        json_error(resp, 400, "Invalid JSON body");
        return;
    }
    const char *prompt = vjson_str(vjson_obj_get(body, "prompt"));
    if (!prompt) {
        json_error(resp, 400, "'prompt' is required");
        vjson_free(body);
        return;
    }
    /* 方案 2：原始请求体 SM3（出证/验证均基于它）。 */
    char bodyhash[65];
    vatt_body_digest(req->body, req->body_len, bodyhash);
    double temperature = 0.7, top_p = 0.9, min_p = ctx->min_p;
    int top_k = ctx->top_k;
    int thinking = resolve_thinking(body);
    int max_tokens = 64;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER) temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER) top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "min_p")) && jv->type == VJ_NUMBER) min_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_k")) && jv->type == VJ_NUMBER) {
        top_k = (int)jv->u.number;
        if (top_k < 0) top_k = 0;
    }
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number;
        if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }
    const char *user = vjson_str(vjson_obj_get(body, "user"));
    char *full = build_completion_prompt(ctx, prompt, thinking);
    if (!full) {
        json_error(resp, 500, "Prompt build failed");
        vjson_free(body);
        return;
    }

    /* Same admission control as chat: bounded queue (429) + RAM (503).
     * Batch mode: never blocks on inf_lock; the scheduler owns the engine. */
    const int batch_mode = (ctx->batch_max >= 2);
    if (batch_mode) batch_ensure(ctx);
    int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
    if (g == -1) {
        json_error(resp, 429, "Too many queued requests, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (g == -2) {
        json_error(resp, 503, "Insufficient memory, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (g == -3) {
        json_error(resp, 503, "Model is unloading, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (batch_mode) {
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        VBatchReq r; memset(&r, 0, sizeof(r));
        r.prompt = full;
        r.temperature = temperature;
        r.top_p = top_p;
        r.min_p = min_p;
        r.top_k = top_k;
        r.thinking = thinking;
        r.max_tokens = max_tokens;
        r.stream = 0;
        r.met = &met;
        char *out_txt = NULL; int out_len = 0;
        r.out_text = &out_txt;
        r.out_len = &out_len;
        r.body_sha = bodyhash;
        int rc = vllm_batch_run(ctx->batch, &r);
        infer_done_batch(ctx);
        if (rc == -2) {
            json_error(resp, 400, "Prompt exceeds the context limit");
        } else if (rc != 0) {
            json_error(resp, 500, "Inference failed");
        } else {
            char *body = (char *)malloc(1 << 20);
            if (!body) {
                json_error(resp, 500, "Out of memory");
            } else {
                VJson *root = vjson_new_object();
                vjson_obj_set(root, "id", vjson_new_string("cmpl-00000000"));
                vjson_obj_set(root, "object", vjson_new_string("text_completion"));
                vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                VJson *choices = vjson_new_array();
                VJson *ch = vjson_new_object();
                vjson_obj_set(ch, "index", vjson_new_number(0));
                vjson_obj_set(ch, "text", vjson_new_string(out_txt ? out_txt : ""));
                vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                vjson_array_push(choices, ch);
                vjson_obj_set(root, "choices", choices);
                vjson_obj_set(root, "usage", usage_json(&met));
                VJson *mj = vjson_new_object();
                vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                vjson_obj_set(mj, "tok_s", vjson_new_number(
                    met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                vjson_obj_set(root, "metrics", mj);
                /* 方案 2：legacy 批处理非流式出证。 */
                attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                              met.prompt_tokens, met.n_tokens, "stop",
                              temperature, top_p, min_p, top_k, thinking,
                              max_tokens);
                size_t n = vjson_serialize(root, body, 1 << 20);
                vjson_free(root);
                resp->status = 200;
                resp->content_type = "application/json";
                resp->body = body;
                resp->body_len = n;
                resp->body_owned = body;
            }
        }
        free(out_txt);
        free(full);
        vjson_free(body);
        return;
    }
    /* Per-user L3 isolation (reap offline users, bind this request's file). */
    l3_user_apply(ctx, user);

    StrBuf out; sb_init(&out);
    VLLMMetrics met; memset(&met, 0, sizeof(met));
    int rc = run_completion(ctx, full, NULL, NULL, 0, NULL, 0, NULL,
                            temperature, top_p, min_p,
                            top_k, thinking, max_tokens, 0, NULL, 0, &out, &met, NULL,
                            NULL, user, bodyhash);
    infer_done(ctx);
    if (rc == -2) {
        json_error(resp, 400, "Prompt exceeds the context limit");
    } else if (rc != 0) {
        json_error(resp, 500, "Inference failed");
    } else {
        char *body = (char *)malloc(1 << 20);
        if (!body) {
            json_error(resp, 500, "Out of memory");
        } else {
            VJson *root = vjson_new_object();
            vjson_obj_set(root, "id", vjson_new_string("cmpl-00000000"));
            vjson_obj_set(root, "object", vjson_new_string("text_completion"));
            vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
            vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
            VJson *choices = vjson_new_array();
            VJson *ch = vjson_new_object();
            vjson_obj_set(ch, "index", vjson_new_number(0));
            vjson_obj_set(ch, "text", vjson_new_string(out.s ? out.s : ""));
            vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
            vjson_array_push(choices, ch);
            vjson_obj_set(root, "choices", choices);
            vjson_obj_set(root, "usage", usage_json(&met));
            /* 方案 2：legacy 串行非流式出证。 */
            attest_attach(root, user, bodyhash, out.s ? out.s : "",
                          met.prompt_tokens, met.n_tokens, "stop",
                          temperature, top_p, min_p, top_k, thinking,
                          max_tokens);
            size_t n = vjson_serialize(root, body, 1 << 20);
            vjson_free(root);
            resp->status = 200;
            resp->content_type = "application/json";
            resp->body = body;
            resp->body_len = n;
            resp->body_owned = body;
        }
    }
    sb_free(&out);
    free(full);
    vjson_free(body);
}

/* ---------- /v1/messages (Anthropic-compatible) ---------- */

/* Extract plain text from an Anthropic content field (string or
 * [{type:"text",text:...}, ...]); appends to sb. */
static void anthropic_content_to_text(const VJson *content, StrBuf *sb) {
    if (!content) return;
    if (content->type == VJ_STRING) {
        const char *t = vjson_str(content);
        if (t) sb_str(sb, t);
        return;
    }
    if (content->type == VJ_ARRAY) {
        size_t n = vjson_array_len(content);
        for (size_t i = 0; i < n; i++) {
            const VJson *part = vjson_array_get(content, i);
            if (!part) continue;
            const char *ptype = vjson_str(vjson_obj_get(part, "type"));
            if (ptype && strcmp(ptype, "text") == 0) {
                const char *t = vjson_str(vjson_obj_get(part, "text"));
                if (t) sb_str(sb, t);
            }
        }
    }
}

/* Convert the OpenAI chat response body into an Anthropic message body.
 * Returns a malloc'd string (caller frees) or NULL. */
static char *anthropic_from_openai(VLLMServerCtx *ctx, const char *openai_body) {
    VJson *j = vjson_parse(openai_body);
    if (!j) return NULL;
    const VJson *choices = vjson_obj_get(j, "choices");
    const VJson *ch = (choices && choices->type == VJ_ARRAY && vjson_array_len(choices) > 0)
                      ? vjson_array_get(choices, 0) : NULL;
    const char *text = "";
    if (ch) {
        const VJson *msg = vjson_obj_get(ch, "message");
        if (msg) {
            const char *c = vjson_str(vjson_obj_get(msg, "content"));
            if (c) text = c;
        }
    }
    const char *fr = ch ? vjson_str(vjson_obj_get(ch, "finish_reason")) : NULL;
    const char *stop_reason = (fr && strcmp(fr, "length") == 0) ? "max_tokens" : "end_turn";
    int in_tok = 0, out_tok = 0;
    /* vllm_kestrel 的 OpenAI 响应把指标放在顶层 "metrics"（usage 是空对象）。 */
    const VJson *met = vjson_obj_get(j, "metrics");
    if (met) {
        const VJson *mj = vjson_obj_get(met, "prompt_tokens");
        if (mj && mj->type == VJ_NUMBER) in_tok = (int)mj->u.number;
        mj = vjson_obj_get(met, "n_tokens");
        if (mj && mj->type == VJ_NUMBER) out_tok = (int)mj->u.number;
    } else {
        const VJson *usage = vjson_obj_get(j, "usage");
        if (usage) {
            const VJson *mj = vjson_obj_get(usage, "prompt_tokens");
            if (mj && mj->type == VJ_NUMBER) in_tok = (int)mj->u.number;
            mj = vjson_obj_get(usage, "n_tokens");
            if (mj && mj->type == VJ_NUMBER) out_tok = (int)mj->u.number;
        }
    }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "id", vjson_new_string("msg_vllm_kestrel"));
    vjson_obj_set(root, "type", vjson_new_string("message"));
    vjson_obj_set(root, "role", vjson_new_string("assistant"));
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    VJson *content = vjson_new_array();
    VJson *block = vjson_new_object();
    vjson_obj_set(block, "type", vjson_new_string("text"));
    vjson_obj_set(block, "text", vjson_new_string(text));
    vjson_array_push(content, block);
    vjson_obj_set(root, "content", content);
    vjson_obj_set(root, "stop_reason", vjson_new_string(stop_reason));
    vjson_obj_set(root, "stop_sequence", vjson_new_null());
    VJson *u = vjson_new_object();
    vjson_obj_set(u, "input_tokens", vjson_new_number(in_tok));
    vjson_obj_set(u, "output_tokens", vjson_new_number(out_tok));
    vjson_obj_set(root, "usage", u);
    char *buf = (char *)malloc(1 << 20);
    if (!buf) { vjson_free(root); vjson_free(j); return NULL; }
    size_t n = vjson_serialize(root, buf, 1 << 20);
    vjson_free(root);
    vjson_free(j);
    buf[n] = '\0';
    return buf;
}

/* Anthropic /v1/messages -> internal OpenAI chat core -> Anthropic response.
 * Non-streaming only for now; system/messages text parts are supported. */
static void handle_anthropic(VLLMServerCtx *ctx, const VHttpRequest *req,
                             VHttpResponse *resp, VHttpConn *conn) {
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }

    const VJson *messages = vjson_obj_get(body, "messages");
    if (!messages || messages->type != VJ_ARRAY || vjson_array_len(messages) == 0) {
        json_error(resp, 400, "'messages' array is required");
        vjson_free(body); return;
    }
    VJson *jv;
    double temperature = 0.7, top_p = 0.9;
    int max_tokens = 0, stream = 0;
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER)
        max_tokens = (int)jv->u.number;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER)
        temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER)
        top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "stream")) && jv->type == VJ_BOOL)
        stream = jv->u.boolean;
    if (stream) {
        json_error(resp, 501, "Anthropic streaming not supported yet (use stream=false)");
        vjson_free(body); return;
    }
    if (max_tokens <= 0) {
        json_error(resp, 400, "'max_tokens' is required");
        vjson_free(body); return;
    }

    /* Convert to an OpenAI-format body so the existing chat core runs it. */
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    VJson *omsg = vjson_new_array();
    const VJson *sys = vjson_obj_get(body, "system");
    if (sys) {
        StrBuf sb; sb_init(&sb);
        anthropic_content_to_text(sys, &sb);
        if (sb.s && sb.s[0]) {
            VJson *m = vjson_new_object();
            vjson_obj_set(m, "role", vjson_new_string("system"));
            vjson_obj_set(m, "content", vjson_new_string(sb.s));
            vjson_array_push(omsg, m);
        }
        sb_free(&sb);
    }
    size_t n = vjson_array_len(messages);
    for (size_t i = 0; i < n; i++) {
        const VJson *msg = vjson_array_get(messages, i);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) continue;
        StrBuf sb; sb_init(&sb);
        anthropic_content_to_text(content, &sb);
        VJson *m = vjson_new_object();
        vjson_obj_set(m, "role", vjson_new_string(role));
        vjson_obj_set(m, "content", vjson_new_string(sb.s ? sb.s : ""));
        vjson_array_push(omsg, m);
        sb_free(&sb);
    }
    vjson_obj_set(o, "messages", omsg);
    vjson_obj_set(o, "max_tokens", vjson_new_number(max_tokens));
    vjson_obj_set(o, "temperature", vjson_new_number(temperature));
    vjson_obj_set(o, "top_p", vjson_new_number(top_p));
    vjson_obj_set(o, "stream", vjson_new_bool(0));

    char *obuf = (char *)malloc(4 << 20);
    if (!obuf) { json_error(resp, 500, "Out of memory"); vjson_free(o); vjson_free(body); return; }
    size_t olen = vjson_serialize(o, obuf, 4 << 20);
    vjson_free(o);
    if (olen == 0) {
        json_error(resp, 500, "body convert failed");
        free(obuf); vjson_free(body); return;
    }

    VHttpRequest fake;
    memset(&fake, 0, sizeof(fake));
    fake.method = "POST";
    fake.path = "/v1/chat/completions";
    fake.body = obuf;
    fake.body_len = olen;
    handle_chat(ctx, &fake, resp, conn);
    free(obuf);
    vjson_free(body);

    /* Translate the OpenAI response into the Anthropic message format. */
    if (resp->status == 200 && resp->body && resp->body_owned) {
        char *conv = anthropic_from_openai(ctx, resp->body);
        if (conv) {
            free((void *)resp->body);
            resp->body = conv;
            resp->body_owned = conv;   /* 必须同步：http 层发送后 free(body_owned) */
            resp->body_len = strlen(conv);
        }
    }
}

/* ---------- /health ---------- */

static void handle_health(VLLMServerCtx *ctx, VHttpResponse *resp) {
    char *body = (char *)malloc(1024);
    if (!body) { json_error(resp, 500, "Out of memory"); return; }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "status", vjson_new_string("ok"));
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    vhttp_mutex_lock(ctx->stat_lock);
    vjson_obj_set(root, "busy", vjson_new_bool(ctx->busy));
    vjson_obj_set(root, "queued", vjson_new_number((double)ctx->queued_requests));
    vjson_obj_set(root, "active", vjson_new_number((double)ctx->active_requests));
    vjson_obj_set(root, "total_requests", vjson_new_number((double)ctx->total_requests));
    vhttp_mutex_unlock(ctx->stat_lock);
    /* Model may not be loaded yet (manual-load mode): cfg is NULL then. */
    if (ctx->cfg)
        vjson_obj_set(root, "max_seq", vjson_new_number((double)ctx->cfg->max_seq_len));
    else
        vjson_obj_set(root, "max_seq", vjson_new_number(0));
    vjson_obj_set(root, "max_queued", vjson_new_number((double)ctx->max_queued));
    size_t n = vjson_serialize(root, body, 1024);
    vjson_free(root);
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body = body;
    resp->body_len = n;
    resp->body_owned = body;
}

/* ---------- /v1/attest (可验证推理的公开参数) ----------
 * 下发设备公钥 + 算法/schema/SM2 用户 ID，供验证方（对话页的浏览器内自验、
 * tools/verify_attest.py）离线复算与验签；不含任何机密（私钥永不离开设备）。*/
static void handle_attest_info(VLLMServerCtx *ctx, VHttpResponse *resp) {
    char *body = (char *)malloc(1024);
    if (!body) { json_error(resp, 500, "Out of memory"); return; }
    char pub[129];
    int have_pub = vatt_pub_hex(pub, sizeof(pub));
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "attest", vjson_new_bool(ctx->attest_on ? 1 : 0));
    vjson_obj_set(root, "active", vjson_new_bool(vatt_active() ? 1 : 0));
    vjson_obj_set(root, "pub", vjson_new_string(have_pub ? pub : ""));
    vjson_obj_set(root, "algo", vjson_new_string(VATT_ALGO));
    vjson_obj_set(root, "schema", vjson_new_number((double)VATT_SCHEMA));
    vjson_obj_set(root, "user_id", vjson_new_string(VATT_USER_ID));
    vjson_obj_set(root, "dir",
                  vjson_new_string(ctx->attest_dir[0] ? ctx->attest_dir : "."));
    size_t n = vjson_serialize(root, body, 1024);
    vjson_free(root);
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body = body;
    resp->body_len = n;
    resp->body_owned = body;
}

/* ---------- /chat/ (inference client page) ---------- */

static const char *chat_html_path(void) {
    const char *e = getenv("VLLM_CHAT_HTML");
    if (e && e[0]) return e;
    FILE *f = fopen("./chat.html", "rb");
    if (f) { fclose(f); return "./chat.html"; }
    return "/NewVLLM/chat.html";
}

static void handle_chat_page(VHttpResponse *resp) {
    /* Embedded chat.html (self-contained exe): used when no file is deployed
     * next to the engine. VLLM_CHAT_HTML overrides to a custom file. */
    size_t elen = 0;
    const char *ehtml = embedded_chat_html(&elen);
    FILE *f = fopen(chat_html_path(), "rb");
    if (!f) {
        if (ehtml && elen) {
            resp->status = 200;
            resp->content_type = "text/html; charset=utf-8";
            resp->body = ehtml;
            resp->body_len = elen;
            return;
        }
        resp->status = 500;
        resp->content_type = "text/html; charset=utf-8";
        const char *msg =
            "<html><body><h3>chat.html not found</h3>"
            "<p>Deploy chat.html next to the engine (e.g. /NewVLLM/chat.html) "
            "or set VLLM_CHAT_HTML.</p></body></html>";
        resp->body = msg;
        resp->body_len = strlen(msg);
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 8 * 1024 * 1024) {
        fclose(f);
        resp->status = 500;
        resp->body = "chat.html too large";
        return;
    }
    char *html = (char *)malloc((size_t)sz + 1);
    if (!html) {
        fclose(f);
        resp->status = 500;
        resp->body = "out of memory";
        return;
    }
    size_t rd = fread(html, 1, (size_t)sz, f);
    fclose(f);
    html[rd] = '\0';
    resp->status = 200;
    resp->content_type = "text/html; charset=utf-8";
    resp->body_owned = html;
    resp->body = html;
    resp->body_len = rd;
}

/* ---------- POST /v1/systemone (K-LAYA-001) ----------
 * Independent System-1 decision path (ModernBERT encoder + decision head),
 * decoupled from the causal-LM engine. The ~1.7 GB F32 arena is loaded once
 * from ctx->laya_pack on the first request and reused; requests are serialized
 * on a dedicated lock (the runner holds a single reusable state).
 * Body  : {"state":..., "questions":{qid:{type,instructions,criteria}}}
 * Reply : {"model":"laya","answers":{...}}
 * Disabled (404) when the server was started without --laya-pack. */
static LayaRunner *g_laya_runner = NULL;
static pthread_mutex_t g_laya_lock = PTHREAD_MUTEX_INITIALIZER;

static void handle_systemone(VLLMServerCtx *ctx, const VHttpRequest *req,
                             VHttpResponse *resp) {
    if (!ctx->laya_pack) {
        static const char e[] =
            "{\"error\":{\"message\":\"systemone disabled: start with --laya-pack\","
            "\"type\":\"invalid_request_error\",\"code\":404}}";
        resp->status = 404; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    if (!req->body || req->body_len == 0) {
        static const char e[] =
            "{\"error\":{\"message\":\"empty body\",\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    const size_t cap = 1 << 20;
    char *out = (char *)malloc(cap);
    if (!out) {
        static const char e[] =
            "{\"error\":{\"message\":\"out of memory\",\"type\":\"server_error\",\"code\":500}}";
        resp->status = 500; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    pthread_mutex_lock(&g_laya_lock);
    if (!g_laya_runner) {
        g_laya_runner = laya_runner_new(ctx->laya_pack);
        if (g_laya_runner)
            fprintf(stderr, "[LAYA] systemone runner ready (%s)\n", ctx->laya_pack);
    }
    int rc = g_laya_runner
                 ? laya_runner_decide(g_laya_runner, req->body, out, cap)
                 : -1;
    pthread_mutex_unlock(&g_laya_lock);
    if (rc != 0) {
        free(out);
        static const char e[] =
            "{\"error\":{\"message\":\"decide failed (bad request or load error)\","
            "\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body_owned = out;
    resp->body = out;
    resp->body_len = 0;   /* 0 = strlen(body) */
}

/* ---------- POST /v1/detect (K-YOLO-001) ----------
 * Independent detection path (own-format VQF v4 + CPU/NPU graph executor),
 * decoupled from the causal-LM engine. Body: image bytes (Content-Type
 * image/*) OR JSON with a base64 data URL ("...base64,<payload>...").
 * Reply : {"model":"yolo","detections":[{cls,name,score,box:[x1,y1,x2,y2]}]}
 * Disabled (404) when the server was started without --yolo-model. */
static StYoloModel *g_yolo = NULL;
static pthread_mutex_t g_yolo_lock = PTHREAD_MUTEX_INITIALIZER;

/* K-YOLO-003：按 --yolo-backend 选择加载器（0=VQF v4 默认，1=官方 .rknn opt-in）。
 * 返回值 NULL = 未启用/加载失败（各 handler 自行显式报错，绝不静默降级）。 */
static StYoloModel *yolo_open_model(VLLMServerCtx *ctx) {
    if (!ctx->yolo_model) return NULL;
    if (ctx->yolo_backend == 1)
        return st_yolo_load_rknn(ctx->yolo_model, ctx->yolo_npu, ctx->yolo_core_mask);
    return st_yolo_load(ctx->yolo_model);
}

static void handle_detect(VLLMServerCtx *ctx, const VHttpRequest *req,
                          VHttpResponse *resp) {
    if (!ctx->yolo_model) {
        static const char e[] =
            "{\"error\":{\"message\":\"detect disabled: start with --yolo-model\","
            "\"type\":\"invalid_request_error\",\"code\":404}}";
        resp->status = 404; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    if (!req->body || req->body_len == 0) {
        static const char e[] =
            "{\"error\":{\"message\":\"empty body\",\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    /* 取编码图像：JSON data URL 的 base64 段，或整体二进制（image/*） */
    uint8_t *enc = NULL; size_t enc_len = 0;
    const char *mark = (req->body[0] == '{' || req->body[0] == '\n')
                       ? strstr(req->body, "base64,") : NULL;
    if (mark) {
        const char *b = mark + 7;
        const char *q = strchr(b, '"');
        size_t n = q ? (size_t)(q - b) : strlen(b);
        char *s = (char *)malloc(n + 1);
        if (!s) goto oom;
        memcpy(s, b, n); s[n] = 0;
        enc = media_base64_decode(s, &enc_len);
        free(s);
    } else {
        enc = (uint8_t *)malloc(req->body_len);
        if (!enc) goto oom;
        memcpy(enc, req->body, req->body_len);
        enc_len = req->body_len;
    }
    if (!enc || enc_len == 0) {
        free(enc);
        static const char e[] =
            "{\"error\":{\"message\":\"no image payload (send image bytes or base64 data URL)\","
            "\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    int w = 0, h = 0;
    uint8_t *rgb = media_load_image_memory(enc, enc_len, &w, &h);
    free(enc);
    if (!rgb) {
        static const char e[] =
            "{\"error\":{\"message\":\"image decode failed\",\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }

    pthread_mutex_lock(&g_yolo_lock);
    if (!g_yolo) {
        g_yolo = yolo_open_model(ctx);
        if (g_yolo) fprintf(stderr, "[YOLO] detect route ready (%s)\n", ctx->yolo_model);
    }
    int n = -1;
    YoloDet dets[300];
    if (g_yolo) n = st_yolo_detect(g_yolo, rgb, w, h, 0.f, 0.f, dets, 300);
    pthread_mutex_unlock(&g_yolo_lock);
    media_free_image(rgb);
    if (n < 0) {
        static const char e[] =
            "{\"error\":{\"message\":\"detect failed\",\"type\":\"server_error\",\"code\":500}}";
        resp->status = 500; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    size_t cap = 256 + (size_t)n * 160;
    char *out = (char *)malloc(cap);
    if (!out) goto oom;
    size_t o = (size_t)snprintf(out, cap,
        "{\"model\":\"yolo\",\"width\":%d,\"height\":%d,\"detections\":[", w, h);
    for (int i = 0; i < n; i++) {
        char cn[64];
        if (st_yolo_class_name(g_yolo, dets[i].cls, cn, sizeof(cn)) != 0) snprintf(cn, sizeof(cn), "?");
        o += (size_t)snprintf(out + o, cap - o,
            "%s{\"cls\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
            i ? "," : "", dets[i].cls, cn, dets[i].score,
            dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }
    o += (size_t)snprintf(out + o, cap - o, "]}");
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body_owned = out;
    resp->body = out;
    resp->body_len = 0;
    return;
oom:
    {
        static const char e[] =
            "{\"error\":{\"message\":\"out of memory\",\"type\":\"server_error\",\"code\":500}}";
        resp->status = 500; resp->body = e; resp->body_len = sizeof(e) - 1;
    }
}

/* ---------- 场景叙事：YOLO 检测框 → 结构化文本（§2.1 组件2，文本级协作） ----------
 * 格式：`<class> at [x1,y1,x2,y2] conf <s>; ...`（无检测 → 明确占位）。
 * 返回堆分配 NUL 结尾串；失败 NULL。 */
static char *scene_narrative(StYoloModel *yolo, const YoloDet *dets, int n) {
    size_t cap = 128 + (size_t)(n > 0 ? n : 1) * 160;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    if (n <= 0) {
        snprintf(out, cap, "（图中未检测到物体）");
        return out;
    }
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        char cn[64];
        if (st_yolo_class_name(yolo, dets[i].cls, cn, sizeof(cn)) != 0)
            snprintf(cn, sizeof(cn), "class_%d", dets[i].cls);
        o += (size_t)snprintf(out + o, cap - o,
            "%s%s at [%.0f,%.0f,%.0f,%.0f] conf %.2f",
            i ? "; " : "", cn,
            dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2, dets[i].score);
    }
    return out;
}

/* ---------- POST /v1/scene (K-COLLAB-001 P3 感知接入) ----------
 * 场景理解链：图 → YOLO 检测 → 场景叙事 → 主 LLM 综合回答。
 * Body  : {"image":"<base64 或 data URL>", "query":"...", "max_tokens":M}
 *         （非 JSON 体按原始图像字节处理，query 缺省"请描述图中场景"）
 * Reply : {"model":...,"detections":[{cls,name,score,box}], "scene":"...", "answer":"..."}
 * 依赖  : --yolo-model（感知）+ 主 chat 模型（深度思考）；任一未启用/失败显式报错。 */
static void handle_scene(VLLMServerCtx *ctx, const VHttpRequest *req,
                         VHttpResponse *resp) {
    if (!ctx->yolo_model) {
        static const char e[] =
            "{\"error\":{\"message\":\"scene disabled: start with --yolo-model\","
            "\"type\":\"invalid_request_error\",\"code\":404}}";
        resp->status = 404; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }
    if (!req->body || req->body_len == 0) {
        static const char e[] =
            "{\"error\":{\"message\":\"empty body\",\"type\":\"invalid_request_error\",\"code\":400}}";
        resp->status = 400; resp->body = e; resp->body_len = sizeof(e) - 1;
        return;
    }

    /* 解析请求体：JSON（image + query）或原始图像字节 */
    const char *query = NULL;
    const char *img_b64 = NULL;
    VJson *body = NULL;
    if (req->body[0] == '{') {
        body = vjson_parse(req->body);
        if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }
        img_b64 = vjson_str(vjson_obj_get(body, "image"));
        query = vjson_str(vjson_obj_get(body, "query"));
        if (!img_b64 || !img_b64[0]) {
            json_error(resp, 400, "'image' is required (base64 or data URL)");
            vjson_free(body); return;
        }
    }
    if (!query || !query[0]) query = "请描述图中场景";

    /* 提取 base64 段（可选 data URL 前缀 "base64,"） */
    uint8_t *enc = NULL; size_t enc_len = 0;
    if (img_b64) {
        const char *mark = strstr(img_b64, "base64,");
        const char *b = mark ? mark + 7 : img_b64;
        size_t bn = strlen(b);
        char *s = (char *)malloc(bn + 1);
        if (!s) { vjson_free(body); json_error(resp, 500, "Out of memory"); return; }
        memcpy(s, b, bn); s[bn] = 0;
        enc = media_base64_decode(s, &enc_len);
        free(s);
    } else {
        enc = (uint8_t *)malloc(req->body_len);
        if (!enc) { json_error(resp, 500, "Out of memory"); return; }
        memcpy(enc, req->body, req->body_len);
        enc_len = req->body_len;
    }
    if (!enc || enc_len == 0) {
        free(enc); vjson_free(body);
        json_error(resp, 400, "no image payload");
        return;
    }

    int w = 0, h = 0;
    uint8_t *rgb = media_load_image_memory(enc, enc_len, &w, &h);
    free(enc);
    if (!rgb) {
        vjson_free(body);
        json_error(resp, 400, "image decode failed");
        return;
    }

    /* YOLO 检测 + 场景叙事（独立感知路径，g_yolo_lock 串行） */
    pthread_mutex_lock(&g_yolo_lock);
    if (!g_yolo) {
        g_yolo = yolo_open_model(ctx);
        if (g_yolo) fprintf(stderr, "[YOLO] scene route ready (%s)\n", ctx->yolo_model);
    }
    int n = -1;
    YoloDet dets[300];
    if (g_yolo) n = st_yolo_detect(g_yolo, rgb, w, h, 0.f, 0.f, dets, 300);
    char *scene = (n >= 0) ? scene_narrative(g_yolo, dets, n) : NULL;
    pthread_mutex_unlock(&g_yolo_lock);
    media_free_image(rgb);
    if (n < 0 || !scene) {
        free(scene); vjson_free(body);
        json_error(resp, 500, "detect failed");
        return;
    }

    /* 拼 prompt：场景叙事作为上下文，query 作为问题 */
    size_t plen = strlen(scene) + strlen(query) + 128;
    char *prompt = (char *)malloc(plen);
    if (!prompt) {
        free(scene); vjson_free(body); json_error(resp, 500, "Out of memory"); return;
    }
    snprintf(prompt, plen,
        "场景检测结果：%s\n\n问题：%s\n请仅根据检测结果回答问题。", scene, query);

    int max_tokens = 256;
    VJson *jv;
    if (body && (jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number;
        if (max_tokens <= 0) max_tokens = 1; if (max_tokens > 4096) max_tokens = 4096;
    }

    if (ensure_model_ready(ctx, resp) != 0) {
        free(scene); free(prompt); vjson_free(body); return;
    }
    int g = infer_gate(ctx);
    if (g != 0) {
        free(scene); free(prompt); vjson_free(body);
        if (g == -1) json_error(resp, 429, "Too many queued requests, retry later");
        else if (g == -2) json_error(resp, 503, "Insufficient memory, retry later");
        else json_error(resp, 503, "Model is unloading, retry later");
        return;
    }

    char *chat_prompt = build_completion_prompt(ctx, prompt, 0);
    free(prompt);
    StrBuf out; sb_init(&out);
    VLLMMetrics met; memset(&met, 0, sizeof(met));
    int rc = chat_prompt
                 ? run_completion(ctx, chat_prompt, NULL, NULL, 0, NULL, 0, NULL,
                                  0.0, 1.0, 0.0, 0, 0, max_tokens,
                                  0, NULL, 0, &out, &met, NULL, NULL, NULL, NULL)
                 : -1;
    free(chat_prompt);
    infer_done(ctx);
    if (rc != 0) {
        sb_free(&out); free(scene); vjson_free(body);
        json_error(resp, 500, rc == -2 ? "Prompt exceeds the context limit" : "Inference failed");
        return;
    }

    /* 构造响应 JSON */
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "scene"));
    VJson *arr = vjson_new_array();
    for (int i = 0; i < n; i++) {
        char cn[64];
        if (st_yolo_class_name(g_yolo, dets[i].cls, cn, sizeof(cn)) != 0)
            snprintf(cn, sizeof(cn), "class_%d", dets[i].cls);
        VJson *it = vjson_new_object();
        vjson_obj_set(it, "cls", vjson_new_number((double)dets[i].cls));
        vjson_obj_set(it, "name", vjson_new_string(cn));
        vjson_obj_set(it, "score", vjson_new_number((double)dets[i].score));
        VJson *box = vjson_new_array();
        vjson_array_push(box, vjson_new_number((double)dets[i].x1));
        vjson_array_push(box, vjson_new_number((double)dets[i].y1));
        vjson_array_push(box, vjson_new_number((double)dets[i].x2));
        vjson_array_push(box, vjson_new_number((double)dets[i].y2));
        vjson_obj_set(it, "box", box);
        vjson_array_push(arr, it);
    }
    vjson_obj_set(root, "detections", arr);
    vjson_obj_set(root, "scene", vjson_new_string(scene));
    vjson_obj_set(root, "answer", vjson_new_string(out.s ? out.s : ""));

    size_t cap = 4096 + strlen(out.s ? out.s : "") * 2 + strlen(scene) * 2
                 + (size_t)n * 160;
    char *rbody = (char *)malloc(cap);
    if (!rbody) {
        sb_free(&out); free(scene); vjson_free(root); vjson_free(body);
        json_error(resp, 500, "Out of memory"); return;
    }
    size_t rn = vjson_serialize(root, rbody, cap);
    vjson_free(root);
    sb_free(&out); free(scene); vjson_free(body);
    if (rn > 0) {
        resp->status = 200;
        resp->content_type = "application/json";
        resp->body = rbody; resp->body_len = rn; resp->body_owned = rbody;
    } else {
        free(rbody);
        json_error(resp, 500, "Response too large");
    }
}

/* ================================================================
 * 视频源检测（K-E2E-005）：MP4 文件 / 图像序列目录 / MJPEG·快照 URL / RTSP
 *
 * 两种形态：
 *   1) 单次抽帧 POST /v1/detect_video  —— 请求内抽 N 帧逐帧检测，返回逐帧框 + 汇总；
 *   2) 常驻拉流 POST /v1/stream/start|stop + GET /v1/stream/status
 *                                     —— 后台线程持续抽帧检测，只保留最近一帧结果。
 *
 * 边界（显式回报，不静默）：引擎自带 H.264 软解只支持 I 片 ⇒ MP4/RTSP 只能抽
 * **IDR 帧**，被跳过的非 IDR 在 summary.skipped_non_idr 与日志中回报。
 * ================================================================ */

typedef struct { int cls; char name[64]; float max_score; int frames; } VClsAgg;

/* 一帧 YOLO（复用全局 g_yolo + g_yolo_lock；与 /v1/detect 同一模型实例）。 */
static int yolo_detect_frame(VLLMServerCtx *ctx, const uint8_t *rgb, int w, int h,
                             YoloDet *dets, int max_out) {
    pthread_mutex_lock(&g_yolo_lock);
    if (!g_yolo) {
        g_yolo = yolo_open_model(ctx);
        if (g_yolo) fprintf(stderr, "[YOLO] video route ready (%s)\n", ctx->yolo_model);
    }
    int n = g_yolo ? st_yolo_detect(g_yolo, rgb, w, h, 0.f, 0.f, dets, max_out) : -1;
    pthread_mutex_unlock(&g_yolo_lock);
    return n;
}

static void agg_add(VClsAgg *agg, int *nagg, int cls, const char *name, float score) {
    for (int i = 0; i < *nagg; i++) {
        if (agg[i].cls == cls) {
            if (score > agg[i].max_score) agg[i].max_score = score;
            agg[i].frames++;
            return;
        }
    }
    if (*nagg >= 256) return;
    agg[*nagg].cls = cls;
    snprintf(agg[*nagg].name, sizeof(agg[*nagg].name), "%s", name);
    agg[*nagg].max_score = score;
    agg[*nagg].frames = 1;
    (*nagg)++;
}

static void vsrc_sleep_ms(int ms) {
    if (ms <= 0) return;
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* ---------- POST /v1/detect_video ----------
 * Body : {"source":"<file|dir|http://|rtsp://>", "max_frames":N, "timeout_ms":T,
 *         "interval_ms":I, "include_frames":true}
 *   source 缺省回落 --yolo-video-src（管理页配置的默认视频源）。
 * Reply: {source,kind,mode,live,width,height,frames:[{i,detections:[...]}],
 *         summary:{frames,detected_frames,skipped_non_idr,classes:[...],ms}} */
static void handle_detect_video(VLLMServerCtx *ctx, const VHttpRequest *req,
                                VHttpResponse *resp) {
    if (!ctx->yolo_model) {
        json_error(resp, 404, "video detect disabled: start with --yolo-model");
        return;
    }
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }
    const char *source = vjson_str(vjson_obj_get(body, "source"));
    if (!source || !source[0]) source = ctx->yolo_video_src;
    if (!source || !source[0]) {
        vjson_free(body);
        json_error(resp, 400, "'source' is required (or configure --yolo-video-src)");
        return;
    }
    /* 先拷一份：下面会 vjson_free(body)，source 可能指向 body 内部（UAF 防护）。 */
    char src_copy[600];
    snprintf(src_copy, sizeof(src_copy), "%s", source);
    source = src_copy;
    int max_frames = (ctx->yolo_video_frames > 0) ? ctx->yolo_video_frames : 8;
    int timeout_ms = 20000;
    int interval_ms = 0;
    int include_frames = 1;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "max_frames")) && jv->type == VJ_NUMBER) {
        max_frames = (int)jv->u.number;
        if (max_frames <= 0) max_frames = 1;
        if (max_frames > 300) max_frames = 300;
    }
    if ((jv = vjson_obj_get(body, "timeout_ms")) && jv->type == VJ_NUMBER) {
        timeout_ms = (int)jv->u.number;
        if (timeout_ms < 1000) timeout_ms = 1000;
        if (timeout_ms > 120000) timeout_ms = 120000;
    }
    if ((jv = vjson_obj_get(body, "interval_ms")) && jv->type == VJ_NUMBER) {
        interval_ms = (int)jv->u.number;
        if (interval_ms < 0) interval_ms = 0;
        if (interval_ms > 10000) interval_ms = 10000;
    }
    if ((jv = vjson_obj_get(body, "include_frames")) && jv->type == VJ_BOOL)
        include_frames = jv->u.boolean ? 1 : 0;

    char err[256] = "";
    VVideoSrc *vs = vsrc_open(source, err, sizeof(err));
    if (!vs) {
        char msg[320];
        snprintf(msg, sizeof(msg), "video source open failed: %s", err);
        fprintf(stderr, "[VDETECT] open failed source=%s err=%s\n", source, err);
        fflush(stderr);
        vjson_free(body);
        json_error(resp, 400, msg);
        return;
    }
    const int live = vsrc_live(vs);
    fprintf(stderr, "[VDETECT] open source=%s kind=%s mode=%s live=%d\n",
            source, vsrc_kind_name(vs), vsrc_mode(vs), live);
    fflush(stderr);

    double t0 = st_now_sec();
    VJson *frames = vjson_new_array();
    VClsAgg agg[256]; int nagg = 0;
    int produced = 0, det_frames = 0, first_w = 0, first_h = 0;
    int stop_reason = 0;   /* 0=达到帧数上限 1=源结束/超时 2=总超时 */
    while (produced < max_frames) {
        if ((st_now_sec() - t0) * 1000.0 > (double)timeout_ms) { stop_reason = 2; break; }
        uint8_t *rgb = NULL; int w = 0, h = 0;
        int rc = vsrc_next(vs, &rgb, &w, &h);
        if (rc < 0) {
            fprintf(stderr, "[VDETECT] decode error (produced=%d)\n", produced);
            fflush(stderr);
            break;
        }
        if (rc == 0) {
            /* live 每轮重读：HTTP 源首帧后才知道是不是 MJPEG（见 vsrc_live 注释） */
            if (vsrc_live(vs) && (st_now_sec() - t0) * 1000.0 < (double)timeout_ms) {
                vsrc_sleep_ms(interval_ms > 0 ? interval_ms : 100);
                continue;                       /* 直播源：等下一帧 */
            }
            stop_reason = 1;                    /* 有限源结束（或直播超时） */
            break;
        }
        if (first_w == 0) { first_w = w; first_h = h; }
        YoloDet dets[300];
        int n = yolo_detect_frame(ctx, rgb, w, h, dets, 300);
        media_free_image(rgb);
        if (n < 0) {
            fprintf(stderr, "[VDETECT] detect failed at frame %d\n", produced);
            fflush(stderr);
            break;
        }
        if (n > 0) det_frames++;
        for (int i = 0; i < n; i++) {
            char cn[64];
            if (st_yolo_class_name(g_yolo, dets[i].cls, cn, sizeof(cn)) != 0)
                snprintf(cn, sizeof(cn), "class_%d", dets[i].cls);
            agg_add(agg, &nagg, dets[i].cls, cn, dets[i].score);
        }
        if (include_frames) {
            VJson *fr = vjson_new_object();
            vjson_obj_set(fr, "i", vjson_new_number((double)produced));
            VJson *da = vjson_new_array();
            for (int i = 0; i < n; i++) {
                char cn[64];
                if (st_yolo_class_name(g_yolo, dets[i].cls, cn, sizeof(cn)) != 0)
                    snprintf(cn, sizeof(cn), "class_%d", dets[i].cls);
                VJson *it = vjson_new_object();
                vjson_obj_set(it, "cls", vjson_new_number((double)dets[i].cls));
                vjson_obj_set(it, "name", vjson_new_string(cn));
                vjson_obj_set(it, "score", vjson_new_number((double)dets[i].score));
                VJson *box = vjson_new_array();
                vjson_array_push(box, vjson_new_number((double)dets[i].x1));
                vjson_array_push(box, vjson_new_number((double)dets[i].y1));
                vjson_array_push(box, vjson_new_number((double)dets[i].x2));
                vjson_array_push(box, vjson_new_number((double)dets[i].y2));
                vjson_obj_set(it, "box", box);
                vjson_array_push(da, it);
            }
            vjson_obj_set(fr, "detections", da);
            vjson_array_push(frames, fr);
        }
        produced++;
        if (interval_ms > 0 && produced < max_frames) vsrc_sleep_ms(interval_ms);
    }
    double ms = (st_now_sec() - t0) * 1000.0;
    int skipped = vsrc_skipped(vs);
    int dec_failed = vsrc_decode_failed(vs);
    char kind[16], mode[48];
    snprintf(kind, sizeof(kind), "%s", vsrc_kind_name(vs));
    snprintf(mode, sizeof(mode), "%s", vsrc_mode(vs));
    int total_hint = vsrc_total_hint(vs);
    int live_final = vsrc_live(vs);     /* 首帧后重读（HTTP 源此时才知是否 MJPEG） */
    vsrc_close(vs);
    vjson_free(body);

    VJson *root = vjson_new_object();
    vjson_obj_set(root, "model", vjson_new_string("yolo"));
    vjson_obj_set(root, "source", vjson_new_string(source));
    vjson_obj_set(root, "kind", vjson_new_string(kind));
    vjson_obj_set(root, "mode", vjson_new_string(mode));
    vjson_obj_set(root, "live", vjson_new_bool(live_final));
    vjson_obj_set(root, "width", vjson_new_number((double)first_w));
    vjson_obj_set(root, "height", vjson_new_number((double)first_h));
    vjson_obj_set(root, "frames", frames);
    VJson *sum = vjson_new_object();
    vjson_obj_set(sum, "frames", vjson_new_number((double)produced));
    vjson_obj_set(sum, "detected_frames", vjson_new_number((double)det_frames));
    vjson_obj_set(sum, "skipped_non_idr", vjson_new_number((double)skipped));
    vjson_obj_set(sum, "decoded_failed", vjson_new_number((double)dec_failed));
    if (total_hint >= 0) vjson_obj_set(sum, "total_hint", vjson_new_number((double)total_hint));
    vjson_obj_set(sum, "stop", vjson_new_string(
        stop_reason == 2 ? "timeout" : stop_reason == 1 ? "source_end" : "frame_budget"));
    vjson_obj_set(sum, "ms", vjson_new_number(ms));
    VJson *cls = vjson_new_array();
    for (int i = 0; i < nagg; i++) {
        VJson *c = vjson_new_object();
        vjson_obj_set(c, "cls", vjson_new_number((double)agg[i].cls));
        vjson_obj_set(c, "name", vjson_new_string(agg[i].name));
        vjson_obj_set(c, "max_score", vjson_new_number((double)agg[i].max_score));
        vjson_obj_set(c, "frames", vjson_new_number((double)agg[i].frames));
        vjson_array_push(cls, c);
    }
    vjson_obj_set(sum, "classes", cls);
    vjson_obj_set(root, "summary", sum);
    fprintf(stderr, "[VDETECT] done source=%s frames=%d detected=%d skipped_non_idr=%d decoded_failed=%d stop=%d %.1fms\n",
            source, produced, det_frames, skipped, dec_failed, stop_reason, ms);
    fflush(stderr);

    size_t cap = 65536 + (size_t)produced * 8192 + strlen(source) * 2;
    char *rb = (char *)malloc(cap);
    if (!rb) { vjson_free(root); json_error(resp, 500, "Out of memory"); return; }
    size_t rn = vjson_serialize(root, rb, cap);
    vjson_free(root);
    if (rn > 0) {
        resp->status = 200;
        resp->content_type = "application/json";
        resp->body = rb; resp->body_len = rn; resp->body_owned = rb;
    } else {
        free(rb);
        json_error(resp, 500, "Response too large");
    }
}

/* ---------- 常驻拉流：后台线程持续抽帧检测 ---------- */
typedef struct {
    int             running;
    volatile int    stop;
    pthread_t       thr;
    pthread_mutex_t lock;
    VLLMServerCtx  *ctx;
    char            source[600];
    char            kind[16];
    char            mode[48];
    int             interval_ms;
    int             max_frames;     /* 0 = 不限 */
    int             frames, det_frames, skipped, dec_failed;
    int             width, height;
    double          started;
    char            err[256];
    char            last[8192];     /* 最近一帧的 detections JSON 片段 */
    int             last_n;
    int             last_i;
} VVideoStream;

static VVideoStream g_vstream;
static pthread_mutex_t g_vstream_spawn_lock = PTHREAD_MUTEX_INITIALIZER;

static void *vstream_thread(void *arg) {
    VVideoStream *v = (VVideoStream *)arg;
    char err[256] = "";
    VVideoSrc *vs = vsrc_open(v->source, err, sizeof(err));
    if (!vs) {
        pthread_mutex_lock(&v->lock);
        snprintf(v->err, sizeof(v->err), "open failed: %s", err);
        v->running = 0;
        pthread_mutex_unlock(&v->lock);
        fprintf(stderr, "[VSTREAM] open failed source=%s err=%s\n", v->source, err);
        fflush(stderr);
        return NULL;
    }
    pthread_mutex_lock(&v->lock);
    snprintf(v->kind, sizeof(v->kind), "%s", vsrc_kind_name(vs));
    snprintf(v->mode, sizeof(v->mode), "%s", vsrc_mode(vs));
    pthread_mutex_unlock(&v->lock);
    fprintf(stderr, "[VSTREAM] started source=%s kind=%s mode=%s interval=%dms\n",
            v->source, vsrc_kind_name(vs), vsrc_mode(vs), v->interval_ms);
    fflush(stderr);

    YoloDet dets[300];
    while (!v->stop) {
        uint8_t *rgb = NULL; int w = 0, h = 0;
        int rc = vsrc_next(vs, &rgb, &w, &h);
        if (rc < 0) {
            pthread_mutex_lock(&v->lock);
            snprintf(v->err, sizeof(v->err), "decode/transport error after %d frames", v->frames);
            v->running = 0;
            pthread_mutex_unlock(&v->lock);
            break;
        }
        if (rc == 0) {
            /* live 每轮重读（HTTP 源首帧后才知道是不是 MJPEG） */
            if (vsrc_live(vs)) { vsrc_sleep_ms(v->interval_ms > 0 ? v->interval_ms : 200); continue; }
            pthread_mutex_lock(&v->lock);
            snprintf(v->err, sizeof(v->err), "source end (finite source)");
            v->running = 0;
            pthread_mutex_unlock(&v->lock);
            break;
        }
        int n = yolo_detect_frame(v->ctx, rgb, w, h, dets, 300);
        media_free_image(rgb);
        if (n < 0) {
            pthread_mutex_lock(&v->lock);
            snprintf(v->err, sizeof(v->err), "detect failed at frame %d", v->frames);
            v->running = 0;
            pthread_mutex_unlock(&v->lock);
            break;
        }
        char frag[8192];
        size_t o = (size_t)snprintf(frag, sizeof(frag), "[");
        for (int i = 0; i < n && o + 200 < sizeof(frag); i++) {
            char cn[64];
            if (st_yolo_class_name(g_yolo, dets[i].cls, cn, sizeof(cn)) != 0)
                snprintf(cn, sizeof(cn), "class_%d", dets[i].cls);
            o += (size_t)snprintf(frag + o, sizeof(frag) - o,
                "%s{\"cls\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                i ? "," : "", dets[i].cls, cn, dets[i].score,
                dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
        }
        snprintf(frag + o, sizeof(frag) - o, "]");
        pthread_mutex_lock(&v->lock);
        v->frames++;
        if (n > 0) v->det_frames++;
        v->width = w; v->height = h;
        v->last_n = n;
        v->last_i = v->frames - 1;
        /* 刷新 kind/mode：HTTP 源首帧后才确定是 MJPEG 还是单帧快照 */
        snprintf(v->kind, sizeof(v->kind), "%s", vsrc_kind_name(vs));
        snprintf(v->mode, sizeof(v->mode), "%s", vsrc_mode(vs));
        snprintf(v->last, sizeof(v->last), "%s", frag);
        if (v->max_frames > 0 && v->frames >= v->max_frames) {
            snprintf(v->err, sizeof(v->err), "frame budget reached (%d)", v->max_frames);
            v->running = 0;
            pthread_mutex_unlock(&v->lock);
            break;
        }
        pthread_mutex_unlock(&v->lock);
        vsrc_sleep_ms(v->interval_ms > 0 ? v->interval_ms : 200);
    }
    pthread_mutex_lock(&v->lock);
    v->skipped = vsrc_skipped(vs);
    v->dec_failed = vsrc_decode_failed(vs);
    pthread_mutex_unlock(&v->lock);
    vsrc_close(vs);
    fprintf(stderr, "[VSTREAM] stopped source=%s frames=%d det_frames=%d skipped_non_idr=%d decoded_failed=%d\n",
            v->source, v->frames, v->det_frames, v->skipped, v->dec_failed);
    fflush(stderr);
    return NULL;
}

/* POST /v1/stream/start  {source?, interval_ms?, max_frames?} */
static void handle_stream_start(VLLMServerCtx *ctx, const VHttpRequest *req,
                                VHttpResponse *resp) {
    if (!ctx->yolo_model) { json_error(resp, 404, "disabled: start with --yolo-model"); return; }
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }
    const char *source = vjson_str(vjson_obj_get(body, "source"));
    if (!source || !source[0]) source = ctx->yolo_video_src;
    if (!source || !source[0]) {
        vjson_free(body);
        json_error(resp, 400, "'source' is required (or configure --yolo-video-src)");
        return;
    }
    int interval_ms = (ctx->yolo_video_interval_ms > 0) ? ctx->yolo_video_interval_ms : 500;
    int max_frames = 0;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "interval_ms")) && jv->type == VJ_NUMBER) {
        interval_ms = (int)jv->u.number;
        if (interval_ms < 0) interval_ms = 0;
        if (interval_ms > 60000) interval_ms = 60000;
    }
    if ((jv = vjson_obj_get(body, "max_frames")) && jv->type == VJ_NUMBER) {
        max_frames = (int)jv->u.number;
        if (max_frames < 0) max_frames = 0;
        if (max_frames > 1000000) max_frames = 1000000;
    }

    pthread_mutex_lock(&g_vstream_spawn_lock);
    if (g_vstream.running) {
        pthread_mutex_unlock(&g_vstream_spawn_lock);
        vjson_free(body);
        json_error(resp, 409, "a video stream is already running (POST /v1/stream/stop first)");
        return;
    }
    memset(&g_vstream, 0, sizeof(g_vstream));
    pthread_mutex_init(&g_vstream.lock, NULL);
    snprintf(g_vstream.source, sizeof(g_vstream.source), "%s", source);
    g_vstream.ctx = ctx;
    g_vstream.interval_ms = interval_ms;
    g_vstream.max_frames = max_frames;
    g_vstream.started = st_now_sec();
    g_vstream.running = 1;
    g_vstream.stop = 0;
    int prc = pthread_create(&g_vstream.thr, NULL, vstream_thread, &g_vstream);
    if (prc != 0) {
        g_vstream.running = 0;
        pthread_mutex_unlock(&g_vstream_spawn_lock);
        vjson_free(body);
        json_error(resp, 500, "failed to start stream thread");
        return;
    }
    pthread_mutex_unlock(&g_vstream_spawn_lock);
    vjson_free(body);

    char rb[512];
    int n = snprintf(rb, sizeof(rb),
        "{\"started\":true,\"source\":\"%s\",\"interval_ms\":%d,\"max_frames\":%d}",
        g_vstream.source, interval_ms, max_frames);
    char *out = (char *)malloc((size_t)n + 1);
    if (!out) { json_error(resp, 500, "Out of memory"); return; }
    memcpy(out, rb, (size_t)n + 1);
    resp->status = 200; resp->content_type = "application/json";
    resp->body = out; resp->body_len = (size_t)n; resp->body_owned = out;
}

/* POST /v1/stream/stop */
static void handle_stream_stop(VLLMServerCtx *ctx, const VHttpRequest *req,
                               VHttpResponse *resp) {
    (void)ctx; (void)req;
    pthread_mutex_lock(&g_vstream_spawn_lock);
    if (!g_vstream.running) {
        pthread_mutex_unlock(&g_vstream_spawn_lock);
        json_error(resp, 409, "no video stream is running");
        return;
    }
    g_vstream.stop = 1;
    pthread_join(g_vstream.thr, NULL);
    g_vstream.running = 0;
    char rb[256];
    int n = snprintf(rb, sizeof(rb), "{\"stopped\":true,\"frames\":%d}", g_vstream.frames);
    pthread_mutex_unlock(&g_vstream_spawn_lock);
    char *out = (char *)malloc((size_t)n + 1);
    if (!out) { json_error(resp, 500, "Out of memory"); return; }
    memcpy(out, rb, (size_t)n + 1);
    resp->status = 200; resp->content_type = "application/json";
    resp->body = out; resp->body_len = (size_t)n; resp->body_owned = out;
}

/* GET /v1/stream/status */
static void handle_stream_status(VLLMServerCtx *ctx, const VHttpRequest *req,
                                 VHttpResponse *resp) {
    (void)ctx; (void)req;
    pthread_mutex_lock(&g_vstream.lock);
    int running = g_vstream.running;
    char rb[12288];
    int n = snprintf(rb, sizeof(rb),
        "{\"running\":%s,\"source\":\"%s\",\"kind\":\"%s\",\"mode\":\"%s\","
        "\"interval_ms\":%d,\"frames\":%d,\"detected_frames\":%d,\"skipped_non_idr\":%d,"
        "\"decoded_failed\":%d,"
        "\"width\":%d,\"height\":%d,\"uptime_s\":%.1f,\"last_index\":%d,\"last_count\":%d,"
        "\"error\":\"%s\",\"last\":%s}",
        running ? "true" : "false", g_vstream.source, g_vstream.kind, g_vstream.mode,
        g_vstream.interval_ms, g_vstream.frames, g_vstream.det_frames, g_vstream.skipped,
        g_vstream.dec_failed,
        g_vstream.width, g_vstream.height,
        g_vstream.started > 0 ? st_now_sec() - g_vstream.started : 0.0,
        g_vstream.last_i, g_vstream.last_n,
        g_vstream.err, g_vstream.last[0] ? g_vstream.last : "[]");
    pthread_mutex_unlock(&g_vstream.lock);
    if (n <= 0 || n >= (int)sizeof(rb)) { json_error(resp, 500, "status too large"); return; }
    char *out = (char *)malloc((size_t)n + 1);
    if (!out) { json_error(resp, 500, "Out of memory"); return; }
    memcpy(out, rb, (size_t)n + 1);
    resp->status = 200; resp->content_type = "application/json";
    resp->body = out; resp->body_len = (size_t)n; resp->body_owned = out;
}

/* K-COLLAB-001 J5：协作链分段计时开关（env 门控，默认关）。 */
static int collab_timing_on(void) {
    const char *e = getenv("VLLM_COLLAB_TIMING");
    return e && e[0] && e[0] != '0';
}

/* ---- K-COLLAB-001 P1：单条文本 → last-token pooling 向量（复刻 handle_embeddings
 * 的裸文本编码路径）。st/cfg/tok/nl 由调用方选定（专用 embed 槽位或主模型）。
 * 返回 0 成功，-1 失败。调用方须已持 inf_lock（与推理串行）。 */
static int rag_embed_text(VLLMServerCtx *ctx, STQwenInferenceState *st,
                          STModelConfig *cfg, QwenTokenizer *tok, int nl,
                          const char *txt, float *vec) {
    (void)ctx;
    int dim = cfg->dim;
    int max_seq = cfg->max_seq_len;
    int *ids = (int *)malloc(((size_t)max_seq + 16) * sizeof(int));
    if (!ids) return -1;
    for (int l = 0; l < nl; l++) st->cache_len[l] = 0;
    st->seq_len = 0; st->mrope_pos = 0;
    st_qwen_kv_rebuild_freed(st);
    int n = qwen_tokenizer_encode(tok, txt, ids, max_seq - 1);
    if (n <= 0) { free(ids); return -1; }
    if (n > max_seq - 8) n = max_seq - 8;
    if (n < max_seq && ids[n-1] != tok->eos_id) ids[n++] = tok->eos_id;
    st_embed_request(vec, dim);
    int rc = st_qwen_model_prefill_batch(st, ids, n);
    free(ids);
    return rc;
}

/* K-ADMIN-001：管理面复用的「单条文本 → 向量」入口（与 /v1/rag 同路径）。
 * 不涉及 HTTP 响应对象；失败以负值返回，由管理面转成 JSON 错误。
 * 注意：不调用 ensure_model_ready(ctx, NULL)（其失败分支会写 resp）。 */
int vllm_serve_embed_text(VLLMServerCtx *ctx, const char *txt, float *out) {
    if (!ctx || !txt || !out) return -1;
    if (ctx->load_state != 2) {
        if (ctx->auto_load && ctx->load_state != 1) {
            if (vllm_serve_load_on_use(ctx) != 0) return -2;
        }
        if (ctx->load_state != 2) return -2;
    }
    int g = infer_gate(ctx);
    if (g != 0) return -3;

    STQwenInferenceState *st; STModelConfig *cfg; QwenTokenizer *tok; int nl;
    if (ctx->embed_model_dir) {
        pthread_mutex_lock(&ctx->embed_lock);
        if (ctx->embed_loaded == 0 && vllm_embed_load(ctx) != 0) {
            pthread_mutex_unlock(&ctx->embed_lock);
            infer_done(ctx);
            return -4;
        }
        pthread_mutex_unlock(&ctx->embed_lock);
        st = ctx->embed_ist; cfg = ctx->embed_cfg; tok = ctx->embed_tok;
        nl = ctx->embed_w->n_layers_allocated;
    } else {
        st = ctx->ist; cfg = ctx->cfg; tok = ctx->tok; nl = ctx->w->n_layers_allocated;
    }
    int rc = rag_embed_text(ctx, st, cfg, tok, nl, txt, out);
    infer_done(ctx);
    if (rc != 0) return -5;
    return cfg->dim;
}

/* ---------- 转人工回调（K-E2E-004） ----------
 * 拒答（证据不足）与路由 escalate 时，POST 一条 JSON 到 ctx->handoff_url。
 * 契约：① 绝不改变主响应（只做旁路通知）；② 成功/失败都打 [HANDOFF] 日志——
 * 失败不静默（对齐公理 silent_fallback_false_pass）；③ 未配置 = 完全不触发（零回归）。
 * 复用 vhttp_post_json（仅明文 http://，https 需前置 TLS 网关）。 */
static size_t json_escape_into(const char *s, char *out, size_t cap);

static void handoff_notify(VLLMServerCtx *ctx, const char *cause, const char *query,
                           double evidence, double threshold) {
    if (!ctx || !ctx->handoff_url || !ctx->handoff_url[0]) return;
    const char *q = query ? query : "";
    size_t ql = strlen(q) * 6 + 16;
    char *esc = (char *)malloc(ql);
    if (!esc) { fprintf(stderr, "[HANDOFF] skip: out of memory\n"); fflush(stderr); return; }
    json_escape_into(q, esc, ql);
    size_t blen = ql + 256;
    char *jb = (char *)malloc(blen);
    if (!jb) { free(esc); fprintf(stderr, "[HANDOFF] skip: out of memory\n"); fflush(stderr); return; }
    snprintf(jb, blen,
             "{\"cause\":\"%s\",\"query\":\"%s\",\"evidence\":%.4f,\"threshold\":%.4f,"
             "\"model\":\"%s\",\"route\":\"handoff_to_human\"}",
             cause ? cause : "unknown", esc, evidence, threshold,
             ctx->model_id ? ctx->model_id : "");
    free(esc);
    char *rb = NULL; int status = 0;
    int rc = vhttp_post_json(ctx->handoff_url, NULL, jb, strlen(jb), &rb, &status, NULL, 0);
    fprintf(stderr, "[HANDOFF] cause=%s url=%s rc=%d status=%d\n",
            cause ? cause : "unknown", ctx->handoff_url, rc, status);
    fflush(stderr);
    free(rb); free(jb);
}

/* ---------- POST /v1/rag (K-COLLAB-001 P1) ----------
 * 知识库检索增强问答：embed(query) → 向量库 top-k → 拼 prompt → 主模型回答。
 * Body  : {"query":"...","kb_dir":"<.vkb path>","top_k":N,"max_tokens":M}
 * Reply : {"retrieved":[{"index":i,"meta":...,"score":...}],"answer":"..."}
 * 依赖  : --embed-model（专用 embedding；缺省复用主模型）+ 主 chat 模型。 */
static void handle_rag(VLLMServerCtx *ctx, const VHttpRequest *req,
                       VHttpResponse *resp) {
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }
    const char *query = vjson_str(vjson_obj_get(body, "query"));
    const char *kb_dir = vjson_str(vjson_obj_get(body, "kb_dir"));
    if (!query || !query[0] || !kb_dir || !kb_dir[0]) {
        json_error(resp, 400, "'query' and 'kb_dir' are required");
        vjson_free(body); return;
    }
    int top_k = 5, max_tokens = 128;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "top_k")) && jv->type == VJ_NUMBER) {
        top_k = (int)jv->u.number; if (top_k <= 0) top_k = 1; if (top_k > 16) top_k = 16;
    }
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number; if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }

    int g = infer_gate(ctx);
    if (g == -1) { json_error(resp, 429, "Too many queued requests, retry later"); vjson_free(body); return; }
    if (g == -2) { json_error(resp, 503, "Insufficient memory, retry later"); vjson_free(body); return; }
    if (g == -3) { json_error(resp, 503, "Model is unloading, retry later"); vjson_free(body); return; }

    /* 回退透明（J4 + silent_fallback_false_pass）：RAG 检索依赖 embedding 质量，
     * 缺省 --embed-model 时复用主模型（4B 非 embedding 训练）会静默降级检索质量。
     * 显式回退纯 LLM（不拼检索上下文）+ 打日志。 */
    const int plain_fallback = (ctx->embed_model_dir == NULL);
    if (plain_fallback) {
        fprintf(stderr, "[RAG] no --embed-model, plain-LLM fallback (no retrieval)\n");
        fflush(stderr);
    }

    double t_c0 = st_now_sec(), t_embed = 0.0, t_search = 0.0;
    float *qvec = NULL;
    VLLMKVStore kb; vllm_kv_init(&kb, 0);
    int k = 0; int *idx = NULL; float *sc = NULL;

    if (!plain_fallback) {
        /* 1) 专用 embedding 模型出 query 向量 */
        pthread_mutex_lock(&ctx->embed_lock);
        if (ctx->embed_loaded == 0 && vllm_embed_load(ctx) != 0) {
            pthread_mutex_unlock(&ctx->embed_lock);
            infer_done(ctx); json_error(resp, 503, ctx->embed_error); vjson_free(body); return;
        }
        pthread_mutex_unlock(&ctx->embed_lock);
        STQwenInferenceState *st = ctx->embed_ist;
        STModelConfig *cfg = ctx->embed_cfg;
        QwenTokenizer *tok = ctx->embed_tok;
        int nl = ctx->embed_w->n_layers_allocated;
        int dim = cfg->dim;
        qvec = (float *)malloc((size_t)dim * sizeof(float));
        if (!qvec) { infer_done(ctx); json_error(resp, 500, "Out of memory"); vjson_free(body); return; }
        if (rag_embed_text(ctx, st, cfg, tok, nl, query, qvec) != 0) {
            infer_done(ctx); free(qvec); json_error(resp, 500, "Embedding failed"); vjson_free(body); return;
        }
        t_embed = st_now_sec();

        /* 2) 加载向量库 + 检索 */
        if (vllm_kv_load(&kb, kb_dir) != 0) {
            infer_done(ctx); free(qvec); json_error(resp, 404, "vector store load failed"); vjson_free(body); return;
        }
        if (kb.dim != dim) {
            infer_done(ctx); free(qvec); vllm_kv_free(&kb);
            json_error(resp, 400, "vector store dim mismatch"); vjson_free(body); return;
        }
        k = top_k < kb.n ? top_k : kb.n;
        if (k <= 0) { infer_done(ctx); free(qvec); vllm_kv_free(&kb); json_error(resp, 400, "empty vector store"); vjson_free(body); return; }
        int kc = k;                                  /* 候选数（重排开时放大） */
        {
            int rn = vllm_kb_rerank_candidates();
            if (rn > kc) kc = (rn < kb.n) ? rn : kb.n;
        }
        idx = (int *)malloc((size_t)kc * sizeof(int));
        sc = (float *)malloc((size_t)kc * sizeof(float));
        if (!idx || !sc) {
            infer_done(ctx); free(qvec); vllm_kv_free(&kb); free(idx); free(sc);
            json_error(resp, 500, "Out of memory"); vjson_free(body); return;
        }
        vllm_kv_search(&kb, qvec, kc, idx, sc);
        if (kc > k) {                                /* 词面混合重排（显式打日志，不静默） */
            vllm_kb_rerank(query, &kb, idx, sc, kc);
            fprintf(stderr, "[RAG] rerank on: candidates=%d -> top%d\n", kc, k);
            fflush(stderr);
        }
        t_search = st_now_sec();
    }

    /* 3) 拒答门（K-E2E-003/004，--kb-abstain 阈值 才开，默认关=零回归）。
     * 只判「知识库里有没有这条依据」，不判答案内容；证据不足则跳过生成、显式拒答
     * 并标 action=handoff_to_human。阈值由板端正/负样本标定（先冻结判据后采数）。 */
    int abstained = 0;
    double ev_value = 0.0, ev_thr = 0.0, ev_cover = 0.0;
    float ev_top1 = 0.0f;
    if (!plain_fallback && k > 0) {
        ev_thr = (ctx->kb_abstain > 0.f) ? (double)ctx->kb_abstain : 0.0;
        ev_top1 = sc[0];
        ev_cover = vllm_kb_coverage(query, kb.meta[idx[0]]);
        ev_value = vllm_kb_evidence(query, kb.meta[idx[0]], sc[0]);
        if (ev_thr > 0.0 && ev_value < ev_thr) abstained = 1;
        fprintf(stderr, "[RAG] evidence=%.3f (top1cos=%.3f cover=%.3f) gate=%s\n",
                ev_value, (double)ev_top1, ev_cover,
                ev_thr > 0.0 ? (abstained ? "ABSTAIN" : "pass") : "off");
        fflush(stderr);
    }

    /* 4) 拼 prompt（回退：不拼检索上下文；拒答：不生成） */
    char *prompt = NULL;
    if (abstained) {
        /* 跳过 prompt 构建与生成 */
    } else if (plain_fallback) {
        size_t plen = strlen(query) + 64;
        char *rag_prompt = (char *)malloc(plen);
        if (!rag_prompt) {
            infer_done(ctx); json_error(resp, 500, "Out of memory"); vjson_free(body); return;
        }
        snprintf(rag_prompt, plen, "问题：%s\n回答：", query);
        prompt = build_completion_prompt(ctx, rag_prompt, 0 /* 直接回答，不思考 */);
        free(rag_prompt);
    } else {
        size_t plen = 128;
        for (int i = 0; i < k; i++) plen += strlen(kb.meta[idx[i]]) + 1;
        plen += strlen(query) + 64;
        char *rag_prompt = (char *)malloc(plen);
        if (!rag_prompt) {
            infer_done(ctx); free(qvec); vllm_kv_free(&kb); free(idx); free(sc);
            json_error(resp, 500, "Out of memory"); vjson_free(body); return;
        }
        size_t po = 0;
        po += (size_t)snprintf(rag_prompt + po, plen - po, "参考资料：\n");
        for (int i = 0; i < k; i++)
            po += (size_t)snprintf(rag_prompt + po, plen - po, "%s\n", kb.meta[idx[i]]);
        po += (size_t)snprintf(rag_prompt + po, plen - po, "\n问题：%s\n回答：", query);
        prompt = build_completion_prompt(ctx, rag_prompt, 0 /* 直接回答，不思考 */);
        free(rag_prompt);
    }

    /* 5) chat 生成（贪心，确定性）；拒答时已无 prompt，跳过 */
    StrBuf out; sb_init(&out);
    VLLMMetrics met; memset(&met, 0, sizeof(met));
    int rc = 0;
    if (!abstained) {
        rc = prompt ? run_completion(ctx, prompt, NULL, NULL, 0, NULL, 0, NULL,
                                     0.0, 1.0, 0.0, 0, 0, max_tokens,
                                     0, NULL, 0, &out, &met, NULL, NULL, NULL, NULL)
                    : -1;
    }
    free(prompt);
    infer_done(ctx);
    double t_llm = st_now_sec();
    if (collab_timing_on()) {
        if (plain_fallback) {
            fprintf(stderr, "[COLLAB-T] rag plain-fallback llm=%.1fms total=%.1fms\n",
                    (t_llm - t_c0) * 1000.0, (t_llm - t_c0) * 1000.0);
        } else {
            fprintf(stderr, "[COLLAB-T] rag embed=%.1fms search=%.1fms llm=%.1fms total=%.1fms\n",
                    (t_embed - t_c0) * 1000.0, (t_search - t_embed) * 1000.0,
                    (t_llm - t_search) * 1000.0, (t_llm - t_c0) * 1000.0);
        }
        fflush(stderr);
    }

    if (rc != 0) {
        sb_free(&out); free(qvec); vllm_kv_free(&kb); free(idx); free(sc);
        json_error(resp, 500, rc == -2 ? "Prompt exceeds the context limit" : "Inference failed");
        vjson_free(body); return;
    }

    /* 6) 构造响应 JSON */
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "rag"));
    if (plain_fallback) vjson_obj_set(root, "fallback", vjson_new_string("plain_llm"));
    if (abstained) {
        handoff_notify(ctx, "low_evidence", query, ev_value, ev_thr);
        vjson_obj_set(root, "abstained", vjson_new_bool(1));
        vjson_obj_set(root, "cause", vjson_new_string("low_evidence"));
        vjson_obj_set(root, "action", vjson_new_string("handoff_to_human"));
        VJson *ej = vjson_new_object();
        vjson_obj_set(ej, "evidence", vjson_new_number(ev_value));
        vjson_obj_set(ej, "threshold", vjson_new_number(ev_thr));
        vjson_obj_set(ej, "top1_cos", vjson_new_number((double)ev_top1));
        vjson_obj_set(ej, "coverage", vjson_new_number(ev_cover));
        vjson_obj_set(root, "evidence", ej);
    }
    VJson *arr = vjson_new_array();
    for (int i = 0; i < k; i++) {
        VJson *it = vjson_new_object();
        vjson_obj_set(it, "index", vjson_new_number((double)idx[i]));
        vjson_obj_set(it, "meta", vjson_new_string(kb.meta[idx[i]]));
        vjson_obj_set(it, "score", vjson_new_number((double)sc[i]));
        vjson_array_push(arr, it);
    }
    vjson_obj_set(root, "retrieved", arr);
    vjson_obj_set(root, "answer", vjson_new_string(abstained
        ? "知识库中未找到足以支撑回答的依据，已拒绝作答（建议转人工处理）。"
        : (out.s ? out.s : "")));
    size_t cap = 4096 + (size_t)strlen(out.s ? out.s : "") * 2;
    for (int i = 0; i < k; i++) cap += strlen(kb.meta[idx[i]]) * 2 + 64;
    char *rbody = (char *)malloc(cap);
    if (!rbody) {
        sb_free(&out); free(qvec); vllm_kv_free(&kb); free(idx); free(sc);
        vjson_free(root); json_error(resp, 500, "Out of memory"); vjson_free(body); return;
    }
    size_t n = vjson_serialize(root, rbody, cap);
    vjson_free(root);
    sb_free(&out); free(qvec); vllm_kv_free(&kb); free(idx); free(sc);
    if (n > 0) {
        resp->status = 200;
        resp->content_type = "application/json";
        resp->body = rbody;
        resp->body_len = n;
        resp->body_owned = rbody;
    } else {
        free(rbody);
        json_error(resp, 500, "Response too large");
    }
    vjson_free(body);
}

/* ---------- POST /v1/ask (K-COLLAB-001 P4 统一入口，轻量规则路由) ----------
 * 四链统一入口：按显式请求字段做**轻量规则**路由（§6 决策 2）。
 * Body : {query, image?, kb_dir?, top_k?, max_tokens?, questions?, state?}
 *   有 image     → scene  场景理解链（YOLO 感知 → 场景叙事 → LLM）
 *   有 kb_dir    → rag    检索增强链（embedding → 向量库 top-k → LLM）
 *   有 questions → decide 决策链（Laya /v1/systemone）
 *   否则         → chat   深度推理（纯 LLM）
 * Reply: 子链响应 + "route" 字段（自描述：本次由哪条链作答）。
 * 说明 : 路由默认只依据显式字段，不做语义猜测（避免 silent mis-route）；子链失败
 *        显式透传其错误码（J4）；不支持流式（本端点用于非流式编排验证）。
 *        —— K-E2E-002：若配置了 env VLLM_ROUTE_POLICY（策略文件），则对**仅带
 *        query** 的请求先做一次 Laya 决策，再按映射层分派（kb / local / escalate），
 *        响应带 route_source=policy + department/confidence/churn_risk/low_conf，
 *        使 /v1/ask 成为单入口语义路由；未配置时行为与以往完全一致。 */
static void ask_tag_route(VHttpResponse *resp, const char *route) {
    if (resp->status != 200 || !resp->body || resp->body[0] != '{') return;
    size_t blen = resp->body_len ? resp->body_len : strlen(resp->body);
    if (blen < 2) return;
    char ins[48];
    int il = snprintf(ins, sizeof(ins), "\"route\":\"%s\",", route);
    char *nb = (char *)malloc(blen + (size_t)il + 1);
    if (!nb) return;
    memcpy(nb, resp->body, 1);                       /* '{' */
    memcpy(nb + 1, ins, (size_t)il);
    memcpy(nb + 1 + il, resp->body + 1, blen - 1);   /* 余下 + '\0' */
    nb[blen + il] = '\0';
    if (resp->body_owned) free(resp->body_owned);
    resp->body = nb;
    resp->body_owned = nb;
    resp->body_len = blen + (size_t)il;
}

/* 在 JSON 响应体开头注入一段原文字段（frag 需自带结尾逗号）。 */
static void ask_inject_raw(VHttpResponse *resp, const char *frag) {
    if (!frag || !frag[0]) return;
    if (resp->status != 200 || !resp->body || resp->body[0] != '{') return;
    size_t blen = resp->body_len ? resp->body_len : strlen(resp->body);
    size_t il = strlen(frag);
    if (blen < 2) return;
    char *nb = (char *)malloc(blen + il + 1);
    if (!nb) return;
    memcpy(nb, resp->body, 1);
    memcpy(nb + 1, frag, il);
    memcpy(nb + 1 + il, resp->body + 1, blen - 1);
    nb[blen + il] = '\0';
    if (resp->body_owned) free(resp->body_owned);
    resp->body = nb;
    resp->body_owned = nb;
    resp->body_len = blen + il;
}

/* ---------- K-E2E-002：/v1/ask 语义路由（可选，默认关） ----------
 * env VLLM_ROUTE_POLICY=<policy.json> 时，对**仅带 query** 的请求先跑一次 Laya
 * 决策（原生三问），再按映射层把「决策 → 执行链」分派，使 /v1/ask 成为单入口的
 * **语义**路由（不配置时行为与以往完全一致：只按显式字段路由）。
 * policy.json = { "kb_dir":"<.vkb>", "top_k":3, "conf_min":0.5, "churn_max":0.5,
 *                 "questions": { <与 /v1/systemone 同构的 questions 对象> } }
 * 映射层（与离线验证一致，见 K-E2E-001）：
 *   churn_risk >= churn_max            → escalate（转人工，不做生成）
 *   department != "other"              → kb（policy.kb_dir 检索增强链）
 *   其余                                → local（本地生成链）
 * 显式降级（不静默）：confidence < conf_min 时不走 kb，改 local 并标 "low_conf":true。 */
typedef struct {
    int   on;
    char  kb_dir[512];
    int   top_k;
    float conf_min;
    float churn_max;
    char *qbody;        /* "questions" 对象的原文（拼 decide 请求用） */
} AskRoutePolicy;
static AskRoutePolicy g_askrp;

/* 定位文本中 "key" 之后的值区间（跟踪字符串/转义，{} [] 配对）。 */
static int json_value_span(const char *t, const char *key,
                           const char **out, size_t *outlen) {
    const char *k = strstr(t, key);
    if (!k) return -1;
    const char *p = strchr(k + strlen(key), ':');
    if (!p) return -1;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    const char *s = p;
    if (*p == '{' || *p == '[') {
        char open = *p;
        char close = (open == '{') ? '}' : ']';
        int depth = 0, instr = 0, esc = 0;
        for (; *p; p++) {
            char c = *p;
            if (instr) {
                if (esc) esc = 0;
                else if (c == '\\') esc = 1;
                else if (c == '"') instr = 0;
            } else if (c == '"') instr = 1;
            else if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) { p++; break; } }
        }
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != '\n') p++;
    }
    *out = s;
    *outlen = (size_t)(p - s);
    return 0;
}

static void askrp_load(void) {
    static int tried = 0;
    if (tried) return;
    tried = 1;
    const char *p = getenv("VLLM_ROUTE_POLICY");
    if (!p || !p[0]) return;                       /* 默认关 */
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "[ASK] route policy not readable: %s\n", p); fflush(stderr); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1 << 20)) { fclose(f); fprintf(stderr, "[ASK] route policy size out of range\n"); fflush(stderr); return; }
    char *txt = (char *)malloc((size_t)sz + 1);
    if (!txt) { fclose(f); return; }
    size_t rd = fread(txt, 1, (size_t)sz, f);
    fclose(f);
    txt[rd] = '\0';
    VJson *root = vjson_parse(txt);
    if (!root || root->type != VJ_OBJECT) {
        fprintf(stderr, "[ASK] route policy: invalid JSON object\n"); fflush(stderr);
        free(txt); vjson_free(root); return;
    }
    VJson *v;
    const char *kd = vjson_str(vjson_obj_get(root, "kb_dir"));
    if (kd && kd[0]) snprintf(g_askrp.kb_dir, sizeof(g_askrp.kb_dir), "%s", kd);
    g_askrp.top_k = 3;
    if ((v = vjson_obj_get(root, "top_k")) && v->type == VJ_NUMBER) g_askrp.top_k = (int)vjson_num(v);
    g_askrp.conf_min = 0.5f;
    if ((v = vjson_obj_get(root, "conf_min")) && v->type == VJ_NUMBER) g_askrp.conf_min = (float)vjson_num(v);
    g_askrp.churn_max = 0.5f;
    if ((v = vjson_obj_get(root, "churn_max")) && v->type == VJ_NUMBER) g_askrp.churn_max = (float)vjson_num(v);
    const char *qs = NULL; size_t ql = 0;
    if (json_value_span(txt, "\"questions\"", &qs, &ql) == 0 && ql > 2) {
        g_askrp.qbody = (char *)malloc(ql + 1);
        if (g_askrp.qbody) { memcpy(g_askrp.qbody, qs, ql); g_askrp.qbody[ql] = '\0'; }
    }
    vjson_free(root);
    free(txt);
    if (g_askrp.qbody && g_askrp.kb_dir[0]) {
        g_askrp.on = 1;
        fprintf(stderr, "[ASK] semantic route policy ON: kb_dir=%s top_k=%d conf_min=%.2f churn_max=%.2f\n",
                g_askrp.kb_dir, g_askrp.top_k, g_askrp.conf_min, g_askrp.churn_max);
    } else {
        fprintf(stderr, "[ASK] route policy ignored (need both 'questions' and 'kb_dir')\n");
    }
    fflush(stderr);
}

/* JSON 字符串转义（仅用于把 query 拼进 decide 请求）。返回写入长度。 */
static size_t json_escape_into(const char *s, char *out, size_t cap) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p && o + 8 < cap; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20) { o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c); }
        else out[o++] = (char)c;
    }
    out[o] = '\0';
    return o;
}

/* 语义决策：返回 0 成功（route_out 填 kb/local/escalate；extra 填决策字段片段）。 */
static int askrp_decide(VLLMServerCtx *ctx, const char *query,
                        char *route_out, size_t rcap,
                        char *extra, size_t ecap) {
    if (!g_askrp.on || !g_askrp.qbody) return -1;
    size_t ql = strlen(query) * 6 + 32;
    char *esc = (char *)malloc(ql);
    if (!esc) return -1;
    json_escape_into(query, esc, ql);
    size_t blen = strlen(esc) + strlen(g_askrp.qbody) + 64;
    char *body = (char *)malloc(blen);
    if (!body) { free(esc); return -1; }
    snprintf(body, blen, "{\"state\":\"%s\",\"questions\":%s}", esc, g_askrp.qbody);
    free(esc);

    pthread_mutex_lock(&g_laya_lock);
    if (!g_laya_runner) {
        g_laya_runner = laya_runner_new(ctx->laya_pack);
        if (g_laya_runner)
            fprintf(stderr, "[ASK] laya runner ready for semantic routing (%s)\n", ctx->laya_pack);
    }
    char out[8192];
    out[0] = '\0';
    int rc = g_laya_runner ? laya_runner_decide(g_laya_runner, body, out, sizeof(out)) : -1;
    pthread_mutex_unlock(&g_laya_lock);
    free(body);
    if (rc != 0 || !out[0]) return -1;

    VJson *root = vjson_parse(out);
    if (!root) return -1;
    VJson *ans = vjson_obj_get(root, "answers");
    char dept[64] = "";                     /* ⚠ 必须在 vjson_free(root) 前拷出，
                                             * choice 指向 root 内部字符串（否则 UAF） */
    double conf = 0.0, churn = 0.0, urg = 0.0;
    if (ans && ans->type == VJ_OBJECT) {
        VJson *d = vjson_obj_get(ans, "department");
        if (d && d->type == VJ_OBJECT) {
            const char *c = vjson_str(vjson_obj_get(d, "choice"));
            if (c) snprintf(dept, sizeof(dept), "%s", c);
            VJson *cf = vjson_obj_get(d, "confidence");
            if (cf && cf->type == VJ_NUMBER) conf = vjson_num(cf);
        }
        VJson *ch = vjson_obj_get(ans, "churn_risk");
        if (ch && ch->type == VJ_OBJECT) {
            VJson *n = vjson_obj_get(ch, "noul");
            if (n && n->type == VJ_NUMBER) churn = vjson_num(n);
        }
        VJson *u = vjson_obj_get(ans, "urgency");
        if (u && u->type == VJ_OBJECT) {
            VJson *s = vjson_obj_get(u, "score");
            if (s && s->type == VJ_NUMBER) urg = vjson_num(s);
        }
    }
    vjson_free(root);

    const char *route;
    int low_conf = 0;
    if (churn >= (double)g_askrp.churn_max) {
        route = "escalate";
    } else if (strcmp(dept, "other") != 0 && dept[0]) {
        if (conf < (double)g_askrp.conf_min) { route = "local"; low_conf = 1; }
        else route = "kb";
    } else {
        route = "local";
    }
    snprintf(route_out, rcap, "%s", route);
    char dept_esc[128];
    json_escape_into(dept, dept_esc, sizeof(dept_esc));   /* 防御：字段原样进响应 JSON */
    snprintf(extra, ecap,
             "\"route_source\":\"policy\",\"department\":\"%s\",\"confidence\":%.4f,"
             "\"churn_risk\":%.4f,\"urgency\":%.2f,\"low_conf\":%s,",
             dept_esc, conf, churn, urg, low_conf ? "true" : "false");
    fprintf(stderr, "[ASK] semantic route=%s dept=%s conf=%.3f churn=%.3f low_conf=%d\n",
            route, dept, conf, churn, low_conf);
    fflush(stderr);
    return 0;
}

static void handle_ask(VLLMServerCtx *ctx, const VHttpRequest *req,
                       VHttpResponse *resp) {
    if (!req->body || req->body_len == 0) {
        json_error(resp, 400, "empty body"); return;
    }
    if (req->body[0] != '{') {
        json_error(resp, 400, "body must be a JSON object"); return;
    }
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }

    const char *image  = vjson_str(vjson_obj_get(body, "image"));
    const char *kb_dir = vjson_str(vjson_obj_get(body, "kb_dir"));
    const char *query  = vjson_str(vjson_obj_get(body, "query"));
    VJson *questions   = vjson_obj_get(body, "questions");
    int has_image = image && image[0];
    int has_kb    = kb_dir && kb_dir[0];
    int has_q     = questions && questions->type == VJ_OBJECT;
    char *qdup    = query ? strdup(query) : NULL;
    int thinking  = resolve_thinking(body);
    int max_tokens = 256;
    VJson *jmt = vjson_obj_get(body, "max_tokens");
    if (jmt && jmt->type == VJ_NUMBER) {
        max_tokens = (int)jmt->u.number;
        if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }

    double t0 = st_now_sec();
    const char *route = has_image ? "scene" : (has_kb ? "rag" : (has_q ? "decide" : "chat"));
    /* K-E2E-002：仅带 query 且配置了路由策略时，先做 Laya 语义决策再分派（默认关）。 */
    char rp_extra[256] = "";
    char *rp_ragbody = NULL;
    int rp_escalate = 0;
    if (!has_image && !has_kb && !has_q && qdup && qdup[0]) {
        askrp_load();
        if (g_askrp.on) {
            char sroute[24], extra[256];
            if (askrp_decide(ctx, qdup, sroute, sizeof(sroute), extra, sizeof(extra)) == 0) {
                snprintf(rp_extra, sizeof(rp_extra), "%s", extra);
                if (strcmp(sroute, "escalate") == 0) {
                    rp_escalate = 1;
                    route = "escalate";
                } else if (strcmp(sroute, "kb") == 0 && !strstr(extra, "\"low_conf\":true")) {
                    size_t ql2 = strlen(qdup) * 6 + 32;
                    char *esc2 = (char *)malloc(ql2);
                    if (esc2) {
                        json_escape_into(qdup, esc2, ql2);
                        size_t bl = strlen(esc2) + strlen(g_askrp.kb_dir) + 96;
                        rp_ragbody = (char *)malloc(bl);
                        if (rp_ragbody)
                            snprintf(rp_ragbody, bl,
                                     "{\"query\":\"%s\",\"kb_dir\":\"%s\",\"top_k\":%d,\"max_tokens\":%d}",
                                     esc2, g_askrp.kb_dir, g_askrp.top_k, max_tokens);
                        free(esc2);
                        if (rp_ragbody) route = "rag";
                    }
                } else {
                    route = "local";       /* 低置信 / 域外 → 本地生成（low_conf 已显式标注） */
                }
            } else {
                route = "local";
                snprintf(rp_extra, sizeof(rp_extra),
                         "\"route_source\":\"policy\",\"decision_failed\":true,");
            }
        }
    }
    fprintf(stderr, "[ASK] route=%s\n", route);
    fflush(stderr);
    /* 转人工旁路通知：必须在 vjson_free(body) 之前（query 指向 body 内字符串）。 */
    if (rp_escalate) handoff_notify(ctx, "policy_escalate", query, 0.0, 0.0);
    vjson_free(body);

    if (rp_escalate) {
        /* 转人工：不做生成，直接产出显式响应（§映射层） */
        VJson *root = vjson_new_object();
        vjson_obj_set(root, "model", vjson_new_string("policy-router"));
        vjson_obj_set(root, "route", vjson_new_string("escalate"));
        vjson_obj_set(root, "action", vjson_new_string("handoff_to_human"));
        vjson_obj_set(root, "answer", vjson_new_string(""));
        size_t cap = 512 + strlen(rp_extra) * 2;
        char *rb = (char *)malloc(cap);
        if (!rb) { vjson_free(root); free(qdup); json_error(resp, 500, "Out of memory"); return; }
        size_t n2 = vjson_serialize(root, rb, cap);
        vjson_free(root);
        free(qdup);
        if (n2 > 0) {
            resp->status = 200; resp->content_type = "application/json";
            resp->body = rb; resp->body_len = n2; resp->body_owned = rb;
            ask_inject_raw(resp, rp_extra);
        } else { free(rb); json_error(resp, 500, "Response too large"); }
        return;
    }
    if (rp_ragbody) {
        VHttpRequest tmp;
        memset(&tmp, 0, sizeof(tmp));
        tmp.method = "POST";
        tmp.path = "/v1/rag";
        tmp.body = rp_ragbody;
        tmp.body_len = strlen(rp_ragbody);
        handle_rag(ctx, &tmp, resp);
        free(rp_ragbody);
        free(qdup);
        ask_tag_route(resp, "rag");
        ask_inject_raw(resp, rp_extra);
        return;
    }

    if (has_image) {
        handle_scene(ctx, req, resp);
        ask_tag_route(resp, route);
    } else if (has_kb) {
        handle_rag(ctx, req, resp);
        ask_tag_route(resp, route);
    } else if (has_q) {
        handle_systemone(ctx, req, resp);
        ask_tag_route(resp, route);
    } else {
        /* chat 深度推理（纯 LLM，贪心确定性） */
        if (!qdup || !qdup[0]) {
            free(qdup);
            json_error(resp, 400,
                "'query' is required (or image / kb_dir / questions for other chains)");
            return;
        }
        if (ensure_model_ready(ctx, resp) != 0) { free(qdup); return; }
        int g = infer_gate(ctx);
        if (g != 0) {
            free(qdup);
            if (g == -1) json_error(resp, 429, "Too many queued requests, retry later");
            else if (g == -2) json_error(resp, 503, "Insufficient memory, retry later");
            else json_error(resp, 503, "Model is unloading, retry later");
            return;
        }
        char *prompt = build_completion_prompt(ctx, qdup, thinking);
        free(qdup);
        StrBuf out; sb_init(&out);
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        int rc = prompt ? run_completion(ctx, prompt, NULL, NULL, 0, NULL, 0, NULL,
                                         0.0, 1.0, 0.0, 0, 0, max_tokens,
                                         0, NULL, 0, &out, &met, NULL, NULL, NULL, NULL)
                        : -1;
        free(prompt);
        infer_done(ctx);
        if (rc != 0) {
            sb_free(&out);
            json_error(resp, 500, rc == -2 ? "Prompt exceeds the context limit"
                                           : "Inference failed");
            return;
        }
        VJson *root = vjson_new_object();
        vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "ask"));
        vjson_obj_set(root, "route", vjson_new_string(route));
        vjson_obj_set(root, "answer", vjson_new_string(out.s ? out.s : ""));
        size_t cap = 1024 + (size_t)strlen(out.s ? out.s : "") * 2;
        char *rbody = (char *)malloc(cap);
        if (!rbody) { sb_free(&out); vjson_free(root); json_error(resp, 500, "Out of memory"); return; }
        size_t n = vjson_serialize(root, rbody, cap);
        vjson_free(root); sb_free(&out);
        if (n > 0) {
            resp->status = 200; resp->content_type = "application/json";
            resp->body = rbody; resp->body_len = n; resp->body_owned = rbody;
            ask_inject_raw(resp, rp_extra);   /* 语义路由回填：route_source/low_conf 等 */
        } else {
            free(rbody); json_error(resp, 500, "Response too large");
        }
    }

    if (collab_timing_on() && resp->status == 200) {
        fprintf(stderr, "[COLLAB-T] ask route=%s total=%.1fms\n",
                route, (st_now_sec() - t0) * 1000.0);
        fflush(stderr);
    }
}

/* ---------- router ---------- */

/* Defined in vllm_admin.c (web management surface under /admin). */
void vllm_admin_route(VLLMServerCtx *ctx, const VHttpRequest *req,
                      VHttpResponse *resp);

/* Defined in vllm_admin.c: 审查台独立页（/review），与 /admin/ 分离。 */
void vllm_review_page(VHttpResponse *resp);

static void server_handler(const VHttpRequest *req, VHttpResponse *resp,
                           VHttpConn *conn, void *ud) {
    VLLMServerCtx *ctx = (VLLMServerCtx *)ud;
    /* 匹配到的路由默认成功：各 handler 自行覆盖（错误分支显式置 400/403/404/5xx）。
     * 未匹配的路由在下方各自的 else 分支里显式置 404 —— 这里不能预置 404，
     * 否则 handler 里的错误码会被"已置 404"状态吞掉/顶替。 */
    resp->status = 200;
    resp->content_type = "application/json";

    /* Unload gate: every HTTP handler is counted (inflight_handlers). The
     * unload waits for this to reach 0 before freeing ctx->ist/cfg/w — a
     * worker may be between ensure_model_ready and infer_gate where it reads
     * ctx->ist/cfg without holding inf_lock, and freeing under it would be a
     * use-after-free. Inference endpoints are additionally rejected (503)
     * while an unload is in progress so no NEW handler can enter that window. */
    const int is_infer_path =
        strcmp(req->path, "/v1/chat/completions") == 0 ||
        strcmp(req->path, "/v1/completions") == 0 ||
        strcmp(req->path, "/v1/embeddings") == 0 ||
        strcmp(req->path, "/v1/messages") == 0 ||
        strcmp(req->path, "/v1/rag") == 0 ||
        strcmp(req->path, "/v1/scene") == 0 ||
        strcmp(req->path, "/v1/ask") == 0;
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->inflight_handlers++;
    int reject_unload = is_infer_path && ctx->unload_pending;
    vhttp_mutex_unlock(ctx->stat_lock);
    if (reject_unload) {
        static const char msg[] =
            "{\"error\":{\"message\":\"Model is unloading, retry later\","
            "\"type\":\"unavailable\",\"code\":503}}";
        resp->status = 503;
        resp->content_type = "application/json";
        resp->body = msg;
        resp->body_len = sizeof(msg) - 1;
        vhttp_mutex_lock(ctx->stat_lock);
        ctx->inflight_handlers--;
        vhttp_mutex_unlock(ctx->stat_lock);
        return;
    }

    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        handle_models(ctx, resp);
    } else if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        handle_health(ctx, resp);
    } else if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/attest") == 0) {
        handle_attest_info(ctx, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/chat/completions") == 0) {
        handle_chat(ctx, req, resp, conn);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/messages") == 0) {
        handle_anthropic(ctx, req, resp, conn);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/completions") == 0) {
        handle_completions(ctx, req, resp, conn);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/embeddings") == 0) {
        handle_embeddings(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/systemone") == 0) {
        handle_systemone(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/detect") == 0) {
        handle_detect(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/detect_video") == 0) {
        handle_detect_video(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/stream/start") == 0) {
        handle_stream_start(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/stream/stop") == 0) {
        handle_stream_stop(ctx, req, resp);
    } else if (strcmp(req->path, "/v1/stream/status") == 0) {
        handle_stream_status(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/rag") == 0) {
        handle_rag(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/scene") == 0) {
        handle_scene(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/ask") == 0) {
        handle_ask(ctx, req, resp);
    } else if (strncmp(req->path, "/admin", 6) == 0) {
        vllm_admin_route(ctx, req, resp);
    } else if (strcmp(req->path, "/review") == 0 ||
               strcmp(req->path, "/review/") == 0) {
        /* 审查台：独立页面（数据仍走同源的 /admin/api/*，单一管理面 API）。 */
        vllm_review_page(resp);
    } else if (strcmp(req->path, "/chat") == 0 || strcmp(req->path, "/chat/") == 0) {
        handle_chat_page(resp);
    } else if (strcmp(req->path, "/") == 0) {
        static const char info[] = "{\"status\":\"ok\",\"server\":\"vllm_kestrel\"}";
        resp->status = 200;
        resp->content_type = "application/json";
        resp->body = info;
        resp->body_len = sizeof(info) - 1;
    } else {
        static const char nf[] =
            "{\"error\":{\"message\":\"Not Found\",\"type\":\"invalid_request_error\",\"code\":404}}";
        resp->status = 404;
        resp->content_type = "application/json";
        resp->body = nf;
        resp->body_len = sizeof(nf) - 1;
    }

    vhttp_mutex_lock(ctx->stat_lock);
    ctx->inflight_handlers--;
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* ================================================================
 * 内存驻留策略（档位阶梯）—— 实现
 *
 * 档位定义与触发语义见 vllm_server.h 顶部注释。动作在 vllm_res_set_level
 * 内执行：KV 档位动作持 inf_lock（与推理互斥）；权重档只改 PTE（madvise）；
 * L0 卸载复用 vllm_serve_unload_model 的安全门（自带 busy 排空）。
 * ================================================================ */

const char *vllm_res_level_name(int lvl) {
    switch (lvl) {
        case 0: return "L0-unloaded";
        case 1: return "L1-initial-layer";
        case 2: return "L2-weight-window";
        case 3: return "L3-kv-evict";
        case 4: return "L4-full-resident";
        default: return "?";
    }
}

/* 权重分层能力：明文 VQF + VLLM_VQF_STREAM 分层模式建表（seg 表）后才有
 * "从高向低释放权重段"动作（L2/L1 档）。 */
static int res_wcap(void) {
    return vqf_stream_active();
}

void vllm_res_defaults(VLLMServerCtx *ctx) {
    VLLMResPolicy *rp = &ctx->res;
    memset(rp, 0, sizeof(*rp));
    rp->level       = 4;               /* 默认全层驻留 */
    rp->idle_s[4]   = 300;             /* L4 -> L3 : 5 分钟 */
    rp->idle_s[3]   = 900;             /* L3 -> L2 : 15 分钟 */
    rp->idle_s[2]   = 1800;            /* L2 -> L1 : 30 分钟 */
    rp->idle_s[1]   = 3600;            /* L1 -> L0 : 60 分钟（落盘+卸载） */
    rp->w_keep      = 8;               /* L2 窗口：保留 0..8 层 */
    rp->mem_low_lvl = 0;               /* 自动动作可到 L0（最终卸载） */
    rp->entered_s   = (long)time(NULL);
    fprintf(stderr, "[RES] defaults: idle_plan=%lds/%lds/%lds/%lds "
                    "w_keep=%d soft=%dMB low_lvl=%d\n",
            rp->idle_s[4], rp->idle_s[3], rp->idle_s[2], rp->idle_s[1],
            rp->w_keep, rp->mem_soft_mb, rp->mem_low_lvl);
}

int vllm_res_level(const VLLMServerCtx *ctx) {
    if (ctx->load_state != 2) return 0;    /* 未加载 = 最省档 */
    int lvl = ctx->res.level;
    if (lvl < 0) lvl = 0;
    if (lvl > 4) lvl = 4;
    if (!res_wcap() && lvl >= 1 && lvl <= 2) lvl = 3;  /* 权重档不可执行 → 3 */
    return lvl;
}

int vllm_res_persist_session(VLLMServerCtx *ctx) {
    /* 文本请求每次完成即 dkv_save（run_completion），此处只兜底：disk-kv
     * 开且内存里还有未落盘的纯文本上下文时补一次保存；多模态不落盘。 */
    if (!ctx->disk_kv || !ctx->last_ids || ctx->last_n <= 0) return 0;
    if (ctx->last_was_mm) return 0;
    dkv_save(ctx, ctx->last_ids, ctx->last_n);
    return 0;
}

/* 开/关 KV 逐层淘汰（L3）。off 时清理 L3 状态、把被置 NULL 的块重新背回
 * 内存并作废上下文（与 ist_reset 的 no-reuse 分支一致：旧 KV 内容在 L3
 * 档已被 Q4 镜像化，无法原样回填 f32/q8 块 → 下一请求全量重 prefill）。
 * 调用前提：持 inf_lock 或串行上下文（tick/加载后）。 */
static void res_kv_l3_set(VLLMServerCtx *ctx, int on) {
    STQwenInferenceState *st = ctx->ist;
    if (!st) return;
    if (on) {
        if (g_l3_evict) return;
        g_l3_evict = 1;
        if (!g_sparse_attn) {
            /* L3 驱逐要求 sparse decode 消费磁盘 Q4 块；不开 sparse 时
             * l3_evict_after_prefill 会告警跳过。策略升 L3 档即同步开，
             * 并记下"由策略强开"，离开 L3 档时恢复原值。 */
            g_sparse_attn = 1;
            ctx->res.l3_forced_sparse = 1;
            fprintf(stderr, "[RES] L3 on: --sparse-attn forced on (L3 依赖)\n");
        }
        fprintf(stderr, "[RES] L3 on: KV 逐层淘汰已启用 (g_l3_evict=1, "
                        "前缀复用随之关闭)\n");
        fflush(stderr);
        return;
    }
    if (!g_l3_evict) return;
    g_l3_evict = 0;
    if (ctx->res.l3_forced_sparse) {
        g_sparse_attn = 0;               /* 恢复策略强开前的 exact attention */
        ctx->res.l3_forced_sparse = 0;
    }
    if (st->l3.fp) {
        fprintf(stderr, "[RES] L3 off: close %s\n", st->l3.path);
        l3_state_free(&st->l3);
    }
    /* 作废内存前缀复用序列：L3 开启期间冷块可能已被释放为 NULL，复用会把
     * 已丢的块当有效上下文（前缀复用与 L3 互斥的既有协议）。块本身留给
     * 下一请求的 ist_reset 重建（st_qwen_kv_rebuild_freed）后全量重写。 */
    if (ctx->last_ids) { free(ctx->last_ids); ctx->last_ids = NULL; }
    /* last_cap 必须同步清零：否则 run_completion 的
     * "total <= 陈旧 last_cap → memcpy(ctx->last_ids, ...)" 会对 NULL 写 →
     * 恢复 L4 后首请求 decode 完成段崩溃（崩溃 A 根因，x64/ARM 同现）。 */
    ctx->last_cap = 0;
    ctx->last_n = 0;
    ctx->last_was_mm = 0;
    fprintf(stderr, "[RES] L3 off: KV 全驻留 (g_l3_evict=0)\n");
    fflush(stderr);
}

/* 释放 layer > keep 的权重段物理页（需 res_wcap；否则 no-op）。 */
static void res_w_evict(int keep) {
    if (!res_wcap()) return;
    vqf_stream_evict_above(keep);
}

/* 把目标档位所需的全部配置动作落地（调用前提：模型已加载 / 持锁串行）。 */
static void res_apply(VLLMServerCtx *ctx, int lvl) {
    switch (lvl) {
        case 4:  res_kv_l3_set(ctx, 0); break;              /* 全层驻留 */
        case 3:  res_kv_l3_set(ctx, 1); break;              /* KV 逐层淘汰 */
        case 2:  res_kv_l3_set(ctx, 1); res_w_evict(ctx->res.w_keep); break;
        case 1:  res_kv_l3_set(ctx, 1); res_w_evict(0); break;
        case 0:
        default: break;      /* L0 由 set_level 走卸载路径 */
    }
}

int vllm_res_set_level(VLLMServerCtx *ctx, int target) {
    if (target < 0) target = 0;
    if (target > 4) target = 4;
    if (!res_wcap() && (target == 1 || target == 2)) target = 3; /* 能力钳制 */
    int cur = vllm_res_level(ctx);
    if (target == cur) return cur;

    /* 到 L0：先 KV 落盘善后，再卸载（busy 则拒绝并保持原档）。
     * 用 self 豁免包装：HTTP handler 内（policy/set）卸载时扣除调用者自身
     * 的 inflight 计数，否则忙检视会把"自己"误判为在途推理而永远 -3。 */
    if (target == 0) {
        vllm_res_persist_session(ctx);
        int rc = vllm_serve_unload_model_self(ctx);
        if (rc == 0) {
            ctx->res.level = 0;
            ctx->res.entered_s = (long)time(NULL);
            fprintf(stderr, "[RES] %s -> L0-unloaded: KV 落盘善后 + 整模卸载\n",
                    vllm_res_level_name(cur));
            fflush(stderr);
            return 0;
        }
        fprintf(stderr, "[RES] L0 unload refused (rc=%d: busy/loading/no-model), "
                        "staying at %s\n", rc, vllm_res_level_name(cur));
        return (rc == -3) ? -3 : cur;
    }

    /* 升档且模型未加载：主动加载（需 model_dir；auto-load/load-on-use 模式
     * 保证可用，manual 模式由 admin 先设置模型目录）。 */
    if (target >= 1 && ctx->load_state != 2) {
        int rc = vllm_serve_load_on_use(ctx);
        if (rc != 0) {
            fprintf(stderr, "[RES] upgrade to %s needs a model load (rc=%d); "
                            "set 模型目录 or enable 用时加载 first\n",
                    vllm_res_level_name(target), rc);
            return -1;
        }
        /* 新加载权重全驻留（mmap 懒页在首次推理后触达）；档位从 L4 语义起 */
        res_kv_l3_set(ctx, 0);
    }

    vhttp_mutex_lock(ctx->inf_lock);
    res_apply(ctx, target);
    vhttp_mutex_unlock(ctx->inf_lock);

    ctx->res.level = target;
    ctx->res.entered_s = (long)time(NULL);
    fprintf(stderr, "[RES] level %s -> %s\n", vllm_res_level_name(cur),
            vllm_res_level_name(target));
    fflush(stderr);
    return target;
}

void vllm_res_tick(VLLMServerCtx *ctx) {
    VLLMResPolicy *rp = &ctx->res;
    long now = (long)time(NULL);
    int cur = vllm_res_level(ctx);
    int loaded = (ctx->load_state == 2);

    /* 档位与加载状态同步：模型被卸载（含策略卸载后又被 auto-load/手动重载）
     * 时，把档位对齐到实际可执行状态。 */
    if (!loaded) {
        if (rp->level != 0) { rp->level = 0; rp->entered_s = now; }
        return;
    }
    if (rp->level == 0 && loaded) {
        /* 已加载但策略档仍 L0：回退到 CLI 引擎配置对应档位（auto-load /
         * 手动加载后新会话从起点档重新计空闲）。CLI --l3-evict 保持 L3，
         * 无 --l3-evict 回 L4 全层驻留。 */
        int start_lvl = g_l3_evict ? 3 : 4;
        fprintf(stderr, "[RES] model loaded while policy L0 -> resume %s\n",
                vllm_res_level_name(start_lvl));
        vllm_res_set_level(ctx, start_lvl);
        return;
    }

    /* 1) 空闲超时逐级降档（auto_idle 开、当前档有定义的等待时间） */
    if (rp->auto_idle && cur >= 1) {
        long wait = rp->idle_s[cur];
        if (wait > 0 && (now - ctx->last_infer_s) >= wait) {
            long idle_since = now - ctx->last_infer_s;
            int target = cur - 1;
            if (!res_wcap() && target <= 2) target = 0;  /* 权重档不可用 → 直达卸载 */
            int prev = vllm_res_level(ctx);
            int rc = vllm_res_set_level(ctx, target);
            if (rc != prev && rc >= 0) {
                rp->steps++;
                fprintf(stderr, "[RES] idle %lds >= %lds -> auto degraded "
                                "%s -> %s (steps=%d)\n",
                        idle_since, wait, vllm_res_level_name(prev),
                        vllm_res_level_name(rc), rp->steps);
            }
            rp->entered_s = now;   /* 成败均重计时：避免每 tick 重复尝试刷日志 */
        }
    }

    /* 2) 软水位：可用内存低于 mem_soft_mb → 主动降一档（15s 冷却），最低到
     *    mem_low_lvl。硬门（请求入口 503）由 ctx->min_free_mb 另行兜底。 */
    if (rp->mem_soft_mb > 0 && cur >= 1 && (now - rp->last_mem_s) >= 15) {
        long avail = st_avail_mem_mb();
        if (avail >= 0 && avail < rp->mem_soft_mb) {
            rp->last_mem_s = now;
            int target = cur - 1;
            if (target < rp->mem_low_lvl) target = rp->mem_low_lvl;
            if (!res_wcap() && target >= 1 && target <= 2) target = 0;
            if (target != cur) {
                int prev = vllm_res_level(ctx);
                int rc = vllm_res_set_level(ctx, target);
                if (rc != prev && rc >= 0) {
                    rp->steps++;
                    fprintf(stderr, "[RES] free RAM %ld MB < soft=%d MB -> "
                                    "auto degraded %s -> %s (steps=%d)\n",
                            avail, rp->mem_soft_mb, vllm_res_level_name(prev),
                            vllm_res_level_name(rc), rp->steps);
                }
            } else {
                fprintf(stderr, "[RES] free RAM %ld MB < soft=%d MB but already "
                                "at mem_low_lvl=%d (hard gate 503 at min-free-mb)\n",
                        avail, rp->mem_soft_mb, rp->mem_low_lvl);
            }
        }
    }
}

/* Serve policy monitor (1s tick): drives the memory-residency ladder
 * (vllm_res_tick: idle 逐级降档 + 软水位主动降级) and, when the new policy
 * is disabled, keeps the legacy --auto-unload behaviour (不用时卸载).
 * Unload only happens when the engine is fully idle; vllm_serve_unload_model
 * refuses busy states and blocks new inference while freeing. */
static void *idle_unload_thread(void *arg) {
    VLLMServerCtx *ctx = (VLLMServerCtx *)arg;
    while (!ctx->stopping) {
        sleep(1);
        /* 内存驻留策略（档位阶梯：空闲逐级降档 / 软水位主动降级） */
        vllm_res_tick(ctx);
        /* 兼容旧 --auto-unload N：策略空闲降级关闭时的原语义 */
        if (!ctx->res.auto_idle && ctx->auto_unload_s > 0 &&
            ctx->load_state == 2 &&
            (time(NULL) - ctx->last_infer_s) > ctx->auto_unload_s) {
            int rc = vllm_serve_unload_model(ctx);
            if (rc == 0) {
                ctx->auto_unload_count++;
                fprintf(stderr, "[UNLOAD] idle %lds > %lds -> auto-unloaded "
                        "(total unloads=%d)\n",
                        (long)(time(NULL) - ctx->last_infer_s),
                        ctx->auto_unload_s, ctx->unload_count);
                fflush(stderr);
            }
        }
    }
    return NULL;
}

int vllm_server_run(VLLMServerCtx *ctx, int port, int n_threads,
                    void (*on_start)(int actual_port, void *ud)) {
    ctx->inf_lock = vhttp_mutex_new();
    ctx->stat_lock = vhttp_mutex_new();
    if (!ctx->inf_lock || !ctx->stat_lock) {
        fprintf(stderr, "[SRV] mutex init failed\n");
        return 1;
    }
    pthread_mutex_init(&ctx->embed_lock, NULL);   /* K-COLLAB-001 P2：专用 embedding 槽位锁 */
    ctx->total_requests = 0;
    ctx->active_requests = 0;
    ctx->queued_requests = 0;
    ctx->busy = 0;
    ctx->inflight_handlers = 0;
    ctx->next_chunk_id = 0;
    ctx->last_infer_s = (long)time(NULL);   /* never auto-unload before first use */
    ctx->stopping = 0;
    ctx->idle_thr_started = 0;
    /* Policy monitor always runs: it drives the residency ladder whenever
     * res.auto_idle / mem_soft_mb get enabled at runtime by /admin (a 1s
     * wakeup is negligible) and preserves legacy --auto-unload. */
    if (pthread_create(&ctx->idle_thr, NULL, idle_unload_thread, ctx) == 0)
        ctx->idle_thr_started = 1;
    else
        fprintf(stderr, "[SRV] policy monitor thread create failed "
                        "(auto-unload / residency policy disabled)\n");
    int rc = vhttp_serve_ex(port, server_handler, ctx, on_start, n_threads);
    ctx->stopping = 1;
    if (ctx->idle_thr_started) {
        pthread_join(ctx->idle_thr, NULL);
        ctx->idle_thr_started = 0;
    }
    vllm_server_reset_session(ctx);
    vllm_embed_free(ctx);                    /* K-COLLAB-001 P2：释放专用 embedding 槽位 */
    pthread_mutex_destroy(&ctx->embed_lock);
    vhttp_mutex_free(ctx->inf_lock);
    vhttp_mutex_free(ctx->stat_lock);
    return rc;
}
