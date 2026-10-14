/* vllm_kvstore.c - 向量库核心（K-COLLAB-001 P0）。零依赖，标量余弦，确定性。
 * 详见 include/core/vllm_kvstore.h。 */
#include "vllm_kvstore.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define KV_MAGIC 0x31424B56u   /* 'VKB1' (little-endian) */

void vllm_kv_init(VLLMKVStore *kb, int dim) {
    memset(kb, 0, sizeof(*kb));
    kb->dim = dim > 0 ? dim : 0;
}

void vllm_kv_free(VLLMKVStore *kb) {
    if (!kb) return;
    if (kb->meta) {
        for (int i = 0; i < kb->n; i++) free(kb->meta[i]);
        free(kb->meta);
    }
    free(kb->vec);
    memset(kb, 0, sizeof(*kb));
}

static int kv_reserve(VLLMKVStore *kb, int need) {
    if (need <= kb->cap) return 0;
    int ncap = kb->cap ? kb->cap : 8;
    while (ncap < need) ncap <<= 1;
    char **nm = (char **)realloc(kb->meta, (size_t)ncap * sizeof(char *));
    if (!nm) return -1;
    kb->meta = nm;
    float *nv = (float *)realloc(kb->vec, (size_t)ncap * (size_t)kb->dim * sizeof(float));
    if (!nv) return -1;
    kb->vec = nv;
    kb->cap = ncap;
    return 0;
}

int vllm_kv_add(VLLMKVStore *kb, const char *meta, const float *vec) {
    if (!kb || kb->dim <= 0 || !vec) return -1;
    if (kv_reserve(kb, kb->n + 1) != 0) return -1;
    int i = kb->n;
    const char *m = meta ? meta : "";
    size_t ml = strlen(m);
    kb->meta[i] = (char *)malloc(ml + 1);
    if (!kb->meta[i]) return -1;
    memcpy(kb->meta[i], m, ml + 1);
    memcpy(kb->vec + (size_t)i * kb->dim, vec, (size_t)kb->dim * sizeof(float));
    kb->n = i + 1;
    return 0;
}

/* K-ADMIN-001：删除第 idx 条（memmove 压缩 meta/vec）。返回 0 成功 / -1 越界。
 * 保持剩余条目的相对顺序不变（检索确定性依赖下标，删后顺序即新下标序）。 */
int vllm_kv_del(VLLMKVStore *kb, int idx) {
    if (!kb || idx < 0 || idx >= kb->n) return -1;
    free(kb->meta[idx]);
    if (idx < kb->n - 1) {
        memmove(kb->meta + idx, kb->meta + idx + 1,
                (size_t)(kb->n - idx - 1) * sizeof(char *));
        memmove(kb->vec + (size_t)idx * kb->dim,
                kb->vec + (size_t)(idx + 1) * kb->dim,
                (size_t)(kb->n - idx - 1) * (size_t)kb->dim * sizeof(float));
    }
    kb->meta[kb->n - 1] = NULL;
    kb->n--;
    return 0;
}

void vllm_kv_search(const VLLMKVStore *kb, const float *query, int k,
                    int *out_idx, float *out_score) {
    if (!kb || !query || !out_idx || !out_score || k <= 0 || kb->n <= 0) return;
    int K = k < kb->n ? k : kb->n;
    const int dim = kb->dim;
    /* 初始化 top-K 列表：分数 −∞、下标用哨兵 kb->n（大于任何真实 i，保证
     * tie-break「i 较小者在前」在首轮不误触发）。 */
    for (int j = 0; j < K; j++) { out_idx[j] = kb->n; out_score[j] = -INFINITY; }
    for (int i = 0; i < kb->n; i++) {
        const float *v = kb->vec + (size_t)i * dim;
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += query[d] * v[d];   /* 固定累加序 ⇒ 确定性 */
        int pos = 0;
        while (pos < K &&
               !(s > out_score[pos] || (s == out_score[pos] && i < out_idx[pos])))
            pos++;
        if (pos < K) {
            for (int j = K - 1; j > pos; j--) {
                out_idx[j] = out_idx[j - 1];
                out_score[j] = out_score[j - 1];
            }
            out_idx[pos] = i;
            out_score[pos] = s;
        }
    }
}

int vllm_kv_save(const VLLMKVStore *kb, const char *path) {
    if (!kb || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = KV_MAGIC, dim = (uint32_t)kb->dim, n = (uint32_t)kb->n;
    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&dim, 4, 1, f) != 1 ||
        fwrite(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    for (int i = 0; i < kb->n; i++) {
        size_t ml = strlen(kb->meta[i]);
        uint32_t ml32 = (uint32_t)ml;
        if (fwrite(&ml32, 4, 1, f) != 1 ||
            fwrite(kb->meta[i], 1, ml, f) != ml ||
            fwrite(kb->vec + (size_t)i * kb->dim, sizeof(float), (size_t)kb->dim, f)
                != (size_t)kb->dim) {
            fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

int vllm_kv_load(VLLMKVStore *kb, const char *path) {
    if (!kb || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0, dim = 0, n = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1 ||
        fread(&n, 4, 1, f) != 1 || magic != KV_MAGIC || dim == 0) {
        fclose(f); return -1;
    }
    VLLMKVStore tmp;
    vllm_kv_init(&tmp, (int)dim);
    int ok = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ml = 0;
        if (fread(&ml, 4, 1, f) != 1 || ml > (1u << 24)) break;
        char *meta = (char *)malloc((size_t)ml + 1);
        if (!meta) break;
        if (ml && fread(meta, 1, ml, f) != ml) { free(meta); break; }
        meta[ml] = 0;
        float *vec = (float *)malloc((size_t)dim * sizeof(float));
        if (!vec) { free(meta); break; }
        if (fread(vec, sizeof(float), (size_t)dim, f) != (size_t)dim) {
            free(meta); free(vec); break;
        }
        if (vllm_kv_add(&tmp, meta, vec) != 0) { free(meta); free(vec); break; }
        free(meta); free(vec);
        if ((int)i == (int)n - 1) ok = 1;
    }
    fclose(f);
    if (!ok) { vllm_kv_free(&tmp); return -1; }
    vllm_kv_free(kb);
    *kb = tmp;
    return 0;
}

/* ================= 词面混合重排（K-E2E-002，零依赖） ================= */

int vllm_kb_rerank_candidates(void) {
    const char *e = getenv("VLLM_KB_RERANK");
    if (!e || !e[0]) return 0;                 /* 默认关 */
    if (e[0] == '0' && e[1] == '\0') return 0;
    int n = atoi(e);
    if (n <= 0) return 0;
    if (n > 256) n = 256;
    return n;
}

/* 取 query 的词面 token：
 *   ① ASCII 词/数字段（首字符须为字母或数字，小写），权重 3；
 *   ② 中文 2-gram，权重 1 —— 先按 UTF-8 3 字节（CJK 区 U+4E00..U+9FFF ⇒ 首字节
 *      E4..E9）过滤出中文串，再取**相邻中文对**成 bigram。⚠ 跨非中文字符也成对
 *      （如 "多少 GB 降到" ⇒ "少降"），与原型实现（Python 先过滤再取相邻对）同一语义，
 *      否则引擎与原型排序会不一致（实测过：不一致会让 top1 从 10/18 掉到 6/18）。 */
#define RR_MAXTOK 128
#define RR_TOKLEN 32

static int rr_is_alnum(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int rr_is_ascii_word(unsigned char c) {
    return rr_is_alnum(c) || c == '_' || c == '.' || c == '-';
}

static int rr_tokenize(const char *q, char tok[RR_MAXTOK][RR_TOKLEN], int weights[RR_MAXTOK]) {
    int nt = 0;
    char cjk[RR_MAXTOK][4];
    int ncjk = 0;
    const unsigned char *p = (const unsigned char *)q;
    while (*p && nt < RR_MAXTOK) {
        if (rr_is_alnum(*p)) {                    /* ① ASCII 词/数字段 */
            char buf[RR_TOKLEN];
            int l = 0;
            while (*p && rr_is_ascii_word(*p) && l < RR_TOKLEN - 1) {
                unsigned char c = *p++;
                buf[l++] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
            }
            buf[l] = '\0';
            snprintf(tok[nt], RR_TOKLEN, "%s", buf);
            weights[nt] = 3;
            nt++;
        } else if (p[0] >= 0xE4 && p[0] <= 0xE9 && p[1] && p[2]) {   /* ② 收集中文 */
            if (ncjk < RR_MAXTOK) {
                cjk[ncjk][0] = (char)p[0]; cjk[ncjk][1] = (char)p[1];
                cjk[ncjk][2] = (char)p[2]; cjk[ncjk][3] = '\0';
                ncjk++;
            }
            p += 3;
        } else {
            p++;
        }
    }
    for (int i = 0; i + 1 < ncjk && nt < RR_MAXTOK; i++) {
        snprintf(tok[nt], RR_TOKLEN, "%s%s", cjk[i], cjk[i + 1]);
        weights[nt] = 1;
        nt++;
    }
    /* 去重（与原型一致：token 按集合计分，重复出现不重复加权）。 */
    int m = 0;
    for (int i = 0; i < nt; i++) {
        int dup = 0;
        for (int j = 0; j < m; j++) if (strcmp(tok[i], tok[j]) == 0) { dup = 1; break; }
        if (!dup) {
            if (m != i) {
                char tmp[RR_TOKLEN];
                memcpy(tmp, tok[i], RR_TOKLEN);
                memcpy(tok[m], tmp, RR_TOKLEN);
                weights[m] = weights[i];
            }
            m++;
        }
    }
    return m;
}

/* 大小写无关子串查找（仅用于 ASCII token；中文 token 走 strstr）。 */
static int rr_ci_contains(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) return 0;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        while (i < nl) {
            char a = p[i], b = needle[i];
            if (!a) return 0;
            if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
            if (b >= 'A' && b <= 'Z') b = (char)(b + 32);
            if (a != b) break;
            i++;
        }
        if (i == nl) return 1;
    }
    return 0;
}

static double rr_score(const char tok[RR_MAXTOK][RR_TOKLEN], const int *w,
                       int nt, const char *text) {
    if (!text) return 0.0;
    double s = 0.0;
    for (int i = 0; i < nt; i++) {
        if (w[i] == 3) { if (rr_ci_contains(text, tok[i])) s += 3.0; }
        else if (strstr(text, tok[i])) s += 1.0;
    }
    return s;
}

void vllm_kb_rerank(const char *query, const VLLMKVStore *kb,
                    int *idx, float *score, int n) {
    if (!query || !query[0] || !kb || !idx || n <= 1) return;
    static char tok[RR_MAXTOK][RR_TOKLEN];
    static int w[RR_MAXTOK];
    int nt = rr_tokenize(query, tok, w);
    if (nt == 0) return;
    /* 稳定插入排序：按词面分降序，并列保持原（向量）序。 */
    for (int i = 1; i < n; i++) {
        int ci = idx[i]; float cs = score[i];
        double ck = rr_score(tok, w, nt, kb->meta[ci]);
        int j = i - 1;
        while (j >= 0) {
            double pk = rr_score(tok, w, nt, kb->meta[idx[j]]);
            if (pk < ck) { idx[j + 1] = idx[j]; score[j + 1] = score[j]; j--; }
            else break;
        }
        idx[j + 1] = ci; score[j + 1] = cs;
    }
}

/* ================= 拒答门证据分（K-E2E-003，零依赖） ================= */

double vllm_kb_coverage(const char *query, const char *meta) {
    if (!query || !query[0] || !meta) return 0.0;
    static char tok[RR_MAXTOK][RR_TOKLEN];
    static int w[RR_MAXTOK];
    int nt = rr_tokenize(query, tok, w);
    if (nt <= 0) return 0.0;
    double tot = 0.0;
    for (int i = 0; i < nt; i++) tot += (double)w[i];
    if (tot <= 0.0) return 0.0;
    double hit = rr_score(tok, w, nt, meta);
    if (hit > tot) hit = tot;                 /* 防御：覆盖率归一到 [0,1] */
    return hit / tot;
}

double vllm_kb_evidence(const char *query, const char *meta, float top1cos) {
    return 0.5 * (double)top1cos + 0.5 * vllm_kb_coverage(query, meta);
}
