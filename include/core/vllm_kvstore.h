/* vllm_kvstore.h - 向量库（多模型协作平台「记忆层」的核心存储，K-COLLAB-001 P0）
 *
 * 零依赖的句子向量知识库：存储 (meta 文本 + L2 归一向量)，暴力余弦检索 top-k，
 * 二进制持久化。本模块**不涉及 embedding 前向**——向量由上游（embedding 模型）
 * 产出后灌入；检索只用「向量余弦 = 内积」这一条假设（约定向量已 L2 归一）。
 *
 * 确定性约定（J1）：标量余弦累加序固定（for i 升序），top-k 用稳定插入选择 ⇒
 * 同输入同输出。不做 NEON（微基准证 NEON 与标量非位级，max|d|=1.37e-6）。
 */
#ifndef VLLM_KVSTORE_H
#define VLLM_KVSTORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VLLMKVStore {
    int    dim;      /* 向量维度（如 1024） */
    int    n;        /* 已存条数 */
    int    cap;      /* meta/vec 容量（按需增长） */
    char **meta;     /* [n] 每条 doc 的文本（UTF-8，自有副本） */
    float *vec;      /* [n * dim] 连续存储的 L2 归一向量 */
} VLLMKVStore;

/* 初始化/释放。dim 须 > 0。 */
void vllm_kv_init(VLLMKVStore *kb, int dim);
void vllm_kv_free(VLLMKVStore *kb);

/* 追加一条。meta 可 NULL（记为 ""）；vec 须含 dim 个 float（按 L2 归一语义，
 * 本模块**不重算归一**，直接原样存储——调用方保证已归一）。返回 0 成功 / -1 OOM。 */
int  vllm_kv_add(VLLMKVStore *kb, const char *meta, const float *vec);

/* 删除第 idx 条（memmove 压缩；剩余条目相对顺序不变）。返回 0 成功 / -1 越界。
 * 持久化需调用方随后 vllm_kv_save。 */
int  vllm_kv_del(VLLMKVStore *kb, int idx);

/* 检索：对 query（dim 维，已 L2 归一）暴力算内积（=余弦），返回 top-k 的下标与
 * 分数（降序）。k<=0 或 k>n 时取 min(k,n)。out_idx/out_score 至少 k 个元素。
 * 稳定：并列时取较小下标在前。 */
void vllm_kv_search(const VLLMKVStore *kb, const float *query, int k,
                    int *out_idx, float *out_score);

/* 持久化（二进制）：magic 'VKB1' | u32 dim | u32 n | n×(u32 meta_len | meta | dim×f32)。
 * 返回 0 成功 / -1 失败。load 会先 free 已有内容。 */
int  vllm_kv_save(const VLLMKVStore *kb, const char *path);
int  vllm_kv_load(VLLMKVStore *kb, const char *path);

/* 返回条数（便捷访问）。 */
static inline int vllm_kv_count(const VLLMKVStore *kb) { return kb ? kb->n : 0; }

/* ---- 词面混合重排（K-E2E-002）：零依赖、默认关 ----
 * 动机：纯向量检索对「数字/术语/专有名词」型查询排序不稳（实测 top1 38.9%）。
 * 做法：先取向量 top-N 候选，再用 query 的 ASCII 词与数字段 + 中文 2-gram 在
 * chunk 文本上的覆盖度重排（ASCII/数字命中权重 3.0、中文 bigram 1.0），
 * 并列时保持原向量序（稳定）。实测 top1 → 55.6%、R@5 66.7→72.2%、MRR +0.13。
 * env VLLM_KB_RERANK=N 控制候选数（0/未设 = 关闭，默认关闭；N 上限 256）。 */
int  vllm_kb_rerank_candidates(void);

/* 就地重排前 n 个候选（idx/score 由 vllm_kv_search 按向量序填好；
 * query 可为 NULL/空 → 不做任何改动）。 */
void vllm_kb_rerank(const char *query, const VLLMKVStore *kb,
                    int *idx, float *score, int n);

/* ---- 拒答门证据分（K-E2E-003）：零依赖、默认关 ----
 * 动机：检索「命中但无依据」时主模型会顺着检索上下文编造（hallucination-by-context）。
 * 证据分只回答「知识库里到底有没有这条依据」，不评判答案内容：
 *   ev = 0.5 * top1 余弦 + 0.5 * 词面覆盖率（覆盖 token 权重 / 全部 token 权重）
 * 阈值由板端实测标定（正样本 18 / 负样本 10，判据先冻结后采数）。 */
double vllm_kb_coverage(const char *query, const char *meta);
double vllm_kb_evidence(const char *query, const char *meta, float top1cos);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_KVSTORE_H */
