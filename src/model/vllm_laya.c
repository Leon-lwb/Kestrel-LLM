/**
 * vllm_laya.c - Laya decision model: ModernBERT-large encoder + decision head.
 *
 * Faithful scalar-F32 port of transformers ModernBertModel + the Laya
 * DecisionModel head (rl_common.py). Correctness-first (no SIMD yet).
 *
 * Pinned semantics (transformers/models/modernbert/modeling_modernbert.py):
 *   embeddings : h = LayerNorm(tok_embed(ids))                 (norm_bias=false)
 *   layer i    : attn_norm(i==0 -> Identity), attention, residual;
 *                mlp_norm, MLP, residual
 *   attention  : bidirectional; sliding layers limit keys to |q-kv| <= local/2 (=64)
 *   RoPE       : inv_freq[j]=1/theta^(2j/hd); emb=cat(freqs,freqs); theta full=160000 sliding=10000
 *   MLP        : input,gate = Wi(x).chunk(2); Wo(gelu(input)*gate)   [exact erf gelu]
 *   head       : h += type_emb[qtype]; 2x TransformerEncoderLayer(norm_first, relu);
 *                scorer = LN -> Linear -> GELU -> Linear(->1) on each marker position
 */
#include "vllm_laya.h"
#include "vllm_platform.h"         /* ST_HAVE_NEON / ST_ARCH_X86 (SIMD 选择, K-LAYA-002) */
#include "vllm_tokenizer_qwen.h"   /* K-LAYA-001: reuse engine BPE (vocab.bin+merges.bin) */
#include "vllm_http.h"             /* vjson: request/response for --laya-decide */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* K-LAYA-002 M4：位置级并行的线程号（无 OpenMP 时恒 0 ⇒ 纯串行、逐位不变）。 */
#ifdef _OPENMP
#include <omp.h>
#define OTID omp_get_thread_num()
#else
#define OTID 0
#endif

/* K-LAYA-002 M6（诊断）：env 门控的阶段计时（VLLM_LAYA_TIMING=1）。
 * 仅测量、不改行为；关闭时开销为 0。仅在 laya_forward 外层（串行上下文）累加。 */
enum { TT_EMB, TT_NORM, TT_MM, TT_ROPE, TT_ATTN, TT_ACT, TT_RESID, TT_SCO, TT_N };
static double g_tt[TT_N] = {0};
static double g_tt_total = 0;
static int    g_timing = -1;
static double lay_wtime(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
}
static inline double tmark(void) { return g_timing == 1 ? lay_wtime() : 0.0; }
static inline void tacc(int i, double t0) { if (g_timing == 1) g_tt[i] += lay_wtime() - t0; }

#define LAYA_D     1024
#define LAYA_H     16
#define LAYA_HD    64
#define LAYA_INTER 2624
#define LAYA_NL    28
#define LAYA_EPS   1e-5f
#define LAYA_HFFN  4096
#define LAYA_SLIDE 64
#define LAYA_NT_MAX 16   /* K-LAYA-002 M4：位置并行的最大线程数（scratch 池上限） */
#define LAYA_QB    128   /* K-LAYA-002 M5：int8 量化块大小（元素/块）。
                          * M5d：板端微基准 t4r1——块 32→128 提速 ~2.0×（27→55 GF/s，
                          * 每块 scale 开销被摊薄）；256/1024 边际递减。代价：粒度 32→128
                          * （公理 collapse_priority_rule 张力点，须以 J2 决策等价实测守门）。
                          * 注：SCALE 块 = LAYA_QB；vdotq 处理 CHUNK 仍恒 32 字节。 */

typedef struct {
    char   name[96];
    int    ndim;
    int    ne[4];
    int    is_f32;
    size_t off;
    size_t nbytes;
    const uint16_t *f16; /* K-LAYA-002 M2：F16 张量指向 blob（不再展开 F32 arena） */
    const float    *f32; /* F32 张量指向 blob（如 temperature） */
    const int8_t   *i8;  /* K-LAYA-002 M5：2D 权重的 int8 表示（per-32 块对称量化） */
    const float    *qsc; /* int8 权重的 per-32 块 scale（行主序，每行 nblk 个） */
    int    rowlen;       /* M5：行宽（2D 为 ne[1]=K，行主序连续维） */
    int    nblk;         /* M5：每行的量化块数 = ceil(rowlen/32) */
    size_t n_elem;
} LayaEnt;

struct LayaModel {
    unsigned char *blob;     /* K-LAYA-002 M2：整个 laya_f16.bin 常驻（843 MB），M5 载入后释放 */
    size_t         blob_size;
    float         *f32pool;  /* 1D 张量（norm/bias）展开的 F32 小池（<0.5 MB） */
    int8_t        *i8pool;   /* K-LAYA-002 M5：全部 2D 权重的 int8 arena（~421 MB） */
    float         *qscpool;  /* K-LAYA-002 M5：int8 权重的 per-32 块 scale（~53 MB） */
    LayaEnt       *ent;
    int            n_ent;
};

/* ---------------- fp16 -> fp32 ---------------- */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t o;
    if (exp == 0) {
        if (man == 0) { o = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            o = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        o = sign | 0x7F800000u | (man << 13);
    } else {
        o = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &o, 4);
    return f;
}

static const LayaEnt *lay_find(const LayaModel *m, const char *name) {
    for (int i = 0; i < m->n_ent; i++)
        if (strcmp(m->ent[i].name, name) == 0) return &m->ent[i];
    fprintf(stderr, "[LAYA] missing tensor: %s\n", name);
    return NULL;
}

static inline float lay_at(const LayaModel *m, const LayaEnt *e, size_t i) {
    (void)m;
    if (e->i8) {   /* M5：int8 反量化（按行主序定位块 scale） */
        size_t r = i / (size_t)e->rowlen, c = i % (size_t)e->rowlen;
        return (float)e->i8[i] * e->qsc[r * (size_t)e->nblk + (c / LAYA_QB)];
    }
    return e->f16 ? f16_to_f32(e->f16[i]) : e->f32[i];
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M1：F32 SIMD 点积（8-lane FMA）。
 * 公理 superposition_addition_base（lane 规约）+ conservation_law_under_collapse
 * （精度守恒：与标量同语义，仅规约顺序不同 ⇒ 容差门内）。
 * aarch64 → NEON vfmaq_f32；x86-64(-mavx2 -mfma) → AVX2 _mm256_fmadd_ps；
 * 其它 → 标量。尾数走标量补齐（LAYA 主维均 4/8 的倍数，实际无尾）。
 * ------------------------------------------------------------------ */
static inline float lay_dot(const float *a, const float *b, int n) {
#if ST_HAVE_NEON
    /* 4 累加器 × unroll16：隐藏 A76 FMA 延迟（4c），提高发射密度 */
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i),      vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    __m256 s8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s8);
    __m128 hi = _mm256_extractf128_ps(s8, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M2：F16 权重 · F32 激活 点积（F32 累加）。
 * 公理 collapse_to_classic（F16 存储）+ conservation_law_under_collapse
 * （F16→F32 转换无损 ⇒ 与 M1 的 F32 arena 逐位同值，仅累加顺序不同）。
 * aarch64：vcvt_f32_f16(vld1_f16) + vfmaq_f32；x8664：_mm256_cvtph_ps + fmadd。
 * 尾数走标量补齐。
 * ------------------------------------------------------------------ */
static inline float lay_dot_f16(const uint16_t *w, const float *x, int n) {
#if ST_HAVE_NEON
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i))),      vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 4))),  vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 8))),  vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 12))), vld1q_f32(x + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i))), vld1q_f32(x + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),      _mm256_loadu_ps(x + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),  _mm256_loadu_ps(x + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 16))), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 24))), _mm256_loadu_ps(x + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))), _mm256_loadu_ps(x + i), a0);
    __m256 s8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s8);
    __m128 hi = _mm256_extractf128_ps(s8, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#endif
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M5：int8 权重 · int8 激活 点积（per-32 块对称量化，int32 累加）。
 * 公理 collapse_to_classic + collapse_priority_rule（细粒度块 scale ⇒ 避免单
 * scale 的不可逆失真）；conservation_law_under_collapse（confidence=0.65 弱约束
 * ⇒ 以 J2 决策输出等价实测验证，不靠理论保证）。
 * aarch64：vcvt_f32_f16 无；用 vdotq_s32（ARMv8.2 dotprod，A76/RK3588 支持）
 * 每指令 16 int8 → 4×int32 累加；逐块 diag(sw)*diag(sx) 缩放后 F32 累加。
 * x86/无 dotprod：标量兜底。
 * ------------------------------------------------------------------ */
static inline float lay_dot_i8(const int8_t *w, const float *sw,
                               const int8_t *x, const float *sx, int K) {
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sumv = vdupq_n_f32(0.f);
    int b = 0;
    for (; b + LAYA_QB <= K; b += LAYA_QB) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int c = 0; c < LAYA_QB; c += 32) {   /* SCALE 块内按 32 字节 CHUNK 做 vdotq */
            acc = vdotq_s32(acc, vld1q_s8(w + b + c),      vld1q_s8(x + b + c));
            acc = vdotq_s32(acc, vld1q_s8(w + b + c + 16), vld1q_s8(x + b + c + 16));
        }
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(acc), sw[b / LAYA_QB] * sx[b / LAYA_QB]);
    }
    float s = vaddvq_f32(sumv);
    for (; b < K; b++) s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
    return s;
#else
    float s = 0.0f;
    for (int b = 0; b + LAYA_QB <= K; b += LAYA_QB)
        for (int j = 0; j < LAYA_QB; j++) s += (float)w[b + j] * (float)x[b + j] * sw[b / LAYA_QB] * sx[b / LAYA_QB];
    for (int b = (K / LAYA_QB) * LAYA_QB; b < K; b++) s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
    return s;
#endif
}

/* K-LAYA-002 M5：把一行 F32 激活量化为 int8 + per-32 块对称 scale（写入 xq/xsc）。
 * M5c：NEON 向量化（maxabs=vmax，量化=vdivq+vcvtnq_s32_f32+vqmovn 饱和窄化）。
 * 用真除法 vdivq_f32 + round-to-nearest-even ⇒ 与标量 lrintf(x/d) **位级同值**
 * （|x/d|≤127 ⇒ [-128,127] 饱和与标量 [-127,127] 等效）。 */
static void lay_quant_act(const float *x, int8_t *xq, float *xsc, int K) {
    for (int b = 0; b < K; b += LAYA_QB) {
        int n = (b + LAYA_QB <= K) ? LAYA_QB : (K - b);
        float d;
#if ST_HAVE_NEON
        float32x4_t vmx = vdupq_n_f32(0.f);
        int j = 0;
        for (; j + 4 <= n; j += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(x + b + j)));
        float mx = vmaxvq_f32(vmx);
        for (; j < n; j++) { float a = fabsf(x[b + j]); if (a > mx) mx = a; }
        d = mx > 0.f ? mx / 127.f : 1.f;
        xsc[b / LAYA_QB] = d;
        float32x4_t vd = vdupq_n_f32(d);
        j = 0;
        for (; j + 16 <= n; j += 16) {
            int32x4_t q0 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j),      vd));
            int32x4_t q1 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 4),  vd));
            int32x4_t q2 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 8),  vd));
            int32x4_t q3 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 12), vd));
            int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(xq + b + j, vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1)));
        }
        for (; j < n; j++) {
            int q = (int)lrintf(x[b + j] / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            xq[b + j] = (int8_t)q;
        }
#else
        float mx = 0.f;
        for (int j = 0; j < n; j++) { float a = fabsf(x[b + j]); if (a > mx) mx = a; }
        d = mx > 0.f ? mx / 127.f : 1.f;
        xsc[b / LAYA_QB] = d;
        for (int j = 0; j < n; j++) {
            int q = (int)lrintf(x[b + j] / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            xq[b + j] = (int8_t)q;
        }
#endif
    }
}

/* K-LAYA-002 M5：把一行 F16 权重按 per-32 块对称量化成 int8 + scale（载入期一次性）。 */
static void lay_quant_f16_row(const uint16_t *src, int8_t *dst, float *sc, int K) {
    for (int b = 0; b < K; b += LAYA_QB) {
        int n = (b + LAYA_QB <= K) ? LAYA_QB : (K - b);
        float mx = 0.f;
        for (int j = 0; j < n; j++) { float a = fabsf(f16_to_f32(src[b + j])); if (a > mx) mx = a; }
        float d = mx > 0.f ? mx / 127.f : 1.f;
        sc[b / LAYA_QB] = d;
        for (int j = 0; j < n; j++) {
            int q = (int)lrintf(f16_to_f32(src[b + j]) / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            dst[b + j] = (int8_t)q;
        }
    }
}

/* y[o] = sum_i W[o][i]*x[i] (+B[o]); W shape [n_out, n_in]
 * 注（K-LAYA-002 M2b 实测 2026-10-10）：NEON 4 行寄存器分块（16 累加器）在
 * A76 上**退化 27%**（19.3s vs 15.3s，寄存器溢出）⇒ 已回退，逐行点积保留。
 * M5：权重/激活均为 int8（per-32 块 scale），走 lay_dot_i8。 */
static void lay_linear(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                       const float *x, float *y, int n_out, int n_in) {
    (void)m;
    if (W->i8) {
        int nblk = (n_in + LAYA_QB - 1) / LAYA_QB;
        int8_t *xq = (int8_t *)malloc((size_t)n_in);
        float  *xs = (float *)malloc(sizeof(float) * (size_t)nblk);
        if (!xq || !xs) { free(xq); free(xs); for (int o = 0; o < n_out; o++) y[o] = 0.f; return; }
        lay_quant_act(x, xq, xs, n_in);
        for (int o = 0; o < n_out; o++) {
            float acc = lay_dot_i8(W->i8 + (size_t)o * n_in, W->qsc + (size_t)o * nblk, xq, xs, n_in);
            if (B) acc += lay_at(m, B, (size_t)o);
            y[o] = acc;
        }
        free(xq); free(xs);
        return;
    }
    const uint16_t *w16 = W->f16;
    const float    *w32 = W->f32;
    for (int o = 0; o < n_out; o++) {
        float acc = w16 ? lay_dot_f16(w16 + (size_t)o * n_in, x, n_in)
                        : lay_dot(w32 + (size_t)o * n_in, x, n_in);
        if (B) acc += lay_at(m, B, (size_t)o);
        y[o] = acc;
    }
}

static void lay_ln(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                   const float *x, float *y, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
    var /= n;
    const float invf  = (float)(1.0 / sqrt(var + LAYA_EPS));
    const float meanf = (float)mean;
    const float *w = W->f32;
    const float *bb = B ? B->f32 : NULL;
    /* K-LAYA-002 M1：归一化+缩放+偏置向量化（mean/var 保持 double 累加，与标量同口径） */
#if ST_HAVE_NEON
    float32x4_t vmean = vdupq_n_f32(meanf), vinv = vdupq_n_f32(invf);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(x + i), vmean);
        v = vmulq_f32(vmulq_f32(v, vinv), vld1q_f32(w + i));
        if (bb) v = vaddq_f32(v, vld1q_f32(bb + i));
        vst1q_f32(y + i, v);
    }
    for (; i < n; i++) {
        float v = (x[i] - meanf) * invf * w[i];
        if (bb) v += bb[i];
        y[i] = v;
    }
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
    __m256 vmean = _mm256_set1_ps(meanf), vinv = _mm256_set1_ps(invf);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        v = _mm256_mul_ps(_mm256_mul_ps(v, vinv), _mm256_loadu_ps(w + i));
        if (bb) v = _mm256_add_ps(v, _mm256_loadu_ps(bb + i));
        _mm256_storeu_ps(y + i, v);
    }
    for (; i < n; i++) {
        float v = (x[i] - meanf) * invf * w[i];
        if (bb) v += bb[i];
        y[i] = v;
    }
#else
    for (int i = 0; i < n; i++) {
        float v = (float)((x[i] - mean) * invf) * lay_at(m, W, (size_t)i);
        if (B) v += lay_at(m, B, (size_t)i);
        y[i] = v;
    }
#endif
}

static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

static void rope_table(float *cos_t, float *sin_t, int n, float theta) {
    for (int p = 0; p < n; p++) {
        for (int j = 0; j < LAYA_HD / 2; j++) {
            double inv = 1.0 / pow((double)theta, (2.0 * j) / LAYA_HD);
            float a = (float)(p * inv);
            float c = cosf(a), s = sinf(a);
            cos_t[(size_t)p * LAYA_HD + j] = c;
            sin_t[(size_t)p * LAYA_HD + j] = s;
            cos_t[(size_t)p * LAYA_HD + LAYA_HD / 2 + j] = c;
            sin_t[(size_t)p * LAYA_HD + LAYA_HD / 2 + j] = s;
        }
    }
}

/* in-place rope on one head vector (head_dim) at position pos */
static void rope_head(float *v, const float *cos_t, const float *sin_t, int pos) {
    const float *c = cos_t + (size_t)pos * LAYA_HD;
    const float *s = sin_t + (size_t)pos * LAYA_HD;
    float lo[LAYA_HD / 2], hi[LAYA_HD / 2];
    for (int j = 0; j < LAYA_HD / 2; j++) { lo[j] = v[j]; hi[j] = v[LAYA_HD / 2 + j]; }
    for (int j = 0; j < LAYA_HD / 2; j++) {
        v[j] = v[j] * c[j] + (-hi[j]) * s[j];
        v[LAYA_HD / 2 + j] = v[LAYA_HD / 2 + j] * c[LAYA_HD / 2 + j] + lo[j] * s[LAYA_HD / 2 + j];
    }
}

/* ---------------- load ---------------- */
LayaModel *laya_load(const char *pack_dir) {
    char pbin[1024], pman[1024];
    snprintf(pbin, sizeof(pbin), "%s/laya_f16.bin", pack_dir);
    snprintf(pman, sizeof(pman), "%s/laya_manifest.txt", pack_dir);
    FILE *fb = fopen(pbin, "rb");
    if (!fb) { fprintf(stderr, "[LAYA] cannot open %s\n", pbin); return NULL; }
    fseek(fb, 0, SEEK_END);
    long sz = ftell(fb);
    fseek(fb, 0, SEEK_SET);
    FILE *fm = fopen(pman, "r");
    if (!fm) { fprintf(stderr, "[LAYA] cannot open %s\n", pman); fclose(fb); return NULL; }
    int cap = 512, n = 0;
    LayaEnt *ent = (LayaEnt *)calloc((size_t)cap, sizeof(LayaEnt));
    char line[512];
    while (fgets(line, sizeof(line), fm)) {
        char dtype[24];
        char *tok = strtok(line, " \t\r\n");
        if (!tok) continue;
        LayaEnt e;
        memset(&e, 0, sizeof(e));
        snprintf(e.name, sizeof(e.name), "%s", tok);
        tok = strtok(NULL, " \t\r\n"); e.ndim = atoi(tok);
        for (int i = 0; i < e.ndim; i++) { tok = strtok(NULL, " \t\r\n"); e.ne[i] = atoi(tok); }
        tok = strtok(NULL, " \t\r\n"); snprintf(dtype, sizeof(dtype), "%s", tok);
        tok = strtok(NULL, " \t\r\n"); e.off = (size_t)strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\r\n"); e.nbytes = (size_t)strtoull(tok, NULL, 10);
        e.is_f32 = strstr(dtype, "32") ? 1 : 0;
        if (n >= cap) { cap <<= 1; ent = (LayaEnt *)realloc(ent, sizeof(LayaEnt) * (size_t)cap); }
        ent[n++] = e;
    }
    fclose(fm);
    /* K-LAYA-002 M5：载入策略再次收窄——1D（norm/bias）+ F32 小张量（temperature）
     * 展开/拷贝到 F32 池；**全部 2D 权重按行做 per-32 块对称量化，存入 int8 arena
     * + scale 池**，随后释放 843 MB 的 F16 blob ⇒ RSS 843 MB → ~474 MB。
     * 公理 collapse_to_classic + collapse_priority_rule（细粒度块 scale，避免单 scale
     * 不可逆失真）、conservation_law_under_collapse（以 J2 决策输出等价实测守门）。 */
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        size_t el = 1;
        for (int d = 0; d < e->ndim; d++) el *= (size_t)e->ne[d];
        e->n_elem = el;
        e->rowlen = (e->ndim >= 2) ? e->ne[1] : (e->ndim == 1 ? e->ne[0] : 1);
        if (e->rowlen < 1) e->rowlen = 1;
        e->nblk = (e->rowlen + LAYA_QB - 1) / LAYA_QB;
    }
    unsigned char *blob = (unsigned char *)malloc((size_t)sz);
    if (!blob) { fprintf(stderr, "[LAYA] OOM blob (%.0f MB)\n", sz / 1e6); fclose(fb); free(ent); return NULL; }
    if (fread(blob, 1, (size_t)sz, fb) != (size_t)sz) {
        fprintf(stderr, "[LAYA] blob read failed %s\n", pbin);
        free(blob); fclose(fb); free(ent); return NULL;
    }
    fclose(fb);
    size_t f32n = 0, i8n = 0, blkn = 0;
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        if (e->is_f32 || e->ndim <= 1) f32n += e->n_elem;          /* 展开/拷贝为 F32 */
        else {                                                     /* 2D → int8 + scales */
            size_t rows = e->n_elem / (size_t)e->rowlen;
            i8n  += e->n_elem;
            blkn += rows * (size_t)e->nblk;
        }
    }
    float  *f32pool = f32n ? (float *)malloc(sizeof(float) * f32n) : NULL;
    int8_t *i8pool  = i8n ? (int8_t *)malloc(i8n) : NULL;
    float  *qscpool = blkn ? (float *)malloc(sizeof(float) * blkn) : NULL;
    if ((f32n && !f32pool) || (i8n && !i8pool) || (blkn && !qscpool)) {
        fprintf(stderr, "[LAYA] OOM pools\n");
        free(blob); free(f32pool); free(i8pool); free(qscpool); free(ent); return NULL;
    }
    size_t f32cur = 0, i8cur = 0, sccur = 0;
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        const unsigned char *src = blob + e->off;
        if (e->is_f32) {                   /* F32 小张量（temperature）：拷贝到 F32 池 */
            float *dst = f32pool + f32cur;
            memcpy(dst, src, sizeof(float) * e->n_elem);
            e->f32 = dst; e->f16 = NULL; e->i8 = NULL;
            f32cur += e->n_elem;
        } else if (e->ndim == 1) {         /* 1D F16 → F32 池（norm/bias） */
            float *dst = f32pool + f32cur;
            const uint16_t *h = (const uint16_t *)src;
            for (size_t j = 0; j < e->n_elem; j++) dst[j] = f16_to_f32(h[j]);
            e->f32 = dst; e->f16 = NULL; e->i8 = NULL;
            f32cur += e->n_elem;
        } else {                           /* 2D F16 → int8 + per-32 块 scale */
            int8_t *q = i8pool + i8cur;
            float  *sc = qscpool + sccur;
            const uint16_t *h = (const uint16_t *)src;
            size_t rows = e->n_elem / (size_t)e->rowlen;
            for (size_t r = 0; r < rows; r++)
                lay_quant_f16_row(h + r * (size_t)e->rowlen, q + r * (size_t)e->rowlen,
                                  sc + r * (size_t)e->nblk, e->rowlen);
            e->i8 = q; e->qsc = sc; e->f16 = NULL; e->f32 = NULL;
            i8cur += e->n_elem;
            sccur += rows * (size_t)e->nblk;
        }
    }
    free(blob); blob = NULL;   /* F16 blob 不再需要 */
    LayaModel *m = (LayaModel *)calloc(1, sizeof(LayaModel));
    m->blob = NULL; m->blob_size = 0; m->ent = ent; m->n_ent = n;
    m->f32pool = f32pool; m->i8pool = i8pool; m->qscpool = qscpool;
    printf("[LAYA] loaded %d tensors (int8 %.0f MB + scales %.1f MB + F32 pool %.2f MB)\n",
           n, (double)i8n / 1e6, (double)(sizeof(float) * blkn) / 1e6,
           (double)(sizeof(float) * f32n) / 1e6);
    return m;
}

void laya_free(LayaModel *m) {
    if (!m) return;
    free(m->blob); free(m->f32pool); free(m->i8pool); free(m->qscpool); free(m->ent); free(m);
}

/* run attention for one position over selected keys; returns nothing, writes out_heads[d] */
static void attn_pos(const float *Qp, const float *K, const float *V, int n, int p,
                     int is_full, float *att, float *out_heads) {
    const float kscale = 1.0f / 8.0f;   /* head_dim^-0.5 */
    /* K-LAYA-002 M7：滑动窗口区间裁剪——滑窗层只遍历有效 key 区间 [lo,hi)，
     * 不再逐 j 判断 abs(p-j)>SLIDE（原实现三个 pass 仍扫全 n，浪费 ~2×）。
     * 数学上逐位等价：被裁剪的 j 原取 att=-1e30 ⇒ exp(-1e30-maxs)=0 ⇒ 对 max/sum/V 零贡献。 */
    int lo = 0, hi = n;
    if (!is_full) {
        lo = p - LAYA_SLIDE; if (lo < 0) lo = 0;
        hi = p + LAYA_SLIDE + 1; if (hi > n) hi = n;
    }
    const int m = hi - lo;
    for (int h = 0; h < LAYA_H; h++) {
        int off = h * LAYA_HD;
        const float *q = Qp + off;
        float maxs = -1e30f;
        for (int j = lo; j < hi; j++) {
            const float *k = K + (size_t)j * LAYA_D + off;
            float s = lay_dot(q, k, LAYA_HD) * kscale;   /* K-LAYA-002 M1：SIMD 点积 */
            att[j - lo] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            float e = expf(att[j] - maxs);
            att[j] = e; sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *oh = out_heads + off;
        memset(oh, 0, sizeof(float) * LAYA_HD);
        for (int j = 0; j < m; j++) {
            float w = att[j] * inv;
            if (w == 0.0f) continue;
            const float *v = V + (size_t)(lo + j) * LAYA_D + off;
            /* K-LAYA-002 M1：加权和向量化（oh += v * w） */
#if ST_HAVE_NEON
            int t = 0;
            for (; t + 4 <= LAYA_HD; t += 4) {
                float32x4_t o = vld1q_f32(oh + t);
                o = vfmaq_n_f32(o, vld1q_f32(v + t), w);
                vst1q_f32(oh + t, o);
            }
            for (; t < LAYA_HD; t++) oh[t] += w * v[t];
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
            __m256 vw = _mm256_set1_ps(w);
            int t = 0;
            for (; t + 8 <= LAYA_HD; t += 8)
                _mm256_storeu_ps(oh + t, _mm256_fmadd_ps(_mm256_loadu_ps(v + t), vw,
                                                         _mm256_loadu_ps(oh + t)));
            for (; t < LAYA_HD; t++) oh[t] += w * v[t];
#else
            for (int t = 0; t < LAYA_HD; t++) oh[t] += w * v[t];
#endif
        }
    }
}

/* K-LAYA-002 M3：分块 MatMul（Y[T,N] = X[T,K]·W[N,K]^T + B）。
 * 4 token × 2 输出行分块：权重行在 4 个 token 间复用、x 向量在 2 输出行间复用，
 * 内层 load:FMA 由 2:1 降到 0.75:1。板端微基准（真实形状 T=55）：
 * GEMV 8.0 → MatMul 12.1~13.2 GFLOP/s（1.51~1.66×），max|Δ|≈0（同容差）。
 * 按 token 块并行（各块写不同 Y 行，互不重叠 ⇒ 与串行同序、J3 位级门保持）。 */
static void lay_matmul(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                       const float *X, float *Y, int T, int K, int N) {
    if (W->i8) {   /* K-LAYA-002 M5：int8 权重 + int8 激活（逐行量化，按 token 并行） */
        int nblk = (K + LAYA_QB - 1) / LAYA_QB;
        int8_t *Xq = (int8_t *)malloc((size_t)T * (size_t)K);
        float  *Xs = (float *)malloc(sizeof(float) * (size_t)T * (size_t)nblk);
        if (!Xq || !Xs) { free(Xq); free(Xs); for (size_t i = 0; i < (size_t)T * N; i++) Y[i] = 0.f; return; }
#pragma omp parallel for schedule(static) if(T > 2)
        for (int t = 0; t < T; t++)
            lay_quant_act(X + (size_t)t * K, Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
        const int8_t *Wi = W->i8;
        const float  *Ws = W->qsc;
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
        /* K-LAYA-002 M5b：4 token × 1 输出行分块——权重行在 4 个 token 间复用
         * （板端微基准 1.46–1.87× over 逐行；int32 累加精确 ⇒ 与逐行位级同值）。 */
        if (K % 32 == 0) {   /* vdotq CHUNK 需 32 对齐；SCALE 尾块（< LAYA_QB）逐元素补 */
            int Kfull = (K / LAYA_QB) * LAYA_QB;
            int ntile = T / 4;
#pragma omp parallel for schedule(static) if(ntile > 2)
            for (int tb = 0; tb < ntile; tb++) {
                int t = tb * 4;
                const int8_t *x0 = Xq + (size_t)(t + 0) * K, *x1 = Xq + (size_t)(t + 1) * K;
                const int8_t *x2 = Xq + (size_t)(t + 2) * K, *x3 = Xq + (size_t)(t + 3) * K;
                const float  *xs0 = Xs + (size_t)(t + 0) * nblk, *xs1 = Xs + (size_t)(t + 1) * nblk;
                const float  *xs2 = Xs + (size_t)(t + 2) * nblk, *xs3 = Xs + (size_t)(t + 3) * nblk;
                for (int o = 0; o < N; o++) {
                    const int8_t *w  = Wi + (size_t)o * K;
                    const float  *ws = Ws + (size_t)o * nblk;
                    float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
                    for (int b = 0; b < Kfull; b += LAYA_QB) {
                        /* 注（K-LAYA-002 M8 实测 2026-10-11）：每 token 双交替 int32 累加链
                         * （8 个累加器）在 A76 上**退化 ~4.6%**（寄存器压力，同 M2b/t4r2 教训）
                         * ⇒ 已回退，保留单链。 */
                        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
                        for (int c = 0; c < LAYA_QB; c += 32) {   /* 块内 32 字节 CHUNK */
                            int8x16_t w0 = vld1q_s8(w + b + c), w1 = vld1q_s8(w + b + c + 16);
                            a0 = vdotq_s32(a0, w0, vld1q_s8(x0 + b + c));  a0 = vdotq_s32(a0, w1, vld1q_s8(x0 + b + c + 16));
                            a1 = vdotq_s32(a1, w0, vld1q_s8(x1 + b + c));  a1 = vdotq_s32(a1, w1, vld1q_s8(x1 + b + c + 16));
                            a2 = vdotq_s32(a2, w0, vld1q_s8(x2 + b + c));  a2 = vdotq_s32(a2, w1, vld1q_s8(x2 + b + c + 16));
                            a3 = vdotq_s32(a3, w0, vld1q_s8(x3 + b + c));  a3 = vdotq_s32(a3, w1, vld1q_s8(x3 + b + c + 16));
                        }
                        int bi = b / LAYA_QB;
                        float wsb = ws[bi];
                        s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(a0), wsb * xs0[bi]);
                        s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(a1), wsb * xs1[bi]);
                        s2 = vmlaq_n_f32(s2, vcvtq_f32_s32(a2), wsb * xs2[bi]);
                        s3 = vmlaq_n_f32(s3, vcvtq_f32_s32(a3), wsb * xs3[bi]);
                    }
                    float r0 = vaddvq_f32(s0), r1 = vaddvq_f32(s1);
                    float r2 = vaddvq_f32(s2), r3 = vaddvq_f32(s3);
                    for (int q = Kfull; q < K; q++) {   /* SCALE 尾块补（本模型仅 K=2624 时 64 个） */
                        int bi = q / LAYA_QB;
                        float wq = (float)w[q] * ws[bi];
                        r0 += wq * (float)x0[q] * xs0[bi];
                        r1 += wq * (float)x1[q] * xs1[bi];
                        r2 += wq * (float)x2[q] * xs2[bi];
                        r3 += wq * (float)x3[q] * xs3[bi];
                    }
                    Y[(size_t)(t + 0) * N + o] = r0;
                    Y[(size_t)(t + 1) * N + o] = r1;
                    Y[(size_t)(t + 2) * N + o] = r2;
                    Y[(size_t)(t + 3) * N + o] = r3;
                }
            }
            for (int t = (T / 4) * 4; t < T; t++)   /* token 尾（T%4 ≤ 3） */
                for (int o = 0; o < N; o++)
                    Y[(size_t)t * N + o] = lay_dot_i8(Wi + (size_t)o * K, Ws + (size_t)o * nblk,
                                                      Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
        } else
#endif
        {
#pragma omp parallel for schedule(static) if(T > 2)
            for (int t = 0; t < T; t++)
                for (int o = 0; o < N; o++)
                    Y[(size_t)t * N + o] =
                        lay_dot_i8(Wi + (size_t)o * K, Ws + (size_t)o * nblk,
                                   Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
        }
        free(Xq); free(Xs);
        if (B)
            for (int t = 0; t < T; t++)
                for (int o = 0; o < N; o++)
                    Y[(size_t)t * N + o] += lay_at(m, B, (size_t)o);
        return;
    }
    const uint16_t *Wb = W->f16;
    if (Wb) {
        int ntile = T / 4;
#if ST_HAVE_NEON
#pragma omp parallel for schedule(static) if(ntile > 2)
        for (int tb = 0; tb < ntile; tb++) {
            int t = tb * 4;
            const float *x0 = X + (size_t)(t + 0) * K, *x1 = X + (size_t)(t + 1) * K;
            const float *x2 = X + (size_t)(t + 2) * K, *x3 = X + (size_t)(t + 3) * K;
            int o = 0;
            for (; o + 2 <= N; o += 2) {
                const uint16_t *w0 = Wb + (size_t)(o + 0) * K, *w1 = Wb + (size_t)(o + 1) * K;
                float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0), a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
                float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0), a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
                int k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t v0 = vcvt_f32_f16(vld1_f16((const __fp16 *)(w0 + k)));
                    float32x4_t v1 = vcvt_f32_f16(vld1_f16((const __fp16 *)(w1 + k)));
                    float32x4_t u0 = vld1q_f32(x0 + k), u1 = vld1q_f32(x1 + k);
                    float32x4_t u2 = vld1q_f32(x2 + k), u3 = vld1q_f32(x3 + k);
                    a00 = vfmaq_f32(a00, v0, u0); a01 = vfmaq_f32(a01, v0, u1);
                    a02 = vfmaq_f32(a02, v0, u2); a03 = vfmaq_f32(a03, v0, u3);
                    a10 = vfmaq_f32(a10, v1, u0); a11 = vfmaq_f32(a11, v1, u1);
                    a12 = vfmaq_f32(a12, v1, u2); a13 = vfmaq_f32(a13, v1, u3);
                }
                float s0[4] = { vaddvq_f32(a00), vaddvq_f32(a01), vaddvq_f32(a02), vaddvq_f32(a03) };
                float s1[4] = { vaddvq_f32(a10), vaddvq_f32(a11), vaddvq_f32(a12), vaddvq_f32(a13) };
                for (int r = 0; r < 4; r++) {
                    const float *xr = X + (size_t)(t + r) * K;
                    for (int q = k; q < K; q++) {   /* K 尾（本模型 K∈{1024,2624,4096}，均 4 的倍数 ⇒ 空） */
                        float xv = xr[q];
                        s0[r] += f16_to_f32(w0[q]) * xv;
                        s1[r] += f16_to_f32(w1[q]) * xv;
                    }
                    Y[(size_t)(t + r) * N + o]     = s0[r];
                    Y[(size_t)(t + r) * N + o + 1] = s1[r];
                }
            }
            for (; o < N; o++)
                for (int r = 0; r < 4; r++)
                    Y[(size_t)(t + r) * N + o] = lay_dot_f16(Wb + (size_t)o * K, X + (size_t)(t + r) * K, K);
        }
#else
        (void)ntile;
#endif
        for (int t = (T / 4) * 4; t < T; t++)   /* token 尾（T%4 ≤ 3） */
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] = lay_dot_f16(Wb + (size_t)o * K, X + (size_t)t * K, K);
    } else {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] = lay_dot(W->f32 + (size_t)o * K, X + (size_t)t * K, K);
    }
    if (B) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] += lay_at(m, B, (size_t)o);
    }
}

int laya_forward(LayaModel *m, const int *ids, int n,
                 const int *markers, int nm, int qtype, float *logits_out) {
    if (n <= 0 || nm <= 0) return -1;
    char nmb[128];
    const LayaEnt *e_tok  = lay_find(m, "encoder.embeddings.tok_embeddings.weight");
    const LayaEnt *e_embn = lay_find(m, "encoder.embeddings.norm.weight");
    const LayaEnt *e_finn = lay_find(m, "encoder.final_norm.weight");
    const LayaEnt *e_te   = lay_find(m, "type_emb.weight");
    if (!e_tok || !e_embn || !e_finn || !e_te) return -1;

    size_t nd = (size_t)n * LAYA_D;
    float *x    = malloc(sizeof(float) * nd);
    float *Q    = malloc(sizeof(float) * nd);
    float *K    = malloc(sizeof(float) * nd);
    float *V    = malloc(sizeof(float) * nd);
    float *nrm  = malloc(sizeof(float) * nd);
    float *nn_a = malloc(sizeof(float) * nd);                      /* [n,D] norm 输出 */
    float *qkv_a = malloc(sizeof(float) * nd * 3);                 /* [n,3D] Wqkv / in_proj 输出 */
    float *wi_a = malloc(sizeof(float) * (size_t)n * 2 * LAYA_INTER);
    float *mh_a = malloc(sizeof(float) * (size_t)n * LAYA_INTER);
    float *ff_a = malloc(sizeof(float) * (size_t)n * LAYA_HFFN);
    float *tmp_a = malloc(sizeof(float) * nd);
    float *aou_a = malloc(sizeof(float) * nd);   /* M5c: [n,D] attention 输出，供 Wo/out_proj 分块 MatMul */
    float *sc  = malloc(sizeof(float) * LAYA_D);
    float *sc2 = malloc(sizeof(float) * LAYA_D);
    float *cosF = malloc(sizeof(float) * nd);
    float *sinF = malloc(sizeof(float) * nd);
    float *cosS = malloc(sizeof(float) * nd);
    float *sinS = malloc(sizeof(float) * nd);

    /* K-LAYA-002 M4/M5c：位置级并行的线程私有 scratch（只余 attention 的 att）；
     * attention 输出改为 token 维共享缓冲 aou_a，Wo/out_proj 走 lay_matmul 分块路径。
     * 逐 p/逐块计算与串行同序 ⇒ 结果与线程数无关（J3 位级门）。 */
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
    if (nth < 1) nth = 1;
    if (nth > LAYA_NT_MAX) nth = LAYA_NT_MAX;
#endif
    if (n < 4) nth = 1;
    float *att_t[LAYA_NT_MAX];
    for (int t = 0; t < LAYA_NT_MAX; t++) att_t[t] = NULL;
    for (int t = 0; t < nth; t++) {
        att_t[t] = malloc(sizeof(float) * (size_t)n);
        if (!att_t[t]) { nth = t; break; }
    }
    int ok = x && Q && K && V && nrm && nn_a && qkv_a && wi_a && mh_a && ff_a && tmp_a && aou_a &&
             sc && sc2 && cosF && sinF && cosS && sinS && nth >= 1;
    if (!ok) goto done;
    if (g_timing < 0) g_timing = getenv("VLLM_LAYA_TIMING") ? 1 : 0;
    double _tw0 = tmark();
    { double _t = tmark();
    rope_table(cosF, sinF, n, 160000.0f);
    rope_table(cosS, sinS, n, 10000.0f);
    tacc(TT_ROPE, _t); }

    {
    double _t = tmark();
    for (int p = 0; p < n; p++)
        for (int i = 0; i < LAYA_D; i++)
            x[(size_t)p * LAYA_D + i] = lay_at(m, e_tok, (size_t)ids[p] * LAYA_D + i);
    for (int p = 0; p < n; p++)
        lay_ln(m, e_embn, NULL, x + (size_t)p * LAYA_D, x + (size_t)p * LAYA_D, LAYA_D);
    tacc(TT_EMB, _t);
    }

    /* K-LAYA-002 L1b：权重指针一次性预解析（消除每位置/每层循环内的
     * snprintf + lay_find 线性扫描；对结果无影响，纯寻址开销消除）。 */
    const LayaEnt *L_attn_norm[LAYA_NL], *L_wqkv[LAYA_NL], *L_wo_attn[LAYA_NL];
    const LayaEnt *L_mlp_norm[LAYA_NL], *L_wi[LAYA_NL], *L_wo_mlp[LAYA_NL];
    for (int l = 0; l < LAYA_NL; l++) {
        L_attn_norm[l] = NULL;
        if (l > 0) {                       /* 层 0 attn_norm = Identity（无该张量） */
            snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn_norm.weight", l);
            L_attn_norm[l] = lay_find(m, nmb);
        }
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn.Wqkv.weight", l);
        L_wqkv[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn.Wo.weight", l);
        L_wo_attn[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp_norm.weight", l);
        L_mlp_norm[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp.Wi.weight", l);
        L_wi[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp.Wo.weight", l);
        L_wo_mlp[l] = lay_find(m, nmb);
        if ((l > 0 && !L_attn_norm[l]) || !L_wqkv[l] || !L_wo_attn[l] ||
            !L_mlp_norm[l] || !L_wi[l] || !L_wo_mlp[l]) { ok = 0; goto done; }
    }

    for (int l = 0; l < LAYA_NL; l++) {
        int is_full = (l % 3 == 0);
        const float *cos_t = is_full ? cosF : cosS;
        const float *sin_t = is_full ? sinF : sinS;

        /* A. attn_norm（层 0 = Identity）+ Wqkv MatMul（权重行在 token 间复用） */
        { double _t = tmark();
        if (l == 0) memcpy(nn_a, x, sizeof(float) * nd);
        else
#pragma omp parallel for schedule(static) if(nth > 1)
            for (int p = 0; p < n; p++)
                lay_ln(m, L_attn_norm[l], NULL, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wqkv[l], NULL, nn_a, qkv_a, n, LAYA_D, 3 * LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *qkv = qkv_a + (size_t)p * 3 * LAYA_D;
            for (int h = 0; h < LAYA_H; h++) {
                int off = h * LAYA_HD;
                memcpy(Q + (size_t)p * LAYA_D + off, qkv + off, sizeof(float) * LAYA_HD);
                memcpy(K + (size_t)p * LAYA_D + off, qkv + LAYA_D + off, sizeof(float) * LAYA_HD);
                memcpy(V + (size_t)p * LAYA_D + off, qkv + 2 * LAYA_D + off, sizeof(float) * LAYA_HD);
                rope_head(Q + (size_t)p * LAYA_D + off, cos_t, sin_t, p);
                rope_head(K + (size_t)p * LAYA_D + off, cos_t, sin_t, p);
            }
        }
        tacc(TT_ROPE, _t); }
        /* B. attention（位置间混合，逐位置 → aou_a）+ Wo 走分块 MatMul（M5c） */
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            attn_pos(Q + (size_t)p * LAYA_D, K, V, n, p, is_full, att_t[OTID], aou_a + (size_t)p * LAYA_D);
        tacc(TT_ATTN, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wo_attn[l], NULL, aou_a, tmp_a, n, LAYA_D, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
        /* C. mlp：norm → Wi MatMul → GELU-gate → Wo MatMul → 残差 */
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            lay_ln(m, L_mlp_norm[l], NULL, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wi[l], NULL, nn_a, wi_a, n, LAYA_D, 2 * LAYA_INTER);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *wi = wi_a + (size_t)p * 2 * LAYA_INTER;
            float *mh = mh_a + (size_t)p * LAYA_INTER;
            for (int i = 0; i < LAYA_INTER; i++) mh[i] = gelu_erf(wi[i]) * wi[LAYA_INTER + i];
        }
        tacc(TT_ACT, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wo_mlp[l], NULL, mh_a, tmp_a, n, LAYA_INTER, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
    }
    { double _t = tmark();
    for (int p = 0; p < n; p++)
        lay_ln(m, e_finn, NULL, x + (size_t)p * LAYA_D, x + (size_t)p * LAYA_D, LAYA_D);

    /* type embedding added to every position */
    for (int p = 0; p < n; p++)
        for (int i = 0; i < LAYA_D; i++)
            x[(size_t)p * LAYA_D + i] += lay_at(m, e_te, (size_t)qtype * LAYA_D + i);
    tacc(TT_NORM, _t); }

    /* 2x transformer encoder layer (norm_first, relu) */
    for (int hl = 0; hl < 2; hl++) {
        char p1[96], p2[96], pi[96], po[96], l1[96], l2[96], nb[96];
        snprintf(p1, sizeof(p1), "head.layers.%d.norm1.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.norm1.bias", hl);
        const LayaEnt *n1w = lay_find(m, p1), *n1b = lay_find(m, nb);
        snprintf(p2, sizeof(p2), "head.layers.%d.norm2.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.norm2.bias", hl);
        const LayaEnt *n2w = lay_find(m, p2), *n2b = lay_find(m, nb);
        snprintf(pi, sizeof(pi), "head.layers.%d.self_attn.in_proj_weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.self_attn.in_proj_bias", hl);
        const LayaEnt *ipw = lay_find(m, pi), *ipb = lay_find(m, nb);
        snprintf(po, sizeof(po), "head.layers.%d.self_attn.out_proj.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.self_attn.out_proj.bias", hl);
        const LayaEnt *opw = lay_find(m, po), *opb = lay_find(m, nb);
        snprintf(l1, sizeof(l1), "head.layers.%d.linear1.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.linear1.bias", hl);
        const LayaEnt *f1w = lay_find(m, l1), *f1b = lay_find(m, nb);
        snprintf(l2, sizeof(l2), "head.layers.%d.linear2.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.linear2.bias", hl);
        const LayaEnt *f2w = lay_find(m, l2), *f2b = lay_find(m, nb);
        if (!n1w || !n1b || !n2w || !n2b || !ipw || !ipb || !opw || !opb || !f1w || !f1b || !f2w || !f2b)
            { ok = 0; goto done; }

        { double _t = tmark();
        for (int p = 0; p < n; p++)
            lay_ln(m, n1w, n1b, x + (size_t)p * LAYA_D, nrm + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, ipw, ipb, nrm, qkv_a, n, LAYA_D, 3 * LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *qkv = qkv_a + (size_t)p * 3 * LAYA_D;
            memcpy(Q + (size_t)p * LAYA_D, qkv, sizeof(float) * LAYA_D);
            memcpy(K + (size_t)p * LAYA_D, qkv + LAYA_D, sizeof(float) * LAYA_D);
            memcpy(V + (size_t)p * LAYA_D, qkv + 2 * LAYA_D, sizeof(float) * LAYA_D);
        }
        tacc(TT_ROPE, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            attn_pos(Q + (size_t)p * LAYA_D, K, V, n, p, 1, att_t[OTID], aou_a + (size_t)p * LAYA_D);
        tacc(TT_ATTN, _t); }
        { double _t = tmark();
        lay_matmul(m, opw, opb, aou_a, tmp_a, n, LAYA_D, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            lay_ln(m, n2w, n2b, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, f1w, f1b, nn_a, ff_a, n, LAYA_D, LAYA_HFFN);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            float *ff1 = ff_a + (size_t)p * LAYA_HFFN;
            for (int i = 0; i < LAYA_HFFN; i++) ff1[i] = relu_f(ff1[i]);
        }
        tacc(TT_ACT, _t); }
        { double _t = tmark();
        lay_matmul(m, f2w, f2b, ff_a, tmp_a, n, LAYA_HFFN, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
    }

    /* scorer */
    const LayaEnt *s0w = lay_find(m, "scorer.0.weight");
    const LayaEnt *s0b = lay_find(m, "scorer.0.bias");
    const LayaEnt *s1w = lay_find(m, "scorer.1.weight");
    const LayaEnt *s1b = lay_find(m, "scorer.1.bias");
    const LayaEnt *s3w = lay_find(m, "scorer.3.weight");
    const LayaEnt *s3b = lay_find(m, "scorer.3.bias");
    if (!s0w || !s0b || !s1w || !s1b || !s3w || !s3b) { ok = 0; goto done; }
    { double _t = tmark();
    for (int k = 0; k < nm; k++) {
        const float *mv = x + (size_t)markers[k] * LAYA_D;
        lay_ln(m, s0w, s0b, mv, sc, LAYA_D);
        lay_linear(m, s1w, s1b, sc, sc2, LAYA_D, LAYA_D);
        for (int i = 0; i < LAYA_D; i++) sc[i] = gelu_erf(sc2[i]);
        lay_linear(m, s3w, s3b, sc, &logits_out[k], 1, LAYA_D);
    }
    tacc(TT_SCO, _t); }

    if (g_timing == 1) {
        double rest = (_tw0 > 0.0) ? (lay_wtime() - _tw0) : 0.0;
        double sum = 0.0; for (int i = 0; i < TT_N; i++) sum += g_tt[i];
        fprintf(stderr, "[LAYA-TIME] n=%d total=%.1fms | emb=%.1f norm=%.1f mm=%.1f rope=%.1f attn=%.1f act=%.1f resid=%.1f sco=%.1f | sum=%.1f (%.0f%%) misc=%.1f\n",
                n, rest * 1e3, g_tt[TT_EMB]*1e3, g_tt[TT_NORM]*1e3, g_tt[TT_MM]*1e3, g_tt[TT_ROPE]*1e3,
                g_tt[TT_ATTN]*1e3, g_tt[TT_ACT]*1e3, g_tt[TT_RESID]*1e3, g_tt[TT_SCO]*1e3,
                sum * 1e3, rest > 0.0 ? 100.0 * sum / rest : 0.0, (rest - sum) * 1e3);
        for (int i = 0; i < TT_N; i++) g_tt[i] = 0.0;
        g_tt_total += rest;
    }

done:
    free(x); free(Q); free(K); free(V); free(nrm);
    free(nn_a); free(qkv_a); free(wi_a); free(mh_a); free(ff_a); free(tmp_a); free(aou_a);
    free(sc); free(sc2);
    free(cosF); free(sinF); free(cosS); free(sinS);
    for (int t = 0; t < LAYA_NT_MAX; t++) free(att_t[t]);
    return ok ? 0 : -1;
}

/* K-LAYA-001: dump ModernBERT token ids for each line of a file (gate input). */
int laya_tok_dump(const char *dir, const char *path) {
    static QwenTokenizer tok;
    if (qwen_tokenizer_load(&tok, dir) != 0) {
        fprintf(stderr, "[LTOK] tokenizer load failed: %s\n", dir);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[LTOK] cannot open %s\n", path); return -1; }
    static char line[1 << 20];
    int i = 0;
    while (fgets(line, sizeof(line), f)) {
        int L = (int)strlen(line);
        while (L > 0 && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = '\0';
        int *ids = (int *)malloc(sizeof(int) * 65536);
        int n = qwen_tokenizer_encode_gpt2(&tok, line, ids, 65536);
        printf("[LTOK] i=%d n=%d:", i, n);
        for (int j = 0; j < n; j++) printf(" %d", ids[j]);
        printf("\n");
        free(ids);
        i++;
    }
    fclose(f);
    return 0;
}

int laya_run_cases(const char *pack_dir, const char *cases_path) {
    LayaModel *m = laya_load(pack_dir);
    if (!m) return -1;
    FILE *f = fopen(cases_path, "r");
    if (!f) { fprintf(stderr, "[LAYA] cannot open %s\n", cases_path); laya_free(m); return -1; }
    char line[1 << 16];
    int qi = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        char *tok = strtok(p, " \t\r\n");
        if (!tok) continue;
        int qtype = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); int n = atoi(tok);
        int ids[8192];
        for (int i = 0; i < n; i++) { tok = strtok(NULL, " \t\r\n"); ids[i] = atoi(tok); }
        tok = strtok(NULL, " \t\r\n"); int nmm = atoi(tok);
        int mk[256];
        for (int i = 0; i < nmm; i++) { tok = strtok(NULL, " \t\r\n"); mk[i] = atoi(tok); }
        float lg[256];
        if (laya_forward(m, ids, n, mk, nmm, qtype, lg) != 0) {
            printf("[LAYA] q=%d FAILED\n", qi);
        } else {
            printf("[LAYA] q=%d n=%d nm=%d logits=[", qi, n, nmm);
            for (int i = 0; i < nmm; i++) printf("%s%.6f", i ? ", " : "", lg[i]);
            printf("]\n");
        }
        qi++;
    }
    fclose(f);
    laya_free(m);
    return 0;
}

/* ================= K-LAYA-001: end-to-end decide ================= */
#define LAYA_TOK_CLS  50281
#define LAYA_TOK_SEP  50282
#define LAYA_TOK_MASK 50284
#define LAYA_HEAD_MAXLEN 192
#define LAYA_MAXLEN      512

static char *lay_read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t r = fread(b, 1, (size_t)sz, f);
    b[r] = '\0';
    fclose(f);
    return b;
}

static char *lay_remove_all(const char *src, const char *pat) {
    size_t pl = strlen(pat), sl = strlen(src), o = 0;
    char *out = (char *)malloc(sl + 1);
    const char *p = src;
    while (*p) {
        if (pl && strncmp(p, pat, pl) == 0) { p += pl; continue; }
        out[o++] = *p++;
    }
    out[o] = '\0';
    return out;
}

static int lay_temp_for(VJson *byopt, int qt, int k, double *out) {
    char key[32];
    const char *nm = (qt == 0) ? "choice" : (qt == 1) ? "score" : "noul";
    const char *sz = (k <= 2) ? "2" : (k <= 5) ? "3-5" : (k <= 10) ? "6-10" : "11+";
    snprintf(key, sizeof(key), "%s:%s", nm, sz);
    VJson *v = byopt ? vjson_obj_get(byopt, key) : NULL;
    if (v && v->type == VJ_NUMBER) { *out = vjson_num(v); return 1; }
    return 0;
}

/* render one option i (0-based) into a freshly malloc'd string */
static char *lay_option(const char *tname, VJson *crit, int i, int *know_k, const char **keys) {
    char buf[1024];
    if (strcmp(tname, "choice") == 0 && crit) {
        if (crit->type == VJ_OBJECT) {
            *know_k = (int)crit->u.obj.n;
            const char *key = crit->u.obj.pairs[i].key;
            VJson *v = crit->u.obj.pairs[i].val;
            if (keys) keys[i] = key;
            if (v && v->type == VJ_STRING && vjson_str(v)[0])
                snprintf(buf, sizeof(buf), "%s: %s", key, vjson_str(v));
            else
                snprintf(buf, sizeof(buf), "%s", key);
            return strdup(buf);
        }
        if (crit->type == VJ_ARRAY) {
            *know_k = (int)vjson_array_len(crit);
            const char *key = vjson_str(vjson_array_get(crit, (size_t)i));
            if (keys) keys[i] = key;
            snprintf(buf, sizeof(buf), "%s", key ? key : "");
            return strdup(buf);
        }
    }
    if (strcmp(tname, "score") == 0 && crit && crit->type == VJ_ARRAY) {
        *know_k = (int)vjson_array_len(crit);
        const char *c = vjson_str(vjson_array_get(crit, (size_t)i));
        snprintf(buf, sizeof(buf), "level %d: %s", i, c ? c : "");
        return strdup(buf);
    }
    /* noul */
    *know_k = 2;
    const char *fd = "no, the statement does not hold";
    const char *td = "yes, the statement holds";
    if (crit && crit->type == VJ_OBJECT) {
        VJson *f = vjson_obj_get(crit, "false");
        VJson *t = vjson_obj_get(crit, "true");
        if (f && f->type == VJ_STRING) fd = vjson_str(f);
        if (t && t->type == VJ_STRING) td = vjson_str(t);
    }
    snprintf(buf, sizeof(buf), "%s: %s", i == 0 ? "false" : "true", i == 0 ? fd : td);
    return strdup(buf);
}

/* ================================================================
 * Runner: model + tokenizer + temperature config, loaded once and reused.
 * Backs both the CLI (--laya-decide) and HTTP /v1/systemone (the HTTP server
 * must not reload the ~1.7 GB arena per request).
 * ================================================================ */
struct LayaRunner {
    LayaModel     *m;
    QwenTokenizer  tok;
    VJson         *cfg;      /* owned parsed rl_agent_config.json (may be NULL) */
    char          *cfg_txt;  /* owned raw text */
    VJson         *byopt;    /* borrowed: cfg.temperature_by_options (may be NULL) */
    double         tarr[3];
};

LayaRunner *laya_runner_new(const char *pack_dir) {
    LayaRunner *r = (LayaRunner *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->m = laya_load(pack_dir);
    if (!r->m) { free(r); return NULL; }
    if (qwen_tokenizer_load(&r->tok, pack_dir) != 0) {
        fprintf(stderr, "[LAYA] cannot load tokenizer from %s\n", pack_dir);
        laya_free(r->m); free(r); return NULL;
    }
    char cfgp[1024];
    snprintf(cfgp, sizeof(cfgp), "%s/rl_agent_config.json", pack_dir);
    r->cfg_txt = lay_read_file(cfgp);
    r->cfg = r->cfg_txt ? vjson_parse(r->cfg_txt) : NULL;
    r->tarr[0] = r->tarr[1] = r->tarr[2] = 1.0;
    VJson *t3 = r->cfg ? vjson_obj_get(r->cfg, "temperature") : NULL;
    if (t3 && t3->type == VJ_ARRAY)
        for (size_t i = 0; i < 3 && i < vjson_array_len(t3); i++)
            r->tarr[i] = vjson_num(vjson_array_get(t3, i));
    r->byopt = r->cfg ? vjson_obj_get(r->cfg, "temperature_by_options") : NULL;
    return r;
}

void laya_runner_free(LayaRunner *r) {
    if (!r) return;
    if (r->m) laya_free(r->m);
    qwen_tokenizer_free(&r->tok);
    if (r->cfg) vjson_free(r->cfg);
    free(r->cfg_txt);
    free(r);
}

/* tiny bounds-checked output writer (JSON assembled by hand, as before) */
typedef struct { char *b; size_t cap, n; int ovf; } LayaW;
static void w_raw(LayaW *w, const char *s) {
    size_t l = strlen(s);
    if (w->n + l + 1 > w->cap) { w->ovf = 1; return; }
    memcpy(w->b + w->n, s, l);
    w->n += l;
    w->b[w->n] = '\0';
}
static void w_fmt(LayaW *w, const char *fmt, ...) {
    char t[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    w_raw(w, t);
}

int laya_runner_decide(LayaRunner *r, const char *req_text, char *out, size_t cap) {
    if (cap) out[0] = '\0';
    VJson *root = vjson_parse(req_text);
    if (!root) { fprintf(stderr, "[LAYA] bad JSON request\n"); return -1; }
    VJson *questions = vjson_obj_get(root, "questions");
    if (!questions || questions->type != VJ_OBJECT) {
        fprintf(stderr, "[LAYA] request needs a 'questions' object\n");
        vjson_free(root);
        return -1;
    }
    VJson *state = vjson_obj_get(root, "state");

    char *state_str;
    if (state && state->type == VJ_STRING) {
        state_str = strdup(vjson_str(state));
    } else {
        size_t scap = 1 << 20;
        char *b = (char *)malloc(scap);
        if (!state || vjson_serialize(state, b, scap) == 0) b[0] = '\0';
        state_str = b;
    }
    char *state_clean = lay_remove_all(state_str, "[MASK]");

    int *ids = (int *)malloc(sizeof(int) * 8192);
    int *tmp = (int *)malloc(sizeof(int) * 8192);
    int *mk  = (int *)malloc(sizeof(int) * 128);
    float *lg = (float *)malloc(sizeof(float) * 128);

    LayaW w = { out, cap, 0, 0 };
    w_raw(&w, "{\"model\":\"laya\",\"answers\":{");
    for (size_t qi = 0; qi < questions->u.obj.n; qi++) {
        const char *qid = questions->u.obj.pairs[qi].key;
        VJson *q = questions->u.obj.pairs[qi].val;
        VJson *jt = vjson_obj_get(q, "type");
        const char *tname = (jt && jt->type == VJ_STRING) ? vjson_str(jt) : "noul";
        int qt = (strcmp(tname, "choice") == 0) ? 0 : (strcmp(tname, "score") == 0) ? 1 : 2;
        VJson *jins = vjson_obj_get(q, "instructions");
        const char *ins = (jins && jins->type == VJ_STRING) ? vjson_str(jins) : "";
        VJson *crit = vjson_obj_get(q, "criteria");

        /* options */
        char *opts[64];
        const char *keys[64];
        int have_opts = 0;
        for (int i = 0; i < 64; i++) keys[i] = "";
        int k = 0;
        for (int i = 0; i < 64; i++) {
            if (i > 0 && i >= k) break;      /* do not probe past the criteria array */
            int kk = 0;
            char *o = lay_option(tname, crit, i, &kk, keys);
            if (i == 0) { k = kk; if (k <= 0 || k > 64) { free(o); break; } }
            opts[i] = o;
            have_opts = i + 1;
        }
        if (k <= 0 || k > 64) {
            w_fmt(&w, "\"%s\":{\"error\":\"bad criteria\"}", qid);
            continue;
        }

        /* option token ids: [MASK] + tok(" " + opt)[:48] */
        int *optid[64];
        int optn[64], tot = 0, have_optid = 0;
        for (int i = 0; i < k; i++) {
            char *o = lay_remove_all(opts[i], "[MASK]");
            char *sp = (char *)malloc(strlen(o) + 2);
            sp[0] = ' ';
            strcpy(sp + 1, o);
            optid[i] = (int *)malloc(sizeof(int) * 64);
            have_optid = i + 1;
            int n = qwen_tokenizer_encode_gpt2(&r->tok, sp, optid[i] + 0, 48);
            /* prepend MASK: shift */
            for (int j = n; j > 0; j--) optid[i][j] = optid[i][j - 1];
            optid[i][0] = LAYA_TOK_MASK;
            optn[i] = n + 1;
            tot += optn[i];
            free(sp); free(o);
        }
        int budget = LAYA_HEAD_MAXLEN - tot;
        if (budget < 16) {
            int per = (LAYA_HEAD_MAXLEN - 16) / k;
            if (per < 4) per = 4;
            tot = 0;
            for (int i = 0; i < k; i++) { if (optn[i] > per) optn[i] = per; tot += optn[i]; }
            budget = LAYA_HEAD_MAXLEN - tot;
        }
        if (budget < 8) budget = 8;

        char hbuf[4096];
        char *ins_clean = lay_remove_all(ins, "[MASK]");
        snprintf(hbuf, sizeof(hbuf), "%s question: %s", tname, ins_clean);
        int hn = qwen_tokenizer_encode_gpt2(&r->tok, hbuf, tmp, 2048);
        if (hn > budget) hn = budget;

        int n = 0, nmk = 0;
        ids[n++] = LAYA_TOK_CLS;
        for (int i = 0; i < hn && n < 8000; i++) ids[n++] = tmp[i];
        ids[n++] = LAYA_TOK_SEP;
        for (int i = 0; i < k; i++) {
            mk[nmk++] = n;
            for (int j = 0; j < optn[i] && n < 8000; j++) ids[n++] = optid[i][j];
        }
        ids[n++] = LAYA_TOK_SEP;
        int room = LAYA_MAXLEN - n - 1;
        if (room < 0) room = 0;
        int sn = qwen_tokenizer_encode_gpt2(&r->tok, state_clean, tmp, 8192);
        int use = sn < room ? sn : room;
        for (int i = 0; i < use && n < 8191; i++) ids[n++] = tmp[i];
        ids[n++] = LAYA_TOK_SEP;
        if (n > LAYA_MAXLEN) n = LAYA_MAXLEN;
        int nmark = 0;
        for (int i = 0; i < nmk; i++) if (mk[i] < n) nmark++;

        int fwd_ok = (laya_forward(r->m, ids, n, mk, nmark, qt, lg) == 0);

        free(ins_clean);
        for (int i = 0; i < have_optid; i++) free(optid[i]);
        for (int i = 0; i < have_opts; i++) free(opts[i]);

        if (!fwd_ok) {
            w_fmt(&w, "\"%s\":{\"error\":\"forward failed\"}", qid);
            continue;
        }
        double T = r->tarr[qt], tv;
        if (lay_temp_for(r->byopt, qt, k, &tv)) T = tv;
        double mx = lg[0];
        for (int i = 1; i < k; i++) if (lg[i] > mx) mx = lg[i];
        double sum = 0.0, p[64];
        for (int i = 0; i < k; i++) { p[i] = exp(((double)lg[i] - mx) / T); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] /= sum;

        if (qi) w_raw(&w, ",");
        if (qt == 0) {
            int best = 0;
            for (int i = 1; i < k; i++) if (p[i] > p[best]) best = i;
            double ent = 0.0;
            for (int i = 0; i < k; i++) if (p[i] > 0) ent -= p[i] * log(p[i]);
            double conf = (k > 1) ? 1.0 - ent / log((double)k) : 1.0;
            w_fmt(&w, "\"%s\":{\"type\":\"choice\",\"choice\":\"%s\",\"probabilities\":{",
                  qid, keys[best] ? keys[best] : "");
            for (int i = 0; i < k; i++)
                w_fmt(&w, "%s\"%s\":%.4f", i ? "," : "", keys[i] ? keys[i] : "", p[i]);
            w_fmt(&w, "},\"confidence\":%.4f}", conf);
        } else if (qt == 1) {
            double ev = 0.0;
            for (int i = 0; i < k; i++) ev += i * p[i];
            double ent = 0.0;
            for (int i = 0; i < k; i++) if (p[i] > 0) ent -= p[i] * log(p[i]);
            double conf = (k > 1) ? 1.0 - ent / log((double)k) : 1.0;
            w_fmt(&w, "\"%s\":{\"type\":\"score\",\"score\":%.4f,\"probabilities\":{", qid, ev);
            for (int i = 0; i < k; i++) w_fmt(&w, "%s\"%d\":%.4f", i ? "," : "", i, p[i]);
            w_fmt(&w, "},\"confidence\":%.4f}", conf);
        } else {
            w_fmt(&w, "\"%s\":{\"type\":\"noul\",\"noul\":%.4f}", qid, p[1]);
        }
    }
    w_raw(&w, "}}");

    free(state_str); free(state_clean);
    free(ids); free(tmp); free(mk); free(lg);
    vjson_free(root);
    if (w.ovf) { fprintf(stderr, "[LAYA] output buffer too small\n"); return -1; }
    return 0;
}

int laya_decide(const char *pack_dir, const char *req_path) {
    LayaRunner *r = laya_runner_new(pack_dir);
    if (!r) return -1;
    char *reqtxt = lay_read_file(req_path);
    if (!reqtxt) { fprintf(stderr, "[LAYA] cannot read %s\n", req_path); laya_runner_free(r); return -1; }
    size_t cap = 1 << 20;
    char *out = (char *)malloc(cap);
    int rc = laya_runner_decide(r, reqtxt, out, cap);
    if (rc == 0) { fputs(out, stdout); fputc('\n', stdout); fflush(stdout); }
    else fprintf(stderr, "[LAYA] decide failed\n");
    free(out); free(reqtxt); laya_runner_free(r);
    return rc;
}
