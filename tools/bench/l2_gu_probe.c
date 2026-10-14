/* l2_gu_probe.c — K-PF-005 L2 探针：ARM gu 内层的「真实可达上界」
 *
 * 目的（计划书 §11.5）：L2（结构性内核重写）"理论 10–40×" 是**推算**来的，缺实测支撑。
 * 本探针在板端用同一工具链，逐形态量 GMAC/s，界定可达上界。
 *
 * 忠实复刻对象 = arm2_gu_worker_exact 的批式结构（VLLM_MOE_BATCH 默认路径）：
 *   任务 (e,g) = (专家, 4 行组)；每任务：for cb(0..nbG-1){ 解包 gate/up 两个 72B tile →
 *   wfg/wfu[32]（×scale）；for k(0..m-1){ for kk(0..31){ xv=splat(x[k][cb*32+kk]);
 *   acc[k] += w[kk]*xv } } }
 *
 * 形态（同 MAC 数，直接比 GMAC/s）：
 *   A1 faithful · 1 线程（runtime m → VLA；mul+add；权重 113MB/投影 真流式）
 *   A4 同 A1 · 4 线程静态切分（镜像 vllm_tp_parfor）→ A1/A4 之比 = 多核可扩展性
 *   B1 fma_reg  · 1 线程（m 编译期 16 → 寄存器累加器；vfmaq；真流式）
 *   C1 alu_only · 1 线程（wf 预解包 + 权重只读同一小块 → 去访存与解包代价）
 *   E1 fma_roof · 1 线程（纯 f32 FMA 流，数据 L2 常驻 → 本芯片 FP 吞吐经验屋顶）
 *
 * 编译（板端）：gcc -O3 -march=armv8.2-a -mtune=cortex-a76 -pthread l2_gu_probe.c -o l2_gu_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <arm_neon.h>

#define D     2048
#define NBG   (D / 32)      /* 64  */
#define EF    768
#define ROWG  (EF / 4)      /* 192 */
#define NEXP  128
#define CBB   72
#define GEB   (NBG * CBB)   /* 4608：单个 (专家,行组) 的 gate 段字节 */
#define MVAL  16            /* 每 (专家,行组) 的归属 token 数（实测量级 ~14.75）*/
#define NTH   4
/* dn 侧（同 d、moe_ffn=EF；dn 权重按 d 分行组、按专家分块） */
#define FF_TOTAL (NEXP * EF)            /* 98304 = 128×768 */
#define GDB      ((FF_TOTAL / 32) * CBB)/* 221184：单个 d 行组跨全部专家的字节 */
#define CBE      (EF / 32)              /* 24：每专家的 dn tile 数 */
#define NSEL     (NEXP * MVAL)          /* 2048：(token,rank) 归属数 */

static volatile int   g_m    = MVAL;
static volatile float g_sink = 0.0f;
static int            g_nth  = NTH;   /* 本次运行的线程数（worker 按它切分专家区间） */

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* 与 vllm_safetensors.c:q4x4_tile32_wf_neon 逐条同形（scale 用 vcvt_f32_f16）。 */
static inline void unpack_tile(const uint8_t *tile, float32x4_t wf[32]) {
    static const uint8_t GRP[16] = { 0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15 };
    const uint8x16_t grp   = vld1q_u8(GRP);
    const uint8x16_t m15   = vdupq_n_u8(15);
    const uint8x16_t xor88 = vdupq_n_u8(0x88);
    const int32x4_t  k8    = vdupq_n_s32(8);
    const float32x4_t scales = vcvt_f32_f16(vld1_f16((const __fp16 *)tile));
    const uint8_t *qs = tile + 8;
    const uint8x16_t C0 = veorq_u8(vld1q_u8(qs +  0), xor88);
    const uint8x16_t C1 = veorq_u8(vld1q_u8(qs + 16), xor88);
    const uint8x16_t C2 = veorq_u8(vld1q_u8(qs + 32), xor88);
    const uint8x16_t C3 = veorq_u8(vld1q_u8(qs + 48), xor88);
#define WF_CHUNK(CQ, KB, HALF)                                                  \
    do {                                                                        \
        uint8x16_t b_ = vqtbl1q_u8((CQ), grp);                                  \
        uint8x16_t n_ = (HALF) ? vandq_u8(vshrq_n_u8(b_, 4), m15)               \
                               : vandq_u8(b_, m15);                             \
        uint16x8_t u0_ = vmovl_u8(vget_low_u8(n_));                             \
        uint16x8_t u1_ = vmovl_u8(vget_high_u8(n_));                            \
        int32x4_t s0_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u0_)));    \
        int32x4_t s1_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u0_)));   \
        int32x4_t s2_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u1_)));    \
        int32x4_t s3_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u1_)));   \
        wf[(KB) + 0] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s0_, k8)), scales);    \
        wf[(KB) + 1] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s1_, k8)), scales);    \
        wf[(KB) + 2] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s2_, k8)), scales);    \
        wf[(KB) + 3] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s3_, k8)), scales);    \
    } while (0)
    WF_CHUNK(C0,  0, 0); WF_CHUNK(C1,  4, 0);
    WF_CHUNK(C2,  8, 0); WF_CHUNK(C3, 12, 0);
    WF_CHUNK(C0, 16, 1); WF_CHUNK(C1, 20, 1);
    WF_CHUNK(C2, 24, 1); WF_CHUNK(C3, 28, 1);
#undef WF_CHUNK
}

static const uint8_t *gG, *gU;
static const float   *gX;

/* 消费全部累加器，防编译器把 k>=1 的链当死代码删掉。 */
static inline float reduce_all(const float32x4_t *gacc, const float32x4_t *uacc, int m) {
    float32x4_t s = vdupq_n_f32(0.f);
    for (int k = 0; k < m; k++) s = vaddq_f32(s, vaddq_f32(gacc[k], uacc[k]));
    return vaddvq_f32(s);
}
static inline float reduce_one(const float32x4_t *acc, int m) {
    float32x4_t s = vdupq_n_f32(0.f);
    for (int k = 0; k < m; k++) s = vaddq_f32(s, acc[k]);
    return vaddvq_f32(s);
}

/* A：忠实形态（runtime m → VLA；mul+add）。处理专家区间 [e0,e1)。 */
static void *worker_faithful(void *arg) {
    long id = (long)arg;
    int e0 = (int)(NEXP * id / g_nth), e1 = (int)(NEXP * (id + 1) / g_nth);
    float sink = 0.f;
    for (int e = e0; e < e1; e++) {
        for (int g = 0; g < ROWG; g++) {
            const uint8_t *ge = gG + ((size_t)e * ROWG + g) * GEB;
            const uint8_t *ue = gU + ((size_t)e * ROWG + g) * GEB;
            int m = g_m;
            float32x4_t gacc[m], uacc[m];
            for (int k = 0; k < m; k++) { gacc[k] = vdupq_n_f32(0.f); uacc[k] = vdupq_n_f32(0.f); }
            for (int cb = 0; cb < NBG; cb++) {
                float32x4_t wfg[32], wfu[32];
                unpack_tile(ge + (size_t)cb * CBB, wfg);
                unpack_tile(ue + (size_t)cb * CBB, wfu);
                for (int k = 0; k < m; k++) {
                    const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                    for (int kk = 0; kk < 32; kk++) {
                        float32x4_t xv = vdupq_n_f32(x[kk]);
                        gacc[k] = vaddq_f32(gacc[k], vmulq_f32(wfg[kk], xv));
                        uacc[k] = vaddq_f32(uacc[k], vmulq_f32(wfu[kk], xv));
                    }
                }
            }
            sink += reduce_all(gacc, uacc, m);
        }
    }
    g_sink += sink;
    return NULL;
}

/* B：m 编译期常量 16 → 寄存器累加器 + vfmaq；真流式。 */
static void *worker_fma_reg(void *arg) {
    long id = (long)arg;
    int e0 = (int)(NEXP * id / g_nth), e1 = (int)(NEXP * (id + 1) / g_nth);
    float sink = 0.f;
    for (int e = e0; e < e1; e++) {
        for (int g = 0; g < ROWG; g++) {
            const uint8_t *ge = gG + ((size_t)e * ROWG + g) * GEB;
            const uint8_t *ue = gU + ((size_t)e * ROWG + g) * GEB;
            float32x4_t gacc[MVAL], uacc[MVAL];
            for (int k = 0; k < MVAL; k++) { gacc[k] = vdupq_n_f32(0.f); uacc[k] = vdupq_n_f32(0.f); }
            for (int cb = 0; cb < NBG; cb++) {
                float32x4_t wfg[32], wfu[32];
                unpack_tile(ge + (size_t)cb * CBB, wfg);
                unpack_tile(ue + (size_t)cb * CBB, wfu);
                for (int k = 0; k < MVAL; k++) {
                    const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                    for (int kk = 0; kk < 32; kk++) {
                        float32x4_t xv = vdupq_n_f32(x[kk]);
                        gacc[k] = vfmaq_f32(gacc[k], wfg[kk], xv);
                        uacc[k] = vfmaq_f32(uacc[k], wfu[kk], xv);
                    }
                }
            }
            sink += reduce_all(gacc, uacc, MVAL);
        }
    }
    g_sink += sink;
    return NULL;
}

/* F：**与引擎同任务序**（idx = g*ne + e，e 最快变化）+ 同 vllm_tp_parfor 静态切分。
 * 与 A 的唯一差别 = 权重访问的遍历顺序（A 是 e 外/g 内 = 顺序；F 是引擎的 864 KB 跨距）。
 * 用于判定 L2 那 2.55× 是否来自「任务序/访存局部性」。 */
static void *worker_engine_order(void *arg) {
    long t = (long)arg;
    int nt = g_nth;
    int n = NEXP * ROWG;
    int base = n / nt, rem = n % nt;
    int lo = (int)t * base + (int)(t < rem ? t : rem);
    int hi = lo + base + (int)(t < rem ? 1 : 0);
    float sink = 0.f;
    for (int idx = lo; idx < hi; idx++) {
        int g = idx / NEXP, e = idx - g * NEXP;
        const uint8_t *ge = gG + ((size_t)e * ROWG + g) * GEB;
        const uint8_t *ue = gU + ((size_t)e * ROWG + g) * GEB;
        int m = g_m;
        float32x4_t gacc[m], uacc[m];
        for (int k = 0; k < m; k++) { gacc[k] = vdupq_n_f32(0.f); uacc[k] = vdupq_n_f32(0.f); }
        for (int cb = 0; cb < NBG; cb++) {
            float32x4_t wfg[32], wfu[32];
            unpack_tile(ge + (size_t)cb * CBB, wfg);
            unpack_tile(ue + (size_t)cb * CBB, wfu);
            for (int k = 0; k < m; k++) {
                const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                for (int kk = 0; kk < 32; kk++) {
                    float32x4_t xv = vdupq_n_f32(x[kk]);
                    gacc[k] = vaddq_f32(gacc[k], vmulq_f32(wfg[kk], xv));
                    uacc[k] = vaddq_f32(uacc[k], vmulq_f32(wfu[kk], xv));
                }
            }
        }
        sink += reduce_all(gacc, uacc, m);
    }
    g_sink += sink;
    return NULL;
}

/* V2（K-ACTQ-009 候选②）：**gate/up 分两趟** —— 峰值存活权重向量由 **64 降到 32**。
 * 每累加器的 kk 升序不变、表达式不变 ⇒ **与本文件 F1 的每个累加器位级一致**（A≡C），
 * 故两者之差只能归因于寄存器压力（栈溢出），而非算术重排或访存量（总字节数相同）。
 * 用途：验证「精确轨每 cb 预计算 wfg[32]+wfu[32] = 64 个 float32x4_t > A76 的 32 个
 * NEON 寄存器」是否为 gu 效率偏低的真因。 */
static void *worker_split_gu(void *arg) {
    long t = (long)arg;
    int nt = g_nth;
    int n = NEXP * ROWG;
    int base = n / nt, rem = n % nt;
    int lo = (int)t * base + (int)(t < rem ? t : rem);
    int hi = lo + base + (int)(t < rem ? 1 : 0);
    float sink = 0.f;
    for (int idx = lo; idx < hi; idx++) {
        int g = idx / NEXP, e = idx - g * NEXP;
        const uint8_t *ge = gG + ((size_t)e * ROWG + g) * GEB;
        const uint8_t *ue = gU + ((size_t)e * ROWG + g) * GEB;
        int m = g_m;
        float32x4_t gacc[m], uacc[m];
        for (int k = 0; k < m; k++) { gacc[k] = vdupq_n_f32(0.f); uacc[k] = vdupq_n_f32(0.f); }
        for (int cb = 0; cb < NBG; cb++) {          /* 趟 1：gate，只活 wfg[32] */
            float32x4_t wfg[32];
            unpack_tile(ge + (size_t)cb * CBB, wfg);
            for (int k = 0; k < m; k++) {
                const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                for (int kk = 0; kk < 32; kk++)
                    gacc[k] = vaddq_f32(gacc[k], vmulq_f32(wfg[kk], vdupq_n_f32(x[kk])));
            }
        }
        for (int cb = 0; cb < NBG; cb++) {          /* 趟 2：up，只活 wfu[32] */
            float32x4_t wfu[32];
            unpack_tile(ue + (size_t)cb * CBB, wfu);
            for (int k = 0; k < m; k++) {
                const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                for (int kk = 0; kk < 32; kk++)
                    uacc[k] = vaddq_f32(uacc[k], vmulq_f32(wfu[kk], vdupq_n_f32(x[kk])));
            }
        }
        sink += reduce_all(gacc, uacc, m);
    }
    g_sink += sink;
    return NULL;
}

/* D：dn 忠实形态（**单投影**：每 (k,kk) 1 dup + 1 mul + 1 add = 4 MAC；gu 是 8 MAC）。
 * `av` 行集**只依赖专家 e**（该专家的 m 个 (t,j) 归属），并跨 rg 复用 —— 与引擎同构。
 *   模式 0 = contig   ：av 用连续小块（计算上界）
 *   模式 1 = rg-major ：task = (rg,e)，e 最快（引擎现状）
 *   模式 2 = exp-major：task = (e,rg)，e 最慢 ⇒ 同一专家的 av 行在扫 rg 时驻留
 */
static const uint8_t *gD;
static const float   *gDnAV;
static void *dn_body(void *arg, int mode) {
    long id = (long)arg;
    int nt = g_nth;
    int NRG = D / 4;                     /* 512 */
    int n = NRG * NEXP;
    int base = n / nt, rem = n % nt;
    int lo = (int)id * base + (int)(id < rem ? id : rem);
    int hi = lo + base + (int)(id < rem ? 1 : 0);
    float sink = 0.f;
    for (int idx = lo; idx < hi; idx++) {
        int rg, e;
        if (mode == 2) { e = idx / NRG; rg = idx - e * NRG; }
        else           { rg = idx / NEXP; e = idx - rg * NEXP; }
        const uint8_t *db = gD + (size_t)rg * GDB + (size_t)((e * EF) >> 5) * CBB;
        int m = g_m;
        float32x4_t acc[m];
        for (int k = 0; k < m; k++) acc[k] = vdupq_n_f32(0.f);
        for (int cbi = 0; cbi < CBE; cbi++) {
            float32x4_t wfd[32];
            unpack_tile(db + (size_t)cbi * CBB, wfd);
            for (int k = 0; k < m; k++) {
                size_t s = (mode == 0) ? (size_t)k
                                       : (size_t)((e * MVAL + k) % NSEL);
                const float *av = gDnAV + s * EF + (size_t)cbi * 32;
                for (int kk = 0; kk < 32; kk++) {
                    float32x4_t avv = vdupq_n_f32(av[kk]);
                    acc[k] = vaddq_f32(acc[k], vmulq_f32(wfd[kk], avv));
                }
            }
        }
        sink += reduce_one(acc, m);
    }
    g_sink += sink;
    return NULL;
}
static void *worker_dn_contig(void *arg) { return dn_body(arg, 0); }
static void *worker_dn_rgmajor(void *arg) { return dn_body(arg, 1); }
static void *worker_dn_exmajor(void *arg) { return dn_body(arg, 2); }

/* C：无访存/无解包 —— wf 预解包一次，全程只读同一小块（纯内层算力上界）。 */
static float32x4_t CG[32], CU[32];
static void *worker_alu_only(void *arg) {
    long id = (long)arg;
    int e0 = (int)(NEXP * id / g_nth), e1 = (int)(NEXP * (id + 1) / g_nth);
    float sink = 0.f;
    for (int e = e0; e < e1; e++) {
        for (int g = 0; g < ROWG; g++) {
            float32x4_t gacc[MVAL], uacc[MVAL];
            for (int k = 0; k < MVAL; k++) { gacc[k] = vdupq_n_f32(0.f); uacc[k] = vdupq_n_f32(0.f); }
            for (int cb = 0; cb < NBG; cb++) {
                for (int k = 0; k < MVAL; k++) {
                    const float *x = gX + (size_t)k * D + (size_t)cb * 32;
                    for (int kk = 0; kk < 32; kk++) {
                        float32x4_t xv = vdupq_n_f32(x[kk]);
                        gacc[k] = vfmaq_f32(gacc[k], CG[kk], xv);
                        uacc[k] = vfmaq_f32(uacc[k], CU[kk], xv);
                    }
                }
            }
            sink += reduce_all(gacc, uacc, MVAL);
        }
    }
    g_sink += sink;
    return NULL;
}

/* E：纯 f32 FMA 吞吐（操作数寄存器常驻，无访存）→ FP 屋顶。 */
static void run_fma_roof(size_t rounds) {
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    float32x4_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const float32x4_t w0 = vdupq_n_f32(1.0001f), w1 = vdupq_n_f32(0.9999f);
    const float32x4_t w2 = vdupq_n_f32(1.0002f), w3 = vdupq_n_f32(0.9998f);
    for (size_t r = 0; r < rounds; r++) {
        a0 = vfmaq_f32(a0, w0, w1);
        a1 = vfmaq_f32(a1, w1, w2);
        a2 = vfmaq_f32(a2, w2, w3);
        a3 = vfmaq_f32(a3, w3, w0);
        a4 = vfmaq_f32(a4, w0, w2);
        a5 = vfmaq_f32(a5, w1, w3);
        a6 = vfmaq_f32(a6, w2, w0);
        a7 = vfmaq_f32(a7, w3, w1);
    }
    g_sink += vgetq_lane_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)), 0)
            + vgetq_lane_f32(vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7)), 0);
}

typedef void *(*wfn)(void *);
static double run_threads(wfn fn, int nthreads) {
    pthread_t th[NTH];
    g_nth = nthreads;
    double t0 = now_s();
    for (int i = 1; i < nthreads; i++) pthread_create(&th[i], NULL, fn, (void *)(long)i);
    fn((void *)0);
    for (int i = 1; i < nthreads; i++) pthread_join(th[i], NULL);
    return now_s() - t0;
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 3;
    size_t seg = (size_t)NEXP * ROWG * GEB;          /* 113 MB / 投影 */
    uint8_t *G = (uint8_t *)aligned_alloc(64, seg);
    uint8_t *U = (uint8_t *)aligned_alloc(64, seg);
    float   *X = (float *)aligned_alloc(64, (size_t)MVAL * D * sizeof(float));
    if (!G || !U || !X) { fprintf(stderr, "alloc fail\n"); return 1; }
    srand(12345);
    for (size_t i = 0; i < seg; i++) { G[i] = (uint8_t)rand(); U[i] = (uint8_t)rand(); }
    /* tile 头 8B 是 f16 scale：写成固定良性值（0x3000≈0.125），避免随机位产生 NaN/Inf */
    for (int e = 0; e < NEXP; e++)
        for (int g = 0; g < ROWG; g++)
            for (int cb = 0; cb < NBG; cb++) {
                size_t o = ((size_t)e * ROWG + g) * GEB + (size_t)cb * CBB;
                for (int b = 0; b < 8; b++) {
                    G[o + b] = (b & 1) ? 0x30 : 0x00;   /* 0x0030 = 小正数 f16 */
                    U[o + b] = (b & 1) ? 0x2c : 0x00;
                }
            }
    for (int i = 0; i < MVAL * D; i++) X[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    gG = G; gU = U; gX = X;
    unpack_tile(G, CG); unpack_tile(U, CU);

    double macs = (double)NEXP * ROWG * NBG * MVAL * 32.0 * 8.0;
    printf("[probe] seg/投影=%.0f MB  m=%d  tasks=%d  reps=%d  MAC/轮=%.2fe9\n",
           seg / 1048576.0, MVAL, NEXP * ROWG, reps, macs / 1e9);

    double t;
    double a1 = 0, a2 = 0, a4 = 0;
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_faithful, 1); if (u < t) t = u; }
    a1 = t;
    printf("A1 faithful     1线程 : %6.3f s  %6.2f GMAC/s\n", t, macs / t / 1e9);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_faithful, 2); if (u < t) t = u; }
    a2 = t;
    printf("A2 faithful     2线程 : %6.3f s  %6.2f GMAC/s  扩展比=%.2fx\n", t, macs / t / 1e9, a1 / t);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_faithful, NTH); if (u < t) t = u; }
    a4 = t;
    printf("A4 faithful     4线程 : %6.3f s  %6.2f GMAC/s  扩展比=%.2fx\n", t, macs / t / 1e9, a1 / t);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_engine_order, 1); if (u < t) t = u; }
    printf("F1 engine_order 1线程 : %6.3f s  %6.2f GMAC/s\n", t, macs / t / 1e9);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_engine_order, NTH); if (u < t) t = u; }
    printf("F4 engine_order 4线程 : %6.3f s  %6.2f GMAC/s  相对A4=%.2fx\n", t, macs / t / 1e9, a4 / t);
    /* V2：寄存器压力对照（K-ACTQ-009 候选②）。V1 = engine_order（F1/F4）。 */
    double v1_1 = 0, v1_4 = 0;
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_engine_order, 1); if (u < t) t = u; }
    v1_1 = t;
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_engine_order, NTH); if (u < t) t = u; }
    v1_4 = t;
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_split_gu, 1); if (u < t) t = u; }
    printf("V2 split_gate/up 1线程: %6.3f s  %6.2f GMAC/s  vs V1(1T)=%.2fx  (峰值权重向量 64→32)\n",
           t, macs / t / 1e9, v1_1 / t);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_split_gu, NTH); if (u < t) t = u; }
    printf("V4 split_gate/up 4线程: %6.3f s  %6.2f GMAC/s  vs V1(4T)=%.2fx\n",
           t, macs / t / 1e9, v1_4 / t);
    {   /* --- dn：单投影形态 --- */
        size_t dsz = (size_t)(D / 4) * GDB;
        uint8_t *Dp = (uint8_t *)aligned_alloc(64, dsz);
        float   *AV = (float *)aligned_alloc(64, (size_t)NSEL * EF * sizeof(float));
        for (size_t i = 0; i < dsz; i++) Dp[i] = (uint8_t)rand();
        for (size_t i = 0; i < (size_t)NSEL * EF; i++) AV[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        gD = Dp; gDnAV = AV;
        double dmacs = (double)(D / 4) * NEXP * CBE * MVAL * 32.0 * 4.0;
        printf("[probe] dn: 段/投影=%.0f MB  tasks=%d  MAC/轮=%.2fe9\n",
               dsz / 1048576.0, (D / 4) * NEXP, dmacs / 1e9);
        t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_dn_contig, 1); if (u < t) t = u; }
        printf("Dc  dn_contig   1线程 : %6.3f s  %6.2f GMAC/s  (计算上界)\n", t, dmacs / t / 1e9);
        t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_dn_rgmajor, 1); if (u < t) t = u; }
        double r1 = t;
        printf("Dr1 dn_rg-major 1线程 : %6.3f s  %6.2f GMAC/s  (引擎现状：e 最快)\n", t, dmacs / t / 1e9);
        t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_dn_rgmajor, NTH); if (u < t) t = u; }
        double r4 = t;
        printf("Dr4 dn_rg-major 4线程 : %6.3f s  %6.2f GMAC/s  扩展比=%.2fx\n", t, dmacs / t / 1e9, r1 / t);
        t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_dn_exmajor, 1); if (u < t) t = u; }
        double e1 = t;
        printf("De1 dn_exp-major1线程 : %6.3f s  %6.2f GMAC/s  (e 最慢：av 驻留)\n", t, dmacs / t / 1e9);
        t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_dn_exmajor, NTH); if (u < t) t = u; }
        printf("De4 dn_exp-major4线程 : %6.3f s  %6.2f GMAC/s  扩展比=%.2fx  相对Dr4=%.2fx\n",
               t, dmacs / t / 1e9, e1 / t, r4 / t);
        free(Dp); free(AV);
    }
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_fma_reg, 1); if (u < t) t = u; }
    printf("B1 fma_reg      1线程 : %6.3f s  %6.2f GMAC/s\n", t, macs / t / 1e9);
    t = 1e30; for (int r = 0; r < reps; r++) { double u = run_threads(worker_alu_only, 1); if (u < t) t = u; }
    printf("C1 alu_only     1线程 : %6.3f s  %6.2f GMAC/s  (去访存+去解包)\n", t, macs / t / 1e9);

    {   /* E：纯 FMA 吞吐（寄存器操作数）→ FP 屋顶 */
        size_t rounds = 2000000000UL;
        double rmac = (double)rounds * 32.0;   /* 每轮 8 vfmaq × 4 lane = 32 MAC */
        t = 1e30; for (int r = 0; r < reps; r++) { double u = now_s(); run_fma_roof(rounds); u = now_s() - u; if (u < t) t = u; }
        printf("E1 fma_roof     1线程 : %6.3f s  %6.2f GMAC/s  (FP 吞吐经验屋顶)\n", t, rmac / t / 1e9);
    }
    printf("[sink] %.3f\n", (double)g_sink);
    free(G); free(U); free(X);
    return 0;
}
