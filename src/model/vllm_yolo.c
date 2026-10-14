/**
 * vllm_yolo.c - YOLO 目标检测引擎实现（自有格式 VQF v4，纯 CPU F32 执行器）
 *
 * 本文件 = 图/权重的 C 侧执行器，与 tools/yolo/yolo_vqf_run.py（numpy 参考执行器）
 * 语义一一对应：同格式、同算子、同后处理。规格见
 * docs/立项-K-YOLO-001-G0-NPU卷积检测适配.md 附录 A。
 *
 * M0：CPU F32，正确性优先。
 * M2：kernel>1 卷积经 im2col 展开为 int8 GEMM，复用已标定的 NPU DIRECT gemm_gw；
 *     失败/不适用时显式回退 CPU（公理 silent_fallback_false_pass）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vqf_format.h"
#include "vllm_yolo.h"
#include "vllm_npu.h"
#include "vllm_platform.h"         /* ST_HAVE_NEON (int8 dotprod CPU 卷积, K-YOLO-002) */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>
static double ynow(void) {
#ifdef __linux__
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* ---- 图算子（与 tools/yolo/yolo_to_vqf.py 冻结表一致） ---- */
#define YOP_CONV      1
#define YOP_ADD       2
#define YOP_CONCAT    3
#define YOP_MAXPOOL   4
#define YOP_UPSAMPLE  5
#define YOP_SIGMOID   6
#define YOP_MUL       7
#define YOP_SLICE     8
#define YOP_SOFTMAX   9
#define YOP_ATTN_PSA  10
#define YOP_DETECT    11

#define YGRAPH_MAGIC  0x50524759u   /* 'YGRP' */
#define YACT_SILU     1

/* ================================================================
 * 模型结构
 * ================================================================ */
typedef struct {
    char     name[32];
    uint32_t qtype, rows, cols;
    uint64_t off, bytes;
    const uint8_t *ptr;
} YTens;

typedef struct {
    uint32_t op, nin, nparam, out;
    const uint32_t *ins;    /* -> graph blob */
    const uint32_t *prm;
} YOp;

struct StYoloModel {
    uint8_t  *blob;
    size_t    size;
    VQFArchDet det;
    int       n_tens;
    YTens    *tens;
    const uint8_t *graph;
    int       n_ops;
    YOp      *ops;
    int       n_slots;
    /* 卷积权重索引（按 cid 直查张量）：F32 或 int8 + per-channel scale */
    const float  **cw;      /* F32 权重（Q8 模型为 NULL） */
    const int8_t **cq;      /* int8 权重（Q8） */
    const float  **cs;      /* per-channel scale [oc]（Q8） */
    const float  **cb;      /* bias（恒 F32；无则 NULL） */
    int          *crow, *ccol;   /* 权重几何（rows=oc, cols=ic*kh*kw） */
    int        max_cid;
    /* M2：NPU（im2col→GEMM）权重重排缓存（惰性，按 cid） */
    vllm_npu_t  *npu;            /* M2 NPU 句柄（NULL=纯 CPU） */
    int8_t     **nw;             /* [cid] 补零到 Kp 的 int8 权重 [OC][Kp] */
    float      **nbs;            /* [cid] per-(oc,block) scale [OC*n_blocks] */
    int         *nkp;            /* [cid] 补齐后的 K（= 32 的倍数） */
    int8_t     **ncw;            /* [cid] M3 原生卷积权重 cube（NC1HWC2 域） */
    long        npu_convs, cpu_convs, npu_fallbacks, npu_submits, npu_native;
    double      t_i2c, t_npucall;/* M2 诊断：im2col 构建 / NPU 调用耗时 */
    /* 类别名（names blob） */
    const char *names;
    size_t      names_len;
    /* K-YOLO-003：官方 .rknn 整网检测路径（opt-in 第二路径；NULL = VQF v4 路径） */
    vllm_npu_rknn *rknn;
    /* 诊断 */
    double    t_conv, t_detect, t_attn;
    /* Phase-0（K-YOLO-002）CPU 分解计时（env VLLM_YOLO_CPU_TIMING 门控；默认不累加） */
    double    t_elem, t_lb, t_post, t_fwd;
    long      n_conv_ops, n_elem_ops;
    double    t_opk[12];   /* Phase 2：逐算子类型耗时（index = opcode） */
    long      n_opk[12];
};

/* M2：全局默认 NPU 句柄（由 st_yolo_set_default_npu 设置；st_yolo_load 取用）。 */
static vllm_npu_t *g_yolo_npu_def = NULL;

void st_yolo_set_default_npu(vllm_npu_t *npu) { g_yolo_npu_def = npu; }

/* ================================================================
 * VQF v4 加载器（FOPEN 读整文件 + 结构/FNV 自校验）
 * ================================================================ */
static uint64_t yfnv(const uint64_t *h0, const uint8_t *p, size_t n) {
    uint64_t h = *h0;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ull; }
    return h;
}

static int ytens_find(const StYoloModel *m, const char *name) {
    for (int i = 0; i < m->n_tens; i++)
        if (!strcmp(m->tens[i].name, name)) return i;
    return -1;
}

static const float *ytens_f32(const StYoloModel *m, int idx, size_t *nelem) {
    if (idx < 0) return NULL;
    const YTens *t = &m->tens[idx];
    if (t->qtype == VQF_QT_F32) {
        if (nelem) *nelem = (size_t)t->rows * t->cols;
        return (const float *)t->ptr;
    }
    return NULL;                 /* F16/Q8 在 M1 接入 */
}

StYoloModel *st_yolo_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[YOLO] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= (long)sizeof(VQFHeader)) { fclose(f); return NULL; }
    StYoloModel *m = (StYoloModel *)calloc(1, sizeof(*m));
    m->blob = (uint8_t *)malloc((size_t)sz);
    if (!m->blob || fread(m->blob, 1, (size_t)sz, f) != (size_t)sz) {
        free(m->blob); free(m); fclose(f); return NULL;
    }
    fclose(f);
    m->size = (size_t)sz;

    const VQFHeader *h = (const VQFHeader *)m->blob;
    if (h->magic != VQF_MAGIC) {
        fprintf(stderr, "[YOLO] bad magic (not a VQF file)\n");
        goto fail;
    }
    if (h->version != VQF_VERSION_DET) {
        fprintf(stderr, "[YOLO] version %u is not a DETECTION model (need %u)\n",
                h->version, VQF_VERSION_DET);
        goto fail;
    }
    if (!(h->flags & VQF_FLAG_DET))
        fprintf(stderr, "[YOLO] warning: VQF_FLAG_DET not set\n");
    memcpy(&m->det, &h->arch, sizeof(m->det));

    uint64_t data_off = h->data_offset, file_len = h->file_len;
    size_t tag_len = (h->flags & VQF_FLAG_ENC) ? 32 : 8;
    if (h->flags & VQF_FLAG_ENC) {
        fprintf(stderr, "[YOLO] encrypted VQF not supported yet (M0)\n");
        goto fail;
    }
    if (file_len != 0 && m->size != file_len + tag_len) {
        fprintf(stderr, "[YOLO] size mismatch: file=%zu header=%llu\n",
                m->size, (unsigned long long)file_len);
        goto fail;
    }
    if (data_off < 448 || data_off > m->size) {
        fprintf(stderr, "[YOLO] bad data_offset %llu\n", (unsigned long long)data_off);
        goto fail;
    }
    if (m->size < data_off + 8) goto fail;

    /* FNV-1a（目录 + 数据区）对照尾部 8B —— 与 python 写出侧同规 */
    {
        size_t dir_off = 448, dn = (size_t)h->n_tensors * 64;
        if (dir_off + dn > m->size) { fprintf(stderr, "[YOLO] dir overflow\n"); goto fail; }
        uint64_t sum = 0xCBF29CE484222325ull;
        sum = yfnv(&sum, m->blob + dir_off, dn);
        sum = yfnv(&sum, m->blob + data_off, (size_t)(file_len - data_off));
        uint64_t want; memcpy(&want, m->blob + file_len, 8);
        if (sum != want) {
            fprintf(stderr, "[YOLO] FNV mismatch (got %016llx want %016llx) - corrupt?\n",
                    (unsigned long long)sum, (unsigned long long)want);
            goto fail;
        }
    }

    /* 目录 */
    m->n_tens = (int)h->n_tensors;
    m->tens = (YTens *)calloc((size_t)m->n_tens, sizeof(YTens));
    for (int i = 0; i < m->n_tens; i++) {
        const VQFTensor *t = (const VQFTensor *)(m->blob + 448 + (size_t)i * 64);
        YTens *d = &m->tens[i];
        memcpy(d->name, t->name, 32); d->name[31] = 0;
        d->qtype = t->qtype; d->rows = t->rows; d->cols = t->cols;
        d->off = t->offset; d->bytes = t->bytes;
        if (t->offset + t->bytes > m->size) {
            fprintf(stderr, "[YOLO] tensor %s out of range\n", d->name);
            goto fail;
        }
        d->ptr = m->blob + t->offset;
    }

    /* graph */
    {
        int gi = ytens_find(m, "graph");
        if (gi < 0 || m->tens[gi].qtype != VQF_QT_BLOB) {
            fprintf(stderr, "[YOLO] graph tensor missing\n"); goto fail;
        }
        m->graph = m->tens[gi].ptr;
        uint32_t magic, gver, nops;
        memcpy(&magic, m->graph, 4); memcpy(&gver, m->graph + 4, 4);
        memcpy(&nops, m->graph + 8, 4);
        if (magic != YGRAPH_MAGIC) { fprintf(stderr, "[YOLO] bad graph magic\n"); goto fail; }
        if (gver != 1) { fprintf(stderr, "[YOLO] graph version %u\n", gver); goto fail; }
        m->n_ops = (int)nops;
        m->ops = (YOp *)calloc((size_t)m->n_ops, sizeof(YOp));
        const uint8_t *p = m->graph + 12, *end = m->graph + m->tens[gi].bytes;
        int maxslot = 0, maxcid = -1;
        for (int i = 0; i < m->n_ops; i++) {
            if (p + 16 > end) { fprintf(stderr, "[YOLO] graph truncated @%d\n", i); goto fail; }
            YOp *o = &m->ops[i];
            memcpy(&o->op, p, 4); memcpy(&o->nin, p + 4, 4);
            memcpy(&o->nparam, p + 8, 4); memcpy(&o->out, p + 12, 4);
            p += 16;
            if (p + 4 * ((size_t)o->nin + o->nparam) > end) {
                fprintf(stderr, "[YOLO] graph op %d overflow\n", i); goto fail;
            }
            o->ins = (const uint32_t *)p; p += 4 * o->nin;
            o->prm = (const uint32_t *)p; p += 4 * o->nparam;
            if ((int)o->out > maxslot) maxslot = (int)o->out;
            if (o->op == YOP_CONV && o->nparam >= 1 && (int)o->prm[0] > maxcid)
                maxcid = (int)o->prm[0];
            if (o->op == YOP_ATTN_PSA && o->nparam >= 6)
                for (int k = 3; k < 6; k++)
                    if ((int)o->prm[k] > maxcid) maxcid = (int)o->prm[k];
        }
        m->n_slots = maxslot + 1;
        m->max_cid = maxcid;
        m->cw = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->cq = (const int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->cs = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->cb = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->crow = (int *)calloc((size_t)maxcid + 1, sizeof(int));
        m->ccol = (int *)calloc((size_t)maxcid + 1, sizeof(int));
        m->nw  = (int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->nbs = (float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->nkp = (int *)calloc((size_t)maxcid + 1, sizeof(int));
        m->ncw = (int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->npu = g_yolo_npu_def;   /* M2：NPU 句柄（可 NULL） */
        for (int c = 0; c <= maxcid; c++) {
            char nm[32];
            snprintf(nm, sizeof(nm), "c%04d_w", c);
            int wi = ytens_find(m, nm);
            if (wi >= 0) {
                m->crow[c] = (int)m->tens[wi].rows;
                m->ccol[c] = (int)m->tens[wi].cols;
                if (m->tens[wi].qtype == VQF_QT_F32)
                    m->cw[c] = (const float *)m->tens[wi].ptr;
                else if (m->tens[wi].qtype == VQF_QT_Q8_0)
                    m->cq[c] = (const int8_t *)m->tens[wi].ptr;
                else
                    fprintf(stderr, "[YOLO] conv %s qtype %u unsupported (M1: F32/Q8)\n",
                            nm, m->tens[wi].qtype);
            }
            snprintf(nm, sizeof(nm), "c%04d_s", c);
            m->cs[c] = ytens_f32(m, ytens_find(m, nm), NULL);
            snprintf(nm, sizeof(nm), "c%04d_b", c);
            m->cb[c] = ytens_f32(m, ytens_find(m, nm), NULL);
        }
        {
            int nq8 = 0, nf32 = 0;
            for (int c = 0; c <= maxcid; c++) { if (m->cq[c]) nq8++; else if (m->cw[c]) nf32++; }
            if (nq8) fprintf(stderr, "[YOLO] conv weights: F32=%d int8=%d (per-channel scale)\n",
                             nf32, nq8);
            if (m->npu)
                fprintf(stderr, "[YOLO] NPU: %s (M2 im2col->int8 GEMM; prefill convs)\n",
                        vllm_npu_available(m->npu) ? "direct backend available"
                                                   : "unavailable - CPU only");
        }
    }

    /* names */
    {
        int ni = ytens_find(m, "names");
        if (ni >= 0) { m->names = (const char *)m->tens[ni].ptr; m->names_len = m->tens[ni].bytes; }
    }

    fprintf(stderr, "[YOLO] loaded %s: family=%u nc=%u reg_max=%u in=%ux%u "
                    "ops=%d slots=%d tensors=%d\n",
            path, m->det.family, m->det.n_classes, m->det.reg_max,
            m->det.input_w, m->det.input_h, m->n_ops, m->n_slots, m->n_tens);
    return m;
fail:
    st_yolo_free(m);
    return NULL;
}

void st_yolo_free(StYoloModel *m) {
    if (!m) return;
    free(m->tens); free(m->ops);
    free(m->cw); free(m->cq); free(m->cs); free(m->cb);
    free(m->crow); free(m->ccol);
    if (m->nw || m->nbs) {
        int nc = m->max_cid + 1;
        for (int c = 0; c < nc; c++) { free(m->nw[c]); free(m->nbs[c]); }
        free(m->nw); free(m->nbs); free(m->nkp);
    }
    if (m->ncw) {
        int nc = m->max_cid + 1;
        for (int c = 0; c < nc; c++) free(m->ncw[c]);
        free(m->ncw);
    }
    free(m->blob);
    if (m->rknn) vllm_npu_rknn_close(m->rknn);   /* K-YOLO-003 */
    free(m);
}

int st_yolo_input_size(const StYoloModel *m, int *w, int *h) {
    if (!m) return -1;
    if (w) *w = (int)m->det.input_w;
    if (h) *h = (int)m->det.input_h;
    return 0;
}
int st_yolo_num_classes(const StYoloModel *m) { return m ? (int)m->det.n_classes : 0; }
int st_yolo_family(const StYoloModel *m) { return m ? (int)m->det.family : -1; }

/* ---------- K-YOLO-003：官方 .rknn 整网检测模型（opt-in 第二路径） ----------
 * 复用 vllm_npu_rknn（dlopen librknnrt）加载整网 .rknn；家族解码在下方。
 * 与 VQF 路径互斥：m->rknn != NULL ⇒ st_yolo_detect 走 RKNN 分支。
 * 官方 .rknn 不含类别名，RKNN 路径统一用 COCO-80 表（ultralytics 顺序）。 */
static const char COCO80_NAMES[] =
    "person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat\n"
    "traffic light\nfire hydrant\nstop sign\nparking meter\nbench\nbird\ncat\n"
    "dog\nhorse\nsheep\ncow\nelephant\nbear\nzebra\ngiraffe\nbackpack\numbrella\n"
    "handbag\ntie\nsuitcase\nfrisbee\nskis\nsnowboard\nsports ball\nkite\n"
    "baseball bat\nbaseball glove\nskateboard\nsurfboard\ntennis racket\nbottle\n"
    "wine glass\ncup\nfork\nknife\nspoon\nbowl\nbanana\napple\nsandwich\norange\n"
    "broccoli\ncarrot\nhot dog\npizza\ndonut\ncake\nchair\ncouch\npotted plant\n"
    "bed\ndining table\ntoilet\ntv\nlaptop\nmouse\nremote\nkeyboard\ncell phone\n"
    "microwave\noven\ntoaster\nsink\nrefrigerator\nbook\nclock\nvase\nscissors\n"
    "teddy bear\nhair drier\ntoothbrush";

StYoloModel *st_yolo_load_rknn(const char *path, vllm_npu_t *npu, int core_mask) {
    if (!path || !npu) return NULL;
    vllm_npu_rknn *r = vllm_npu_rknn_open(npu, path, core_mask);
    if (!r) return NULL;
    vllm_npu_rknn_dump(r);
    int n_out = vllm_npu_rknn_n_outputs(r);
    if (n_out != 9) {
        fprintf(stderr, "[RKNN-YOLO] unsupported n_outputs=%d (expect 9 for v8/v11)\n", n_out);
        vllm_npu_rknn_close(r);
        return NULL;
    }
    int nd = 0, dims[8];
    if (vllm_npu_rknn_output_dims(r, 1, &nd, dims, 8) != 0 || nd != 4) {
        fprintf(stderr, "[RKNN-YOLO] cannot read cls output dims\n");
        vllm_npu_rknn_close(r);
        return NULL;
    }
    int nc = dims[1];                      /* cls: [1, nc, H, W] */
    int ih = 0, iw = 0, ic = 0;
    vllm_npu_rknn_input_hwc(r, &ih, &iw, &ic);
    StYoloModel *m = (StYoloModel *)calloc(1, sizeof(*m));
    if (!m) { vllm_npu_rknn_close(r); return NULL; }
    m->rknn = r;
    m->det.task = VQF_DET_TASK_DETECT;
    m->det.family = VQF_DET_FAM_V8;        /* v8/v11 同为 9 输出 anchor-free DFL */
    m->det.n_classes = (uint32_t)nc;
    m->det.input_w = (uint32_t)iw;
    m->det.input_h = (uint32_t)ih;
    m->det.strides[0] = 8; m->det.strides[1] = 16; m->det.strides[2] = 32;
    m->det.head = 1; m->det.reg_max = 16;
    m->det.conf_thres = 0.25f;
    m->det.iou_thres  = 0.45f;
    m->names = COCO80_NAMES;
    m->names_len = sizeof(COCO80_NAMES) - 1;
    fprintf(stderr, "[YOLO] RKNN backend ready: %s (nc=%d, in %ux%u)\n",
            path, nc, m->det.input_w, m->det.input_h);
    return m;
}

int st_yolo_class_name(const StYoloModel *m, int idx, char *buf, int buflen) {
    if (!m || !m->names || idx < 0 || !buf || buflen <= 0) return -1;
    const char *p = m->names;
    for (int i = 0; i < idx; i++) {
        const char *nl = memchr(p, '\n', m->names_len - (size_t)(p - m->names));
        if (!nl) return -1;
        p = nl + 1;
    }
    const char *nl = memchr(p, '\n', m->names_len - (size_t)(p - m->names));
    size_t n = nl ? (size_t)(nl - p) : strlen(p);
    if ((int)n >= buflen) n = (size_t)buflen - 1;
    memcpy(buf, p, n); buf[n] = 0;
    return 0;
}

/* ================================================================
 * 算子
 * ================================================================ */
static void yconv(const float *x, int C, int H, int W,
                  const float *w, const float *b, int OC, int KH, int KW,
                  int sh, int sw, int ph, int pw, int G, int act,
                  float *y, int OH, int OW) {
    const int icg = C / G, ocg = OC / G;
    if (KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0) {
        const int P = H * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OC * P > 262144)
#endif
        for (int oc = 0; oc < OC; oc++) {
            int g = oc / ocg, ic0 = g * icg;
            const float *wo = w + (size_t)oc * icg;
            float bias = b ? b[oc] : 0.f;
            float *yo = y + (size_t)oc * P;
            for (int q = 0; q < P; q++) {
                float acc = bias;
                for (int icl = 0; icl < icg; icl++)
                    acc += wo[icl] * x[(size_t)(ic0 + icl) * P + q];
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                yo[q] = acc;
            }
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 16)
#endif
    for (int oc = 0; oc < OC; oc++) {
        int g = oc / ocg, ic0 = g * icg;
        const float *wo = w + (size_t)oc * icg * KH * KW;
        float bias = b ? b[oc] : 0.f;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = bias;
                for (int icl = 0; icl < icg; icl++) {
                    const float *xc = x + (size_t)(ic0 + icl) * H * W;
                    const float *wk = wo + (size_t)icl * KH * KW;
                    for (int ky = 0; ky < KH; ky++) {
                        int iy = oy * sh - ph + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < KW; kx++) {
                            int ix = ox * sw - pw + kx;
                            if (ix < 0 || ix >= W) continue;
                            acc += wk[ky * KW + kx] * xc[iy * W + ix];
                        }
                    }
                }
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                y[((size_t)oc * OH + oy) * OW + ox] = acc;
            }
        }
    }
}

/* int8 权重卷积（per-output-channel scale；激活与 F32 路径同式）。
 * 量化坍缩公理：collapse_to_classic + collapse_priority_rule（细粒度 per-channel scale）。
 * 注：int32 累加后一次性乘 scale（与 python 参考执行器同语义）。 */
static void yconv_q8(const float *x, int C, int H, int W,
                     const int8_t *wq, const float *wsc, const float *b,
                     int OC, int KH, int KW, int sh, int sw, int ph, int pw,
                     int G, int act, float *y, int OH, int OW) {
    const int icg = C / G, ocg = OC / G;
    if (KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0) {
        const int P = H * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OC * P > 262144)
#endif
        for (int oc = 0; oc < OC; oc++) {
            int g = oc / ocg, ic0 = g * icg;
            const int8_t *wo = wq + (size_t)oc * icg;
            float sc = wsc ? wsc[oc] : 1.f;
            float bias = b ? b[oc] : 0.f;
            float *yo = y + (size_t)oc * P;
            for (int q = 0; q < P; q++) {
                float acc = 0.f;
                for (int icl = 0; icl < icg; icl++)
                    acc += (float)wo[icl] * x[(size_t)(ic0 + icl) * P + q];
                acc = acc * sc + bias;
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                yo[q] = acc;
            }
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 16)
#endif
    for (int oc = 0; oc < OC; oc++) {
        int g = oc / ocg, ic0 = g * icg;
        const int8_t *wo = wq + (size_t)oc * icg * KH * KW;
        float sc = wsc ? wsc[oc] : 1.f;
        float bias = b ? b[oc] : 0.f;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = 0.f;
                for (int icl = 0; icl < icg; icl++) {
                    const float *xc = x + (size_t)(ic0 + icl) * H * W;
                    const int8_t *wk = wo + (size_t)icl * KH * KW;
                    for (int ky = 0; ky < KH; ky++) {
                        int iy = oy * sh - ph + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < KW; kx++) {
                            int ix = ox * sw - pw + kx;
                            if (ix < 0 || ix >= W) continue;
                            acc += (float)wk[ky * KW + kx] * xc[iy * W + ix];
                        }
                    }
                }
                acc = acc * sc + bias;
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                y[((size_t)oc * OH + oy) * OW + ox] = acc;
            }
        }
    }
}

/* M2：为单个卷积选每-pass 的 (Kb, Nt)，最小化 NPU 提交数 ceil(P/Nt)*ceil(Kp/Kb)。
 * 依据（实测 + 参考）：该驱动每次 submit ≈0.5 ms，卷积-as-GEMM 的墙钟由**提交数**
 * 主导（而非 FLOPs）；G(=Kb) 越大 → K 块越少，Nt 越小 → 权重立方越大但可容纳。
 * 约束（vllm_npu_direct.c 的 regcmd 生成器）：CBUF=12×32768 B 同时装特征 (M×Kb) 与
 * 权重 (Np_tile×Kb)；int8 特征需 +1 bank slack；Nt ∈[64,1024] 且 32 对齐。 */
static void yolo_pick_tile(int OC, int Kp, int P, int *out_kb, int *out_nt) {
    const int BANK = 32768, BANKS = 12;
    int best_kb = 0, best_nt = 0; long best_sub = 0;
    for (int Kb = 32; Kb <= Kp; Kb += 32) {
        if (Kp % Kb) continue;
        const int nb = Kp / Kb;
        const int fd_banks = (OC * Kb + BANK - 1) / BANK;
        const int data_bank = fd_banks + 1;              /* int8 DMA slack */
        if (data_bank >= BANKS) continue;
        const int wbank = BANKS - data_bank;
        int nt = 0;
        for (int t = 1024; t >= 64; t -= 32) {           /* largest Nt that fits */
            if ((t * Kb + BANK - 1) / BANK <= wbank) { nt = t; break; }
        }
        if (!nt) continue;
        long sub = (long)((P + nt - 1) / nt) * nb;
        if (best_kb == 0 || sub < best_sub) { best_sub = sub; best_kb = Kb; best_nt = nt; }
    }
    *out_kb = best_kb; *out_nt = best_nt;
}

/* ---- M2：im2col → int8 GEMM 上 NPU（复用已标定的 DIRECT gemm_gw） ----
 * 公理：isomorphic_superposition（卷积↔GEMM 同构坍缩）、block_matrix_assoc_natural、
 *       MI_fusion_rule（int8 权重带宽收益）、silent_fallback_false_pass（禁止静默降级）。
 *
 * **转置取向（关键）**：DIRECT gemm_gw 的 CNA 几何只标定小 M（特征行数 ≤ ~204，
 * 且 M 的 CNA_DATA_SIZE0 仅 11-bit、CON2 10-bit）、大 K（按 32 块）× 大 N（按 Nt 分块）；
 * 若把 im2col 的 M=空间位置（数千）当 GEMM 的 M，会溢出 M 编码并撑爆 CBUF。
 * 因此把卷积**转置**成 out[oc][p] = Σ_k Wconv[oc][k]·im2col[p][k]：
 *   M = OC（16..512，%4 满足 HW 契约）、N = P = OH·OW（driver 原生支持大 N 分块）、
 *   K = C·KH·KW（补齐 Kp=32 倍数）。于是：
 *     - “特征 A” = 卷积权重 nw[cid]=[OC][Kp]（行主序，行距 Kp），a_scale = cs[oc] 复制到每 K 块；
 *     - “权重 W” = im2col [P][Kp]（行主序），b_scale = 逐像素（行）对称 int8 scale；
 *   ⇒ out[oc*P + p] 与引擎槽位 [oc][OH][OW] 布局**完全一致**（无需转置）。
 *   per-channel scale 复制到各块 ⇒ 与 CPU per-channel 反量化**语义一致**。
 * 注：im2col 在 driver 内作为“权重”每调用重打包（wkey=0 禁缓存，避免陈旧 im2col）。
 * 返回 1 = 已在 NPU 完成；0 = 不适用/失败（调用方跑 CPU 路径，J4 显式回退）。 */
/* im2col + 逐像素（行）对称 int8 量化（K-YOLO-001 M2 / K-YOLO-002 Phase 1 共用）。
 * **单遍**：gather→per-thread 行缓冲（同时求行 max|·|），再从行缓冲量化 ⇒ x 只读一遍。
 * 1x1/s1/p0（无 padding）走无边界特化（YOLO 中约半数为 1×1）。
 * 输出：imc[P][Kp] int8（K..Kp 补 0），rowsc[P] 逐像素 scale（s = max|row|/127）。
 * 返回 0 成功，-1 失败（OOM）。 */
static int yolo_im2col_q8(const float *x, int C, int H, int W,
                          int KH, int KW, int sh, int sw, int ph, int pw,
                          int OH, int OW, int K, int Kp,
                          int8_t *imc, float *rowsc) {
    const int is11 = (KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0);
#ifdef _OPENMP
    int nth = omp_get_max_threads(); if (nth < 1) nth = 1;
#else
    int nth = 1;
#endif
    float **rb = (float **)calloc((size_t)nth, sizeof(float *));
    if (!rb) return -1;
    for (int i = 0; i < nth; i++) {
        rb[i] = (float *)malloc((size_t)K * sizeof(float));
        if (!rb[i]) { for (int j = 0; j < i; j++) free(rb[j]); free(rb); return -1; }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OH * OW > 1024)
#endif
    for (int oy = 0; oy < OH; oy++) {
#ifdef _OPENMP
        float *row = rb[omp_get_thread_num()];
#else
        float *row = rb[0];
#endif
        for (int ox = 0; ox < OW; ox++) {
            const int pp = oy * OW + ox;
            int8_t *arow = imc + (size_t)pp * Kp;
            float mx = 0.f;
            if (is11) {
                const int q0 = oy * W + ox;
                for (int ic = 0; ic < C; ic++) {
                    float v = x[(size_t)ic * H * W + q0];
                    row[ic] = v;
                    float av = fabsf(v); if (av > mx) mx = av;
                }
            } else {
                int k = 0;
                const int iy0 = oy * sh - ph, ix0 = ox * sw - pw;
                const int interior = (iy0 >= 0 && iy0 + KH - 1 <= H - 1 &&
                                      ix0 >= 0 && ix0 + KW - 1 <= W - 1);
                if (interior) {                       /* 内部像素：无边界判断 */
                    for (int ic = 0; ic < C; ic++) {
                        const float *xc = x + (size_t)ic * H * W;
                        for (int ky = 0; ky < KH; ky++) {
                            const float *xr = xc + (size_t)(iy0 + ky) * W + ix0;
                            for (int kx = 0; kx < KW; kx++, k++) {
                                float v = xr[kx];
                                row[k] = v;
                                float av = fabsf(v); if (av > mx) mx = av;
                            }
                        }
                    }
                } else {
                    for (int ic = 0; ic < C; ic++) {
                        const float *xc = x + (size_t)ic * H * W;
                        for (int ky = 0; ky < KH; ky++) {
                            const int iy = iy0 + ky;
                            const float *xr = (iy >= 0 && iy < H) ? xc + (size_t)iy * W : NULL;
                            for (int kx = 0; kx < KW; kx++, k++) {
                                float v = 0.f;
                                if (xr) { const int ix = ix0 + kx; if (ix >= 0 && ix < W) v = xr[ix]; }
                                row[k] = v;
                                float av = fabsf(v); if (av > mx) mx = av;
                            }
                        }
                    }
                }
            }
            const float s = (mx > 0.f) ? (mx / 127.f) : 1.f;
            const float inv = 1.f / s;
            if (Kp > K) memset(arow + K, 0, (size_t)(Kp - K));
            /* Quantise row -> int8. `(int)(f + 2^23) - 2^23` is bit-identical
             * to lrintf() under the default FE_TONEAREST mode for |f| < 2^22
             * (all our values: |f| <= 127) and avoids a per-element libm call
             * (the im2col build is the single largest CPU cost of the M2 path,
             * K-YOLO-001 §9 #2). NOTE (K-YOLO-002 Phase 2): an explicit NEON
             * version (vmul+vadd(magic)+vcvtq) measured 2% SLOWER than GCC's
             * `omp simd` codegen, and plain `vcvtq_s32_f32` (FCVTZS = truncate)
             * is WRONG here (breaks round-to-nearest -> detections collapse);
             * the auto-vectorised scalar loop is kept. */
            const float quant_magic = 12582912.0f;   /* 2^23 */
#pragma omp simd
            for (int j = 0; j < K; j++) {
                int q = (int)(row[j] * inv + quant_magic) - 12582912;
                if (q > 127) q = 127; else if (q < -127) q = -127;
                arow[j] = (int8_t)q;
            }
            rowsc[pp] = s;
        }
    }
    for (int i = 0; i < nth; i++) free(rb[i]);
    free(rb);
    return 0;
}

/* ================================================================
 * PC 侧 int8 dotprod 卷积（K-YOLO-002 Phase 1）
 *
 * 与 M2（NPU）同量化语义：复用 yolo_im2col_q8 的 im2col int8 + 逐像素 scale，
 * 权重用同一份补齐到 Kp 的 int8 重排 m->nw[cid]=[OC][Kp]。GEMM 转置取向同 M2：
 *   out[oc][p] = Σ_k W[oc][k]·im2col[p][k]
 * 整数域用 vdotq_s32（ARMv8.2 dotprod，A76）；per-(oc,位置) 反量化 = cs[oc]·rowsc[p]。
 * 数学等价性：M2 的 per-block 反量化中 a_scale=b_scale 逐块同值 ⇒ 合成 scale
 *   cs[oc]·rowsc[p] 与「先整段 int32 点积、再一次乘 scale」**数值等价**（int32 精确）。
 * 返回 1 = 已处理；0 = 不适用（调用方回退标量 yconv_q8，J-CPU-4 显式）。 */

static int yconv_cpu_dotprod(StYoloModel *m, const float *x, int C, int H, int W,
                             uint32_t cid, const float *b, int OC, int KH, int KW,
                             int sh, int sw, int ph, int pw, int G, int act,
                             float *y, int OH, int OW) {
    if (G != 1) return 0;                              /* 分组卷积：回退标量 */
    if ((int)cid > m->max_cid || !m->cq[cid] || !m->cs[cid]) return 0;  /* 需 int8 权重 */
    const int K = C * KH * KW;
    const int Kp = (K + 31) & ~31;
    const int P = OH * OW;
    if (K <= 0 || P <= 0 || OC <= 0) return 0;

    /* 惰性构建补齐到 Kp 的 int8 权重 [OC][Kp]（与 NPU 路径同布局，供两条路复用）。 */
    if (!m->nw[cid]) {
        int8_t *w = (int8_t *)calloc((size_t)OC * Kp, 1);
        if (!w) return 0;
        const int8_t *src = m->cq[cid];
        for (int oc = 0; oc < OC; oc++)
            memcpy(w + (size_t)oc * Kp, src + (size_t)oc * K, (size_t)K);
        m->nw[cid] = w;
    }

    int8_t *imc = (int8_t *)malloc((size_t)P * Kp);
    float  *rowsc = (float *)malloc((size_t)P * sizeof(float));
    if (!imc || !rowsc) { free(imc); free(rowsc); return 0; }

    double _t0 = ynow();
    if (yolo_im2col_q8(x, C, H, W, KH, KW, sh, sw, ph, pw, OH, OW, K, Kp,
                       imc, rowsc) != 0) {
        free(imc); free(rowsc); return 0;
    }
    m->t_i2c += ynow() - _t0;

    const int8_t *Wq = m->nw[cid];
    const float  *cs = m->cs[cid];
    const int Kp16 = Kp & ~15;              /* Kp 为 32 倍数 ⇒ Kp16 == Kp（k=16 步长尾） */

    /* Phase 1.2：4-way ILP —— 同时算 4 个位置、4 条独立累加链，掩盖 vdotq 与
     * vaddvq_s32（水平归约）的延迟（小 Kp 时归约占比高，单链效率仅 ~50%）。
     * 每个位置的点积和顺序不变 ⇒ 逐位等价。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 4)
#endif
    for (int oc = 0; oc < OC; oc++) {
        const int8_t *wrow = Wq + (size_t)oc * Kp;
        const float oc_s = cs[oc];
        const float bias = b ? b[oc] : 0.f;
        float *yo = y + (size_t)oc * P;
        int p = 0;
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
        for (; p + 4 <= P; p += 4) {
            const int8_t *r0 = imc + (size_t)(p + 0) * Kp;
            const int8_t *r1 = imc + (size_t)(p + 1) * Kp;
            const int8_t *r2 = imc + (size_t)(p + 2) * Kp;
            const int8_t *r3 = imc + (size_t)(p + 3) * Kp;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            for (int k = 0; k < Kp16; k += 16) {
                int8x16_t wv = vld1q_s8(wrow + k);
                a0 = vdotq_s32(a0, wv, vld1q_s8(r0 + k));
                a1 = vdotq_s32(a1, wv, vld1q_s8(r1 + k));
                a2 = vdotq_s32(a2, wv, vld1q_s8(r2 + k));
                a3 = vdotq_s32(a3, wv, vld1q_s8(r3 + k));
            }
            int32_t s0 = vaddvq_s32(a0), s1 = vaddvq_s32(a1);
            int32_t s2 = vaddvq_s32(a2), s3 = vaddvq_s32(a3);
            for (int k = Kp16; k < Kp; k++) {
                s0 += (int32_t)wrow[k] * (int32_t)r0[k];
                s1 += (int32_t)wrow[k] * (int32_t)r1[k];
                s2 += (int32_t)wrow[k] * (int32_t)r2[k];
                s3 += (int32_t)wrow[k] * (int32_t)r3[k];
            }
            float f0 = (float)s0 * (oc_s * rowsc[p + 0]) + bias;
            float f1 = (float)s1 * (oc_s * rowsc[p + 1]) + bias;
            float f2 = (float)s2 * (oc_s * rowsc[p + 2]) + bias;
            float f3 = (float)s3 * (oc_s * rowsc[p + 3]) + bias;
            if (act == YACT_SILU) {
                f0 = f0 / (1.f + expf(-f0)); f1 = f1 / (1.f + expf(-f1));
                f2 = f2 / (1.f + expf(-f2)); f3 = f3 / (1.f + expf(-f3));
            }
            yo[p + 0] = f0; yo[p + 1] = f1; yo[p + 2] = f2; yo[p + 3] = f3;
        }
#endif
        for (; p < P; p++) {
            const int8_t *irow = imc + (size_t)p * Kp;
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < Kp16; k += 16)
                acc = vdotq_s32(acc, vld1q_s8(wrow + k), vld1q_s8(irow + k));
            int32_t s = vaddvq_s32(acc);
            for (int k = Kp16; k < Kp; k++) s += (int32_t)wrow[k] * (int32_t)irow[k];
#else
            int32_t s = 0;
            for (int k = 0; k < Kp; k++) s += (int32_t)wrow[k] * (int32_t)irow[k];
#endif
            float r = (float)s * (oc_s * rowsc[p]) + bias;
            if (act == YACT_SILU) r = r / (1.f + expf(-r));
            yo[p] = r;
        }
    }
    free(imc); free(rowsc);
    return 1;
}

static int yconv_npu_gemm(StYoloModel *m, const float *x, int C, int H, int W,
                     uint32_t cid, const float *b, int OC, int KH, int KW,
                     int sh, int sw, int ph, int pw, int G, int act,
                     float *y, int OH, int OW) {
    if (!m->npu || !vllm_npu_available(m->npu)) return 0;
    if (G != 1) return 0;                              /* 分组卷积（块对角）：NPU 不支持 */
    if ((int)cid > m->max_cid || !m->cq[cid] || !m->cs[cid]) return 0;  /* 需 int8 权重 */
    const int K = C * KH * KW;
    const int Kp = (K + 31) & ~31;
    const int P = OH * OW;                             /* GEMM 的 N 维 */
    if (K <= 0 || P <= 0 || OC <= 0) return 0;
    if (OC % 4 != 0 || OC > 1022) return 0;            /* HW M 维契约（%4 + 10-bit） */
    /* 体积护栏：driver 内部按 [P][Kp] 打包“权重”，限制单次展开规模。 */
    if ((double)P * (double)Kp > 96.0 * 1024 * 1024) return 0;

    /* 选最小提交数的 (Kb, Nt)（Kb=G 即 K 块宽）。 */
    int Kb = 0, NHint = 0;
    yolo_pick_tile(OC, Kp, P, &Kb, &NHint);
    if (Kb <= 0) return 0;
    const int n_blocks = Kp / Kb;
    long submits;
    {
        submits = (long)((P + NHint - 1) / NHint) * n_blocks;
        m->npu_submits += submits;
        if (getenv("VLLM_YOLO_NPU_DEBUG"))
            fprintf(stderr, "[NPU-SCHED] cid=%u OC=%d P=%d Kp=%d Kb=%d Nt=%d nb=%d submits=%ld\n",
                    cid, OC, P, Kp, Kb, NHint, n_blocks, submits);
    }

    /* 权重重排（惰性缓存，作为“特征 A”）：nw[cid]=[OC][Kp]（K..Kp 补 0）。 */
    if (!m->nw[cid]) {
        int8_t *w = (int8_t *)calloc((size_t)OC * Kp, 1);
        if (!w) return 0;
        const int8_t *src = m->cq[cid];
        for (int oc = 0; oc < OC; oc++)
            memcpy(w + (size_t)oc * Kp, src + (size_t)oc * K, (size_t)K);
        m->nw[cid] = w;
    }

    int8_t *imc = (int8_t *)malloc((size_t)P * Kp);              /* im2col（“权重 W”） */
    float  *rowsc = (float *)malloc((size_t)P * sizeof(float));  /* 逐像素 scale（单值） */
    float  *wsc = (float *)malloc((size_t)P * n_blocks * sizeof(float)); /* 权重角色 b_scale */
    float  *asc = (float *)malloc((size_t)OC * n_blocks * sizeof(float)); /* 特征角色 a_scale */
    float  *outf = (float *)calloc((size_t)OC * P, sizeof(float));       /* gemm_gw 累加，须零初始化 */
    if (!imc || !rowsc || !wsc || !asc || !outf) {
        free(imc); free(rowsc); free(wsc); free(asc); free(outf); return 0;
    }

    /* im2col + 逐像素（行）对称 int8 量化（共享助手 yolo_im2col_q8）。 */
    double _t0 = ynow();
    if (yolo_im2col_q8(x, C, H, W, KH, KW, sh, sw, ph, pw, OH, OW, K, Kp,
                       imc, rowsc) != 0) {
        free(imc); free(rowsc); free(wsc); free(asc); free(outf);
        return 0;
    }
    /* 展开为 per-block scale（每块同值 ⇒ 与 CPU per-channel 反量化语义一致）。 */
    for (int oc = 0; oc < OC; oc++) {
        const float s = m->cs[cid][oc];
        for (int g = 0; g < n_blocks; g++) asc[(size_t)oc * n_blocks + g] = s;
    }
    for (int pp = 0; pp < P; pp++) {
        const float s = rowsc[pp];
        for (int g = 0; g < n_blocks; g++) wsc[(size_t)pp * n_blocks + g] = s;
    }

    /* Aq=卷积权重（特征），Wq=im2col；M=OC，N=P，K=Kp，G=Kb，Nt=NHint。
     * wkey=0（im2col 每调用变化，禁权重缓存）。 */
    m->t_i2c += ynow() - _t0;
    double _t1 = ynow();
    int ok = vllm_npu_gemm_gw_ex(m->npu, outf, m->nw[cid], asc,
                                 imc, wsc, OC, P, Kp, Kb, 0, 0, NHint);
    m->t_npucall += ynow() - _t1;
    if (!ok) {
        m->npu_fallbacks++;
        if (getenv("VLLM_YOLO_NPU_DEBUG"))
            fprintf(stderr, "[NPU] fallback conv cid=%u OC=%d P=%d K=%d Kb=%d Nt=%d\n",
                    cid, OC, P, Kp, Kb, NHint);
        free(imc); free(rowsc); free(wsc); free(asc); free(outf);
        return 0;
    }

    /* 反量化 + bias + 激活；outf[oc*P+p] 即 y[oc*P+p] 布局。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 4)
#endif
    for (int oc = 0; oc < OC; oc++) {
        const float bias = b ? b[oc] : 0.f;
        float *yo = y + (size_t)oc * P;
        const float *oo = outf + (size_t)oc * P;
        for (int pp = 0; pp < P; pp++) {
            float v = oo[pp] + bias;
            if (act == YACT_SILU) v = v / (1.f + expf(-v));
            yo[pp] = v;
        }
    }

    /* J1（M2）：单算子数值门 —— 同输入跑 CPU 卷积对拍（env 门控，仅诊断）。 */
    if (getenv("VLLM_YOLO_NPU_CHECK")) {
        float *yref = (float *)malloc((size_t)OC * P * sizeof(float));
        if (yref) {
            yconv_q8(x, C, H, W, m->cq[cid], m->cs[cid], b, OC, KH, KW,
                     sh, sw, ph, pw, G, act, yref, OH, OW);
            double md = 0.0, mr = 0.0;
            for (size_t i = 0; i < (size_t)OC * P; i++) {
                double d = fabs((double)y[i] - (double)yref[i]);
                if (d > md) md = d;
                double den = fabs((double)yref[i]);
                if (den > 1e-6) { double r = d / den; if (r > mr) mr = r; }
            }
            fprintf(stderr, "[YOLO-J1] npu-conv cid=%u OC=%d P=%d K=%d maxabs=%.3e maxrel=%.3e\n",
                    cid, OC, P, K, md, mr);
            free(yref);
        }
    }
    m->npu_convs++;
    free(imc); free(rowsc); free(wsc); free(asc); free(outf);
    return 1;
}

/* ================================================================
 * M3 Phase-2：原生 CNA 卷积（kernel>1 单卷积直接上 NPU，不经 im2col）
 *
 * 提交数模型：把输出按 OC_band（16 的倍数）与 OH_band（CBUF 反推）分块，
 * submits = ceil(OC/OC_band) * ceil(OH/OH_band)。band 需满足 driver 的 CBUF 约束
 * （fd_banks=ceil(C_al*Hband*Wb/32768)<=10 且 fd_banks+w_banks<=11，w_banks=
 * ceil(KH*KW*C_al*OC_band/32768)）。
 *
 * 量化：激活**逐张量对称 int8**（整张卷积输入一个 max|x|/127）；权重仍用现有
 * per-channel cs[oc]。反量化 val = int32 * feat_scale * cs[oc] + bias（+SiLU）。
 * 特征按 NC1HWC2(C2=16) 预打包，halo 边界补 0，故传给硬件 pad=0（避免上/下越界
 * 的 PAD_TOP 不对称）。失败**显式回退**（silent_fallback_false_pass）。
 * ================================================================ */

/* 原生卷积权重 cube（有符号 int8，无 0x80 偏移）；源 [oc][ic][kh][kw]。 */
static int8_t *ypack_weight_cube(const int8_t *w, int C, int KH, int KW, int OC) {
    const int icb = ((C + 15) / 16) * 16;
    const int OCp = ((OC + 15) / 16) * 16;
    int8_t *cube = (int8_t *)calloc((size_t)KH * KW * OCp * icb, 1);
    if (!cube) return NULL;
    for (int oc = 0; oc < OC; oc++)
        for (int ky = 0; ky < KH; ky++)
            for (int kx = 0; kx < KW; kx++)
                for (int ic = 0; ic < C; ic++) {
                    int pos = (oc / 16) * KH * KW * 16 * icb
                            + (ky * KW + kx) * 16 * icb
                            + (oc % 16) * icb + ic;
                    cube[pos] = w[((oc * C + ic) * KH + ky) * KW + kx];
                }
    return cube;
}

/* 在 CBUF 约束下选最小提交数的 (OC_band, rb)。返回 submits；0 = 不可行。 */
static long yolo_pick_native(int C_al, int KH, int KW, int stride, int pad,
                             int OH, int W, int OC, int *ocb_out, int *rb_out) {
    const long BANK = 32768;
    const int Wb = W + 2 * pad;
    if (Wb <= 0 || OH <= 0) return 0;
    const long per_row = (long)C_al * Wb;
    const int OCp_full = ((OC + 15) / 16) * 16;
    long best = 0; int bo = 0, br = 0;
    for (int ocb = 16; ocb <= OCp_full; ocb += 16) {
        long w_bytes = (long)KH * KW * C_al * ocb;
        int w_banks = (int)((w_bytes + BANK - 1) / BANK);
        if (w_banks > 10) continue;
        int fd_max = 11 - w_banks;             /* fd_banks <= 10 && fd+w <= 11 */
        if (fd_max > 10) fd_max = 10;
        long max_Hband = (long)fd_max * BANK / per_row;
        if (max_Hband < KH) continue;
        int rb = (int)((max_Hband - KH) / stride) + 1;
        if (rb < 1) continue;
        if (rb > OH) rb = OH;
        long sub = (long)((OC + ocb - 1) / ocb) * ((OH + rb - 1) / rb);
        if (best == 0 || sub < best) { best = sub; bo = ocb; br = rb; }
    }
    if (best == 0) return 0;
    *ocb_out = bo; *rb_out = br;
    return best;
}

/* 执行原生卷积（按 ocb/rb 分块逐 band 提交）。返回 1 = 已在 NPU 完成。 */
static int yconv_npu_native(StYoloModel *m, const float *x, int C, int H, int W,
                            uint32_t cid, const float *b, int OC, int KH, int KW,
                            int sh, int sw, int ph, int pw, int G, int act,
                            float *y, int OH, int OW, int ocb, int rb) {
    const int C_al = ((C + 15) / 16) * 16;
    const int OCp_full = ((OC + 15) / 16) * 16;
    const int Wb = W + 2 * ph;
    if (Wb <= 0 || ocb <= 0 || rb <= 0) return 0;

    /* 逐张量对称 int8 激活量化。 */
    float mx = 0.f;
    {
        size_t n = (size_t)C * H * W;
        for (size_t i = 0; i < n; i++) { float v = fabsf(x[i]); if (v > mx) mx = v; }
    }
    const float fs = (mx > 0.f) ? (mx / 127.f) : 1.f;
    const float inv = 1.f / fs;

    /* 权重重排缓存（整 OC 一次；band 用 (oc0/16) 组偏移切片）。 */
    if (!m->ncw[cid]) {
        m->ncw[cid] = ypack_weight_cube(m->cq[cid], C, KH, KW, OC);
        if (!m->ncw[cid]) return 0;
    }
    const int8_t *cube = m->ncw[cid];

    const int Hmax = (rb - 1) * sh + KH;
    int8_t  *pbuf = (int8_t *)malloc((size_t)C_al * Hmax * Wb);
    int32_t *obuf = (int32_t *)malloc((size_t)OCp_full * rb * OW * sizeof(int32_t));
    if (!pbuf || !obuf) { free(pbuf); free(obuf); return 0; }

    long submits = 0;
    int ok = 1;
    for (int oc0 = 0; oc0 < OC && ok; oc0 += ocb) {
        const int ocn = (OC - oc0 < ocb) ? (OC - oc0) : ocb;
        const int8_t *wslice = cube + (size_t)(oc0 / 16) * KH * KW * 16 * C_al;
        for (int y0 = 0; y0 < OH && ok; y0 += rb) {
            const int rbn = (OH - y0 < rb) ? (OH - y0) : rb;
            const int Hband = (rbn - 1) * sh + KH;
            memset(pbuf, 0, (size_t)C_al * Hband * Wb);
            const int iy_top = y0 * sh - ph;
            for (int c = 0; c < C; c++) {
                const float *xc = x + (size_t)c * H * W;
                for (int hh = 0; hh < Hband; hh++) {
                    const int iy = iy_top + hh;
                    if (iy < 0 || iy >= H) continue;         /* 上/下越界 -> 0 */
                    for (int ww = 0; ww < Wb; ww++) {
                        const int iw = ww - ph;
                        float v = 0.f;
                        if (iw >= 0 && iw < W) v = xc[(size_t)iy * W + iw];
                        int q = (int)lrintf(v * inv);
                        if (q > 127) q = 127; else if (q < -127) q = -127;
                        pbuf[((c / 16) * Hband * Wb + hh * Wb + ww) * 16 + (c % 16)] = (int8_t)q;
                    }
                }
            }
            if (vllm_npu_conv_i32(m->npu, obuf, pbuf, wslice, C_al, Hband, Wb,
                                  KH, KW, sh, 0, rbn, OW, ocn, oc0, 0) != 0) {
                ok = 0; break;
            }
            submits++;
            /* 反量化 + bias + 激活。 */
            for (int oc = 0; oc < ocn; oc++) {
                const float sc = fs * m->cs[cid][oc0 + oc];
                const float bias = b ? b[oc0 + oc] : 0.f;
                const int32_t *oo = obuf + (size_t)oc * rbn * OW;
                float *yo = y + (size_t)(oc0 + oc) * OH * OW + (size_t)y0 * OW;
                for (int p = 0; p < rbn * OW; p++) {
                    float v = (float)oo[p] * sc + bias;
                    if (act == YACT_SILU) v = v / (1.f + expf(-v));
                    yo[p] = v;
                }
            }
        }
    }
    free(pbuf); free(obuf);
    if (!ok) {
        m->npu_fallbacks++;
        return 0;
    }
    m->npu_submits += submits;
    m->npu_native++;

    /* J1（M3）：单算子数值门 —— 同输入跑 CPU 卷积对拍（env 门控，仅诊断）。 */
    if (getenv("VLLM_YOLO_NPU_CHECK")) {
        float *yref = (float *)malloc((size_t)OC * OH * OW * sizeof(float));
        if (yref) {
            yconv_q8(x, C, H, W, m->cq[cid], m->cs[cid], b, OC, KH, KW,
                     sh, sw, ph, pw, G, act, yref, OH, OW);
            double md = 0.0, mr = 0.0;
            for (size_t i = 0; i < (size_t)OC * OH * OW; i++) {
                double d = fabs((double)y[i] - (double)yref[i]);
                if (d > md) md = d;
                double den = fabs((double)yref[i]);
                if (den > 1e-6) { double r = d / den; if (r > mr) mr = r; }
            }
            fprintf(stderr, "[YOLO-J1] npu-native cid=%u OC=%d C=%d OH=%d OW=%d %dx%d "
                            "maxabs=%.3e maxrel=%.3e\n",
                    cid, OC, C, OH, OW, KH, KW, md, mr);
            free(yref);
        }
    }
    return 1;
}

/* 卷积派发（NPU）：M2（im2col→GEMM）与 M3 原生卷积按**提交数**择优，取小者；
 * 原生不可行/失败时回退 M2（再失败由 yconv_dispatch 回退 CPU）。 */
static int yconv_npu(StYoloModel *m, const float *x, int C, int H, int W,
                     uint32_t cid, const float *b, int OC, int KH, int KW,
                     int sh, int sw, int ph, int pw, int G, int act,
                     float *y, int OH, int OW) {
    if (!m->npu || !vllm_npu_available(m->npu)) return 0;
    if (G != 1) return 0;                              /* 分组卷积（块对角）：NPU 不支持 */
    if ((int)cid > m->max_cid || !m->cq[cid] || !m->cs[cid]) return 0;

    /* ---- M2 路径提交数 ---- */
    const int K = C * KH * KW;
    const int Kp = (K + 31) & ~31;
    const int P = OH * OW;
    int m2_ok = (K > 0 && P > 0 && OC > 0 && OC % 4 == 0 && OC <= 1022 &&
                 (double)P * (double)Kp <= 96.0 * 1024 * 1024);
    int kb = 0, nt = 0;
    long sub_m2 = 0;
    if (m2_ok) {
        yolo_pick_tile(OC, Kp, P, &kb, &nt);
        if (kb <= 0) m2_ok = 0;
        else sub_m2 = (long)((P + nt - 1) / nt) * (Kp / kb);
    }

    /* ---- M3 原生卷积提交数 ----
     * CONSERVATIVE MODE (default OFF): the Phase-1 register recipe is only
     * validated for the tiny envelope (C=16, 8x8, 3x3, s1, p1); every larger
     * probe on the board wedged the DPU (job timeout -> soft reset) or produced
     * garbage, so the native path must NOT auto-activate. It is opt-in via
     * VLLM_YOLO_NPU_NATIVE=1 AND restricted to the tiny geometry, otherwise
     * yconv_npu always uses the proven M2 (im2col->GEMM) path. */
    long sub_nat = 0; int noc = 0, nrb = 0;
    const char *ne = getenv("VLLM_YOLO_NPU_NATIVE");
    int native_on = (ne && (ne[0] == '1' || ne[0] == 'y' || ne[0] == 'Y'));
    if (native_on && KH == 3 && KW == 3 && sh == 1 && ph == 1 && C == 16 &&
        W + 2 * ph <= 10 && OH <= 8) {
        const int C_al = ((C + 15) / 16) * 16;
        sub_nat = yolo_pick_native(C_al, KH, KW, sh, ph, OH, W, OC, &noc, &nrb);
    }

    if (getenv("VLLM_YOLO_NPU_DEBUG"))
        fprintf(stderr, "[NPU-SCHED] cid=%u OC=%d C=%d %dx%d s=%d p=%d OH=%d OW=%d "
                        "m2=%ld(nb=%d Nt=%d) native=%ld(ocb=%d rb=%d)\n",
                cid, OC, C, KH, KW, sh, ph, OH, OW, sub_m2,
                kb > 0 ? Kp / kb : 0, nt, sub_nat, noc, nrb);

    if (sub_nat > 0 && (!m2_ok || sub_nat < sub_m2)) {
        if (yconv_npu_native(m, x, C, H, W, cid, b, OC, KH, KW, sh, sw, ph, pw,
                             G, act, y, OH, OW, noc, nrb))
            return 1;
        if (getenv("VLLM_YOLO_NPU_DEBUG"))
            fprintf(stderr, "[NPU] native conv failed cid=%u -> M2 fallback\n", cid);
    }
    if (m2_ok)
        return yconv_npu_gemm(m, x, C, H, W, cid, b, OC, KH, KW, sh, sw, ph, pw,
                              G, act, y, OH, OW);
    return 0;
}

/* 卷积派发：优先 NPU（M2），不适用/失败显式回退 CPU（J4）。bias 取 m->cb[cid]。 */
static void yconv_dispatch(StYoloModel *m, const float *x, int C, int H, int W,
                           uint32_t cid, int hb, int OC, int KH, int KW,
                           int sh, int sw, int ph, int pw, int G, int act,
                           float *y, int OH, int OW) {
    const float *b = (hb && (int)cid <= m->max_cid) ? m->cb[cid] : NULL;
    if (m->npu && yconv_npu(m, x, C, H, W, cid, b, OC, KH, KW, sh, sw, ph, pw,
                            G, act, y, OH, OW))
        return;
    m->cpu_convs++;
    if (m->cw[cid]) {
        /* F32 权重：标量 F32 卷积（F32 模型无 int8 权重，dotprod 不适用）。 */
        yconv(x, C, H, W, m->cw[cid], b, OC, KH, KW, sh, sw, ph, pw, G, act, y, OH, OW);
        return;
    }
    /* Phase 1（K-YOLO-002）：CPU int8 dotprod 卷积（env VLLM_YOLO_CPU_DOTPROD，默认 ON）。
     * 不适用（分组/无 int8 权重等）时显式回退标量 yconv_q8（公理
     * silent_fallback_false_pass，J-CPU-4）。 */
    {
        static int g_dp = -1;
        if (g_dp < 0) {
            const char *e = getenv("VLLM_YOLO_CPU_DOTPROD");
            g_dp = (!e || e[0] == '\0' || e[0] != '0');
        }
        if (g_dp && yconv_cpu_dotprod(m, x, C, H, W, cid, b, OC, KH, KW, sh, sw,
                                      ph, pw, G, act, y, OH, OW))
            return;
        if (getenv("VLLM_YOLO_NPU_DEBUG"))
            fprintf(stderr, "[CPU] dotprod n/a cid=%u OC=%d C=%d K=%dx%d G=%d -> scalar\n",
                    cid, OC, C, KH, KW, G);
    }
    yconv_q8(x, C, H, W, m->cq[cid], m->cs[cid], b, OC, KH, KW, sh, sw, ph, pw,
             G, act, y, OH, OW);
}

static void yattn_psa(StYoloModel *m, const float *x, int C, int H, int W,
                      int nh, int hd, int kd,
                      uint32_t cqkv, uint32_t cproj, uint32_t cpe, float *y) {
    const int N = H * W;
    const int hq = nh * (2 * kd + hd);
    float *qkv = (float *)malloc((size_t)hq * N * sizeof(float));
    float *att = (float *)malloc((size_t)nh * N * N * sizeof(float));
    float *vbuf = (float *)malloc((size_t)C * N * sizeof(float));   /* 连续 v（pe 输入） */
    float *vm  = (float *)malloc((size_t)C * N * sizeof(float));    /* 注意力输出 o */
    float *out = (float *)malloc((size_t)C * N * sizeof(float));
    if (!qkv || !att || !vbuf || !vm || !out) {
        free(qkv); free(att); free(vbuf); free(vm); free(out); return;
    }
    yconv_dispatch(m, x, C, H, W, cqkv, m->cb[cqkv] != NULL, hq, 1, 1, 1, 1, 0, 0, 1, 0, qkv, H, W);
    const float scale = 1.f / sqrtf((float)kd);
    /* qkv[c][p], 每头块内布局 [q(kd) k(kd) v(hd)]；v 拷成连续 [nh,hd,N] */
    for (int h = 0; h < nh; h++) {
        const float *qh = qkv + (size_t)(h * (2 * kd + hd)) * N;
        const float *kh_ = qh + (size_t)kd * N;
        const float *vh = qh + (size_t)2 * kd * N;
        float *at = att + (size_t)h * N * N;
        for (int d = 0; d < hd; d++)
            memcpy(vbuf + (size_t)(h * hd + d) * N, vh + (size_t)d * N, (size_t)N * sizeof(float));
        /* K-YOLO-002 Phase 2：QK^T / softmax / att·V 每个 (h,i) 输出独立 ⇒ OMP over i。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float s = 0.f;
                for (int d = 0; d < kd; d++) s += qh[(size_t)d * N + i] * kh_[(size_t)d * N + j];
                at[(size_t)i * N + j] = s * scale;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
        for (int i = 0; i < N; i++) {           /* softmax over j */
            float mx = at[(size_t)i * N];
            for (int j = 1; j < N; j++) { float v = at[(size_t)i * N + j]; if (v > mx) mx = v; }
            float sum = 0.f;
            for (int j = 0; j < N; j++) { float e = expf(at[(size_t)i * N + j] - mx); at[(size_t)i * N + j] = e; sum += e; }
            float inv = 1.f / sum;
            for (int j = 0; j < N; j++) at[(size_t)i * N + j] *= inv;
        }
        for (int d = 0; d < hd; d++) {
            float *od = vm + (size_t)(h * hd + d) * N;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
            for (int i = 0; i < N; i++) {
                float s = 0.f;
                for (int j = 0; j < N; j++) s += vh[(size_t)d * N + j] * at[(size_t)i * N + j];
                od[i] = s;
            }
        }
    }
    /* o += pe(v)（pe 为 depthwise 3×3，NPU 不支持 → CPU） */
    yconv_dispatch(m, vbuf, C, H, W, cpe, m->cb[cpe] != NULL, C, 3, 3, 1, 1, 1, 1, C, 0, out, H, W);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)C * N > 65536)
#endif
    for (long long i = 0; i < (long long)C * N; i++) vm[i] += out[i];
    yconv_dispatch(m, vm, C, H, W, cproj, m->cb[cproj] != NULL, C, 1, 1, 1, 1, 0, 0, 1, 0, out, H, W);
    memcpy(y, out, (size_t)C * N * sizeof(float));
    free(qkv); free(att); free(vbuf); free(vm); free(out);
}

/* ---- 槽位 ---- */
typedef struct { float *buf; int c, h, w; } YSlot;

static float *slot_new(YSlot *S, uint32_t s, int c, int h, int w) {
    size_t n = (size_t)c * h * w;
    S[s].buf = (float *)malloc(n * sizeof(float));
    S[s].c = c; S[s].h = h; S[s].w = w;
    return S[s].buf;
}

/* ================================================================
 * 前向（图解释器）
 * ================================================================ */
static int yolo_forward(StYoloModel *m, const float *in, int iw, int ih,
                        YSlot *S, uint32_t *final_slot, int *final_c, int *final_a) {
    /* Phase-0 CPU 分解（K-YOLO-002）：env 门控，默认关（零开销）。 */
    int cpu_tim = 0;
    { const char *e = getenv("VLLM_YOLO_CPU_TIMING");
      cpu_tim = (e && e[0] && e[0] != '0'); }
    m->t_conv = m->t_detect = m->t_attn = m->t_elem = 0.0;
    m->n_conv_ops = m->n_elem_ops = 0;
    if (cpu_tim) {
        m->t_i2c = 0.0; m->t_npucall = 0.0;
        for (int k = 0; k < 12; k++) { m->t_opk[k] = 0.0; m->n_opk[k] = 0; }
    }
    double _fwd0 = cpu_tim ? ynow() : 0.0;
    slot_new(S, 0, 3, ih, iw);
    memcpy(S[0].buf, in, (size_t)3 * ih * iw * sizeof(float));

    for (int i = 0; i < m->n_ops; i++) {
        const YOp *o = &m->ops[i];
        const uint32_t *P = o->prm;
        double _op0 = cpu_tim ? ynow() : 0.0;
        switch (o->op) {
        case YOP_CONV: {
            uint32_t cid = P[0]; int hb = (int)P[1];
            int sh = (int)P[2], sw = (int)P[3], ph = (int)P[4], pw = (int)P[5];
            int G = (int)P[6], act = (int)P[7], KH = (int)P[8], KW = (int)P[9];
            YSlot *x = &S[o->ins[0]];
            int OH = (x->h + 2 * ph - KH) / sh + 1;
            int OW = (x->w + 2 * pw - KW) / sw + 1;
            int OC = m->crow[cid];
            if (OC <= 0) { fprintf(stderr, "[YOLO] conv cid=%u weight missing\n", cid); return -1; }
            float *y = slot_new(S, o->out, OC, OH, OW);
            if (!y) return -1;
            yconv_dispatch(m, x->buf, x->c, x->h, x->w, cid, hb, OC, KH, KW,
                           sh, sw, ph, pw, G, act, y, OH, OW);
            break;
        }
        case YOP_ADD: {
            YSlot *a = &S[o->ins[0]], *b = &S[o->ins[1]];
            long long n = (long long)a->c * a->h * a->w;
            float *y = slot_new(S, o->out, a->c, a->h, a->w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = a->buf[k] + b->buf[k];
            break;
        }
        case YOP_CONCAT: {
            int C = 0, h = S[o->ins[0]].h, w = S[o->ins[0]].w;
            for (uint32_t k = 0; k < o->nin; k++) C += S[o->ins[k]].c;
            float *y = slot_new(S, o->out, C, h, w);
            const size_t hw = (size_t)h * w;
            /* 各输入写入 y 的不相交区间 ⇒ 可并行（K-YOLO-002 Phase 2；前缀偏移先算）。 */
            if (o->nin <= 32) {
                size_t off[32];
                size_t acc = 0;
                for (uint32_t k = 0; k < o->nin; k++) {
                    off[k] = acc;
                    acc += (size_t)S[o->ins[k]].c * hw;
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(o->nin > 1 && acc > 65536)
#endif
                for (int k = 0; k < (int)o->nin; k++) {
                    YSlot *s = &S[o->ins[k]];
                    memcpy(y + off[k], s->buf, (size_t)s->c * hw * sizeof(float));
                }
            } else {
                size_t off = 0;
                for (uint32_t k = 0; k < o->nin; k++) {
                    YSlot *s = &S[o->ins[k]];
                    memcpy(y + off, s->buf, (size_t)s->c * hw * sizeof(float));
                    off += (size_t)s->c * hw;
                }
            }
            break;
        }
        case YOP_MAXPOOL: {
            YSlot *x = &S[o->ins[0]];
            int k = (int)P[0], st = (int)P[1], p = (int)P[2];
            int OH = (x->h + 2 * p - k) / st + 1, OW = (x->w + 2 * p - k) / st + 1;
            float *y = slot_new(S, o->out, x->c, OH, OW);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(x->c > 2)
#endif
            for (int c = 0; c < x->c; c++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++) {
                        float mx = -INFINITY;
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy * st - p + ky; if (iy < 0 || iy >= x->h) continue;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox * st - p + kx; if (ix < 0 || ix >= x->w) continue;
                                float v = x->buf[((size_t)c * x->h + iy) * x->w + ix];
                                if (v > mx) mx = v;
                            }
                        }
                        y[((size_t)c * OH + oy) * OW + ox] = mx;
                    }
            break;
        }
        case YOP_UPSAMPLE: {
            YSlot *x = &S[o->ins[0]];
            int sc = (int)P[0];
            int OH = x->h * sc, OW = x->w * sc;
            float *y = slot_new(S, o->out, x->c, OH, OW);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(x->c > 2)
#endif
            for (int c = 0; c < x->c; c++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++)
                        y[((size_t)c * OH + oy) * OW + ox] =
                            x->buf[((size_t)c * x->h + oy / sc) * x->w + ox / sc];
            break;
        }
        case YOP_SIGMOID: {
            YSlot *x = &S[o->ins[0]];
            long long n = (long long)x->c * x->h * x->w;
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = 1.f / (1.f + expf(-x->buf[k]));
            break;
        }
        case YOP_MUL: {
            YSlot *a = &S[o->ins[0]], *b = &S[o->ins[1]];
            long long n = (long long)a->c * a->h * a->w;
            float *y = slot_new(S, o->out, a->c, a->h, a->w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = a->buf[k] * b->buf[k];
            break;
        }
        case YOP_SLICE: {
            YSlot *x = &S[o->ins[0]];
            int a = (int)P[1], b2 = (int)P[2];
            int C = b2 - a;
            float *y = slot_new(S, o->out, C, x->h, x->w);
            const float *src = x->buf + (size_t)a * x->h * x->w;
            long long tot = (long long)C * x->h * x->w;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(tot > 65536)
#endif
            for (long long i = 0; i < tot; i++) y[i] = src[i];
            break;
        }
        case YOP_ATTN_PSA: {
            YSlot *x = &S[o->ins[0]];
            int nh = (int)P[0], hd = (int)P[1], kd = (int)P[2];
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            yattn_psa(m, x->buf, x->c, x->h, x->w, nh, hd, kd,
                      P[3], P[4], P[5], y);
            break;
        }
        case YOP_DETECT: {
            int nc = (int)P[0], rm = (int)P[1];
            int st[3] = { (int)P[2], (int)P[3], (int)P[4] };
            /* box slots = ins[0..2], cls slots = ins[3..5] */
            int A = 0, shapes[3][2];
            for (int s = 0; s < 3; s++) {
                YSlot *b = &S[o->ins[s]];
                shapes[s][0] = b->h; shapes[s][1] = b->w;
                A += b->h * b->w;
            }
            int C = 4 + nc;
            float *y = slot_new(S, o->out, C, A, 1);
            int base = 0;
            for (int s = 0; s < 3; s++) {
                YSlot *bx = &S[o->ins[s]];
                YSlot *cl = &S[o->ins[3 + s]];
                int h = shapes[s][0], w = shapes[s][1];
                /* 每个 anchor p 写 y[*][base+p] 互不相交 ⇒ 可并行（K-YOLO-002 Phase 2）。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(h * w > 256)
#endif
                for (int p = 0; p < h * w; p++) {
                    int px = p % w, py = p / w;
                    float ax = px + 0.5f, ay = py + 0.5f;
                    float d4[4];
                    for (int d = 0; d < 4; d++) {
                        const float *bin = bx->buf + (size_t)d * rm * h * w + p;
                        float mx = bin[0], sum = 0.f;
                        for (int j = 1; j < rm; j++) { float v = bin[(size_t)j * h * w]; if (v > mx) mx = v; }
                        float acc = 0.f;
                        for (int j = 0; j < rm; j++) {
                            float e = expf(bin[(size_t)j * h * w] - mx);
                            acc += e * (float)j; sum += e;
                        }
                        d4[d] = acc / sum;
                    }
                    float x1 = ax - d4[0], y1 = ay - d4[1], x2 = ax + d4[2], y2 = ay + d4[3];
                    float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
                    float bw = x2 - x1, bh = y2 - y1;
                    int q = base + p;
                    y[(size_t)0 * A + q] = cx * st[s];
                    y[(size_t)1 * A + q] = cy * st[s];
                    y[(size_t)2 * A + q] = bw * st[s];
                    y[(size_t)3 * A + q] = bh * st[s];
                    for (int c = 0; c < nc; c++) {
                        float v = cl->buf[(size_t)c * h * w + p];
                        y[(size_t)(4 + c) * A + q] = 1.f / (1.f + expf(-v));
                    }
                }
                base += h * w;
            }
            *final_slot = o->out; *final_c = C; *final_a = A;
            break;
        }
        default:
            fprintf(stderr, "[YOLO] unknown op %u @op%d\n", o->op, i);
            return -1;
        }
        if (cpu_tim) {
            const double dt = ynow() - _op0;
            if (o->op == YOP_CONV)      { m->t_conv  += dt; m->n_conv_ops++; }
            else if (o->op == YOP_DETECT){ m->t_detect += dt; }
            else if (o->op == YOP_ATTN_PSA) { m->t_attn += dt; }
            else                        { m->t_elem  += dt; m->n_elem_ops++; }
            if (o->op <= YOP_DETECT) { m->t_opk[o->op] += dt; m->n_opk[o->op]++; }
        }
    }
    if (cpu_tim) {
        m->t_fwd = ynow() - _fwd0;
        fprintf(stderr, "[YOLO-CPU] forward breakdown: total=%.1fms | conv=%.1fms (%ld ops, "
                        "im2col=%.1fms) elem=%.1fms (%ld ops) attn=%.1fms detect=%.1fms | fwd_other=%.1fms\n",
                m->t_fwd * 1e3, m->t_conv * 1e3, m->n_conv_ops, m->t_i2c * 1e3,
                m->t_elem * 1e3, m->n_elem_ops, m->t_attn * 1e3, m->t_detect * 1e3,
                (m->t_fwd - m->t_conv - m->t_elem - m->t_attn - m->t_detect) * 1e3);
        {   /* 逐算子类型（Phase 2 定位） */
            static const char *nm[12] = { "-", "CONV", "ADD", "CONCAT", "MAXPOOL",
                                          "UPSAMPLE", "SIGMOID", "MUL", "SLICE",
                                          "SOFTMAX", "ATTN_PSA", "DETECT" };
            fprintf(stderr, "[YOLO-CPU] ops:");
            for (int k = 1; k <= 11; k++)
                if (m->n_opk[k])
                    fprintf(stderr, " %s=%.1fms/%ld", nm[k], m->t_opk[k] * 1e3, m->n_opk[k]);
            fprintf(stderr, "\n");
        }
    }
    if (m->npu)
        fprintf(stderr, "[YOLO] NPU convs=%ld (native=%ld) cpu=%ld fallback=%ld submits=%ld "
                        "(im2col=%.3fs npu_call=%.3fs)\n",
                m->npu_convs, m->npu_native, m->cpu_convs, m->npu_fallbacks, m->npu_submits,
                m->t_i2c, m->t_npucall);
    return 0;
}

int st_yolo_forward_raw(StYoloModel *m, const float *nchw,
                        float *out, int *out_c, int *out_a) {
    if (!m || !nchw || !out) return -1;
    YSlot *S = (YSlot *)calloc((size_t)m->n_slots, sizeof(YSlot));
    uint32_t fs = 0; int fc = 0, fa = 0;
    int rc = yolo_forward(m, nchw, (int)m->det.input_w, (int)m->det.input_h,
                          S, &fs, &fc, &fa);
    if (rc == 0) {
        memcpy(out, S[fs].buf, (size_t)fc * fa * sizeof(float));
        if (out_c) *out_c = fc;
        if (out_a) *out_a = fa;
    }
    for (int i = 0; i < m->n_slots; i++) free(S[i].buf);
    free(S);
    return rc;
}

/* ================================================================
 * letterbox + 后处理（NMS）
 * ================================================================ */
/* letterbox（rect/auto=True 语义，对齐 ultralytics predict）：
 *   缩放到长边 = tw/th，再只补到 stride 倍数（最小矩形，不补成正方形）。
 * 输出尺寸 (oh, ow) 为 stride 倍数；返回 malloc 的 CHW float（[0,1]，114/255 填充）。 */
static float *letterbox_alloc(const uint8_t *rgb, int w, int h, int tw, int th, int stride,
                              int *out_w, int *out_h, float *scale,
                              float *padx, float *pady) {
    float r = fminf((float)tw / w, (float)th / h);
    int nw = (int)lroundf(w * r), nh = (int)lroundf(h * r);
    if (nw < 1) nw = 1; if (nh < 1) nh = 1;
    int dw = tw - nw, dh = th - nh;
    dw %= stride; dh %= stride;                    /* rect：只补到 stride 倍数 */
    int left = (int)lroundf(dw * 0.5f - 0.1f), right = (int)lroundf(dw * 0.5f + 0.1f);
    int top  = (int)lroundf(dh * 0.5f - 0.1f), bottom = (int)lroundf(dh * 0.5f + 0.1f);
    int ow = nw + left + right, oh = nh + top + bottom;
    float *out = (float *)malloc((size_t)3 * oh * ow * sizeof(float));
    if (!out) return NULL;
    size_t n = (size_t)3 * oh * ow;
    for (size_t i = 0; i < n; i++) out[i] = 114.f / 255.f;
    /* K-YOLO-002 Phase 2：x/y 映射只与 x（或 y）有关，预计算一次——原码把 x 映射
     * 放在每 y 行内重算整整 nh 次（nh×nw 次多余的除/floor/地址算）。同时把 c 提到
     * 最外层，使每个 (c,y) 行的写出连续（原码 c 内层为跨 oh·ow 的 3 路散射写）。
     * 逐点浮点表达式不变 ⇒ **逐位一致**。 */
    int   *xm = (int *)malloc((size_t)nw * 2 * sizeof(int));
    float *xf = (float *)malloc((size_t)nw * sizeof(float));
    int   *ym = (int *)malloc((size_t)nh * 2 * sizeof(int));
    float *yf = (float *)malloc((size_t)nh * sizeof(float));
    if (!xm || !xf || !ym || !yf) {
        free(xm); free(xf); free(ym); free(yf); free(out); return NULL;
    }
    for (int x = 0; x < nw; x++) {
        float sx = (x + 0.5f) * w / nw - 0.5f;
        int x0 = (int)floorf(sx); float fx = sx - x0;
        int x1 = x0 + 1; if (x0 < 0) x0 = 0; if (x1 > w - 1) x1 = w - 1;
        xm[x * 2] = x0; xm[x * 2 + 1] = x1; xf[x] = fx;
    }
    for (int y = 0; y < nh; y++) {
        float sy = (y + 0.5f) * h / nh - 0.5f;
        int y0 = (int)floorf(sy); float fy = sy - y0;
        int y1 = y0 + 1; if (y0 < 0) y0 = 0; if (y1 > h - 1) y1 = h - 1;
        ym[y * 2] = y0; ym[y * 2 + 1] = y1; yf[y] = fy;
    }
    const int crow = 3 * nh;    /* (c,y) 行数：行间写不相交 ⇒ 可并行 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(crow > 64)
#endif
    for (int ci = 0; ci < crow; ci++) {
        const int c = ci / nh, y = ci - c * nh;
        const int y0 = ym[y * 2], y1 = ym[y * 2 + 1];
        const float fy = yf[y], nfy = 1.f - fy;
        const uint8_t *r0 = rgb + (size_t)y0 * w * 3 + c;
        const uint8_t *r1 = rgb + (size_t)y1 * w * 3 + c;
        float *orow = out + ((size_t)c * oh + (y + top)) * ow + left;
        for (int x = 0; x < nw; x++) {
            const int x0 = xm[x * 2], x1 = xm[x * 2 + 1];
            const float fx = xf[x], nfx = 1.f - fx;
            float p00 = r0[(size_t)x0 * 3];
            float p01 = r0[(size_t)x1 * 3];
            float p10 = r1[(size_t)x0 * 3];
            float p11 = r1[(size_t)x1 * 3];
            float v = p00 * nfy * nfx + p01 * nfy * fx +
                      p10 * fy * nfx + p11 * fy * fx;
            orow[x] = v / 255.f;
        }
    }
    free(xm); free(xf); free(ym); free(yf);
    *out_w = ow; *out_h = oh; *scale = r; *padx = (float)left; *pady = (float)top;
    return out;
}

static float iou_xyxy(const float *a, const float *b) {
    float x1 = fmaxf(a[0], b[0]), y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]), y2 = fminf(a[3], b[3]);
    float iw = x2 - x1, ih = y2 - y1;
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return inter / ua;
}

typedef struct { float s; int i; } Cand;
/* 升序（qsort 约定：a<b 返回负）；NMS 从数组末尾（最高分）向前遍历 */
static int cand_cmp(const void *pa, const void *pb) {
    const Cand *a = (const Cand *)pa, *b = (const Cand *)pb;
    return (a->s > b->s) - (a->s < b->s);
}

/* ---------- K-YOLO-003：uint8 NHWC letterbox（官方 .rknn 固定输入） ----------
 * 与 VQF 路径的 rect letterbox 不同：官方 .rknn 输入固定 tw×th，故补满灰边(114)。
 * 双线性取样映射与 letterbox_alloc 同式（(i+0.5)*src/dst-0.5），保证坐标可对齐。 */
static uint8_t *letterbox_u8_alloc(const uint8_t *rgb, int w, int h, int tw, int th,
                                   float *scale, int *padx, int *pady) {
    float r = fminf((float)tw / w, (float)th / h);
    int nw = (int)lroundf(w * r), nh = (int)lroundf(h * r);
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    if (nw > tw) nw = tw;
    if (nh > th) nh = th;
    int left = (tw - nw) / 2, top = (th - nh) / 2;
    uint8_t *out = (uint8_t *)malloc((size_t)tw * th * 3);
    if (!out) return NULL;
    memset(out, 114, (size_t)tw * th * 3);
    for (int y = 0; y < nh; y++) {
        float sy = (y + 0.5f) * h / nh - 0.5f;
        int y0 = (int)floorf(sy); float fy = sy - y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y1 > h - 1) y1 = h - 1;
        uint8_t *orow = out + ((size_t)(y + top) * tw + left) * 3;
        for (int x = 0; x < nw; x++) {
            float sx = (x + 0.5f) * w / nw - 0.5f;
            int x0 = (int)floorf(sx); float fx = sx - x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x1 > w - 1) x1 = w - 1;
            for (int c = 0; c < 3; c++) {
                float p00 = rgb[((size_t)y0 * w + x0) * 3 + c];
                float p01 = rgb[((size_t)y0 * w + x1) * 3 + c];
                float p10 = rgb[((size_t)y1 * w + x0) * 3 + c];
                float p11 = rgb[((size_t)y1 * w + x1) * 3 + c];
                float v = p00 * (1 - fy) * (1 - fx) + p01 * (1 - fy) * fx +
                          p10 * fy * (1 - fx) + p11 * fy * fx;
                int iv = (int)lroundf(v);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                orow[x * 3 + c] = (uint8_t)iv;
            }
        }
    }
    if (scale) *scale = r;
    if (padx) *padx = left;
    if (pady) *pady = top;
    return out;
}

/* ---------- K-YOLO-003：官方 .rknn 整网解码（v8/v11：9 输出） ----------
 * 输出 = 3 scale × {reg[64], cls[nc], sum[1]}，NCHW。cls 已过 sigmoid（直接当概率）；
 * reg 为 DFL 的 16-bin logits（side*16+bin 通道），需 softmax 加权求 l/t/r/b。 */
static int yolo_rknn_detect(StYoloModel *m, const uint8_t *rgb, int w, int h,
                            float conf, float iou, YoloDet *out, int max_out) {
    if (conf <= 0) conf = m->det.conf_thres > 0 ? m->det.conf_thres : 0.25f;
    if (iou  <= 0) iou  = m->det.iou_thres  > 0 ? m->det.iou_thres  : 0.45f;

    int tw = 0, th = 0, tc = 0;
    vllm_npu_rknn_input_hwc(m->rknn, &th, &tw, &tc);
    static const int strides[3] = {8, 16, 32};
    int nc = (int)m->det.n_classes;

    double _l0 = ynow();
    float scale = 1.f; int padx = 0, pady = 0;
    uint8_t *in = letterbox_u8_alloc(rgb, w, h, tw, th, &scale, &padx, &pady);
    if (!in) return -1;
    double t_lb = ynow() - _l0;

    float *outs[16]; int ne[16];
    memset(outs, 0, sizeof(outs));
    memset(ne, 0, sizeof(ne));
    double _t0 = ynow();
    int rc = vllm_npu_rknn_run(m->rknn, in, outs, ne);
    double t_run = ynow() - _t0;
    free(in);
    if (rc != 0) {
        fprintf(stderr, "[RKNN-YOLO] run failed rc=%d (fallback)\n", rc);
        for (int i = 0; i < 16; i++) free(outs[i]);
        return -1;
    }

    /* 诊断/离线解码复核：VLLM_RKNN_DUMP=<dir> 转储 9 个输出 + 几何 */
    const char *dumpdir = getenv("VLLM_RKNN_DUMP");
    if (dumpdir && dumpdir[0]) {
        char p[512];
        FILE *mf = NULL;
        snprintf(p, sizeof(p), "%s/meta.txt", dumpdir);
        mf = fopen(p, "w");
        if (mf) fprintf(mf, "w=%d h=%d scale=%.8f padx=%d pady=%d nc=%d\n",
                        w, h, scale, padx, pady, nc);
        for (int i = 0; i < 9; i++) {
            int nd = 0, d[8];
            vllm_npu_rknn_output_dims(m->rknn, i, &nd, d, 8);
            if (mf) { fprintf(mf, "out%d:", i); for (int j = 0; j < nd; j++) fprintf(mf, " %d", d[j]); fprintf(mf, "\n"); }
            snprintf(p, sizeof(p), "%s/out%d.f32", dumpdir, i);
            FILE *of = fopen(p, "wb");
            if (of) { fwrite(outs[i], sizeof(float), (size_t)ne[i], of); fclose(of); }
        }
        if (mf) fclose(mf);
    }

    /* 收集候选（每 grid 点取最大类） */
    int gh[3] = {0, 0, 0}, gw[3] = {0, 0, 0};
    size_t total = 0;
    for (int s = 0; s < 3; s++) {
        int nd = 0, d[8];
        if (vllm_npu_rknn_output_dims(m->rknn, s * 3 + 1, &nd, d, 8) != 0 || nd != 4) {
            for (int i = 0; i < 16; i++) free(outs[i]);
            return -1;
        }
        gh[s] = d[2]; gw[s] = d[3];
        total += (size_t)gh[s] * gw[s];
    }
    Cand  *cand  = (Cand *)malloc((total > 0 ? total : 1) * sizeof(Cand));
    float *boxes = (float *)malloc((total > 0 ? total : 1) * 4 * sizeof(float));
    int   *clsv  = (int *)malloc((total > 0 ? total : 1) * sizeof(int));
    if (!cand || !boxes || !clsv) {
        free(cand); free(boxes); free(clsv);
        for (int i = 0; i < 16; i++) free(outs[i]);
        return -1;
    }
    int ncan = 0;
    for (int s = 0; s < 3; s++) {
        const float *reg = outs[s * 3 + 0];
        const float *cls = outs[s * 3 + 1];
        int H = gh[s], W = gw[s], st = strides[s];
        if (!reg || !cls || ne[s * 3 + 0] < 64 * H * W || ne[s * 3 + 1] < nc * H * W) continue;
        for (int gy = 0; gy < H; gy++) {
            for (int gx = 0; gx < W; gx++) {
                int pos = gy * W + gx;
                int bc = 0; float bs = -1.f;
                for (int c = 0; c < nc; c++) {
                    float v = cls[(size_t)c * H * W + pos];
                    if (v > bs) { bs = v; bc = c; }
                }
                if (bs < conf) continue;
                float dfl[4];
                for (int k = 0; k < 4; k++) {
                    const float *rp = reg + (size_t)(k * 16) * H * W + pos;
                    float mx = -1e30f;
                    for (int b = 0; b < 16; b++) { float v = rp[(size_t)b * H * W]; if (v > mx) mx = v; }
                    float sum = 0.f, acc = 0.f;
                    for (int b = 0; b < 16; b++) {
                        float e = expf(rp[(size_t)b * H * W] - mx);
                        sum += e; acc += e * (float)b;
                    }
                    dfl[k] = (sum > 0.f) ? acc / sum : 0.f;
                }
                float fx = (float)gx + 0.5f, fy = (float)gy + 0.5f;
                float *bx = boxes + (size_t)ncan * 4;
                bx[0] = (fx - dfl[0]) * st; bx[1] = (fy - dfl[1]) * st;
                bx[2] = (fx + dfl[2]) * st; bx[3] = (fy + dfl[3]) * st;
                clsv[ncan] = bc;
                cand[ncan].s = bs; cand[ncan].i = ncan;
                ncan++;
            }
        }
    }
    for (int i = 0; i < 16; i++) free(outs[i]);
    qsort(cand, (size_t)ncan, sizeof(Cand), cand_cmp);

    /* class-wise NMS（与 VQF 路径同口径） */
    char *dead = (char *)calloc((size_t)(ncan > 0 ? ncan : 1), 1);
    int nout = 0;
    for (int a = ncan - 1; a >= 0 && nout < max_out; a--) {
        int ia = cand[a].i;
        if (dead[ia]) continue;
        for (int b = a - 1; b >= 0; b--) {
            int ib = cand[b].i;
            if (dead[ib] || clsv[ib] != clsv[ia]) continue;
            if (iou_xyxy(boxes + (size_t)ia * 4, boxes + (size_t)ib * 4) > iou)
                dead[ib] = 1;
        }
        float *bx = boxes + (size_t)ia * 4;
        YoloDet *d = &out[nout++];
        d->x1 = (bx[0] - (float)padx) / scale; d->y1 = (bx[1] - (float)pady) / scale;
        d->x2 = (bx[2] - (float)padx) / scale; d->y2 = (bx[3] - (float)pady) / scale;
        if (d->x1 < 0) d->x1 = 0;
        if (d->y1 < 0) d->y1 = 0;
        if (d->x2 > w) d->x2 = (float)w;
        if (d->y2 > h) d->y2 = (float)h;
        d->score = cand[a].s;
        d->cls = clsv[ia];
    }
    free(cand); free(boxes); free(clsv); free(dead);

    /* K3-J4② 生效性凭证：每次 RKNN 推理打印一条 [RKNN-YOLO] 生效日志 */
    fprintf(stderr, "[RKNN-YOLO] run=%.1fms letterbox=%.1fms ncan=%d dets=%d\n",
            t_run * 1e3, t_lb * 1e3, ncan, nout);
    fflush(stderr);
    return nout;
}

int st_yolo_detect(StYoloModel *m, const uint8_t *rgb, int w, int h,
                   float conf, float iou, YoloDet *out, int max_out) {
    if (!m || !rgb || w <= 0 || h <= 0) return -1;
    if (m->rknn) return yolo_rknn_detect(m, rgb, w, h, conf, iou, out, max_out);
    int tw = (int)m->det.input_w, th = (int)m->det.input_h;
    int nc = (int)m->det.n_classes;
    if (conf <= 0) conf = m->det.conf_thres > 0 ? m->det.conf_thres : 0.25f;
    if (iou  <= 0) iou  = m->det.iou_thres  > 0 ? m->det.iou_thres  : 0.7f;

    float *in = NULL;
    float scale = 1.f, pw = 0.f, ph = 0.f;
    int ow = 0, oh = 0;
    int stride = (int)m->det.strides[2] > 0 ? (int)m->det.strides[2] : 32;
    int cpu_tim = 0;
    { const char *e = getenv("VLLM_YOLO_CPU_TIMING");
      cpu_tim = (e && e[0] && e[0] != '0'); }
    double _l0 = cpu_tim ? ynow() : 0.0;
    in = letterbox_alloc(rgb, w, h, tw, th, stride, &ow, &oh, &scale, &pw, &ph);
    if (!in) return -1;
    if (cpu_tim) m->t_lb = ynow() - _l0;
    {   /* 诊断：转储 letterbox 后的 NCHW 输入（VLLM_YOLO_DUMP_LB=<file>） */
        const char *dp = getenv("VLLM_YOLO_DUMP_LB");
        if (dp && dp[0]) {
            FILE *df = fopen(dp, "wb");
            if (df) { fwrite(in, sizeof(float), (size_t)3 * oh * ow, df); fclose(df); }
        }
    }

    YSlot *S = (YSlot *)calloc((size_t)m->n_slots, sizeof(YSlot));
    uint32_t fs = 0; int fc = 0, A = 0;
    int rc = yolo_forward(m, in, ow, oh, S, &fs, &fc, &A);
    free(in);
    if (rc != 0) { free(S); return -1; }
    double _p0 = cpu_tim ? ynow() : 0.0;
    const float *Y = S[fs].buf;

    /* 每 anchor 取最大类 */
    Cand *cand = (Cand *)malloc((size_t)A * sizeof(Cand));
    int ncan = 0;
    float *boxes = (float *)malloc((size_t)A * 4 * sizeof(float));
    int *clsv = (int *)malloc((size_t)A * sizeof(int));
    for (int q = 0; q < A; q++) {
        float best = -1.f; int bc = 0;
        for (int c = 0; c < nc; c++) {
            float v = Y[(size_t)(4 + c) * A + q];
            if (v > best) { best = v; bc = c; }
        }
        if (best < conf) continue;
        float cx = Y[(size_t)0 * A + q], cy = Y[(size_t)1 * A + q];
        float bw = Y[(size_t)2 * A + q], bh = Y[(size_t)3 * A + q];
        float *bx = boxes + (size_t)ncan * 4;
        bx[0] = cx - bw * 0.5f; bx[1] = cy - bh * 0.5f;
        bx[2] = cx + bw * 0.5f; bx[3] = cy + bh * 0.5f;
        clsv[ncan] = bc;
        cand[ncan].s = best; cand[ncan].i = ncan;
        ncan++;
    }
    qsort(cand, (size_t)ncan, sizeof(Cand), cand_cmp);   /* 升序 */
    if (getenv("VLLM_YOLO_DEBUG")) {
        fprintf(stderr, "[YOLO-dbg] ncan=%d conf=%.3f iou=%.3f\n", ncan, conf, iou);
        for (int t = 0; t < 5 && t < ncan; t++) {
            int ii = cand[ncan - 1 - t].i;
            fprintf(stderr, "[YOLO-dbg]   #%d score=%.4f cls=%d box640=[%.2f,%.2f,%.2f,%.2f]\n",
                    t, cand[ncan - 1 - t].s, clsv[ii],
                    boxes[ii*4], boxes[ii*4+1], boxes[ii*4+2], boxes[ii*4+3]);
        }
    }

    /* class-wise NMS（按 score 降序） */
    char *dead = (char *)calloc((size_t)(ncan > 0 ? ncan : 1), 1);
    int nout = 0;
    for (int a = ncan - 1; a >= 0 && nout < max_out; a--) {
        int ia = cand[a].i;
        if (dead[ia]) continue;
        for (int b = a - 1; b >= 0; b--) {
            int ib = cand[b].i;
            if (dead[ib] || clsv[ib] != clsv[ia]) continue;
            if (iou_xyxy(boxes + (size_t)ia * 4, boxes + (size_t)ib * 4) > iou)
                dead[ib] = 1;
        }
        float *bx = boxes + (size_t)ia * 4;
        YoloDet *d = &out[nout++];
        /* letterbox 反变换 → 原图坐标 */
        d->x1 = (bx[0] - pw) / scale; d->y1 = (bx[1] - ph) / scale;
        d->x2 = (bx[2] - pw) / scale; d->y2 = (bx[3] - ph) / scale;
        if (d->x1 < 0) d->x1 = 0; if (d->y1 < 0) d->y1 = 0;
        if (d->x2 > w) d->x2 = (float)w; if (d->y2 > h) d->y2 = (float)h;
        d->score = cand[a].s;
        d->cls = clsv[ia];
    }
    free(cand); free(boxes); free(clsv); free(dead);
    for (int i = 0; i < m->n_slots; i++) free(S[i].buf);
    free(S);
    if (cpu_tim) {
        m->t_post = ynow() - _p0;
        fprintf(stderr, "[YOLO-CPU] letterbox=%.1fms post(NMS)=%.1fms | fwd_total=%.1fms "
                        "| image_total=%.1fms (fwd+lb+post)\n",
                m->t_lb * 1e3, m->t_post * 1e3, m->t_fwd * 1e3,
                (m->t_fwd + m->t_lb + m->t_post) * 1e3);
    }
    return nout;
}
