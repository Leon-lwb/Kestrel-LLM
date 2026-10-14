/**
 * vllm_npu.c - Transparent RK3588 NPU acceleration backend
 *
 * See vllm_npu.h for the design contract. Implementation notes:
 *   - librknnrt.so is discovered with dlopen() and every rknn_* entry is
 *     resolved into a function-pointer table, so the binary runs unchanged
 *     on hosts without the NPU runtime (all entry points return "CPU path").
 *   - RKNN ABI types are declared here (opaque, field-order compatible with
 *     rknn_api.h) so this file compiles without the RKNN headers.
 *   - Operator-level models are loaded lazily per (M, N, K, precision) and
 *     cached in a small registry; a missing model or a shape below the FLOPs
 *     threshold transparently falls back to the CPU (NEON) path.
 */

#ifdef _WIN32
/* ================================================================
 * x86/Windows stub：NPU 为 RK3588 专属（/dev/rknpu + librknnrt）。
 * 所有入口透明降级 CPU 路径（available()==0），语义同"无 NPU 宿主"。
 * ================================================================ */
#include "vllm_npu.h"

int vllm_npu_init(vllm_npu_t **npu, const vllm_npu_cfg_t *cfg) {
    (void)npu; (void)cfg;
    return 0;   /* 成功但不可用：透明回退 */
}
void vllm_npu_destroy(vllm_npu_t *npu) { (void)npu; }
int vllm_npu_available(const vllm_npu_t *npu) { (void)npu; return 0; }
int vllm_npu_sdk_version(const vllm_npu_t *npu, char *buf, int len) {
    (void)npu;
    if (buf && len > 0) buf[0] = '\0';
    return -1;
}
int vllm_npu_gemm_f32(vllm_npu_t *npu, float *out, const float *A, const float *B,
                      int M, int N, int K) {
    (void)npu; (void)out; (void)A; (void)B; (void)M; (void)N; (void)K;
    return 0;   /* CPU 路径 */
}
int vllm_npu_gemm_i8(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *aw,
                     int M, int N, int K, int layer, int proj) {
    (void)npu; (void)out; (void)Aq; (void)aw;
    (void)M; (void)N; (void)K; (void)layer; (void)proj;
    return 0;
}
vllm_npu_backend_t vllm_npu_backend(const vllm_npu_t *npu) {
    (void)npu;
    return VLLM_NPU_BACKEND_AUTO;
}
int vllm_npu_gemm_gw(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *a_scale,
                     const int8_t *Wq, const float *b_scale,
                     int M, int N, int K, int G, int prec,
                     uint64_t wkey) {
    (void)npu; (void)out; (void)Aq; (void)a_scale;
    (void)Wq; (void)b_scale; (void)M; (void)N; (void)K;
    (void)G; (void)prec; (void)wkey;
    return 0;
}
int vllm_npu_gemm_gw_ex(vllm_npu_t *npu, float *out,
                        const int8_t *Aq, const float *a_scale,
                        const int8_t *Wq, const float *b_scale,
                        int M, int N, int K, int G, int prec,
                        uint64_t wkey, int nt_hint) {
    (void)npu; (void)out; (void)Aq; (void)a_scale;
    (void)Wq; (void)b_scale; (void)M; (void)N; (void)K;
    (void)G; (void)prec; (void)wkey; (void)nt_hint;
    return 0;
}
int vllm_npu_selftest(vllm_npu_t *npu) { (void)npu; return 0; }
int vllm_npu_conv_i32(vllm_npu_t *npu, int32_t *out,
                      const int8_t *feat_q, const int8_t *wq,
                      int C_al, int Hband, int W, int KH, int KW,
                      int stride, int pad, int OH_band, int OW, int OC,
                      int OC_off, int wbank_hint) {
    (void)npu; (void)out; (void)feat_q; (void)wq; (void)C_al; (void)Hband;
    (void)W; (void)KH; (void)KW; (void)stride; (void)pad; (void)OH_band;
    (void)OW; (void)OC; (void)OC_off; (void)wbank_hint;
    return -1;   /* no NPU on the x86/Windows stub */
}
/* K-YOLO-003：整网 RKNN（x86/Windows 无 librknnrt ⇒ 一律不可用，调用方回退）。 */
vllm_npu_rknn *vllm_npu_rknn_open(vllm_npu_t *npu, const char *path, int core_mask) {
    (void)npu; (void)path; (void)core_mask;
    return NULL;
}
void vllm_npu_rknn_close(vllm_npu_rknn *m) { (void)m; }
int vllm_npu_rknn_sdk(const vllm_npu_rknn *m, char *buf, int len) {
    (void)m; if (buf && len > 0) buf[0] = '\0'; return -1;
}
int vllm_npu_rknn_input_hwc(const vllm_npu_rknn *m, int *h, int *w, int *c) {
    (void)m; (void)h; (void)w; (void)c; return -1;
}
int vllm_npu_rknn_n_outputs(const vllm_npu_rknn *m) { (void)m; return -1; }
int vllm_npu_rknn_output_dims(const vllm_npu_rknn *m, int i,
                              int *n_dims, int *dims, int max_dims) {
    (void)m; (void)i; (void)n_dims; (void)dims; (void)max_dims; return -1;
}
void vllm_npu_rknn_dump(const vllm_npu_rknn *m) { (void)m; }
int vllm_npu_rknn_run(vllm_npu_rknn *m, const uint8_t *in,
                      float **out_data, int *out_nelems) {
    (void)m; (void)in; (void)out_data; (void)out_nelems; return -1;
}
#else

#include "vllm_npu.h"
#include "vllm_npu_direct.h"
#include "vllm_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

/* ================================================================
 * RKNN ABI (field-order compatible with rknn_api.h - do not reorder)
 * ================================================================ */
/* rknn_context：官方头在 32 位 ARM 上是 uint32，其余（含 aarch64/x86_64）是
 * uint64。旧版固定 int32 ⇒ rknn_init 写 8 字节破坏栈、ctx 被截断（K-YOLO-003 修正）。 */
#if defined(__arm__) && !defined(__aarch64__)
typedef uint32_t rknn_context_t;
#else
typedef uint64_t rknn_context_t;
#endif

/* Query commands — values match rknn_api.h v2.3.2 (K-YOLO-003 修正：
 * 旧表把 SDK_VERSION 当 0，实际 0=IN_OUT_NUM，导致版本查询取错结构）。 */
typedef enum {
    RKNN_QUERY_IN_OUT_NUM   = 0,
    RKNN_QUERY_INPUT_ATTR   = 1,
    RKNN_QUERY_OUTPUT_ATTR  = 2,
    RKNN_QUERY_PERF_RUN     = 4,
    RKNN_QUERY_SDK_VERSION  = 5
} rknn_query_cmd_t;

/* Tensor data types — official order (旧表 INT16/UINT8 互换，已修正). */
typedef enum {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8    = 2,
    RKNN_TENSOR_UINT8   = 3,
    RKNN_TENSOR_INT16   = 4,
    RKNN_TENSOR_UINT16  = 5,
    RKNN_TENSOR_INT32   = 6,
    RKNN_TENSOR_UINT32  = 7,
    RKNN_TENSOR_INT64   = 8,
    RKNN_TENSOR_BOOL    = 9
} rknn_tensor_type_t;

/* Tensor layouts — official order NCHW=0 / NHWC=1 (旧表两者颠倒，已修正). */
typedef enum {
    RKNN_TENSOR_NCHW      = 0,
    RKNN_TENSOR_NHWC      = 1,
    RKNN_TENSOR_NC1HWC2   = 2,
    RKNN_TENSOR_UNDEFINED = 3
} rknn_tensor_format_t;

/* rknn_input (v2.3.2)：{index, buf, size, pass_through, type, fmt}
 * （旧表把 attr 塞进结构且顺序不同，已按官方头修正。） */
typedef struct {
    uint32_t index;
    void    *buf;
    uint32_t size;
    uint8_t  pass_through;
    rknn_tensor_type_t   type;
    rknn_tensor_format_t fmt;
} rknn_input_t;

/* rknn_output (v2.3.2)：{want_float, is_prealloc, index, buf, size}
 * （旧表多出 pass_through/align_size/attr 三个字段，与官方头不符，已修正。） */
typedef struct {
    uint8_t  want_float;
    uint8_t  is_prealloc;
    uint32_t index;
    void    *buf;
    uint32_t size;
} rknn_output_t;

/* rknn_sdk_version (v2.3.2)：两个定长字符串（旧表多了 3 个 int，已修正）。 */
typedef struct {
    char api_version[256];
    char drv_version[256];
} rknn_sdk_version_t;

/* rknn_input_output_num (v2.3.2)。 */
typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num_t;

/* rknn_tensor_attr (v2.3.2) — 供整网模型查询输入/输出几何。 */
typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[16];
    char     name[256];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format_t fmt;
    rknn_tensor_type_t   type;
    uint32_t qnt_type;
    int8_t   fl;
    int32_t  zp;
    float    scale;
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint8_t  pass_through;
    uint32_t h_stride;
} rknn_tensor_attr_t;

typedef int32_t rknn_npu_core_mask_t;

/* Function-pointer table for the resolved librknnrt API. */
typedef struct {
    int (*rknn_init)(rknn_context_t *, void *, uint32_t, uint32_t, void *);
    int (*rknn_query)(rknn_context_t, rknn_query_cmd_t, void *, uint32_t);
    int (*rknn_inputs_set)(rknn_context_t, uint32_t, rknn_input_t *);
    int (*rknn_run)(rknn_context_t, void *);
    int (*rknn_outputs_get)(rknn_context_t, uint32_t, rknn_output_t *, void *);
    int (*rknn_outputs_release)(rknn_context_t, uint32_t, rknn_output_t *);
    int (*rknn_destroy)(rknn_context_t);
    int (*rknn_set_core_mask)(rknn_context_t, rknn_npu_core_mask_t);
} rknn_api_t;

/* ================================================================
 * DIRECT backend - the engine's own minimal rknpu userspace driver
 * (zero third-party dependencies; see vllm_npu_direct.h/.c).
 * Programs the NPU through the stock Rockchip rknpu kernel driver
 * (/dev/rknpu). The engine's Q8_0/Q4_0 block weights feed the hardware
 * as-is - no .rknn/.rkllm conversion, no external NPU library.
 * ================================================================ */

/* Cached operator model. */
#define VLLM_NPU_MAX_MODELS 32
typedef struct {
    int  M, N, K, prec;         /* prec: 0 = fp32 GEMM, 1 = int8 GEMM */
    int  layer, proj;           /* int8: (layer, projection) key */
    int  state;                 /* 0 = not attempted, 1 = loaded, -1 = failed */
    rknn_context_t ctx;
} npu_model_t;

struct vllm_npu_s {
    int         ok;             /* selected backend usable */
    vllm_npu_backend_t backend; /* live backend */
    int         verbosity;
    int         core_mask;
    double      flops_threshold;
    char        model_dir[512];
    char        sdk_version[128];
    int         core_count;
    /* RKNN backend */
    rknn_api_t  api;
    void       *dl;             /* dlopen handle to librknnrt.so */
    npu_model_t models[VLLM_NPU_MAX_MODELS];
    int         n_models;
    /* DIRECT backend (in-tree zero-dependency driver) */
    vllm_npu_direct_t *direct;
};

/* ================================================================
 * Runtime discovery
 * ================================================================ */
static void npu_resolve(vllm_npu_t *n, void *h) {
    /* rknn functions use the C calling convention; dlsym handles it. */
    *(void **)(&n->api.rknn_init)           = dlsym(h, "rknn_init");
    *(void **)(&n->api.rknn_query)          = dlsym(h, "rknn_query");
    *(void **)(&n->api.rknn_inputs_set)     = dlsym(h, "rknn_inputs_set");
    *(void **)(&n->api.rknn_run)            = dlsym(h, "rknn_run");
    *(void **)(&n->api.rknn_outputs_get)    = dlsym(h, "rknn_outputs_get");
    *(void **)(&n->api.rknn_outputs_release)= dlsym(h, "rknn_outputs_release");
    *(void **)(&n->api.rknn_destroy)        = dlsym(h, "rknn_destroy");
    *(void **)(&n->api.rknn_set_core_mask)  = dlsym(h, "rknn_set_core_mask");
    n->ok = (n->api.rknn_init && n->api.rknn_query && n->api.rknn_inputs_set &&
             n->api.rknn_run && n->api.rknn_outputs_get && n->api.rknn_destroy);
}

/* DIRECT backend discovery: open the in-tree zero-dependency driver. */
static void npu_direct_discover(vllm_npu_t *n, int verbosity) {
    n->direct = vllm_npu_direct_open(verbosity, n->core_mask);
}

int vllm_npu_init(vllm_npu_t **out, const vllm_npu_cfg_t *cfg) {
    if (!out) return -1;
    *out = NULL;
    vllm_npu_t *n = (vllm_npu_t *)calloc(1, sizeof(vllm_npu_t));
    if (!n) return -1;

    vllm_npu_backend_t want = cfg ? cfg->backend : VLLM_NPU_BACKEND_AUTO;
    n->verbosity = cfg ? cfg->verbosity : 0;
    n->core_mask = cfg ? cfg->core_mask : 0;
    n->flops_threshold = cfg ? cfg->flops_threshold : 1e8;
    if (cfg && cfg->model_dir)
        snprintf(n->model_dir, sizeof(n->model_dir), "%s", cfg->model_dir);

    if (want == VLLM_NPU_BACKEND_AUTO || want == VLLM_NPU_BACKEND_DIRECT) {
        /* Preferred: the engine's own minimal rknpu driver - zero deps,
         * no model conversion. Falls through to RKNN only when explicitly
         * requested or no /dev/rknpu node exists. */
        npu_direct_discover(n, n->verbosity);
        if (n->direct) {
            n->ok = 1;
            n->backend = VLLM_NPU_BACKEND_DIRECT;
            snprintf(n->sdk_version, sizeof(n->sdk_version),
                     "in-tree direct rknpu driver (/dev/rknpu)");
        } else if (want == VLLM_NPU_BACKEND_DIRECT) {
            fprintf(stderr, "[NPU] direct backend unavailable: no /dev/rknpu "
                    "node (is the rknpu kernel driver loaded?). CPU path.\n");
        }
    }
    if (n->ok == 0 && (want == VLLM_NPU_BACKEND_AUTO || want == VLLM_NPU_BACKEND_RKNN)) {
        n->dl = dlopen(cfg && cfg->rknn_path ? cfg->rknn_path : "librknnrt.so",
                       RTLD_NOW | RTLD_LOCAL);
        if (n->dl) {
            npu_resolve(n, n->dl);
            if (n->ok && n->api.rknn_query) {
                rknn_sdk_version_t ver;
                memset(&ver, 0, sizeof(ver));
                if (n->api.rknn_query(0, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0) {
                    snprintf(n->sdk_version, sizeof(n->sdk_version), "%s (drv %s)",
                             ver.api_version, ver.drv_version);
                } else {
                    snprintf(n->sdk_version, sizeof(n->sdk_version), "queried");
                }
            }
            if (n->ok) n->backend = VLLM_NPU_BACKEND_RKNN;
        }
    }
    if (n->ok && n->verbosity > 0)
        fprintf(stderr, "[NPU] backend=%s (%s)\n",
                n->backend == VLLM_NPU_BACKEND_DIRECT ? "direct" : "rknn",
                n->sdk_version[0] ? n->sdk_version : "?");
    *out = n;
    return 0;
}

void vllm_npu_destroy(vllm_npu_t *n) {
    if (!n) return;
    for (int i = 0; i < n->n_models; i++) {
        if (n->models[i].state == 1 && n->api.rknn_destroy)
            n->api.rknn_destroy(n->models[i].ctx);
    }
    if (n->dl) dlclose(n->dl);
    vllm_npu_direct_close(n->direct);
    free(n);
}

int vllm_npu_available(const vllm_npu_t *n) {
    return (n && n->ok);
}

int vllm_npu_sdk_version(const vllm_npu_t *n, char *buf, int len) {
    if (!n || !n->ok || !buf || len <= 0) return -1;
    int l = (int)strlen(n->sdk_version);
    if (l >= len) l = len - 1;
    memcpy(buf, n->sdk_version, (size_t)l);
    buf[l] = 0;
    return l;
}

vllm_npu_backend_t vllm_npu_backend(const vllm_npu_t *n) {
    return (n && n->ok) ? n->backend : VLLM_NPU_BACKEND_AUTO;
}

/* ================================================================
 * Group-wise quantized GEMM (DIRECT backend) - no model conversion.
 *
 * The engine's Q8_0 (prec=0) / Q4_0 (prec=1) block weights map 1:1 onto the
 * NPU group-wise matmul: group G = 32 == one block, scales per block.
 *   C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)
 * ================================================================ */
int vllm_npu_gemm_gw(vllm_npu_t *n, float *out,
                     const int8_t *Aq, const float *a_scale,
                     const int8_t *Wq, const float *b_scale,
                     int M, int N, int K, int G, int prec,
                     uint64_t wkey) {
    return vllm_npu_gemm_gw_ex(n, out, Aq, a_scale, Wq, b_scale,
                               M, N, K, G, prec, wkey, 0);
}

int vllm_npu_gemm_gw_ex(vllm_npu_t *n, float *out,
                        const int8_t *Aq, const float *a_scale,
                        const int8_t *Wq, const float *b_scale,
                        int M, int N, int K, int G, int prec,
                        uint64_t wkey, int nt_hint) {
    /* Diagnostic: VLLM_NPU_FORCE_CPU=1 keeps the NPU initialized but makes
     * every offload point fall back to the CPU - separates "NPU init side
     * effect" from "NPU output used" as the source of a wrong result. */
    if (getenv("VLLM_NPU_FORCE_CPU")) return 0;
    if (!n || !n->ok || n->backend != VLLM_NPU_BACKEND_DIRECT)
        return 0;
    if (!n->direct) return 0;
    if (!out || !Aq || !a_scale || !Wq || !b_scale) return 0;
    if (M <= 0 || N <= 0 || K <= 0 || G <= 0) return 0;
    if (M % 4 != 0) return 0;         /* HW M%4==0 contract; M==1 stays CPU */
    if (K % G != 0) return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    int rc = vllm_npu_direct_gemm_gw(n->direct, out, Aq, a_scale, Wq, b_scale,
                                     M, N, K, G, prec, wkey, nt_hint);
    return (rc == 0) ? 1 : 0;   /* -2 (uncalibrated regcmd) -> CPU fallback */
}

int vllm_npu_conv_i32(vllm_npu_t *n, int32_t *out,
                      const int8_t *feat_q, const int8_t *wq,
                      int C_al, int Hband, int W, int KH, int KW,
                      int stride, int pad, int OH_band, int OW, int OC,
                      int OC_off, int wbank_hint) {
    if (getenv("VLLM_NPU_FORCE_CPU")) return -1;
    if (!n || !n->ok || n->backend != VLLM_NPU_BACKEND_DIRECT || !n->direct)
        return -1;
    return vllm_npu_direct_conv_i32(n->direct, out, feat_q, wq, C_al, Hband, W,
                                    KH, KW, stride, pad, OH_band, OW, OC,
                                    OC_off, wbank_hint);
}

/* ================================================================
 * Model registry (lazy per-shape load of exported operator .rknn)
 * ================================================================ */
static npu_model_t *npu_find_model(vllm_npu_t *n, int M, int N, int K, int prec,
                                   int layer, int proj) {
    for (int i = 0; i < n->n_models; i++)
        if (n->models[i].M == M && n->models[i].N == N &&
            n->models[i].K == K && n->models[i].prec == prec &&
            n->models[i].layer == layer && n->models[i].proj == proj)
            return &n->models[i];
    return NULL;
}

static npu_model_t *npu_load_model(vllm_npu_t *n, int M, int N, int K, int prec,
                                   int layer, int proj) {
    npu_model_t *m = npu_find_model(n, M, N, K, prec, layer, proj);
    if (m) return (m->state == 1) ? m : NULL;

    if (n->n_models >= VLLM_NPU_MAX_MODELS) return NULL;   /* registry full */
    if (!n->model_dir[0]) return NULL;                     /* no model dir */

    char path[640];
    if (prec == 0) {
        /* generic fp32 MatMul (A + B as runtime inputs), one file per shape */
        snprintf(path, sizeof(path), "%s/gemm_%dx%dx%d.rknn", n->model_dir, M, N, K);
    } else {
        /* per-(layer, projection) int8 GEMM with baked weights */
        snprintf(path, sizeof(path), "%s/gemm_i8_p%d_l%d.rknn", n->model_dir, proj, layer);
    }

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }

    m = &n->models[n->n_models];
    m->M = M; m->N = N; m->K = K; m->prec = prec;
    m->layer = layer; m->proj = proj;
    m->ctx = 0; m->state = -1;
    n->n_models++;

    if (n->api.rknn_init(&m->ctx, buf, (uint32_t)sz, 0, NULL) != 0) {
        if (n->verbosity > 1)
            fprintf(stderr, "[NPU] rknn_init failed: %s\n", path);
        free(buf);
        return NULL;
    }
    free(buf);   /* rknn_init copies the model */
    m->state = 1;

    if (n->api.rknn_set_core_mask && n->core_mask != VLLM_NPU_CORE_AUTO)
        n->api.rknn_set_core_mask(m->ctx, (rknn_npu_core_mask_t)n->core_mask);

    if (n->verbosity > 1)
        fprintf(stderr, "[NPU] model loaded: %s\n", path);
    return m;
}

/* ================================================================
 * GEMM offload entry points (transparent: return 0 = run CPU path)
 * ================================================================ */
int vllm_npu_gemm_f32(vllm_npu_t *n, float *out,
                      const float *A, const float *B,
                      int M, int N, int K) {
    if (!n || !n->ok || !out || !A || !B || M <= 0 || N <= 0 || K <= 0)
        return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    npu_model_t *m = npu_load_model(n, M, N, K, 0, 0, 0);
    if (!m) return 0;

    rknn_input_t in[2];
    memset(in, 0, sizeof(in));
    in[0].index = 0; in[0].buf = (void *)A; in[0].size = (uint32_t)((size_t)M * K * 4);
    in[0].fmt = RKNN_TENSOR_NCHW; in[0].type = RKNN_TENSOR_FLOAT32;
    in[1].index = 1; in[1].buf = (void *)B; in[1].size = (uint32_t)((size_t)K * N * 4);
    in[1].fmt = RKNN_TENSOR_NCHW; in[1].type = RKNN_TENSOR_FLOAT32;

    rknn_output_t outd[1];
    memset(outd, 0, sizeof(outd));
    outd[0].want_float = 1;
    outd[0].buf = out;
    outd[0].size = (uint32_t)((size_t)M * N * 4);
    outd[0].is_prealloc = 1;

    if (n->api.rknn_inputs_set(m->ctx, 2, in) != 0) return 0;
    if (n->api.rknn_run(m->ctx, NULL) != 0) return 0;
    if (n->api.rknn_outputs_get(m->ctx, 1, outd, NULL) != 0) return 0;
    n->api.rknn_outputs_release(m->ctx, 1, outd);
    return 1;
}

int vllm_npu_gemm_i8(vllm_npu_t *n, float *out,
                     const int8_t *Aq, const float *aw,
                     int M, int N, int K, int layer, int proj) {
    if (!n || !n->ok || !out || !Aq || !aw || M <= 0 || N <= 0 || K <= 0)
        return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    npu_model_t *m = npu_load_model(n, M, N, K, 1, layer, proj);
    if (!m) return 0;

    /* Inputs: Aq int8 [M*K], aw fp32 [M]. Output: out fp32 [M*N].
     * The exported graph (tools/npu/npu_export_ops.py) has the int8 weights baked
     * as constants and dequantizes with per-channel scales internally. */
    rknn_input_t in[2];
    memset(in, 0, sizeof(in));
    in[0].index = 0; in[0].buf = (void *)Aq; in[0].size = (uint32_t)((size_t)M * K);
    in[0].fmt = RKNN_TENSOR_NCHW; in[0].type = RKNN_TENSOR_INT8;
    in[1].index = 1; in[1].buf = (void *)aw; in[1].size = (uint32_t)((size_t)M * 4);
    in[1].fmt = RKNN_TENSOR_NCHW; in[1].type = RKNN_TENSOR_FLOAT32;

    rknn_output_t outd[1];
    memset(outd, 0, sizeof(outd));
    outd[0].want_float = 1;
    outd[0].buf = out;
    outd[0].size = (uint32_t)((size_t)M * N * 4);
    outd[0].is_prealloc = 1;

    if (n->api.rknn_inputs_set(m->ctx, 2, in) != 0) return 0;
    if (n->api.rknn_run(m->ctx, NULL) != 0) return 0;
    if (n->api.rknn_outputs_get(m->ctx, 1, outd, NULL) != 0) return 0;
    n->api.rknn_outputs_release(m->ctx, 1, outd);
    return 1;
}

/* ================================================================
 * K-YOLO-003：整网 RKNN 模型（官方 .rknn 检测路径，opt-in 第二路径）
 *
 * 复用上面 dlopen 出来的 librknnrt 函数表，加载 Rockchip 官方**整网** .rknn
 * （如 rknn_model_zoo 的 yolov8.rknn），喂 uint8 NHWC 图像、取回 N 个 fp32 输出；
 * 家族化的解码/NMS 留在 YOLO 层（本层只管张量）。
 * 与 DIRECT/VQF 默认路径并存：打不开就返回 NULL，调用方显式回退，绝不静默降级。
 * ================================================================ */
struct vllm_npu_rknn {
    rknn_api_t         *api;
    rknn_context_t      ctx;
    int                 n_in, n_out;
    int                 in_h, in_w, in_c;
    rknn_tensor_attr_t *out_attr;      /* [n_out] */
    char                sdk[160];
};

vllm_npu_rknn *vllm_npu_rknn_open(vllm_npu_t *n, const char *path, int core_mask) {
    if (!n || !n->ok || !path || !n->api.rknn_init) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[RKNN-YOLO] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void *mb = malloc((size_t)sz);
    if (!mb) { fclose(f); return NULL; }
    if (fread(mb, 1, (size_t)sz, f) != (size_t)sz) { free(mb); fclose(f); return NULL; }
    fclose(f);

    vllm_npu_rknn *m = (vllm_npu_rknn *)calloc(1, sizeof(*m));
    if (!m) { free(mb); return NULL; }
    m->api = &n->api;
    if (n->api.rknn_init(&m->ctx, mb, (uint32_t)sz, 0, NULL) != 0) {
        fprintf(stderr, "[RKNN-YOLO] rknn_init failed: %s\n", path);
        free(mb); free(m); return NULL;
    }
    free(mb);   /* rknn_init copies the model */
    if (n->api.rknn_set_core_mask && core_mask != 0)
        n->api.rknn_set_core_mask(m->ctx, (rknn_npu_core_mask_t)core_mask);

    rknn_sdk_version_t ver; memset(&ver, 0, sizeof(ver));
    if (n->api.rknn_query(m->ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
        snprintf(m->sdk, sizeof(m->sdk), "%s (drv %s)", ver.api_version, ver.drv_version);

    rknn_input_output_num_t ioc; memset(&ioc, 0, sizeof(ioc));
    if (n->api.rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &ioc, sizeof(ioc)) != 0 ||
        ioc.n_input < 1 || ioc.n_output < 1) {
        fprintf(stderr, "[RKNN-YOLO] IN_OUT_NUM query failed\n");
        n->api.rknn_destroy(m->ctx); free(m); return NULL;
    }
    m->n_in = (int)ioc.n_input;
    m->n_out = (int)ioc.n_output;

    rknn_tensor_attr_t ia; memset(&ia, 0, sizeof(ia));
    if (n->api.rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof(ia)) == 0 && ia.n_dims == 4) {
        m->in_h = (int)ia.dims[1]; m->in_w = (int)ia.dims[2]; m->in_c = (int)ia.dims[3];
    }
    if (m->in_h <= 0 || m->in_w <= 0 || m->in_c != 3) {
        fprintf(stderr, "[RKNN-YOLO] unsupported input geometry (h=%d w=%d c=%d)\n",
                m->in_h, m->in_w, m->in_c);
        n->api.rknn_destroy(m->ctx); free(m); return NULL;
    }

    m->out_attr = (rknn_tensor_attr_t *)calloc((size_t)m->n_out, sizeof(*m->out_attr));
    if (!m->out_attr) { n->api.rknn_destroy(m->ctx); free(m); return NULL; }
    for (int i = 0; i < m->n_out; i++) {
        m->out_attr[i].index = (uint32_t)i;
        if (n->api.rknn_query(m->ctx, RKNN_QUERY_OUTPUT_ATTR,
                              &m->out_attr[i], sizeof(m->out_attr[i])) != 0)
            m->out_attr[i].n_dims = 0;
    }
    fprintf(stderr, "[RKNN-YOLO] full-network model ready: %s "
            "(in %dx%dx%d, %d outputs, sdk %s)\n",
            path, m->in_h, m->in_w, m->in_c, m->n_out,
            m->sdk[0] ? m->sdk : "?");
    return m;
}

void vllm_npu_rknn_close(vllm_npu_rknn *m) {
    if (!m) return;
    if (m->api && m->api->rknn_destroy) m->api->rknn_destroy(m->ctx);
    free(m->out_attr);
    free(m);
}

int vllm_npu_rknn_sdk(const vllm_npu_rknn *m, char *buf, int len) {
    if (!m || !buf || len <= 0) return -1;
    int l = (int)strlen(m->sdk);
    if (l >= len) l = len - 1;
    memcpy(buf, m->sdk, (size_t)l); buf[l] = 0;
    return l;
}

int vllm_npu_rknn_input_hwc(const vllm_npu_rknn *m, int *h, int *w, int *c) {
    if (!m) return -1;
    if (h) *h = m->in_h;
    if (w) *w = m->in_w;
    if (c) *c = m->in_c;
    return 0;
}

int vllm_npu_rknn_n_outputs(const vllm_npu_rknn *m) { return m ? m->n_out : -1; }

int vllm_npu_rknn_output_dims(const vllm_npu_rknn *m, int i,
                              int *n_dims, int *dims, int max_dims) {
    if (!m || i < 0 || i >= m->n_out) return -1;
    if (n_dims) *n_dims = (int)m->out_attr[i].n_dims;
    if (dims) {
        int nd = (int)m->out_attr[i].n_dims;
        if (nd > max_dims) nd = max_dims;
        for (int j = 0; j < nd; j++) dims[j] = (int)m->out_attr[i].dims[j];
    }
    return 0;
}

void vllm_npu_rknn_dump(const vllm_npu_rknn *m) {
    if (!m) return;
    fprintf(stderr, "[RKNN-YOLO] input %dx%dx%d (NHWC uint8); %d outputs:\n",
            m->in_h, m->in_w, m->in_c, m->n_out);
    for (int i = 0; i < m->n_out; i++) {
        const rknn_tensor_attr_t *a = &m->out_attr[i];
        fprintf(stderr, "[RKNN-YOLO]   OUT[%d] nd=%u dims=", i, a->n_dims);
        for (uint32_t j = 0; j < a->n_dims && j < 8; j++) fprintf(stderr, "%u ", a->dims[j]);
        fprintf(stderr, "nelems=%u zp=%d scale=%g\n", a->n_elems, a->zp, a->scale);
    }
}

int vllm_npu_rknn_run(vllm_npu_rknn *m, const uint8_t *in,
                      float **out_data, int *out_nelems) {
    if (!m || !in || !out_data || !out_nelems) return -1;
    rknn_input_t inrec; memset(&inrec, 0, sizeof(inrec));
    inrec.index        = 0;
    inrec.buf          = (void *)in;
    inrec.size         = (uint32_t)(m->in_h * m->in_w * m->in_c);
    inrec.pass_through = 0;                       /* RKNN 自行量化 */
    inrec.type         = RKNN_TENSOR_UINT8;
    inrec.fmt          = RKNN_TENSOR_NHWC;
    if (m->api->rknn_inputs_set(m->ctx, 1, &inrec) != 0) return -2;
    if (m->api->rknn_run(m->ctx, NULL) != 0) return -3;

    rknn_output_t *outs = (rknn_output_t *)calloc((size_t)m->n_out, sizeof(rknn_output_t));
    if (!outs) return -4;
    for (int i = 0; i < m->n_out; i++) {
        outs[i].index       = (uint32_t)i;
        outs[i].want_float  = 1;                  /* 取 fp32，省去手写 dequant */
        outs[i].is_prealloc = 0;
    }
    if (m->api->rknn_outputs_get(m->ctx, (uint32_t)m->n_out, outs, NULL) != 0) {
        free(outs); return -5;
    }
    int rc = 0;
    for (int i = 0; i < m->n_out; i++) {
        int ne = (int)(outs[i].size / 4);
        float *d = (float *)malloc((size_t)(ne > 0 ? ne : 1) * sizeof(float));
        if (!d) { rc = -6; break; }
        if (ne > 0 && outs[i].buf) memcpy(d, outs[i].buf, (size_t)ne * sizeof(float));
        out_data[i] = d;
        out_nelems[i] = ne;
    }
    m->api->rknn_outputs_release(m->ctx, (uint32_t)m->n_out, outs);
    free(outs);
    return rc;
}

/* ================================================================
 * Self test (PASS/FAIL, honest failure reporting)
 * ================================================================ */
int vllm_npu_selftest(vllm_npu_t *n) {
    if (!n) { fprintf(stderr, "[NPU-SELFTEST] FAIL no backend\n"); return 1; }
    if (!n->ok) { fprintf(stderr, "[NPU-SELFTEST] SKIP runtime absent\n"); return 0; }

    char ver[128] = "?";
    vllm_npu_sdk_version(n, ver, sizeof(ver));
    int failures = 0;

    if (n->backend == VLLM_NPU_BACKEND_DIRECT) {
        /* The in-tree direct driver has its own staged self-test
         * (BO round-trip + group-wise matmul vs CPU reference). */
        return vllm_npu_direct_selftest(n->direct, ver);
    }

    /* RKNN: fp32 GEMM smoke test: 8x8 identity-like check */
    float A[64], B[64], C[64], ref[64];
    for (int i = 0; i < 64; i++) {
        A[i] = (float)(i % 7) * 0.25f;
        B[i] = (float)((i / 8) == (i % 8) ? 1.0f : 0.0f);  /* identity */
    }
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            double acc = 0.0;
            for (int k = 0; k < 8; k++) acc += (double)A[r * 8 + k] * B[k * 8 + c];
            ref[r * 8 + c] = (float)acc;
        }
    int ok = vllm_npu_gemm_f32(n, C, A, B, 8, 8, 8);
    if (ok) {
        double max_err = 0.0;
        for (int i = 0; i < 64; i++)
            max_err = fmax(max_err, fabs((double)C[i] - (double)ref[i]));
        int pass = (max_err < 1e-3);
        printf("[NPU-SELFTEST] gemm_f32 8x8 %s (max_err=%.2e) [sdk=%s]\n",
               pass ? "PASS" : "FAIL", max_err, ver);
        if (!pass) failures++;
    } else {
        printf("[NPU-SELFTEST] gemm_f32 8x8 SKIP (no model / below threshold)\n");
    }
    return failures;
}
#endif /* !_WIN32 */
