/**
 * vllm_npu.h - Transparent RK3588 NPU acceleration interface
 *
 * Design goal (user constraint): accelerate inference on the RK3588 NPU
 * WITHOUT converting the model format. The safetensors model + the engine's
 * Q8_0/Q4_0 weights stay untouched; the NPU only executes operator-level
 * mini-models (generic MatMul graphs exported once per projection shape by
 * tools/npu/npu_export_ops.py) whose weights are baked as per-channel int8
 * constants. The runtime feeds the already-quantized int8 activations and
 * receives fp32 results - bit-identical to the CPU path is NOT guaranteed
 * (different reduction order/quantization - see vllm_safetensors.c NEON
 * section); consistency is verified at the PPL / greedy-text level.
 *
 * Axiom mapping (axiom_registry.json):
 *   - blas_precision_efficiency_tradeoff (MI_fusion_rule): INT8 weights
 *     (2x-4x bandwidth cut), error bound eps <= eps1*||Y|| + eps2*||X|| + ...
 *   - block_matrix_assoc_natural: per-shape generic MatMul graphs share the
 *     A-block across output rows (shared input fusion).
 *   - Transparent fallback: every entry point returns an int status; 0 means
 *     "not offloaded - run the CPU (NEON) path". No caller is ever required
 *     to know about the NPU (gumbel determinism red line preserved: the
 *     decision is a pure function of availability + shape + FLOPs threshold).
 *
 * Runtime discovery: librknnrt.so is dlopen()ed at vllm_npu_init() and every
 * rknn_* entry point is resolved into a function-pointer table. On platforms
 * without the runtime (Windows, no NPU), the table is left NULL and all
 * entry points report "not available" (available() == 0) - the engine runs
 * exactly as before.
 */

#ifndef VLLM_NPU_H
#define VLLM_NPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vllm_npu_s vllm_npu_t;   /* opaque handle */

/* NPU backend selection.
 *
 * DIRECT (default, zero third-party dependencies): the engine's OWN minimal
 * userspace driver for the Rockchip rknpu kernel driver (a stock system
 * component on RK3588 boards). It opens /dev/rknpu (or /dev/accel/accel0),
 * allocates DMA buffer objects via ioctl, generates the register-command
 * (regcmd) lists for the group-wise int8/int4 GEMM and submits them directly
 * to the NPU frontend. No external runtime, compiler or library is used and
 * no model conversion happens: the engine's Q8_0/Q4_0 block weights feed the
 * hardware as-is. See vllm_npu_direct.c for the driver and its calibration
 * contract.
 *
 * RKNN (optional): the official runtime (librknnrt.so) with operator-level
 * .rknn models exported by tools/npu/npu_export_ops.py. Kept only for A/B.
 */
typedef enum {
    VLLM_NPU_BACKEND_AUTO   = 0,  /* DIRECT */
    VLLM_NPU_BACKEND_DIRECT = 1,  /* in-tree minimal rknpu driver (zero-dep) */
    VLLM_NPU_BACKEND_RKNN   = 2   /* librknnrt + exported .rknn ops */
} vllm_npu_backend_t;

/* RKNN core selection (rknn_api.h RKNN_NPU_CORE_*). */
#define VLLM_NPU_CORE_AUTO_1  0        /* NPU1 only */
#define VLLM_NPU_CORE_AUTO_2  1        /* NPU2 only */
#define VLLM_NPU_CORE_AUTO_3  2        /* NPU3 only */
#define VLLM_NPU_CORE_AUTO_0  8        /* NPU0 only */
#define VLLM_NPU_CORE_AUTO    15       /* auto */

typedef struct {
    vllm_npu_backend_t backend; /* VLLM_NPU_BACKEND_AUTO (default) */
    const char *model_dir;   /* RKNN backend: dir of exported *.rknn op models
                              * (gemm_<M>x<N>x<K>.rknn); NULL disables RKNN */
    const char *rknn_path;   /* path to librknnrt.so; NULL = dlopen default */
    int         core_mask;   /* rknpu core mask (RKNN_NPU_CORE_MASK_*), default
                              * = NPU3 (fastest standalone core) */
    int         verbosity;   /* 0 = silent, 1 = one-line summary, 2 = per-op */
    double      flops_threshold; /* offload only ops >= this many FLOPs
                                  * (default 1e8 = 100 MFLOPs) */
} vllm_npu_cfg_t;

/**
 * Initialize the NPU backend. Always succeeds (on failure the backend simply
 * reports unavailable). Returns 0 on success, -1 on argument error.
 * Caller owns *npu and must call vllm_npu_destroy().
 */
int vllm_npu_init(vllm_npu_t **npu, const vllm_npu_cfg_t *cfg);

/** Release all resources (rknn contexts, dl handles). */
void vllm_npu_destroy(vllm_npu_t *npu);

/** 1 = runtime loaded and usable; 0 = transparent CPU fallback. */
int vllm_npu_available(const vllm_npu_t *npu);

/** Fill buf with the librknnrt SDK version string; returns strlen or -1. */
int vllm_npu_sdk_version(const vllm_npu_t *npu, char *buf, int len);

/**
 * Generic fp32 GEMM offload: out[M][N] = A[M][K] @ B[K][N] (row-major).
 * Uses the exported gemm_<M>x<N>x<K>.rknn model with A and B fed as runtime
 * inputs. Returns 1 if executed on the NPU, 0 if the caller must run the
 * CPU path (no model / below threshold / runtime absent).
 */
int vllm_npu_gemm_f32(vllm_npu_t *npu, float *out,
                      const float *A, const float *B,
                      int M, int N, int K);

/* Projection ids used to key the per-layer exported int8 models.
 * MUST match projection_shapes() in tools/npu/npu_export_ops.py. */
#define VLLM_NPU_PROJ_Q      0   /* dim -> dim            */
#define VLLM_NPU_PROJ_K      1   /* dim -> kv_dim (GQA)   */
#define VLLM_NPU_PROJ_V      2   /* dim -> kv_dim (GQA)   */
#define VLLM_NPU_PROJ_O      3   /* dim -> dim            */
#define VLLM_NPU_PROJ_GATE   4   /* dim -> ffn_dim        */
#define VLLM_NPU_PROJ_UP     5   /* dim -> ffn_dim        */
#define VLLM_NPU_PROJ_DOWN   6   /* ffn_dim -> dim        */
#define VLLM_NPU_PROJ_LMHEAD 7   /* dim -> vocab          */

/**
 * INT8 GEMM offload: out[M][N] = sum_k Aq[m][k] * Wq[k][n] * (aw[m] * ws[n]).
 * The per-channel int8 weights + scales are BAKED into the exported model
 * gemm_i8_p<proj>_l<layer>.rknn (tools/npu/npu_export_ops.py), so the weight
 * stream to the NPU is ~4x smaller than fp32 (the actual RKNN speedup
 * path). Aq is the engine's already-quantized int8 activation [M*K], aw is
 * the per-token scale [M]. Returns 1 on offload, 0 on CPU fallback.
 */
int vllm_npu_gemm_i8(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *aw,
                     int M, int N, int K, int layer, int proj);

/** Report which backend is live (VLLM_NPU_BACKEND_*), or AUTO when none. */
vllm_npu_backend_t vllm_npu_backend(const vllm_npu_t *npu);

/**
 * Group-wise quantized GEMM offload (DIRECT backend - no model conversion).
 *
 * Feeds the engine's OWN block-quantized weights to the NPU:
 *   C[m][n] = Σ_g a_scale[m][g] * b_scale[n][g] * (int dot of K-group g)
 * with g = G (32 for the engine's Q8_0/Q4_0 blocks). This is exactly the
 * Q8_0/Q4_0 block format held in RAM - the model is never converted.
 *
 *   prec 0 = int8 codes (Q8_0: Wq payload = 32 int8 per block, header-less)
 *   prec 1 = int4 codes (Q4_0: Wq payload = 16 nibble bytes per block)
 *   Aq/A scale are the caller's already-quantized activations (token-major).
 *
 * Returns 1 if executed on the NPU, 0 = run the CPU (NEON/AVX) path.
 * NOTE: M==1 (decode) is broken on the NPU hardware geometry and is padded
 * to M=4 internally; decode therefore stays on the CPU kernels. This entry
 * targets batched prefill (M = mini-batch size >= 4, multiple of 4).
 *
 * wkey (0 = no caching) is a stable per-weight identity forwarded to the
 * DIRECT backend's persistent weight-BO cache (see vllm_npu_direct.h).
 */
int vllm_npu_gemm_gw(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *a_scale,
                     const int8_t *Wq, const float *b_scale,
                     int M, int N, int K, int G, int prec,
                     uint64_t wkey);

/**
 * Same as vllm_npu_gemm_gw but with an explicit N-per-pass hint (nt_hint).
 * Called by workloads made of MANY SMALL GEMMs (conv-as-GEMM): there the
 * wall time is the SUBMIT COUNT (~0.5 ms each), not the FLOPs, so the caller
 * picks a smaller Nt together with a larger G to minimise ceil(N/Nt)*K/G.
 * nt_hint 0 (or out of range / misaligned) = default/env behaviour, exactly
 * like vllm_npu_gemm_gw.
 */
int vllm_npu_gemm_gw_ex(vllm_npu_t *npu, float *out,
                        const int8_t *Aq, const float *a_scale,
                        const int8_t *Wq, const float *b_scale,
                        int M, int N, int K, int G, int prec,
                        uint64_t wkey, int nt_hint);

/**
 * K-YOLO-001 M3 Phase-2: native (CNA CONV_MODE=0) regular convolution, one
 * band per submit, DIRECT backend only. The caller pre-packs the NC1HWC2
 * (C2=16) int8 feature band [C_al][Hband][W] and a signed-int8 weight cube
 * for OC channels (already sliced for the OC band). Writes the raw int32
 * accumulator into out[oc*OH_band*OW + oh*OW + ow]. Returns 0 on success,
 * nonzero -> caller MUST fall back (no retry). On non-Linux/without the driver
 * it always returns nonzero.
 *   feat_q : [C_al][Hband][W] int8, C_al % 16 == 0
 *   wq     : weight cube, see vllm_npu_direct_pack_weight_conv layout
 *   pad    : hardware symmetric top/left pad (0 if the caller pre-padded)
 */
int vllm_npu_conv_i32(vllm_npu_t *npu, int32_t *out,
                      const int8_t *feat_q, const int8_t *wq,
                      int C_al, int Hband, int W,
                      int KH, int KW, int stride, int pad,
                      int OH_band, int OW, int OC, int OC_off, int wbank_hint);

/**
 * Self-test: exercises the active backend (direct: /dev/rknpu open + BO
 * round-trip + a small group-wise matmul against a CPU reference; rknn:
 * exported gemm model). Returns 0 on success, nonzero if a check failed.
 */
int vllm_npu_selftest(vllm_npu_t *npu);

/* ================================================================
 * K-YOLO-003：整网 RKNN 模型（官方 .rknn 检测路径，opt-in 第二路径）
 *
 * 与上面「算子级」RKNN 模型不同，这里加载的是 Rockchip **整网** .rknn（如
 * rknn_model_zoo 的 yolov8.rknn）：喂 uint8 NHWC 图像、取回 N 个 fp32 输出，
 * 家族的 DFL/解码/NMS 由 YOLO 层负责。复用同一 dlopen 函数表。
 *
 * 契约：任何失败（runtime 缺失 / 文件不可读 / rknn_init 失败 / 输入几何不支持）
 * 一律返回 NULL，调用方**显式回退**到 VQF/CPU 路径——绝不静默降级
 * （公理 silent_fallback_false_pass）。
 * ================================================================ */
typedef struct vllm_npu_rknn vllm_npu_rknn;   /* opaque */

/** 加载整网 .rknn。core_mask=0 走 runtime 默认。成功返回句柄，失败 NULL。 */
vllm_npu_rknn *vllm_npu_rknn_open(vllm_npu_t *npu, const char *path, int core_mask);
void           vllm_npu_rknn_close(vllm_npu_rknn *m);

/** 运行时版本串（如 "2.3.2 ... (drv 0.9.6)"）。返回 strlen 或 -1。 */
int vllm_npu_rknn_sdk(const vllm_npu_rknn *m, char *buf, int len);

/** 输入几何（前处理用）：填 NHWC 的 H/W/C。返回 0 成功。 */
int vllm_npu_rknn_input_hwc(const vllm_npu_rknn *m, int *h, int *w, int *c);

/** 输出个数（失败 -1）。 */
int vllm_npu_rknn_n_outputs(const vllm_npu_rknn *m);

/** 第 i 个输出的维度。dims 至少 max_dims 个元素。返回 0 成功。 */
int vllm_npu_rknn_output_dims(const vllm_npu_rknn *m, int i,
                              int *n_dims, int *dims, int max_dims);

/** 打印输入/输出属性（诊断）。 */
void vllm_npu_rknn_dump(const vllm_npu_rknn *m);

/**
 * 推理一次：in = uint8 NHWC（大小 = H*W*C，见 input_hwc）。
 * out_data[i] 由本层 malloc（fp32，out_nelems[i] 个 float）——调用方负责 free。
 * 返回 0 成功；负数表示失败（调用方须回退）。
 */
int vllm_npu_rknn_run(vllm_npu_rknn *m, const uint8_t *in,
                      float **out_data, int *out_nelems);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_NPU_H */
