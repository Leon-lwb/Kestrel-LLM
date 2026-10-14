/**
 * vllm_yolo.h - YOLO 目标检测（自有格式 VQF v4）引擎接口
 *
 * 设计红线（用户约束）：部署/运行时只认**引擎自有格式 VQF v4**（见
 * docs/立项-K-YOLO-001-G0-NPU卷积检测适配.md 附录 A），不引入 ONNX/.rknn。
 * 本模块自带 VQF v4 加载器（LLM 侧 vqf.c 对 v4 显式拒绝，反之亦然）。
 *
 * M0 阶段：纯 CPU F32 前向（正确的 golden 执行器）+ letterbox + NMS。
 * M2：kernel>1 卷积经 im2col 展开为 int8 GEMM，复用已标定的 DIRECT gemm_gw 上 NPU；
 * 非卷积算子（SPPF/Concat/Upsample/DFL/NMS）留 CPU。失败时**显式回退 CPU**
 * （公理 silent_fallback_false_pass：禁止静默降级）。
 */
#ifndef VLLM_YOLO_H
#define VLLM_YOLO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vllm_npu_s;

/**
 * 设置 YOLO 全局 NPU 句柄（M2：im2col→GEMM 上 NPU）。须在 st_yolo_load 之前
 * 调用；NULL = 纯 CPU。非 Linux/无 NPU 平台该句柄不可用，自动回退 CPU。
 */
void st_yolo_set_default_npu(struct vllm_npu_s *npu);

/* 单条检测结果（xyxy，输入图像像素坐标；letterbox 已反变换） */
typedef struct {
    float x1, y1, x2, y2;
    float score;
    int   cls;
} YoloDet;

typedef struct StYoloModel StYoloModel;   /* opaque */

/** 加载 VQF v4 检测模型（自有格式）。非 v4 / 校验失败返回 NULL。 */
StYoloModel *st_yolo_load(const char *path);

/**
 * K-YOLO-003：加载 Rockchip 官方**整网** .rknn 检测模型（opt-in 第二路径）。
 * npu 须为 RKNN 后端句柄（backend=VLLM_NPU_BACKEND_RKNN 且 rknn_path 指向
 * librknnrt.so）。成功返回模型；任何失败返回 NULL（调用方须显式回退 VQF/CPU，
 * 绝不静默降级）。家族解码/NMS 在本模块内完成（v8/v11：9 输出）。
 */
StYoloModel *st_yolo_load_rknn(const char *rknn_path, struct vllm_npu_s *npu,
                               int core_mask);

/** 释放模型。 */
void st_yolo_free(StYoloModel *m);

/** 模型元信息：网络输入尺寸（默认 640×640）。 */
int st_yolo_input_size(const StYoloModel *m, int *w, int *h);

/** 类别数。 */
int st_yolo_num_classes(const StYoloModel *m);

/** 家族（VQF_DET_FAM_*）。 */
int st_yolo_family(const StYoloModel *m);

/** 类别名（names blob，'\n' 分隔）；拷入 buf（NUL 结尾）。成功 0，越界 -1。 */
int st_yolo_class_name(const StYoloModel *m, int idx, char *buf, int buflen);

/**
 * 原始前向（数值门用）：输入 NCHW float [3*ih*iw]（无 letterbox），
 * 输出 [4+nc, A] 追加到 out（容量 op_cap*op_a_cap）。返回 0 成功。
 * 用于与 python/torch golden 逐值对拍（J1/J2 前置）。
 */
int st_yolo_forward_raw(StYoloModel *m, const float *nchw,
                        float *out, int *out_c, int *out_a);

/**
 * 检测：rgb [h*w*3] uint8 → letterbox → 前向 → 后处理(NMS) → 原图坐标检测框。
 * conf/iou ≤ 0 时用模型内置默认。返回检测数（≤ max_out），-1 失败。
 */
int st_yolo_detect(StYoloModel *m, const uint8_t *rgb, int w, int h,
                   float conf, float iou, YoloDet *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_YOLO_H */
