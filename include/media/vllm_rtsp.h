/**
 * vllm_rtsp.h - 零依赖 RTSP 客户端（RTP over TCP interleaved，H.264）
 *
 * 用途：YOLO 视频源接入摄像头（K-E2E-005）。只做取帧，不做音频/RTCP 反馈。
 * 边界（诚实）：
 *   * 仅明文 rtsp://（无 RTSPS/TLS；需要时前置网关）；
 *   * 仅 H.264（PT 由 SDP rtpmap 决定），SDP 必须带 sprop-parameter-sets
 *     （即 SPS/PPS 在带内，无需等 SPS/PPS 包）；
 *   * 传输仅 RTP/AVP/TCP;interleaved（不支持 UDP 与 RTSP-over-HTTP 隧道）；
 *   * 复用引擎既有 H.264 软解 ⇒ **只能解 I 片**（无 P/B、无 8x8；CABAC 富残差+
 *     高频场景位级不可信），故 vrtsp_next_idr 只返回 IDR 访问单元，非 IDR 一律
 *     跳过并计数（实际抽帧率 ≈ 关键帧率，通常 0.5~2 fps）；**IDR 但解不出**的
 *     访问单元单列 vrtsp_dec_failed()（显式回退），不混入 skipped。
 */
#ifndef VLLM_RTSP_H
#define VLLM_RTSP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VRtspClient VRtspClient;

/* 连接并完成 OPTIONS / DESCRIBE / SETUP / PLAY。失败返回 NULL 并写 err。 */
VRtspClient *vrtsp_open(const char *url, char *err, size_t errcap);
void vrtsp_close(VRtspClient *c);

/* 取下一帧 **IDR** 的 RGB888（内部自动跳过非 IDR 访问单元并计数）。
 * 返回 1 = 有帧（*rgb 为 malloc 缓冲，调用方 free）；0 = timeout_ms 内没有 IDR；
 * <0 = 错误。*w/*h 为解码尺寸。 */
int vrtsp_next_idr(VRtspClient *c, uint8_t **rgb, int *w, int *h, int timeout_ms);

int vrtsp_width(const VRtspClient *c);
int vrtsp_height(const VRtspClient *c);
int vrtsp_skipped(const VRtspClient *c);   /* 已跳过的非 IDR 访问单元数 */
int vrtsp_dec_failed(const VRtspClient *c); /* IDR 解码失败访问单元数（显式回退计数） */
int vrtsp_payload_type(const VRtspClient *c);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_RTSP_H */
