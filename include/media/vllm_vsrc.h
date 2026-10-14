/**
 * vllm_vsrc.h - 统一视频源抽象（K-E2E-005，零依赖）
 *
 * 一个入口吃四类源，供 YOLO 视频检测复用（单次抽帧 / 常驻拉流）：
 *   <path>.mp4|.mov   → FILE：自研 MP4 demux + H.264 软解
 *   <dir>             → DIR ：目录内图像序列（媒体网关已落盘的分帧）
 *   http://...        → HTTP：MJPEG 流（multipart/x-mixed-replace）或单帧快照
 *   rtsp://...        → RTSP：RTP over TCP interleaved + H.264
 *
 * 硬边界（诚实，不静默）：引擎自带 H.264 软解**只支持 I 片**（无 P/B、无 8x8
 * 变换；CABAC 富残差+高频场景重建**位级不可信**，实测 40/97 流与 ffmpeg 全等），
 * 故 FILE/RTSP 只能产出 **IDR 帧**（关键帧抽帧），实际帧率 ≈ 关键帧率。
 * 计数语义：被跳过的**非 IDR** 计入 vsrc_skipped()；**IDR 但引擎解不出**的帧
 * 单列 vsrc_decode_failed()（显式回退），二者均随响应回报，不混淆。
 * HTTP(MJPEG)/DIR 不受此限。
 */
#ifndef VLLM_VSRC_H
#define VLLM_VSRC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VSRC_KIND_NONE = 0,
    VSRC_KIND_FILE,   /* 本地视频文件（MP4/H.264） */
    VSRC_KIND_DIR,    /* 图像序列目录 */
    VSRC_KIND_HTTP,   /* MJPEG 流 / 快照 URL */
    VSRC_KIND_RTSP    /* RTSP 摄像头 */
} VVideoSrcKind;

typedef struct VVideoSrc VVideoSrc;

/* 打开视频源。spec 形式见文件头。失败返回 NULL 并写 err（errcap>0）。 */
VVideoSrc *vsrc_open(const char *spec, char *err, size_t errcap);
void       vsrc_close(VVideoSrc *s);

/* 取下一帧 RGB888。
 * 返回 1 = 有帧（*rgb 为 malloc 缓冲，调用方 free；*w/*h 为尺寸）；
 *      0 = 本轮无帧（直播源超时）或已到流末尾；
 *      <0 = 错误。 */
int vsrc_next(VVideoSrc *s, uint8_t **rgb, int *w, int *h);

VVideoSrcKind vsrc_kind(const VVideoSrc *s);
const char   *vsrc_kind_name(const VVideoSrc *s);   /* file|dir|http|rtsp */
/* 解码模式（显式回报进响应）：如 "h264-IDR-only" / "image-sequence" /
 * "mjpeg-http" / "http-snapshot" / "rtsp-h264-IDR-only" */
const char   *vsrc_mode(const VVideoSrc *s);
int vsrc_skipped(const VVideoSrc *s);      /* 跳过的非 IDR 帧数（0 表示未受限） */
int vsrc_decode_failed(const VVideoSrc *s); /* IDR 解码失败帧数（显式回退计数） */
int vsrc_produced(const VVideoSrc *s);     /* 已产出帧数 */
int vsrc_total_hint(const VVideoSrc *s);   /* 已知总帧数（file/dir），否则 -1 */
int vsrc_live(const VVideoSrc *s);         /* 1 = 实时流（RTSP/MJPEG），0 = 有限源 */

#ifdef __cplusplus
}
#endif

#endif /* VLLM_VSRC_H */
