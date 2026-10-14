/**
 * vllm_media.h - 自研媒体解码（零第三方依赖）
 *
 * MP4 (ISO BMFF) 容器解析 + H.264 软件解码器 + YUV→RGB 缩放。
 * 纯 C 标量实现，x86 本地开发 / aarch64 (RK3588) 板上运行同一份源码。
 * 解码帧仅用于视觉推理，采用"从最近 IDR 解码到目标帧"的抽帧策略。
 */
#ifndef VLLM_H264_MEDIA_H
#define VLLM_H264_MEDIA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * MP4 容器解析 (demux)
 * ================================================================ */

#define MPM_MAX_SAMPLES 65536

typedef struct {
    /* 输入（调用方持有） */
    const uint8_t *data;
    size_t         size;

    /* 视频轨道信息 */
    int    width, height;        /* 显示分辨率（crop 后） */
    int    n_frames;             /* 视频 sample 数 */
    int    length_size;          /* AVCC NAL 长度前缀字节数 (1-4) */
    int    timescale;            /* mvhd timescale */
    int64_t duration_ms;         /* 视频时长 (ms) */

    /* SPS / PPS（AVCC 原始 NAL，含 1 字节 NAL header） */
    const uint8_t *sps; int sps_len;
    const uint8_t *pps; int pps_len;

    /* 帧索引 */
    uint32_t *sample_off;        /* [n_frames] 文件内偏移 */
    uint32_t *sample_size;       /* [n_frames] 字节数 */
    uint32_t *sample_pts_ms;     /* [n_frames] 显示时间戳 (ms) */
    uint8_t  *sample_idr;        /* [n_frames] 1 = IDR */

    uint8_t *own_index;          /* 内部一次性分配（含全部数组） */
} MPMovie;

/* 解析 mp4 容器。成功返回 0；data/size 需在解码期间保持有效。 */
int mp4_open(MPMovie *m, const uint8_t *data, size_t size);

/* 释放 mp4_open 分配的内部内存（data 归调用方）。 */
void mp4_close(MPMovie *m);

/* ================================================================
 * H.264 软件解码器
 * ================================================================ */

/* 能力边界（2026-10-13 收口定稿，诚实不夸大）：
 *   ① 仅解 I 片（IDR）；P/B 片未实现（遇到时显式拒绝返回 -1）；
 *   ② 8x8 变换未实现（encoder 用到时解码器打印 WARN，输出不可信）；
 *   ③ CABAC I_NxN 富残差+高频（scan≥10）场景与 encoder 位级不一致：
 *      实测 40/97 个受控流 Y 平面 maxabs=0（与 ffmpeg 解码全等），
 *      失败共性 = 富残差 + 高频。⇒ 返回 0 只代表「流程完成」，
 *      不保证位级正确；富残差/高频视频请用 ffmpeg 侧抽帧。
 * 用途：YOLO/多模态抽帧（关键帧抽帧，实际帧率 ≈ 关键帧率）。 */

/* 解码器上下文（一次初始化，多次逐帧解码） */
typedef struct H264Dec H264Dec;

typedef struct {
    const MPMovie *movie;
    H264Dec *dec;              /* 内部解码状态（h264_dec.c） */

    /* 输出帧缓冲（YUV420P，紧凑布局：w*h + w/2*h/2*2）
     * yuv=Y 起点；y_u/y_v 为 U/V 平面（紧邻 Y，便于无 padding 拷贝） */
    uint8_t *yuv;
    uint8_t *y_u, *y_v;
    int      yuv_cap;
} H264Decoder;

/* 初始化解码器（解析 SPS/PPS）。返回 0 成功。 */
int h264_decoder_init(H264Decoder *d, const MPMovie *movie);

/* 释放解码器资源 */
void h264_decoder_free(H264Decoder *d);

/* 解码 frame_idx（1-based sample id），输出到 d->yuv（YUV420P）。
 * 内部从该帧之前最近的 IDR 开始解码到目标帧。
 * 返回 0 成功，-1 失败。⚠ 能力边界见上方分节注释：只解 I 片、无 8x8、
 * CABAC 富残差+高频位级不可信；返回 0 ≠ 位级正确。 */
int h264_decoder_decode(H264Decoder *d, int frame_idx);

/* ---------- 裸流 / RTSP 推入式接口（K-E2E-005） ----------
 * 解码器内核与 MP4 路径完全相同 ⇒ 能力边界同上（**只支持 I 片**，P/B 未实现，
 * 无 8x8；CABAC 富残差+高频位级不可信，见 h264_dec.c "slice type %d not yet
 * supported (I only)"）。
 * 用法：sps/pps 为**含 1 字节 NAL header 的原始 NAL**（RTSP 的
 * sprop-parameter-sets base64 解出即为该形态）；每帧用 h264_decoder_decode_au
 * 推入一个访问单元，avcc 为 **每 NAL 前 4 字节大端长度前缀**的拼接缓冲。*/
int h264_decoder_init_spspps(H264Decoder *d,
                             const uint8_t *sps, int sps_len,
                             const uint8_t *pps, int pps_len);
int h264_decoder_decode_au(H264Decoder *d, const uint8_t *avcc, size_t len);

/* 解码器几何（未初始化返回 0）与输出重排。
 * 内部输出按 32 字节对齐行距且 U/V 间有 pad ⇒ 供 media_yuv_to_rgb 使用前必须
 * 用 h264_decoder_yuv_compact 转成紧凑 YUV420P（out 大小 = w*h*3/2）。 */
int h264_decoder_width(const H264Decoder *d);
int h264_decoder_height(const H264Decoder *d);
int h264_decoder_yuv_compact(const H264Decoder *d, uint8_t *out);

/* ================================================================
 * 帧处理：YUV420 → RGB 缩放
 * ================================================================ */

/* 把 yuv (src_w x src_h) 缩放为 rgb24 (max_edge 最长边限制)。
 * rgb 由调用方分配 (dst_w*dst_h*3)。返回 0 成功。 */
int media_yuv_to_rgb(const uint8_t *yuv, int src_w, int src_h,
                     uint8_t *rgb, int max_edge, int *dst_w, int *dst_h);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_H264_MEDIA_H */
