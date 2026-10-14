/* vllm_vsrc.c - 统一视频源抽象（K-E2E-005）。详见 include/media/vllm_vsrc.h。
 *
 * 四类源共用一条「取下一帧 RGB888」契约，向上层屏蔽 MP4/目录/MJPEG/RTSP 差异。
 * 所有解码都是引擎自带的零依赖实现（stb_image / 自研 MP4+H.264 / 自研 RTSP）。 */
#include "media/vllm_vsrc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "media/vllm_media.h"    /* MPMovie / H264Decoder / media_yuv_to_rgb */
#include "media/vllm_rtsp.h"     /* VRtspClient */
#include "model/vllm_media.h"    /* media_load_image_memory / media_load_video_frames */
#include "serve/vllm_http.h"     /* vhttp_get_binary */

#define VSRC_FILE_MAX_BYTES (512u << 20)   /* 单个视频文件上限 512MB */
#define VSRC_DIR_MAX_FRAMES 512            /* 目录序列最多载入帧数 */
#define VSRC_HTTP_WINDOW_MS 2000           /* MJPEG 每轮读取窗口 */
#define VSRC_HTTP_MAX_BYTES (4u << 20)     /* MJPEG 单轮读取上限 */

struct VVideoSrc {
    VVideoSrcKind kind;
    char  mode[32];
    char  err[256];
    int   produced;
    int   skipped;
    int   dec_failed;   /* IDR 解码失败帧数（显式回退计数，不混入 skipped_non_idr） */
    int   total_hint;
    int   live;

    /* FILE */
    uint8_t *file_data; size_t file_size;
    MPMovie  movie;
    H264Decoder dec;
    int      dec_ready;
    int      cur_frame;      /* 下一个待检查的 1-based 帧号 */
    int      width, height;
    uint8_t *yuv_compact;
    int      file_opened;

    /* DIR */
    uint8_t **frames; int n_frames; int dir_idx;
    int      dir_w, dir_h;

    /* HTTP */
    char    *url;
    uint8_t *buf; size_t buf_len, buf_cap;
    int      multipart;
    int      http_done;

    /* RTSP */
    VRtspClient *rtsp;
};

static void vsrc_seterr(VVideoSrc *s, const char *m) {
    if (s) snprintf(s->err, sizeof(s->err), "%s", m);
}

/* ---------------- 通用：解码器输出 → 紧凑 YUV → RGB ---------------- */
static int vsrc_dec_rgb(VVideoSrc *s, uint8_t **out, int *ow, int *oh) {
    int w = s->width, h = s->height;
    if (h264_decoder_yuv_compact(&s->dec, s->yuv_compact) != 0) return -1;
    uint8_t *rgb = (uint8_t *)malloc((size_t)w * h * 3);
    if (!rgb) return -1;
    int dw = w, dh = h;
    int me = (w > h) ? w : h;        /* max_edge >= 源长边 ⇒ 不缩放 */
    if (media_yuv_to_rgb(s->yuv_compact, w, h, rgb, me, &dw, &dh) != 0) { free(rgb); return -1; }
    *out = rgb; *ow = dw; *oh = dh;
    return 1;
}

/* ---------------- FILE：MP4 + H.264（仅 IDR） ---------------- */
static int vsrc_file_open(VVideoSrc *s, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { vsrc_seterr(s, "video file not found"); return -1; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); vsrc_seterr(s, "seek failed"); return -1; }
    long sz = ftell(fp);
    if (sz <= 0 || (unsigned long)sz > VSRC_FILE_MAX_BYTES) {
        fclose(fp); vsrc_seterr(s, "video file empty or exceeds 512MB cap"); return -1;
    }
    rewind(fp);
    s->file_data = (uint8_t *)malloc((size_t)sz);
    if (!s->file_data) { fclose(fp); vsrc_seterr(s, "out of memory"); return -1; }
    size_t got = fread(s->file_data, 1, (size_t)sz, fp);
    fclose(fp);
    if (got != (size_t)sz) { vsrc_seterr(s, "short read"); return -1; }
    s->file_size = (size_t)sz;

    if (mp4_open(&s->movie, s->file_data, s->file_size) != 0) {
        vsrc_seterr(s, "not a decodable MP4 (or no H.264 video track)");
        return -1;
    }
    if (h264_decoder_init(&s->dec, &s->movie) != 0) {
        vsrc_seterr(s, "H.264 init from SPS/PPS failed");
        return -1;
    }
    s->dec_ready = 1;
    s->width = h264_decoder_width(&s->dec);
    s->height = h264_decoder_height(&s->dec);
    if (s->width <= 0 || s->height <= 0) { vsrc_seterr(s, "bad SPS geometry"); return -1; }
    s->yuv_compact = (uint8_t *)malloc((size_t)s->width * s->height * 3 / 2);
    if (!s->yuv_compact) { vsrc_seterr(s, "out of memory"); return -1; }
    s->cur_frame = 1;
    s->file_opened = 1;

    int n_idr = 0;
    for (int i = 0; i < s->movie.n_frames; i++) if (s->movie.sample_idr[i]) n_idr++;
    snprintf(s->mode, sizeof(s->mode), n_idr == s->movie.n_frames ? "h264-all-I"
                                                                  : "h264-IDR-only");
    s->total_hint = n_idr;
    return 0;
}

static int vsrc_file_next(VVideoSrc *s, uint8_t **rgb, int *w, int *h) {
    while (s->cur_frame <= s->movie.n_frames) {
        int idx = s->cur_frame++;                 /* 1-based */
        if (!s->movie.sample_idr[idx - 1]) { s->skipped++; continue; }
        if (h264_decoder_decode(&s->dec, idx) != 0) {
            /* 显式回退不可静默：IDR 帧解码失败（引擎限制 I-only / 无 8x8 /
             * CABAC 富残差）单列计数并告警，不混入 skipped_non_idr。 */
            s->dec_failed++;
            fprintf(stderr, "[VSRC] WARN: frame %d (IDR) decode failed; skipped "
                            "(engine limits: I-only / no 8x8 / CABAC rich-residue)\n", idx);
            continue;
        }
        int rc = vsrc_dec_rgb(s, rgb, w, h);
        if (rc == 1) { s->produced++; return 1; }
        s->dec_failed++;
        fprintf(stderr, "[VSRC] WARN: frame %d YUV->RGB conversion failed; skipped\n", idx);
    }
    return 0;   /* EOF */
}

/* ---------------- DIR：图像序列 ---------------- */
static int vsrc_dir_open(VVideoSrc *s, const char *dir) {
    s->frames = media_load_video_frames(dir, &s->dir_w, &s->dir_h, &s->n_frames,
                                        VSRC_DIR_MAX_FRAMES);
    if (!s->frames || s->n_frames <= 0) { vsrc_seterr(s, "no image frames in directory"); return -1; }
    snprintf(s->mode, sizeof(s->mode), "image-sequence");
    s->total_hint = s->n_frames;
    return 0;
}

static int vsrc_dir_next(VVideoSrc *s, uint8_t **rgb, int *w, int *h) {
    if (s->dir_idx >= s->n_frames) return 0;
    size_t bytes = (size_t)s->dir_w * s->dir_h * 3;
    uint8_t *copy = (uint8_t *)malloc(bytes);    /* 复制：统一 malloc 所有权契约 */
    if (!copy) return -1;
    memcpy(copy, s->frames[s->dir_idx], bytes);
    s->dir_idx++; s->produced++;
    *rgb = copy; *w = s->dir_w; *h = s->dir_h;
    return 1;
}

/* ---------------- HTTP：MJPEG 流 / 单帧快照 ---------------- */
static int vsrc_http_fill(VVideoSrc *s) {
    unsigned char *b = NULL; size_t bl = 0; int status = 0;
    char ct[128] = "";
    int rc = vhttp_get_binary(s->url, VSRC_HTTP_WINDOW_MS, VSRC_HTTP_MAX_BYTES,
                              &b, &bl, &status, ct, sizeof(ct));
    if (rc != 0) { free(b); return -1; }
    if (status != 200) { free(b); return -2; }
    if (s->buf_len == 0) {                       /* 首轮判定流类型（显式） */
        s->multipart = (strstr(ct, "multipart") || strstr(ct, "mixed-replace")) ? 1 : 0;
        snprintf(s->mode, sizeof(s->mode), s->multipart ? "mjpeg-http" : "http-snapshot");
        s->live = s->multipart ? 1 : 0;
    }
    if (bl == 0) { free(b); return -3; }
    if (s->buf_len + bl > s->buf_cap) {
        size_t ncap = s->buf_cap ? s->buf_cap : (1u << 16);
        while (ncap < s->buf_len + bl) ncap <<= 1;
        uint8_t *nb = (uint8_t *)realloc(s->buf, ncap);
        if (!nb) { free(b); return -4; }
        s->buf = nb; s->buf_cap = ncap;
    }
    memcpy(s->buf + s->buf_len, b, bl);
    s->buf_len += bl;
    free(b);
    return 0;
}

/* 从缓冲里取出**第一个完整 JPEG**（FFD8 ... FFD9），解成 RGB。 */
static int vsrc_http_take_jpeg(VVideoSrc *s, uint8_t **rgb, int *w, int *h) {
    size_t i = 0;
    while (i + 1 < s->buf_len) {
        if (s->buf[i] == 0xFF && s->buf[i + 1] == 0xD8) break;
        i++;
    }
    if (i + 1 >= s->buf_len) {                    /* 没有 SOI：整块丢弃 */
        s->buf_len = 0;
        return 0;
    }
    size_t j = i + 2;
    while (j + 1 < s->buf_len) {
        if (s->buf[j] == 0xFF && s->buf[j + 1] == 0xD9) break;
        j++;
    }
    if (j + 1 >= s->buf_len) {                    /* 帧未收全：保留从 SOI 起的部分 */
        if (i > 0) { memmove(s->buf, s->buf + i, s->buf_len - i); s->buf_len -= i; }
        return 0;
    }
    size_t flen = j + 2 - i;
    int iw = 0, ih = 0;
    uint8_t *img = media_load_image_memory(s->buf + i, flen, &iw, &ih);
    memmove(s->buf, s->buf + j + 2, s->buf_len - (j + 2));
    s->buf_len -= (j + 2);
    if (!img) return -1;
    *rgb = img; *w = iw; *h = ih;
    return 1;
}

static int vsrc_http_next(VVideoSrc *s, uint8_t **rgb, int *w, int *h) {
    for (;;) {
        int rc = vsrc_http_take_jpeg(s, rgb, w, h);
        if (rc == 1) { s->produced++; return 1; }
        if (rc < 0) return -1;
        if (s->http_done) return 0;
        int fr = vsrc_http_fill(s);
        if (fr != 0) {
            /* 快照源：一次 GET 一帧，取完即结束（非错误） */
            if (!s->multipart && s->produced > 0) { s->http_done = 1; return 0; }
            if (fr == -2 || fr == -3) { s->http_done = 1; return s->produced ? 0 : -1; }
            return -1;
        }
        if (!s->multipart && s->produced > 0) { s->http_done = 1; }  /* 单帧：取到即止 */
    }
}

/* ---------------- 打开 / 关闭 ---------------- */
static int is_dir(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) return 0;
    return (st.st_mode & S_IFMT) == S_IFDIR;
}

static int is_file(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) return 0;
    return (st.st_mode & S_IFMT) == S_IFREG;
}

VVideoSrc *vsrc_open(const char *spec, char *err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!spec || !spec[0]) { if (err && errcap) snprintf(err, errcap, "empty source"); return NULL; }
    VVideoSrc *s = (VVideoSrc *)calloc(1, sizeof(VVideoSrc));
    if (!s) { if (err && errcap) snprintf(err, errcap, "out of memory"); return NULL; }
    s->total_hint = -1;
    int rc = -1;

    if (strncmp(spec, "rtsp://", 7) == 0) {
        s->kind = VSRC_KIND_RTSP;
        snprintf(s->mode, sizeof(s->mode), "rtsp-h264-IDR-only");
        s->live = 1;
        char rerr[256] = "";
        s->rtsp = vrtsp_open(spec, rerr, sizeof(rerr));
        if (!s->rtsp) { vsrc_seterr(s, rerr[0] ? rerr : "rtsp open failed"); goto fail; }
        rc = 0;
    } else if (strncmp(spec, "http://", 7) == 0 || strncmp(spec, "https://", 8) == 0) {
        s->kind = VSRC_KIND_HTTP;
        s->url = (char *)malloc(strlen(spec) + 1);
        if (!s->url) { vsrc_seterr(s, "out of memory"); goto fail; }
        strcpy(s->url, spec);
        snprintf(s->mode, sizeof(s->mode), "http-pending");
        rc = 0;
    } else if (is_dir(spec)) {
        s->kind = VSRC_KIND_DIR;
        rc = vsrc_dir_open(s, spec);
    } else if (is_file(spec)) {
        s->kind = VSRC_KIND_FILE;
        rc = vsrc_file_open(s, spec);
    } else {
        vsrc_seterr(s, "source not found (expect file/dir/http:///rtsp://)");
        goto fail;
    }
    if (rc != 0) goto fail;
    if (err && errcap) snprintf(err, errcap, "ok");
    return s;

fail:
    if (err && errcap) snprintf(err, errcap, "%s", s->err[0] ? s->err : "open failed");
    vsrc_close(s);
    return NULL;
}

void vsrc_close(VVideoSrc *s) {
    if (!s) return;
    if (s->dec_ready) h264_decoder_free(&s->dec);
    if (s->file_opened) mp4_close(&s->movie);
    free(s->file_data);
    free(s->yuv_compact);
    if (s->frames) media_free_video_frames(s->frames, s->n_frames);
    free(s->buf);
    free(s->url);
    if (s->rtsp) vrtsp_close(s->rtsp);
    free(s);
}

int vsrc_next(VVideoSrc *s, uint8_t **rgb, int *w, int *h) {
    if (!s || !rgb || !w || !h) return -1;
    *rgb = NULL;
    switch (s->kind) {
        case VSRC_KIND_FILE: return vsrc_file_next(s, rgb, w, h);
        case VSRC_KIND_DIR:  return vsrc_dir_next(s, rgb, w, h);
        case VSRC_KIND_HTTP: return vsrc_http_next(s, rgb, w, h);
        case VSRC_KIND_RTSP: {
            int rc = vrtsp_next_idr(s->rtsp, rgb, w, h, 3000);
            if (rc == 1) s->produced++;
            return rc;
        }
        default: return -1;
    }
}

VVideoSrcKind vsrc_kind(const VVideoSrc *s) { return s ? s->kind : VSRC_KIND_NONE; }

const char *vsrc_kind_name(const VVideoSrc *s) {
    if (!s) return "none";
    switch (s->kind) {
        case VSRC_KIND_FILE: return "file";
        case VSRC_KIND_DIR:  return "dir";
        case VSRC_KIND_HTTP: return "http";
        case VSRC_KIND_RTSP: return "rtsp";
        default: return "none";
    }
}

const char *vsrc_mode(const VVideoSrc *s) { return s ? s->mode : ""; }
int vsrc_skipped(const VVideoSrc *s) {
    if (!s) return 0;
    if (s->kind == VSRC_KIND_RTSP && s->rtsp) return s->skipped + vrtsp_skipped(s->rtsp);
    return s->skipped;
}
/* 解码失败（IDR 但引擎解不出）帧数：显式回退计数，与 skipped_non_idr 区分 */
int vsrc_decode_failed(const VVideoSrc *s) {
    if (!s) return 0;
    if (s->kind == VSRC_KIND_RTSP && s->rtsp) return s->dec_failed + vrtsp_dec_failed(s->rtsp);
    return s->dec_failed;
}
int vsrc_produced(const VVideoSrc *s) { return s ? s->produced : 0; }
int vsrc_total_hint(const VVideoSrc *s) { return s ? s->total_hint : -1; }
int vsrc_live(const VVideoSrc *s) {
    if (!s) return 0;
    /* HTTP 源的流类型要等首帧（首次 GET 看 Content-Type）才能判定，
     * 期间 mode=http-pending ⇒ live=0（保守）。调用方应**每轮**重读本函数
     * 而不是在循环外缓存，否则 MJPEG 会被误判为有限源而提前结束。 */
    if (s->kind == VSRC_KIND_HTTP) return s->multipart;
    return s->live;
}
