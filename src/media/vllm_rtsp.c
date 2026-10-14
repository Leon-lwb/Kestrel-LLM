/* vllm_rtsp.c - 零依赖 RTSP 客户端（K-E2E-005）。
 * 仅 RTP/AVP/TCP;interleaved + H.264；复用引擎 H.264 软解（I 片）。
 * 详见 include/media/vllm_rtsp.h。 */
#include "media/vllm_rtsp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "media/vllm_media.h"        /* H264Decoder / h264_decoder_* / media_yuv_to_rgb */
#include "model/vllm_media.h"        /* media_base64_decode */

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET vrtsp_fd;
#define VRTSP_INVALID INVALID_SOCKET
#define VRTSP_BAD(f) ((f) == INVALID_SOCKET)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
typedef int vrtsp_fd;
#define VRTSP_INVALID (-1)
#define VRTSP_BAD(f) ((f) < 0)
#endif

#define RTSP_MAX_URL   512
#define RTSP_MAX_HDR   8192
#define RTSP_MAX_SDP   (256 * 1024)
#define RTSP_MAX_PKT   (256 * 1024)
#define RTSP_MAX_AU    (4 * 1024 * 1024)

struct VRtspClient {
    vrtsp_fd fd;
    int      cseq;
    char     base_url[RTSP_MAX_URL];
    char     control[RTSP_MAX_URL];
    char     session[128];
    int      payload_type;

    /* SPS/PPS（原始 NAL，含 1 字节 NAL header） */
    uint8_t  sps[128]; int sps_len;
    uint8_t  pps[128]; int pps_len;

    /* 解码器 */
    H264Decoder dec;
    int         dec_ready;
    int         width, height;
    uint8_t    *yuv_compact;
    int         skipped;
    int         dec_failed;   /* IDR 解码失败访问单元数（显式回退计数，不混入 skipped） */

    /* 访问单元装配（AVCC：每 NAL 前 4 字节大端长度） */
    uint8_t *au; size_t au_len, au_cap;
    int      au_has_idr;

    /* FU-A 分片重组 */
    uint8_t *fu; size_t fu_len, fu_cap;
    int      fu_active;

    char err[256];
};

/* ---------------- 低层 socket ---------------- */
static void vrtsp_close_fd(vrtsp_fd fd) {
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
}

static void vrtsp_set_timeout(vrtsp_fd fd, int ms) {
#ifdef _WIN32
    DWORD t = (DWORD)ms;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&t, sizeof(t));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&t, sizeof(t));
#else
    struct timeval tv;
    tv.tv_sec = ms / 1000; tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

static int rtsp_send_all(vrtsp_fd fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
#ifdef _WIN32
        int w = send(fd, buf + off, (int)(len - off), 0);
#else
        int w = (int)send(fd, buf + off, len - off, 0);
#endif
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

/* 读满 n 字节；返回已读字节数（<n 表示超时/断开） */
static size_t rtsp_recv_n(vrtsp_fd fd, uint8_t *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
#ifdef _WIN32
        int r = recv(fd, (char *)(buf + off), (int)(n - off), 0);
#else
        int r = (int)recv(fd, buf + off, n - off, 0);
#endif
        if (r <= 0) break;
        off += (size_t)r;
    }
    return off;
}

static void seterr(VRtspClient *c, const char *msg) {
    if (c) snprintf(c->err, sizeof(c->err), "%s", msg);
}

/* ---------------- 工具 ---------------- */
static uint16_t be16(const uint8_t *p) { return (uint16_t)((p[0] << 8) | p[1]); }

/* 解析 rtsp://[user:pass@]host[:port]/path → host/port/path（path 保证以 '/' 开头） */
static int rtsp_url_split(const char *url, char *host, size_t hcap, int *port,
                          char *path, size_t pcap) {
    if (!url || strncmp(url, "rtsp://", 7) != 0) return -1;
    const char *p = url + 7;
    const char *slash = strchr(p, '/');
    const char *at = strchr(p, '@');
    if (at && (!slash || at < slash)) p = at + 1;   /* 跳过 user:pass@ */
    const char *host_end = slash ? slash : (p + strlen(p));
    const char *colon = NULL;
    for (const char *q = p; q < host_end; q++) if (*q == ':') colon = q;
    size_t hl = (size_t)((colon ? colon : host_end) - p);
    if (hl == 0 || hl >= hcap) return -1;
    memcpy(host, p, hl); host[hl] = '\0';
    *port = 554;
    if (colon) {
        int v = atoi(colon + 1);
        if (v > 0 && v < 65536) *port = v;
    }
    if (slash) snprintf(path, pcap, "%s", slash);
    else snprintf(path, pcap, "/");
    return 0;
}

/* 发送一个 RTSP 请求并读取响应头；out_body 接收响应体（按 Content-Length），
 * 可为 NULL。返回 HTTP 状态码（200/401/...），<0 表示传输失败。 */
static int rtsp_request(VRtspClient *c, const char *method, const char *url,
                        const char *extra, char *out_body, size_t body_cap,
                        size_t *out_body_len) {
    char req[2048];
    int n = snprintf(req, sizeof(req),
                     "%s %s RTSP/1.0\r\n"
                     "CSeq: %d\r\n"
                     "User-Agent: vllm-kestrel/1.0\r\n"
                     "%s"
                     "\r\n",
                     method, url, ++c->cseq,
                     extra ? extra : "");
    if (n <= 0 || n >= (int)sizeof(req)) return -1;
    if (rtsp_send_all(c->fd, req, (size_t)n) != 0) return -1;

    /* 读响应头 */
    char hdr[RTSP_MAX_HDR];
    size_t hl = 0;
    while (hl + 1 < sizeof(hdr)) {
        uint8_t ch;
        if (rtsp_recv_n(c->fd, &ch, 1) != 1) return -1;
        hdr[hl++] = (char)ch;
        if (hl >= 4 && memcmp(hdr + hl - 4, "\r\n\r\n", 4) == 0) break;
    }
    hdr[hl] = '\0';
    if (hl < 8 || strncmp(hdr, "RTSP/", 5) != 0) return -1;
    int status = atoi(hdr + 8);

    /* Session / Content-Length */
    const char *sess = strstr(hdr, "Session:");
    if (sess) {
        sess += 8;
        while (*sess == ' ') sess++;
        size_t i = 0;
        while (sess[i] && sess[i] != ';' && sess[i] != '\r' && i < sizeof(c->session) - 1) {
            c->session[i] = sess[i]; i++;
        }
        c->session[i] = '\0';
    }
    long clen = 0;
    const char *cl = strstr(hdr, "Content-Length:");
    if (cl) clen = atol(cl + 15);
    if (out_body_len) *out_body_len = 0;
    if (out_body && body_cap > 1 && clen > 0) {
        size_t want = (size_t)clen;
        if (want > body_cap - 1) want = body_cap - 1;
        size_t got = rtsp_recv_n(c->fd, (uint8_t *)out_body, want);
        out_body[got] = '\0';
        if (out_body_len) *out_body_len = got;
        /* 丢弃剩余未读部分（避免污染后续 interleaved 帧） */
        for (size_t skip = (size_t)clen - got; skip > 0; ) {
            uint8_t tmp[4096];
            size_t k = skip > sizeof(tmp) ? sizeof(tmp) : skip;
            size_t r = rtsp_recv_n(c->fd, tmp, k);
            if (r == 0) break;
            skip -= r;
        }
    }
    return status;
}

/* ---------------- AU 装配 ---------------- */
static int au_reserve(VRtspClient *c, size_t need) {
    if (c->au_cap >= need) return 0;
    size_t ncap = c->au_cap ? c->au_cap : 4096;
    while (ncap < need) ncap <<= 1;
    uint8_t *nb = (uint8_t *)realloc(c->au, ncap);
    if (!nb) return -1;
    c->au = nb; c->au_cap = ncap;
    return 0;
}

static int au_append_nal(VRtspClient *c, const uint8_t *nal, size_t nlen) {
    if (nlen == 0 || nlen > 0x00ffffff) return -1;
    if (au_reserve(c, c->au_len + 4 + nlen) != 0) return -1;
    c->au[c->au_len + 0] = (uint8_t)((nlen >> 24) & 0xff);
    c->au[c->au_len + 1] = (uint8_t)((nlen >> 16) & 0xff);
    c->au[c->au_len + 2] = (uint8_t)((nlen >> 8) & 0xff);
    c->au[c->au_len + 3] = (uint8_t)(nlen & 0xff);
    memcpy(c->au + c->au_len + 4, nal, nlen);
    c->au_len += 4 + nlen;
    if ((nal[0] & 0x1f) == 5) c->au_has_idr = 1;   /* IDR slice */
    return 0;
}

static int fu_reserve(VRtspClient *c, size_t need) {
    if (c->fu_cap >= need) return 0;
    size_t ncap = c->fu_cap ? c->fu_cap : 4096;
    while (ncap < need) ncap <<= 1;
    uint8_t *nb = (uint8_t *)realloc(c->fu, ncap);
    if (!nb) return -1;
    c->fu = nb; c->fu_cap = ncap;
    return 0;
}

static int fu_push(VRtspClient *c, const uint8_t *d, size_t n) {
    if (fu_reserve(c, c->fu_len + n) != 0) return -1;
    memcpy(c->fu + c->fu_len, d, n);
    c->fu_len += n;
    return 0;
}

/* H.264 RTP 载荷 → AU 装配。返回 0 成功。 */
static int depack_h264(VRtspClient *c, const uint8_t *pl, size_t pll) {
    if (pll < 1) return 0;
    int nt = pl[0] & 0x1f;
    if (nt >= 1 && nt <= 23) {                       /* 单 NAL */
        return au_append_nal(c, pl, pll);
    }
    if (nt == 24) {                                  /* STAP-A */
        size_t i = 1;
        while (i + 2 <= pll) {
            size_t sz = be16(pl + i);
            i += 2;
            if (sz == 0 || i + sz > pll) break;
            if (au_append_nal(c, pl + i, sz) != 0) return -1;
            i += sz;
        }
        return 0;
    }
    if (nt == 28 || nt == 29) {                      /* FU-A（29=FU-B 少见，尽力） */
        if (pll < 3) return 0;
        int s = (pl[1] >> 7) & 1, e = (pl[1] >> 6) & 1;
        int t = pl[1] & 0x1f;
        size_t skip = (nt == 29) ? 2 : 0;            /* FU-B 有 DON 字段 */
        if (s) {
            c->fu_active = 1;
            c->fu_len = 0;
            uint8_t nh = (uint8_t)((pl[0] & 0xe0) | t);
            if (fu_push(c, &nh, 1) != 0) return -1;
        }
        if (!c->fu_active) return 0;
        if (pll > 2 + skip && fu_push(c, pl + 2 + skip, pll - 2 - skip) != 0) return -1;
        if (e) {
            if (au_append_nal(c, c->fu, c->fu_len) != 0) { c->fu_active = 0; return -1; }
            c->fu_active = 0;
        }
        return 0;
    }
    /* 25(STAP-B)/26..27 等：忽略 */
    return 0;
}

/* ---------------- SDP ---------------- */
/* 在 SDP 中找 rtpmap 的 H264 载荷类型与 sprop-parameter-sets，并解析 control URL。 */
static int parse_sdp(VRtspClient *c, const char *sdp) {
    const char *p = sdp;
    int have_control = 0;
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        while (len && (p[len - 1] == '\r' || p[len - 1] == ' ')) len--;

        if (len > 9 && strncmp(p, "a=control:", 10) == 0) {
            char ctl[RTSP_MAX_URL];
            size_t cl = len - 10;
            if (cl >= sizeof(ctl)) cl = sizeof(ctl) - 1;
            memcpy(ctl, p + 10, cl); ctl[cl] = '\0';
            if (ctl[0] && strcmp(ctl, "*") != 0) {
                if (strncmp(ctl, "rtsp://", 7) == 0) snprintf(c->control, sizeof(c->control), "%s", ctl);
                else snprintf(c->control, sizeof(c->control), "%s/%s", c->base_url, ctl);
                have_control = 1;
            }
        } else if (len > 9 && strncmp(p, "a=rtpmap:", 9) == 0) {
            const char *sp = memchr(p + 9, ' ', len - 9);
            if (sp && strstr(sp, "H264")) {
                c->payload_type = atoi(p + 9);
                if (c->payload_type <= 0) c->payload_type = 96;
            }
        } else if (len > 8 && strncmp(p, "a=fmtp:", 7) == 0) {
            const char *sps = strstr(p, "sprop-parameter-sets=");
            if (sps && (size_t)(sps - p) < len) {
                sps += 21;
                const char *comma = memchr(sps, ',', (size_t)(p + len - sps));
                if (comma) {
                    char b64s[512], b64p[512];
                    size_t l1 = (size_t)(comma - sps);
                    if (l1 >= sizeof(b64s)) l1 = sizeof(b64s) - 1;
                    memcpy(b64s, sps, l1); b64s[l1] = '\0';
                    size_t l2 = (size_t)(p + len - comma - 1);
                    if (l2 >= sizeof(b64p)) l2 = sizeof(b64p) - 1;
                    memcpy(b64p, comma + 1, l2); b64p[l2] = '\0';
                    /* 去掉可能的参数后缀（; 之后） */
                    char *sc = strchr(b64s, ';'); if (sc) *sc = '\0';
                    sc = strchr(b64p, ';'); if (sc) *sc = '\0';
                    size_t n1 = 0, n2 = 0;
                    uint8_t *d1 = media_base64_decode(b64s, &n1);
                    uint8_t *d2 = media_base64_decode(b64p, &n2);
                    if (d1 && n1 > 0 && n1 <= sizeof(c->sps)) {
                        memcpy(c->sps, d1, n1); c->sps_len = (int)n1;
                    }
                    if (d2 && n2 > 0 && n2 <= sizeof(c->pps)) {
                        memcpy(c->pps, d2, n2); c->pps_len = (int)n2;
                    }
                    free(d1); free(d2);
                }
            }
        }
        p = eol ? eol + 1 : NULL;
    }
    return have_control ? 0 : -1;
}

/* ---------------- 解码 ---------------- */
static int decode_au_to_rgb(VRtspClient *c, uint8_t **out_rgb, int *out_w, int *out_h) {
    if (!c->dec_ready) {
        if (h264_decoder_init_spspps(&c->dec, c->sps, c->sps_len, c->pps, c->pps_len) != 0)
            return -1;
        c->width = h264_decoder_width(&c->dec);
        c->height = h264_decoder_height(&c->dec);
        if (c->width <= 0 || c->height <= 0) return -1;
        c->yuv_compact = (uint8_t *)malloc((size_t)c->width * c->height * 3 / 2);
        if (!c->yuv_compact) return -1;
        c->dec_ready = 1;
    }
    if (h264_decoder_decode_au(&c->dec, c->au, c->au_len) != 0) return -2;
    if (h264_decoder_yuv_compact(&c->dec, c->yuv_compact) != 0) return -2;
    int w = c->width, h = c->height;
    uint8_t *rgb = (uint8_t *)malloc((size_t)w * h * 3);
    if (!rgb) return -3;
    int dw = w, dh = h;
    int me = (w > h) ? w : h;      /* max_edge >= 源长边 ⇒ 不缩放 */
    if (media_yuv_to_rgb(c->yuv_compact, w, h, rgb, me, &dw, &dh) != 0) {
        free(rgb); return -2;
    }
    *out_rgb = rgb; *out_w = dw; *out_h = dh;
    return 1;
}

/* ---------------- 对外接口 ---------------- */
VRtspClient *vrtsp_open(const char *url, char *err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    VRtspClient *c = (VRtspClient *)calloc(1, sizeof(VRtspClient));
    if (!c) return NULL;
    c->fd = VRTSP_INVALID;
    c->payload_type = 96;

    char host[256], path[RTSP_MAX_URL];
    int port = 554;
    if (rtsp_url_split(url, host, sizeof(host), &port, path, sizeof(path)) != 0) {
        seterr(c, "invalid rtsp url (expect rtsp://host[:port]/path)");
        goto fail;
    }
    snprintf(c->base_url, sizeof(c->base_url), "rtsp://%s:%d%s", host, port, path);

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) {
        seterr(c, "dns resolution failed");
        goto fail;
    }
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        c->fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (VRTSP_BAD(c->fd)) continue;
#ifdef _WIN32
        if (connect(c->fd, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
#else
        if (connect(c->fd, ai->ai_addr, (socklen_t)ai->ai_addrlen) == 0) break;
#endif
        vrtsp_close_fd(c->fd); c->fd = VRTSP_INVALID;
    }
    freeaddrinfo(res);
    if (VRTSP_BAD(c->fd)) {
        seterr(c, "connect failed");
        goto fail;
    }
    vrtsp_set_timeout(c->fd, 8000);

    /* OPTIONS */
    int st = rtsp_request(c, "OPTIONS", c->base_url, NULL, NULL, 0, NULL);
    if (st < 0) { seterr(c, "OPTIONS failed (no response)"); goto fail; }

    /* DESCRIBE → SDP */
    char *sdp = (char *)malloc(RTSP_MAX_SDP);
    if (!sdp) { seterr(c, "out of memory"); goto fail; }
    size_t sdp_len = 0;
    st = rtsp_request(c, "DESCRIBE", c->base_url,
                      "Accept: application/sdp\r\n", sdp, RTSP_MAX_SDP, &sdp_len);
    if (st < 0 || sdp_len == 0) {
        free(sdp);
        snprintf(c->err, sizeof(c->err), "DESCRIBE failed (status=%d)", st);
        goto fail;
    }
    if (st != 200) {
        snprintf(c->err, sizeof(c->err), "DESCRIBE status=%d", st);
        free(sdp);
        goto fail;
    }
    if (parse_sdp(c, sdp) != 0) { free(sdp); seterr(c, "SDP has no a=control track"); goto fail; }
    free(sdp);
    if (c->sps_len == 0 || c->pps_len == 0) {
        seterr(c, "SDP missing sprop-parameter-sets (SPS/PPS in-band unsupported)");
        goto fail;
    }

    /* SETUP（RTP over TCP interleaved） */
    {
        char extra[256];
        snprintf(extra, sizeof(extra),
                 "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n");
        st = rtsp_request(c, "SETUP", c->control, extra, NULL, 0, NULL);
        if (st < 0 || st != 200) {
            snprintf(c->err, sizeof(c->err), "SETUP status=%d", st);
            goto fail;
        }
    }
    /* PLAY */
    {
        char extra[256];
        snprintf(extra, sizeof(extra), "Session: %s\r\nRange: npt=0.000-\r\n", c->session);
        st = rtsp_request(c, "PLAY", c->base_url, extra, NULL, 0, NULL);
        if (st < 0 || st != 200) {
            snprintf(c->err, sizeof(c->err), "PLAY status=%d", st);
            goto fail;
        }
    }
    if (err && errcap) snprintf(err, errcap, "ok");
    return c;

fail:
    if (err && errcap) snprintf(err, errcap, "%s", c->err[0] ? c->err : "rtsp open failed");
    vrtsp_close(c);
    return NULL;
}

void vrtsp_close(VRtspClient *c) {
    if (!c) return;
    if (!VRTSP_BAD(c->fd)) {
        /* 尽力发送 TEARDOWN，失败无所谓 */
        char extra[192];
        snprintf(extra, sizeof(extra), "Session: %s\r\n", c->session);
        rtsp_request(c, "TEARDOWN", c->base_url, extra, NULL, 0, NULL);
        vrtsp_close_fd(c->fd);
    }
    if (c->dec_ready) h264_decoder_free(&c->dec);
    free(c->yuv_compact);
    free(c->au);
    free(c->fu);
    free(c);
}

/* 读一个 interleaved 包（'$' + channel + 2B len + payload）。
 * 返回 1 成功（*out_ch 为 channel）/ 0 超时 / -1 错。 */
static int rtsp_read_interleaved(VRtspClient *c, int *out_ch,
                                 uint8_t *buf, size_t cap, size_t *out_len) {
    uint8_t hdr[4];
    size_t got = 0;
    /* 同步到 '$'（容忍残留字节） */
    while (got < 4) {
        if (rtsp_recv_n(c->fd, hdr + got, 1) != 1) return got ? -1 : 0;
        if (got == 0 && hdr[0] != '$') continue;
        got++;
    }
    size_t len = be16(hdr + 2);
    if (len == 0 || len > cap) return -1;
    if (rtsp_recv_n(c->fd, buf, len) != len) return -1;
    *out_ch = hdr[1];
    *out_len = len;
    return 1;
}

int vrtsp_next_idr(VRtspClient *c, uint8_t **rgb, int *w, int *h, int timeout_ms) {
    if (!c || VRTSP_BAD(c->fd) || !rgb) return -1;
    *rgb = NULL;
    vrtsp_set_timeout(c->fd, timeout_ms > 0 ? timeout_ms : 3000);

    uint8_t *pkt = (uint8_t *)malloc(RTSP_MAX_PKT);
    if (!pkt) return -1;

    int result = 0;
    for (;;) {
        size_t plen = 0;
        int ch = -1;
        int r = rtsp_read_interleaved(c, &ch, pkt, RTSP_MAX_PKT, &plen);
        if (r == 0) { result = 0; break; }        /* 超时：本轮无帧 */
        if (r < 0) { result = -1; break; }
        if (ch != 0) continue;                    /* channel 1 = RTCP → 丢弃 */
        if (plen < 12) continue;
        int pt = pkt[1] & 0x7f;
        if (c->payload_type && pt != c->payload_type) continue;

        int cc = pkt[0] & 0x0f;
        int x = (pkt[0] >> 4) & 1;
        size_t off = 12 + (size_t)cc * 4;
        if (x) {
            if (plen < off + 4) continue;
            size_t ext = be16(pkt + off + 2);
            off += 4 + ext * 4;
        }
        if (off >= plen) continue;
        int marker = (pkt[1] >> 7) & 1;
        if (depack_h264(c, pkt + off, plen - off) != 0) { result = -1; break; }

        if (marker && c->au_len > 0) {
            if (c->au_has_idr) {
                int rc = decode_au_to_rgb(c, rgb, w, h);
                c->au_len = 0; c->au_has_idr = 0;
                if (rc == 1) { result = 1; break; }
                /* 解码失败（引擎限制 I-only / 无 8x8 / CABAC 富残差）：单列计数，
                 * 随响应显式回报（vrtsp_dec_failed），不混入 skipped（反静默回退）。 */
                c->dec_failed++;
            } else {
                c->skipped++;
                c->au_len = 0; c->au_has_idr = 0;
            }
        }
    }
    free(pkt);
    return result;
}

int vrtsp_width(const VRtspClient *c)  { return c ? c->width : 0; }
int vrtsp_height(const VRtspClient *c) { return c ? c->height : 0; }
int vrtsp_skipped(const VRtspClient *c) { return c ? c->skipped : 0; }
int vrtsp_dec_failed(const VRtspClient *c) { return c ? c->dec_failed : 0; }
int vrtsp_payload_type(const VRtspClient *c) { return c ? c->payload_type : 0; }
