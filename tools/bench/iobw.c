/* iobw.c - 顺序读 vs 跨距读（模拟逐层驻留布局）带宽对照
 *
 * 用途：立项 K-L0-001 §8.4 / 候选 D 的 G0 上界测量 —— 量「同一字节总量、
 * 不同访问形态」在真实介质上的带宽差。**只读，不写**。
 *
 * 用法: iobw <file> seq|stride <extent_mb> [nseg] [nlayers]
 *   seq    : 从文件头顺序读 extent_mb
 *   stride : 把 [0, extent) 切成 nseg 个连续段，逐层(l) 对每段读
 *            (extent/nseg/nl) 字节 —— 即当前 VQF 的 off + l*step 形态
 *   两种模式读取的字节总量相同 => 可直接比 MB/s
 *
 * 口径：配合 shell 端 `sync; echo 3 > /proc/sys/vm/drop_caches` 取冷页缓存读数。
 * 构建：gcc -O2 -o iobw iobw.c
 *
 * 实测（2026-10-08，RK3588 / Orange Pi 5 Plus，3 次/档，1 GB extent）：
 *   SD   (30B-A3B, nseg=11 nl=48): SEQ 64.9 MB/s, STRIDE 62.2 MB/s -> 罚 1.043x
 *   eMMC (8b-q4,    nseg=11 nl=36): SEQ 313.2 MB/s, STRIDE 271.0 MB/s -> 罚 1.156x
 *   => 「跨距读慢 1.5-3x」的假设被否；层主序布局改造（候选 D-a）收益上限过小。
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> seq|stride <extent_mb> [nseg] [nlayers]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int stride = (strcmp(argv[2], "stride") == 0);
    double extent = (double)atoll(argv[3]) * 1048576.0;
    int nseg = (argc > 4) ? atoi(argv[4]) : 11;
    int nl   = (argc > 5) ? atoi(argv[5]) : 48;
    if (nseg < 1) nseg = 1;
    if (nl < 1) nl = 1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return 1; }
    if ((double)st.st_size < extent) extent = (double)st.st_size;

    const size_t BUF = 1u << 20;
    char *buf = (char *)malloc(BUF);
    if (!buf) { fprintf(stderr, "oom\n"); close(fd); return 1; }

    double total = 0.0;
    double t0 = now_s();

    if (!stride) {
        off_t off = 0;
        double left = extent;
        while (left > 0.0) {
            size_t k = (left > (double)BUF) ? BUF : (size_t)left;
            ssize_t r = pread(fd, buf, k, off);
            if (r <= 0) break;
            off += r; left -= (double)r; total += (double)r;
        }
    } else {
        double segsz = extent / (double)nseg;
        double chunk = segsz / (double)nl;
        size_t ck = (size_t)chunk;
        if (ck < 4096) ck = 4096;
        for (int l = 0; l < nl; l++) {
            for (int s = 0; s < nseg; s++) {
                off_t off = (off_t)((double)s * segsz + (double)l * chunk);
                size_t left = ck;
                while (left > 0) {
                    size_t k = (left > BUF) ? BUF : left;
                    ssize_t r = pread(fd, buf, k, off);
                    if (r <= 0) { left = 0; break; }
                    off += r; left -= (size_t)r; total += (double)r;
                }
            }
        }
    }

    double dt = now_s() - t0;
    double mb = total / 1048576.0;
    printf("%-6s extent=%.0fMB nseg=%-2d nl=%-2d bytes=%.1fMB elapsed=%.3fs speed=%.1f MB/s\n",
           stride ? "STRIDE" : "SEQ", extent / 1048576.0, nseg, nl, mb, dt,
           (dt > 0.0) ? (mb / dt) : 0.0);
    fflush(stdout);
    free(buf);
    close(fd);
    return 0;
}
