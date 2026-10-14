/**
 * vllm_tokenizer_qwen.c - Qwen3-VL Tokenizer Implementation
 *
 * Loads vocab.bin binary format and provides UTF-8 longest-prefix
 * tokenization. Designed for the Qwen3-VL-8B-Instruct 151K vocabulary.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vllm_tokenizer_qwen.h"
#include "vllm_util.h"

#ifdef __linux__
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================================================================
 * 词表索引：消除 O(文本长度 × vocab_size) 的最长前缀匹配
 *
 * 背景：原实现每个 token 位置都对整张词表（151936 条）做一次线性扫描
 * （见下方 find_longest_match_linear），实测 7212-token 的 prompt 光分词
 * 就要 ~10.4 s（≈1.44 ms/token；每个词表项约 9.5 ns，花在指针跳转 + memcmp
 * + cache miss 上）。这是"短追问 + 长前缀"复用轮 TTFT 的最大单项。
 *
 * 索引语义必须与线性扫描**逐位一致**：
 *   - 线性扫描保留"最长、且 id 最小"的那条（`tlen > best_len` 严格比较 ⇒
 *     等长时先出现者胜 = 最小 id）；
 *   - 因此索引按 id 升序插入、同串只保留首个（= 最小 id），查询时长度从大到小
 *     回退，第一个命中的长度即最长匹配。
 * 两条路径对同一输入必须产出完全相同的 id 序列（回归对照见
 * tools/bench/tok_ref_check.c，以及引擎内 VLLM_TOK_LINEAR=1 的强制旧路径）。
 * ================================================================ */

/* 前向声明：byte-level BPE 合并表载入（定义见下方 BPE 段）。 */
static int tok_bpe_load_bin(QwenTokenizer *tok, const char *model_dir);

/* FNV-1a：短键够快，且不需要先求长度。 */
static uint32_t tok_key_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

/* 精确匹配（长度相同且字节全同），返回**最小** id；不在词表内返回 -1。 */
static int vocab_index_find(const QwenTokenizer *tok, const char *s, int len) {
    if (!tok->vocab_tab || len <= 0) return -1;
    uint32_t mask = (uint32_t)tok->vocab_tab_mask;
    uint32_t h = tok_key_hash(s, len) & mask;
    for (;;) {
        int slot = tok->vocab_tab[h];
        if (slot == 0) return -1;               /* 空槽：键不在表内 */
        int id = slot - 1;
        if (tok->str_lens[id] == len &&
            memcmp(tok->strings[id], s, (size_t)len) == 0) return id;
        h = (h + 1) & mask;                     /* 容量 ≥ 2×词表，不会走满 */
    }
}

static int vocab_index_build(QwenTokenizer *tok) {
    int cap = 1;
    while (cap < tok->vocab_size * 2) cap <<= 1;
    tok->vocab_tab = calloc((size_t)cap, sizeof(int));
    if (!tok->vocab_tab) return -1;
    tok->vocab_tab_mask = cap - 1;
    memset(tok->first_maxlen, 0, sizeof(tok->first_maxlen));

    uint32_t mask = (uint32_t)tok->vocab_tab_mask;
    for (int id = 0; id < tok->vocab_size; id++) {
        int len = tok->str_lens[id];
        const char *s = tok->strings[id];
        if (!s || len <= 0) continue;
        uint8_t b0 = (uint8_t)s[0];
        if (len > tok->first_maxlen[b0]) tok->first_maxlen[b0] = len;

        uint32_t h = tok_key_hash(s, len) & mask;
        for (;;) {
            int slot = tok->vocab_tab[h];
            if (slot == 0) { tok->vocab_tab[h] = id + 1; break; }
            int other = slot - 1;
            if (tok->str_lens[other] == len &&
                memcmp(tok->strings[other], s, (size_t)len) == 0)
                break;   /* 同串已在表内且 id 更小：保持不动（等长取最小 id） */
            h = (h + 1) & mask;
        }
    }
    return 0;
}

/* ================================================================
 * Load vocab.bin binary format
 * ================================================================ */

int qwen_tokenizer_load(QwenTokenizer *tok, const char *model_dir) {
    memset(tok, 0, sizeof(*tok));
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->pad_id = -1;
    tok->im_start_id = -1;
    tok->im_end_id = -1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/vocab.bin", model_dir);

    FILE *f = st_fopen(path, "rb");
    if (!f) {
        /* Try model directory directly */
        snprintf(path, sizeof(path), "%s/build/vocab.bin", model_dir);
        f = st_fopen(path, "rb");
    }
    if (!f) {
        /* Fallback: 软件根目录（可执行文件所在目录）的统一 vocab.bin。
         * 多模型共享同一 Qwen 词表时，模型目录可省去该文件，由部署目录提供。 */
#ifdef __linux__
        char exe_path[1024];
        ssize_t rl = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (rl > 0) {
            exe_path[rl] = '\0';
            char *slash = strrchr(exe_path, '/');
            if (slash) {
                *slash = '\0';
                snprintf(path, sizeof(path), "%s/vocab.bin", exe_path);
                f = st_fopen(path, "rb");
            }
        }
#endif
    }
    if (!f) {
        fprintf(stderr, "[TOK] Cannot open vocab.bin: %s\n", path);
        return -1;
    }

    /* Read vocab_size */
    uint32_t v32;
    if (fread(&v32, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    tok->vocab_size = v32;

    if (tok->vocab_size <= 0 || tok->vocab_size > QWEN_TOKENIZER_MAX_TOKENS) {
        fprintf(stderr, "[TOK] Invalid vocab_size: %d\n", tok->vocab_size);
        fclose(f);
        return -1;
    }

    /* Allocate string pointers */
    tok->strings  = calloc((size_t)tok->vocab_size, sizeof(char*));
    tok->str_lens = calloc((size_t)tok->vocab_size, sizeof(int));
    if (!tok->strings || !tok->str_lens) {
        fprintf(stderr, "[TOK] OOM allocating %d token pointers\n", tok->vocab_size);
        fclose(f);
        return -1;
    }

    /* First pass: read all entries, calculate total string data size */
    /* We need to store entries by token_id, but file may not be sorted. */
    /* vocab.bin is written in order: tid=0,1,2,...vocab_size-1 */

    size_t total_str_bytes = 0;
    long *offsets = malloc((size_t)tok->vocab_size * sizeof(long));
    int  *lens    = malloc((size_t)tok->vocab_size * sizeof(int));
    int  *ids     = malloc((size_t)tok->vocab_size * sizeof(int));
    if (!offsets || !lens || !ids) {
        fprintf(stderr, "[TOK] OOM during vocab scan\n");
        fclose(f);
        free(offsets); free(lens); free(ids);
        return -1;
    }

    tok->max_str_len = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        uint32_t tid;
        uint16_t slen;
        if (fread(&tid, 4, 1, f) != 1) break;
        if (fread(&slen, 2, 1, f) != 1) break;

        ids[i]    = tid;
        lens[i]   = (int)slen;
        offsets[i] = ftell(f);
        total_str_bytes += (size_t)slen + 1;  /* +1 for null terminator */
        if (lens[i] > tok->max_str_len) tok->max_str_len = lens[i];

        /* Skip string data */
        fseek(f, (long)slen, SEEK_CUR);
    }

    /* Allocate contiguous string storage */
    tok->str_data = calloc(total_str_bytes, 1);
    if (!tok->str_data) {
        fprintf(stderr, "[TOK] OOM allocating string data (%zu bytes)\n", total_str_bytes);
        fclose(f);
        free(offsets); free(lens); free(ids);
        return -1;
    }

    /* Second pass: read strings into str_data and set pointers */
    char *data_ptr = tok->str_data;
    for (int i = 0; i < tok->vocab_size; i++) {
        int tid = ids[i];
        if (tid < 0 || tid >= tok->vocab_size) continue;

        fseek(f, offsets[i], SEEK_SET);
        size_t nread = fread(data_ptr, 1, (size_t)lens[i], f);
        data_ptr[nread] = '\0';

        tok->strings[tid]  = data_ptr;
        tok->str_lens[tid] = lens[i];
        data_ptr += lens[i] + 1;
    }

    fclose(f);
    free(offsets);
    free(lens);
    free(ids);

    tok->is_loaded = 1;
    /* 建词表索引（VLLM_TOK_LINEAR=1 时保持旧线性扫描，用于位级回归对照）。 */
    const char *lin = getenv("VLLM_TOK_LINEAR");
    tok->tok_linear = (lin && lin[0] == '1') ? 1 : 0;
    if (!tok->tok_linear && vocab_index_build(tok) != 0) {
        fprintf(stderr, "[TOK] vocab index build failed (OOM?) -> linear scan\n");
        free(tok->vocab_tab);
        tok->vocab_tab = NULL;
        tok->tok_linear = 1;
    }
    printf("[TOK] Loaded %d tokens from vocab.bin (max_len=%d)\n",
           tok->vocab_size, tok->max_str_len);
    if (tok->tok_linear) {
        printf("[TOK] match mode: linear scan (VLLM_TOK_LINEAR=1; regression oracle)\n");
    } else {
        printf("[TOK] match mode: indexed (slots=%d, %.1f MB, load=%.2f)\n",
               tok->vocab_tab_mask + 1,
               (tok->vocab_tab_mask + 1) * 4.0 / 1048576.0,
               (double)tok->vocab_size / (double)(tok->vocab_tab_mask + 1));
    }
    /* merges.bin（可选）：载入则启用 byte-level BPE；缺失回退贪心最长前缀。
     * VLLM_TOK_LEGACY=1 仍可强制旧路径（位级回归对照）。 */
    if (tok_bpe_load_bin(tok, model_dir) != 0)
        printf("[TOK] merges.bin not found -> greedy longest-prefix (legacy)\n");
    return 0;
}

void qwen_tokenizer_free(QwenTokenizer *tok) {
    if (!tok->is_loaded) return;
    free(tok->vocab_tab);
    free(tok->strings);
    free(tok->str_lens);
    free(tok->str_data);
    free(tok->bpe_str);
    free(tok->bm_off);
    free(tok->bm_llen);
    free(tok->bm_rlen);
    free(tok->bm_tab);
    free(tok->at_id);
    free(tok->at_off);
    free(tok->at_len);
    memset(tok, 0, sizeof(*tok));
}

/* ================================================================
 * Encode: UTF-8 longest-prefix matching
 * ================================================================ */

/* Check if byte starts a UTF-8 continuation byte (0x80-0xBF) */
static int is_utf8_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

/* 旧实现：全词表线性扫描（保留为位级对照基准，VLLM_TOK_LINEAR=1 时启用）。
 * 语义 = 「不动点」：本函数的结果就是索引路径必须复刻的参考真值。 */
static int find_longest_match_linear(QwenTokenizer *tok,
                                     const char *text, int text_len, int max_len,
                                     int *match_len) {
    int best_id = -1;
    int best_len = 0;

    if (max_len > text_len) max_len = text_len;

    /* Simple linear scan — could be optimized with trie */
    for (int i = 0; i < tok->vocab_size; i++) {
        int tlen = tok->str_lens[i];
        if (tlen == 0) continue;
        if (tlen > max_len) continue;
        if (memcmp(tok->strings[i], text, (size_t)tlen) == 0) {
            if (tlen > best_len) {
                best_id  = i;
                best_len = tlen;
            }
        }
    }

    *match_len = best_len;
    return best_id;
}

/* 在词表里找与 text 前缀相同、且长度不超过 max_len 的最长 token。
 * max_len 用来把匹配限制在"下一个特殊 token 之前"，防止贪心长匹配把特殊
 * token 的首字节吞掉（见 QWEN_SPECIAL_STR 上方注释）。
 *
 * 索引路径：长度从大到小回退，第一个命中的长度即最长匹配；同一长度由索引保证
 * 取最小 id。与 find_longest_match_linear 逐位同结果（见文件头的证明性注释）。
 * 起点收紧到 min(max_len, 首字节允许的最长 token, 词表最长 token)：
 * 没有任何 token 以 text[0] 开头时直接判无匹配，与线性扫描一致。 */
static int find_longest_match_qwen(QwenTokenizer *tok,
                                    const char *text, int text_len, int max_len,
                                    int *match_len) {
    if (tok->tok_linear || !tok->vocab_tab)
        return find_longest_match_linear(tok, text, text_len, max_len, match_len);

    *match_len = 0;
    if (max_len > text_len) max_len = text_len;
    if (max_len <= 0) return -1;

    int hi = max_len;
    int bmax = tok->first_maxlen[(uint8_t)text[0]];
    if (bmax < hi) hi = bmax;
    if (tok->max_str_len > 0 && tok->max_str_len < hi) hi = tok->max_str_len;

    for (int L = hi; L >= 1; L--) {
        int id = vocab_index_find(tok, text, L);
        if (id >= 0) {
            *match_len = L;
            return id;
        }
    }
    return -1;
}

/* ---- 特殊 token 最高优先级匹配 ----------------------------------------
 * 本编码器是"逐位置取最长词表前缀"，不是真正的 BPE。当某个字符合成的二元
 * token 恰好以 '<' 结尾时（词表里存在 ".<"(15757)、"。"+"<"(89393) 这类
 * 条目），贪心匹配会把下一个特殊 token 的 '<' 一起吃掉，于是
 * <|im_end|> 被切成 5 个普通 token（"|i"+"m"+"_end"+"|"+">"）而不是单个
 * 151645。后果是 ChatML 模板被破坏：模型收到的是把 <|im_end|> 当纯文本的
 * prompt（训练时从未见过），分布外 → 开头还算连贯、随后坍缩成重复。
 * 因此在这些控制串上先做一次"整串精确匹配"，命中就直接出特殊 token id。
 * 与 HF 分词器行为一致（encode('<|im_end|>', add_special_tokens=False)
 * 同样返回 [151645]）。 */
static const char *const QWEN_SPECIAL_STR[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<think>", "</think>",          /* Qwen3 思考段边界（151667 / 151668） */
};
#define QWEN_N_SPECIAL \
    ((int)(sizeof(QWEN_SPECIAL_STR) / sizeof(QWEN_SPECIAL_STR[0])))

/* 词表里整串精确等于 s 的 token id（长度相同且字节全同），找不到返回 -1。
 * 索引不可用时退回线性扫描（语义相同：返回最小 id）。 */
static int find_exact_token(const QwenTokenizer *tok, const char *s, int len) {
    if (tok->vocab_tab) return vocab_index_find(tok, s, len);
    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->strings[i]) continue;
        if (tok->str_lens[i] != len) continue;
        if (memcmp(tok->strings[i], s, (size_t)len) == 0) return i;
    }
    return -1;
}

/* 惰性解析并缓存特殊 token 的 id（按 tokenizer 指针失效）。 */
static int special_token_id(const QwenTokenizer *tok, int idx) {
    static const QwenTokenizer *cached_tok = NULL;
    static int cached_ids[QWEN_N_SPECIAL];
    if (cached_tok != tok) {
        for (int i = 0; i < QWEN_N_SPECIAL; i++)
            cached_ids[i] = find_exact_token(tok, QWEN_SPECIAL_STR[i],
                                             (int)strlen(QWEN_SPECIAL_STR[i]));
        cached_tok = tok;
    }
    return cached_ids[idx];
}

/* 返回 rem 里"下一个特殊 token"的起始偏移（>0）；不存在则返回 rem_len。
 * 编码时用这个上界截断普通 token 的最长匹配：否则像 "." + "<|im_end|>" 这种
 * 位置，贪心匹配会先吃掉词表里的 ".<"（id 15757），使扫描指针跳过 '<'，
 * 后面的特殊 token 就再也匹配不上了。 */
static int special_start_after(const char *rem, int rem_len) {
    const char *p = rem;
    int left = rem_len;
    while (left > 1) {
        const char *lt = (const char *)memchr(p + 1, '<', (size_t)(left - 1));
        if (!lt) break;
        int k = (int)(lt - rem);
        for (int i = 0; i < QWEN_N_SPECIAL; i++) {
            int l = (int)strlen(QWEN_SPECIAL_STR[i]);
            if (l <= rem_len - k &&
                memcmp(lt, QWEN_SPECIAL_STR[i], (size_t)l) == 0)
                return k;
        }
        left = rem_len - k;
        p = lt;
    }
    return rem_len;
}

/* ================================================================
 * byte-level BPE 编码（merges.bin）—— 与 HF/tokenizers 位级对齐
 *
 * 旧编码器（下方 qwen_tokenizer_encode 的 legacy 分支）是"逐位置取最长词表
 * 前缀"，与 HF 的 BPE 合并序不同（"Explain" → 贪心 "Expla"+"in"，HF 为
 * "Ex"+"plain"）。BPE 路径：
 *   1) 先在文本里整串匹配 added_tokens（<|im_*|>/<think> 等）→ 直接出 id；
 *   2) 否则按 Qwen 预分词正则（GPT2 家族）近似切出一个 piece；
 *   3) raw 字节 → repr 域（GPT2 bytes_to_unicode，'Ġ'/'Ċ'）；
 *   4) 在 repr 序列上按 merges rank 做**最小合并**（左优先）；
 *   5) 每个符号在词表（repr 域）里查 id。
 * merges.bin 缺失时回退 legacy 贪心（行为与改动前一致）。
 * ================================================================ */

#define TK_BPE_MAXSYM 4096
#define TK_BPE_RBUF   (TK_BPE_MAXSYM * 4)

/* raw 字节 → 引擎词表域（与 vocab.bin 的存储域一致）。
 * 实测（对比 tokenizer.json vocab 反查 vocab.bin，151643/151643 一致）：
 * 引擎词表存**原始字节**，仅对 b <= 0x20 的字节做 bytes_to_unicode 编码
 * （即 codepoint = 256+b：space=0x20→U+0120 'Ġ'(C4 A0)、\n=0x0A→U+010A 'Ċ'、
 *  tab=0x09→U+0109 …）；其余字节（含 CJK 的 0x80-0xFF 与 0x7F）全部原样。
 * 注意：这不是标准 GPT2 bytes_to_unicode（后者把 0x7F-0xA0 也重映射），
 * 但与本引擎 vocab.bin 一致，故 BPE 的 merges/vocab 都按此域比较。 */
static uint8_t s_b2u_len[256];
static char    s_b2u[256][4];
static int     s_b2u_ready = 0;

static void b2u_init(void) {
    if (s_b2u_ready) return;
    for (int b = 0; b < 256; b++) {
        if (b > 0x20) {
            /* 原样单字节（含 0x80-0xFF 与 0x7F）：引擎词表按原始字节存。 */
            s_b2u[b][0] = (char)b; s_b2u_len[b] = 1;
            continue;
        }
        /* b <= 0x20：repr 域，codepoint = 256+b，UTF-8 编码（空格/换行 → 'Ġ'/'Ċ'）。 */
        uint32_t cp = (uint32_t)(256 + b);
        s_b2u[b][0] = (char)(0xC0 | (cp >> 6));
        s_b2u[b][1] = (char)(0x80 | (cp & 0x3F));
        s_b2u_len[b] = 2;
    }
    s_b2u_ready = 1;
}

static uint32_t bpe_pair_hash(const char *a, int la, const char *b, int lb) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < la; i++) { h ^= (uint8_t)a[i]; h *= 16777619u; }
    h ^= 0xFFu; h *= 16777619u;
    for (int i = 0; i < lb; i++) { h ^= (uint8_t)b[i]; h *= 16777619u; }
    return h;
}

/* 查 (a,b) 的 merge rank（列表下标）；无合并返回 -1。 */
static int bpe_rank(const QwenTokenizer *tok, const char *a, int la,
                    const char *b, int lb) {
    if (!tok->bm_tab || tok->n_merges <= 0 || la <= 0 || lb <= 0) return -1;
    uint32_t mask = (uint32_t)tok->bm_mask;
    uint32_t h = bpe_pair_hash(a, la, b, lb) & mask;
    for (;;) {
        int slot = tok->bm_tab[h];
        if (slot == 0) return -1;
        int i = slot - 1;
        if (tok->bm_llen[i] == la && tok->bm_rlen[i] == lb &&
            memcmp(tok->bpe_str + tok->bm_off[i], a, (size_t)la) == 0 &&
            memcmp(tok->bpe_str + tok->bm_off[i] + la, b, (size_t)lb) == 0)
            return i;
        h = (h + 1) & mask;
    }
}

/* ---- UTF-8 解码 + Unicode 粗分类（\p{L}/\p{N}/\s 的工程近似）---- */
static int u8_dec(const unsigned char *s, int len, uint32_t *cp) {
    unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int n; uint32_t v;
    if ((c & 0xE0) == 0xC0) { n = 1; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 2; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 3; v = c & 0x07; }
    else { *cp = c; return 1; }
    if (n > len - 1) { *cp = c; return 1; }
    for (int i = 1; i <= n; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = c; return 1; }
        v = (v << 6) | (uint32_t)(s[i] & 0x3F);
    }
    *cp = v;
    return n + 1;
}
static int cp_nl(uint32_t c) { return c == '\n' || c == '\r'; }
/* 非换行空白（\\s 减去 \\r\\n）。注意 \\s 本身含 \\r\\n，故另有 cp_ws 合并。 */
static int cp_sp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == 0x85 ||
           c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}
/* \\s（含 \\r\\n）。 */
static int cp_ws(uint32_t c) { return cp_sp(c) || cp_nl(c); }
/* 近似 \p{N}：ASCII 数字 + 全角数字。 */
static int cp_dig(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19);
}
/* 近似 \p{L}：ASCII 字母，或非 ASCII 且**不落在常见标点/符号块**的码点
 * （覆盖 CJK/拉丁扩展；显式排除全角标点、CJK 标点、通用标点、符号/emoji）。 */
static int cp_let(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (c < 0x80) return 0;
    if (cp_ws(c)) return 0;
    if (c >= 0x00A1 && c <= 0x00BF)          /* ¡¢£…¿（ª º µ 例外为字母） */
        return (c == 0xAA || c == 0xBA || c == 0xB5) ? 1 : 0;
    if (c == 0xD7 || c == 0xF7) return 0;    /* × ÷ */
    if (c >= 0x2000 && c <= 0x206F) return 0;/* General Punctuation */
    if (c >= 0x2190 && c <= 0x2BFF) return 0;/* Arrows/Math/Misc symbols */
    if (c >= 0x2E00 && c <= 0x2E7F) return 0;/* Supplemental Punctuation */
    if (c >= 0x3000 && c <= 0x303F) return 0;/* CJK Symbols & Punctuation */
    if (c >= 0xFE30 && c <= 0xFE6F) return 0;/* CJK Compat Forms / small forms */
    if (c >= 0xFF01 && c <= 0xFF0F) return 0;/* 全角标点第一段 */
    if (c >= 0xFF1A && c <= 0xFF20) return 0;/* ：；＜＝＞？＠ */
    if (c >= 0xFF3B && c <= 0xFF40) return 0;/* ［＼］＾＿｀ */
    if (c >= 0xFF5B && c <= 0xFF65) return 0;/* ｛｜｝～ 及半角 katakana 标点段 */
    if (c >= 0xFFE0 && c <= 0xFFEE) return 0;/* ￠￡… */
    if (c >= 0x1F000 && c <= 0x1FAFF) return 0; /* 符号/emoji */
    return 1;
}

/* 预分词：Qwen3 Split 正则（顺序即优先级）：
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+ |\p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+ |\s+(?!\S) |\s+
 * 返回下一个 piece 的字节长度（≥1）。 */
static int m_g1(const unsigned char *s, int len) {
    if (s[0] != '\'') return 0;
    static const char *c[] = { "s", "t", "re", "ve", "m", "ll", "d" };
    for (int k = 0; k < 7; k++) {
        int l = (int)strlen(c[k]);
        if (1 + l > len) continue;
        int ok = 1;
        for (int i = 0; i < l; i++) {
            char ch = (char)s[1 + i];
            if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
            if (ch != c[k][i]) { ok = 0; break; }
        }
        if (ok) return 1 + l;
    }
    return 0;
}
static int m_g2(const unsigned char *s, int len) {
    uint32_t c; int n = u8_dec(s, len, &c);
    int off = 0;
    if (!cp_let(c)) {
        if (cp_nl(c) || cp_dig(c)) return 0;
        if (n >= len) return 0;
        uint32_t c1; int n1 = u8_dec(s + n, len - n, &c1);
        if (!cp_let(c1)) return 0;
        off = n;
    }
    int p = off;
    while (p < len) {
        uint32_t cc; int nn = u8_dec(s + p, len - p, &cc);
        if (!cp_let(cc)) break;
        p += nn;
    }
    return (p > off) ? p : 0;
}
static int m_g3(const unsigned char *s, int len) {
    uint32_t c; int n = u8_dec(s, len, &c);
    return cp_dig(c) ? n : 0;
}
static int m_g4(const unsigned char *s, int len) {
    int p = (s[0] == ' ') ? 1 : 0;
    int q = p, cnt = 0;
    while (q < len) {
        uint32_t c; int n = u8_dec(s + q, len - q, &c);
        if (cp_ws(c) || cp_let(c) || cp_dig(c)) break;
        q += n; cnt++;
    }
    if (cnt == 0) return 0;
    while (q < len) {
        uint32_t c; int n = u8_dec(s + q, len - q, &c);
        if (!cp_nl(c)) break;
        q += n;
    }
    return q;
}
static int m_g5(const unsigned char *s, int len) {
    int q = 0, last_nl = -1;
    while (q < len) {
        uint32_t c; int n = u8_dec(s + q, len - q, &c);
        if (!cp_ws(c)) break;
        if (cp_nl(c)) last_nl = q;
        q += n;
    }
    return (last_nl >= 0) ? last_nl + 1 : 0;
}
static int m_g6(const unsigned char *s, int len) {
    /* \s+(?!\S)：空白段**后接非空白**时回溯到 R-1（把最后一个空白留给下一段
     * 作为 'Ġ' 前缀），段尾到底才全取。否则 "a    b" 会整段成 'ĠĠĠĠ'，与 HF
     * 的 ['ĠĠĠ','Ġb'] 不一致。 */
    int q = 0;
    while (q < len) {
        uint32_t c; int n = u8_dec(s + q, len - q, &c);
        if (!cp_ws(c)) break;
        q += n;
    }
    if (q <= 0) return 0;
    if (q == len) return q;
    return (q - 1 >= 1) ? (q - 1) : 0;
}
static int m_g7(const unsigned char *s, int len) {
    int q = 0;
    while (q < len) {
        uint32_t c; int n = u8_dec(s + q, len - q, &c);
        if (!cp_ws(c)) break;
        q += n;
    }
    return q;
}
static int pretoken_span(const unsigned char *s, int len) {
    int r;
    if ((r = m_g1(s, len))) return r;
    if ((r = m_g2(s, len))) return r;
    if ((r = m_g3(s, len))) return r;
    if ((r = m_g4(s, len))) return r;
    if ((r = m_g5(s, len))) return r;
    if ((r = m_g6(s, len))) return r;
    if ((r = m_g7(s, len))) return r;
    return 1;   /* 兜底：单字节 */
}

/* 预分词：tokenizers ByteLevel(use_regex=true) = GPT-2 正则（K-LAYA-001）：
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
 * 与 Qwen 版的差异：前缀仅「可选一个空格」（非任意非字母数字字符）、数字成组、
 * 标点段不带尾随换行、无 \s*[\r\n]+ 组。 */
static int pretoken_span_gpt2(const unsigned char *s, int len) {
    int r;
    if ((r = m_g1(s, len))) return r;
    {   /* ' ?\p{L}+' */
        int p = (s[0] == ' ') ? 1 : 0, q = p;
        while (q < len) { uint32_t c; int n = u8_dec(s + q, len - q, &c); if (!cp_let(c)) break; q += n; }
        if (q > p) return q;
    }
    {   /* ' ?\p{N}+' */
        int p = (s[0] == ' ') ? 1 : 0, q = p;
        while (q < len) { uint32_t c; int n = u8_dec(s + q, len - q, &c); if (!cp_dig(c)) break; q += n; }
        if (q > p) return q;
    }
    {   /* ' ?[^\s\p{L}\p{N}]+' */
        int p = (s[0] == ' ') ? 1 : 0, q = p, cnt = 0;
        while (q < len) {
            uint32_t c; int n = u8_dec(s + q, len - q, &c);
            if (cp_ws(c) || cp_let(c) || cp_dig(c)) break;
            q += n; cnt++;
        }
        if (cnt > 0) return q;
    }
    if ((r = m_g6(s, len))) return r;   /* \s+(?!\S) */
    if ((r = m_g7(s, len))) return r;   /* \s+ */
    return 1;
}

/* 载入 merges.bin（见 tools/gen_merges_bin.py 的二进制布局）。 */
static int tok_bpe_load_bin(QwenTokenizer *tok, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/merges.bin", model_dir);
    FILE *f = st_fopen(path, "rb");
    if (!f) {
        snprintf(path, sizeof(path), "%s/build/merges.bin", model_dir);
        f = st_fopen(path, "rb");
    }
    if (!f) {
#ifdef __linux__
        char exe_path[1024];
        ssize_t rl = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (rl > 0) {
            exe_path[rl] = '\0';
            char *slash = strrchr(exe_path, '/');
            if (slash) {
                *slash = '\0';
                snprintf(path, sizeof(path), "%s/merges.bin", exe_path);
                f = st_fopen(path, "rb");
            }
        }
#endif
    }
    if (!f) return -1;

    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != 0x314D4251u) { fclose(f); return -1; }
    int nm = (int)hdr[2], na = (int)hdr[3];
    if (nm <= 0 || nm > 4000000 || na < 0 || na > 200000) { fclose(f); return -1; }

    uint32_t *off = malloc(sizeof(uint32_t) * (size_t)nm);
    uint16_t *ll  = malloc(sizeof(uint16_t) * (size_t)nm);
    uint16_t *lr  = malloc(sizeof(uint16_t) * (size_t)nm);
    size_t cap = 1u << 20, len = 0;
    char *str = malloc(cap);
    if (!off || !ll || !lr || !str) {
        free(off); free(ll); free(lr); free(str); fclose(f); return -1;
    }
    int ok = 1;
    for (int i = 0; i < nm && ok; i++) {
        uint16_t a, b;
        if (fread(&a, 2, 1, f) != 1 || fread(&b, 2, 1, f) != 1) { ok = 0; break; }
        if (len + (size_t)a + b > cap) {
            while (len + (size_t)a + b > cap) cap <<= 1;
            char *ns = realloc(str, cap);
            if (!ns) { ok = 0; break; }
            str = ns;
        }
        off[i] = (uint32_t)len; ll[i] = a; lr[i] = b;
        if (a && fread(str + len, 1, a, f) != a) { ok = 0; break; }
        len += a;
        if (b && fread(str + len, 1, b, f) != b) { ok = 0; break; }
        len += b;
    }
    int tcap = 1;
    while (tcap < nm * 2) tcap <<= 1;
    int *tab = ok ? calloc((size_t)tcap, sizeof(int)) : NULL;
    if (ok && !tab) ok = 0;
    if (ok) {
        uint32_t mask = (uint32_t)(tcap - 1);
        for (int i = 0; i < nm; i++) {
            uint32_t h = bpe_pair_hash(str + off[i], ll[i],
                                       str + off[i] + ll[i], lr[i]) & mask;
            while (tab[h]) h = (h + 1) & mask;
            tab[h] = i + 1;
        }
    }
    uint32_t *aid = NULL; uint32_t *aoff = NULL; uint16_t *alen = NULL;
    if (ok && na > 0) {
        aid  = malloc(sizeof(uint32_t) * (size_t)na);
        aoff = malloc(sizeof(uint32_t) * (size_t)na);
        alen = malloc(sizeof(uint16_t) * (size_t)na);
        if (!aid || !aoff || !alen) ok = 0;
    }
    for (int i = 0; i < na && ok; i++) {
        uint32_t id; uint16_t cl;
        if (fread(&id, 4, 1, f) != 1 || fread(&cl, 2, 1, f) != 1) { ok = 0; break; }
        if (len + cl > cap) {
            while (len + cl > cap) cap <<= 1;
            char *ns = realloc(str, cap);
            if (!ns) { ok = 0; break; }
            str = ns;
        }
        aoff[i] = (uint32_t)len; alen[i] = cl; aid[i] = id;
        if (cl && fread(str + len, 1, cl, f) != cl) { ok = 0; break; }
        len += cl;
    }
    fclose(f);
    if (!ok) {
        free(off); free(ll); free(lr); free(str); free(tab);
        free(aid); free(aoff); free(alen);
        return -1;
    }
    tok->bpe_str = str; tok->bpe_str_len = len;
    tok->bm_off = off; tok->bm_llen = ll; tok->bm_rlen = lr;
    tok->bm_tab = tab; tok->bm_mask = tcap - 1; tok->n_merges = nm;
    tok->at_id = aid; tok->at_off = aoff; tok->at_len = alen; tok->n_added = na;
    tok->bpe_ready = 1;
    printf("[TOK] BPE merges loaded: %d merges, %d added tokens (%zu KB)\n",
           nm, na, len / 1024);
    return 0;
}

/* 从 from 起找最近的 added-token 起始字节偏移（找不到返回 len）。
 * added（<|im_*|>/<think> 等）都以 '<' 开头，只在 '<' 处比对。用作预分词
 * piece 的硬边界：否则 "。" + "<|im_end|>" 会被 g4 当成一段 "。<" 合并，
 * 把特殊 token 的首字节吞掉（模板尾错位）。 */
static int next_added_start(const QwenTokenizer *tok, const char *s, int len,
                            int from) {
    for (int k = from; k < len; k++) {
        /* K-LAYA-001：added token 首字节不一定 '<'（ModernBERT 有 '|'/'['/空格），
         * 故在所有位置试匹配；对 Qwen（全 '<' 前缀）结果不变。 */
        for (int i = 0; i < tok->n_added; i++) {
            int l = tok->at_len[i];
            if (l <= 0 || k + l > len) continue;
            if (memcmp(s + k, tok->bpe_str + tok->at_off[i], (size_t)l) == 0)
                return k;
        }
    }
    return len;
}

static int tok_bpe_encode(QwenTokenizer *tok, const char *text, int text_len,
                          int *token_ids, int max_tokens, int gpt2_span) {
    b2u_init();
    const unsigned char *s = (const unsigned char *)text;
    int pos = 0, num = 0;
    while (pos < text_len && num < max_tokens) {
        /* 1. added/special 整串最长匹配 */
        int best = -1, best_len = 0;
        for (int i = 0; i < tok->n_added; i++) {
            int l = tok->at_len[i];
            if (l <= best_len || l > text_len - pos) continue;
            if (memcmp(text + pos, tok->bpe_str + tok->at_off[i], (size_t)l) == 0) {
                best = i; best_len = l;
            }
        }
        if (best >= 0) {
            token_ids[num++] = (int)tok->at_id[best];
            pos += best_len;
            continue;
        }
        /* 2. 预分词切 piece；超长 piece 切分处理（避免栈内固定缓冲溢出） */
        int wlen = gpt2_span ? pretoken_span_gpt2(s + pos, text_len - pos)
                             : pretoken_span(s + pos, text_len - pos);
        if (wlen <= 0) wlen = 1;
        if (wlen > text_len - pos) wlen = text_len - pos;
        if (wlen > TK_BPE_MAXSYM) wlen = TK_BPE_MAXSYM;
        /* added-token 硬边界：piece 不得跨过后面出现的特殊 token 起始位置。 */
        int cap = next_added_start(tok, text + pos, text_len - pos, 1);
        if (wlen > cap) wlen = cap;

        /* 3. raw → repr */
        char rb[TK_BPE_RBUF];
        int rl = 0;
        for (int i = 0; i < wlen; i++) {
            int L = s_b2u_len[s[pos + i]];
            memcpy(rb + rl, s_b2u[s[pos + i]], (size_t)L);
            rl += L;
        }
        /* 4. BPE 最小合并（左优先） */
        int bnd[TK_BPE_MAXSYM + 1];
        int ns = 0, o = 0;
        for (int i = 0; i < wlen; i++) { bnd[ns++] = o; o += s_b2u_len[s[pos + i]]; }
        bnd[ns] = o;
        for (;;) {
            int bi = -1, brank = 0x7FFFFFFF;
            for (int i = 0; i + 1 < ns; i++) {
                int r = bpe_rank(tok, rb + bnd[i], bnd[i + 1] - bnd[i],
                                      rb + bnd[i + 1], bnd[i + 2] - bnd[i + 1]);
                if (r >= 0 && r < brank) { brank = r; bi = i; }
            }
            if (bi < 0) break;
            for (int j = bi + 1; j < ns; j++) bnd[j] = bnd[j + 1];
            ns--;
        }
        if (getenv("VLLM_TOK_DBG")) {
            fprintf(stderr, "[TOK-DBG] piece=");
            for (int i = 0; i < wlen; i++) fprintf(stderr, "%02x", s[pos + i]);
            fprintf(stderr, " syms=");
            for (int i = 0; i < ns; i++) {
                fprintf(stderr, "[");
                for (int k = bnd[i]; k < bnd[i + 1]; k++)
                    fprintf(stderr, "%02x", (unsigned char)rb[k]);
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
        /* 5. 逐符号查词表 */
        for (int i = 0; i < ns && num < max_tokens; i++) {
            int id = find_exact_token(tok, rb + bnd[i], bnd[i + 1] - bnd[i]);
            token_ids[num++] = (id >= 0) ? id : 0;
        }
        pos += wlen;
    }
    return num;
}

/* K-LAYA-001：GPT-2/ByteLevel 预分词版编码（供 ModernBERT/Laya 使用）。
 * 复用同一 QwenTokenizer（vocab.bin + merges.bin 均为「引擎域」），仅换预分词正则。 */
int qwen_tokenizer_encode_gpt2(QwenTokenizer *tok, const char *text,
                               int *token_ids, int max_tokens) {
    if (!tok->is_loaded || !tok->bpe_ready) return 0;
    return tok_bpe_encode(tok, text, (int)strlen(text), token_ids, max_tokens, 1);
}

int qwen_tokenizer_encode(QwenTokenizer *tok, const char *text,
                           int *token_ids, int max_tokens) {
    if (!tok->is_loaded) return 0;

    const char *legacy = getenv("VLLM_TOK_LEGACY");
    if (tok->bpe_ready && !(legacy && legacy[0] == '1'))
        return tok_bpe_encode(tok, text, (int)strlen(text), token_ids, max_tokens, 0);

    int text_len = (int)strlen(text);
    int pos = 0;
    int num = 0;

    while (pos < text_len && num < max_tokens) {
        /* Skip spaces (but preserve them as position markers) */
        /* We attempt to match with leading space for BPE-style encoding */
        int had_space = 0;
        while (pos < text_len && text[pos] == ' ') {
            pos++;
            had_space = 1;
        }
        if (pos >= text_len) break;

        const char *rem = text + pos;
        int rem_len = text_len - pos;

        int match_len = 0;
        int token_id = -1;

        /* 特殊 token 优先：整串精确命中就直接产出特殊 token id，避免被下面的
         * 贪心最长匹配吞掉首字节（见 QWEN_SPECIAL_STR 上方注释）。 */
        {
            int sp_hit = -1, sp_len = 0;
            for (int i = 0; i < QWEN_N_SPECIAL; i++) {
                int l = (int)strlen(QWEN_SPECIAL_STR[i]);
                if (l > rem_len || l <= sp_len) continue;
                if (memcmp(rem, QWEN_SPECIAL_STR[i], (size_t)l) == 0) {
                    sp_hit = i;
                    sp_len = l;
                }
            }
            if (sp_hit >= 0) {
                int sid = special_token_id(tok, sp_hit);
                if (sid >= 0) {
                    token_ids[num++] = sid;
                    pos += sp_len;
                    continue;
                }
            }
        }

        /* 普通 token 的匹配上界：不得跨过后面出现的特殊 token 起始位置。 */
        int lim = special_start_after(rem, rem_len);

        /* For Qwen2 tokenizer, word continuations typically use "Ġ" prefix (0xC4 0xA0).
         * Try matching with this prefix first if we had a space before this word. */
        if (had_space && num > 0) {
            char spaced_buf[512];
            int buf_len = rem_len + 2;
            if (buf_len > 511) buf_len = 511;
            spaced_buf[0] = '\xC4';
            spaced_buf[1] = '\xA0';
            if (buf_len - 2 > 0) memcpy(spaced_buf + 2, rem, (size_t)(buf_len - 2));
            spaced_buf[buf_len] = '\0';

            token_id = find_longest_match_qwen(tok, spaced_buf, buf_len,
                                               buf_len, &match_len);
            if (token_id >= 0 && match_len > 2 && (match_len - 2) <= lim) {
                pos += (match_len - 2);  /* subtract the 2-byte Ġ prefix */
                token_ids[num++] = token_id;
                continue;
            }
        }

        /* Try matching without space prefix */
        token_id = find_longest_match_qwen(tok, rem, rem_len, lim, &match_len);
        if (token_id >= 0 && match_len > 0) {
            token_ids[num++] = token_id;
            pos += match_len;
        } else {
            /* Fallback: encode single byte as character */
            if (rem[0] == '\n') {
                /* The Qwen BPE represents a literal newline as the "Ċ"
                 * marker (U+010A, bytes C4 8A); vocab.bin has no standalone
                 * 0x0A entry, so a raw '\n' would otherwise fall through to
                 * <unk> (id 0) and corrupt the prompt template.
                 * 连续的换行同样要按「连续多个 Ċ」去匹配：HF 把 "\n\n"
                 * 切成单个 token 271（ĊĊ），逐个吃会变成两个 198，与参考
                 * 分词不一致（Qwen3 非 thinking 模板的 `<think>\n\n</think>\n\n`
                 * 正是这种形状）。 */
                int nl = 0;
                while (nl < rem_len && rem[nl] == '\n') nl++;
                if (nl > 8) nl = 8;
                if (nl > lim) nl = lim;
                if (nl > 0) {
                    char nb[16];
                    for (int i = 0; i < nl; i++) {
                        nb[i * 2] = '\xC4';
                        nb[i * 2 + 1] = '\x8A';
                    }
                    int ml2 = 0;
                    int tid2 = find_longest_match_qwen(tok, nb, nl * 2, nl * 2, &ml2);
                    if (tid2 >= 0 && ml2 >= 2 && (ml2 % 2) == 0) {
                        token_ids[num++] = tid2;
                        pos += ml2 / 2;
                        continue;
                    }
                }
            }
            char c[2] = {rem[0], '\0'};
            int ml;
            int tid = find_longest_match_qwen(tok, c, 1, 1, &ml);
            if (tid >= 0 && ml > 0) {
                token_ids[num++] = tid;
            } else {
                /* Use unk or just skip */
                token_ids[num++] = 0; /* <unk> */
            }
            pos++;
        }
    }

    return num;
}

/* ================================================================
 * Decode
 * ================================================================ */

const char* qwen_tokenizer_decode(const QwenTokenizer *tok, int token_id) {
    if (!tok->is_loaded) return "";
    if (token_id < 0 || token_id >= tok->vocab_size) return "[UNK]";
    if (!tok->strings[token_id]) return "";
    return tok->strings[token_id];
}

/* ================================================================
 * Load special token IDs from config.json
 * ================================================================ */

/* Minimal JSON integer value parser */
static int json_get_int(const char *json, const char *key, int *out) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return -1;
    pos += strlen(search);
    while (*pos == ' ' || *pos == ':' || *pos == '\t') pos++;
    if (*pos == 'n') { *out = -1; return -1; }  /* null */
    char *end;
    long v = strtol(pos, &end, 10);
    if (end == pos) return -1;
    *out = (int)v;
    return 0;
}

int qwen_tokenizer_load_special(QwenTokenizer *tok, const char *model_dir) {
    if (!tok->is_loaded) return -1;

    /* Read config.json */
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TOK] Cannot open config.json for special tokens\n");
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc((size_t)fsize + 1);
    if (!json) { fclose(f); return -1; }
    size_t nread = fread(json, 1, (size_t)fsize, f);
    fclose(f);
    json[nread] = '\0';

    /* Qwen3-VL config has text_config nested object.
     * bos_token_id, eos_token_id are inside text_config. */
    json_get_int(json, "bos_token_id", &tok->bos_id);
    json_get_int(json, "eos_token_id", &tok->eos_id);

    /* Find im_start / im_end from tokenizer_config or hardcode known values */
    /* Qwen3-VL known specials: im_start=151644, im_end=151645, pad=151643 */
    if (tok->bos_id < 0) tok->bos_id = 151643;   /* <|endoftext|> */
    if (tok->eos_id < 0) tok->eos_id = 151645;   /* <|im_end|> */
    tok->im_start_id = 151644;  /* <|im_start|> */
    tok->im_end_id   = 151645;  /* <|im_end|> */
    tok->pad_id      = 151643;  /* <|endoftext|> */

    free(json);

    printf("[TOK] Special tokens: bos=%d eos=%d im_start=%d im_end=%d pad=%d\n",
           tok->bos_id, tok->eos_id, tok->im_start_id, tok->im_end_id, tok->pad_id);
    return 0;
}
