# K-BENCH-019 · 板端 2B 档（Qwen3-VL-2B）：vllm_kestrel vs llama.cpp

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-BENCH-019** |
| **目标** | 给出板端 **2B 档**的核心技术指标（吞吐/延迟/内存/长上下文质量/多轮），并与 llama.cpp 对照；补齐 8B(K-BENCH-017)/MoE(K-BENCH-018) 之外的**最轻档**画像 |
| **日期** | 2026-10-10 |
| **载体** | 板端 `192.168.1.92`（RK3588 OPi5+，4×A76@2.256G + 4×A55@1.8G，RAM 15964 MB） |
| **被测 A** | `vllm_kestrel`（2026-10-10 构建）+ `/mnt/VQF/2b/qwen3vl2b.q4.vqf`（**2.33 GB**，本轮**新构建**，见 §0） |
| **被测 B** | `llama.cpp`（`/root/llama_build_master`）+ `/mnt/emmc/llama/Qwen3-VL-2B-Q4_0.gguf`（**1.05 GB**，1.72B 参数） |
| **判定** | 见 §6 |

---

## 0. 载体构建（2B 档此前**不存在** VQF，本轮从零构建）

**背景**：板端 `/mnt/VQF` 只有 8B 与 30B-A3B 的 VQF；`/mnt/emmc/llama/` 有 Qwen3-VL-2B 的 GGUF（F16/Q8_0/Q4_0），但**引擎只读 VQF** ⇒ 必须先转换。

**流程（全部在板端完成）**：
1. **转换器可移植**：`vqf_convert/`（纯 C，无 x86 intrinsic，只有 Win/Linux 平台层）用板端 `gcc 11.4 (aarch64)` 直接编译通过（`build.sh` 虽写 x86_64）：`gcc -O2 -w -I src src/conv_main.c src/model_cfgio.c src/model_layers.c src/quant_kernels.c src/qk_repack.c src/vqf_vision.c src/vllm_crypto.c -o /tmp/vqf_conv -lm`。
2. **下载权重**：`hf-mirror.com/Qwen/Qwen3-VL-2B-Instruct`（`config.json` + `model.safetensors`，**4 255 140 312 B**）→ `/root/models/qwen3vl2b/`（板端 ~3.5 MB/s，约 20 min）。
3. **转换**：`/tmp/vqf_conv --model /root/models/qwen3vl2b --convert-vqf /mnt/VQF/2b/qwen3vl2b.q4.vqf --wmode q4`（126 s）⇒ **41 tensors / 2 331 316 232 B / flags=0x63**，含 ViT（24 层，Q8_0）。config：text 侧 **28 层 / dim 2048 / 16 heads / 8 kv_heads / hd 128 / ffn 6144**，`tie_word_embeddings=true`（转换器正常处理）。
4. **手工放置**（**关键**）：输出目录须同时有 **`config.json`** 与 **`vocab.bin`**——否则引擎报 `[TOK] Cannot open config.json for special tokens` → `[STREAM] FAIL: tokenizer load`。`vocab.bin` 复用 Qwen3 族共享版（151669 tokens，与 8B 同）。
5. **冒烟**（`--stream-test ctx=256`）：`vqf_load OK (0.40s): dim=2048 layers=28 heads=16 kv=8 ffn=6144 vocab=151936 max_seq=8192`；prefill **75 tok/s**、decode **117 ms/tok**、RSS 0.98 GB；输出连贯中文（"逐层加载推理是一种在深度学习…"）。

> ⚠️ **2B 的 `max_seq`（VQF 头）= 8192**（8B 为 20480）⇒ 本项上下文上限 8K，多轮上限 ~7.8K。

---

## 1. 口径与可比性披露（先冻结再采数）

- governor=`performance`；线程 **t4**；每档预热后测；每轮前 `pkill`；**单会话读数**。
- 引擎口径：`--stream-test --stream-ctx N --bench-seqlen N --stream-n M`（`ctx=8192` 时 `M=8`，其余 `M=32`）。
- llama 口径：`llama-bench -p N -n 32 -t 4 -r 2`（prefill）与 `-n 32 -d N -t 4 -r 2`（decode @depth N）。
- ⚠️ **量化格不同**（VQF **2.33 GB** vs GGUF **1.05 GB**；VQF 主因含 Q8 段与 ViT）⇒ 依 `quantization_grid_bucket_comparability` **只比总时间、不做桶级归因**。
- ⚠️ **chat 模板口径**：2B-Instruct **不接受**引擎的 no-think 后缀（见附录 A）⇒ 本轮全部走**默认（thinking 开）**。

---

## 2. 基线性能对比（默认档，t4）

### 2.1 prefill 吞吐（tok/s，越大越好）

| ctx | vllm_kestrel | llama.cpp | 谁快 |
|---|---|---|---|
| 1K | 73.4 | 78.47 | llama **1.07×** |
| 2K | 61.5 | 66.07 | llama **1.07×** |
| 4K | 43.7 | 50.34 | llama **1.15×** |
| 8K | 27.2 | 34.15 | llama **1.26×** |

⇒ 引擎 prefill 落后 **1.07–1.26×**（**轻于** 8B 档的 1.14–1.38×）。

### 2.2 decode TPOT（ms/tok，越小越好）——**交叉点在 ≤1K**

| ctx | vllm_kestrel | llama.cpp | 谁快 |
|---|---|---|---|
| 1K | 73 | 75.4 | vllm **1.03×** |
| 2K | 101 | 126.6 | vllm **1.25×** |
| 4K | 172 | 233.7 | vllm **1.36×** |
| 8K | 309 | 426.2 | vllm **1.38×** |

- **引擎 decode 全程领先**（8B 档是 ~2K 才反超；2B 档在 1K 附近即打平/反超）。
- llama decode 1K→8K 劣化 **5.65×**（75.4→426.2），引擎 **4.23×**（73→309）。

### 2.3 内存

| 项 | vllm_kestrel | llama.cpp |
|---|---|---|
| 模型文件 | 2.33 GB | 1.05 GB |
| 进程 RSS（8K 会话） | **1.37 GB** | 未单测 |

---

## 3. 长上下文质量（NIAH，gen=128）

**判据**：needle=`"Zara likes the color teal and her favorite food is ramen."`，greedy 输出是否含 `ramen`。⚠️ 2B 为**先推理后作答**型，`--longctx-gen 32` 会**高估失败**（引擎自警告）⇒ 本轮统一 **gen=128**；llama 同为 `-n 128` 且只取 `Answer:` 之后的生成段。

| ctx | 位置 | vllm_kestrel | llama.cpp |
|---|---|---|---|
| 2048 | 0 / 25 / 50 / 75 / 100% | **Y Y Y Y N** | **Y Y Y Y Y** |
| 4096 | 0 / 25 / 50 / 75 / 100% | **Y Y Y Y Y** | **Y Y Y Y Y** |

- 引擎 @2K `pos=100%` 为**真失**：模型把 "Zara" 误读为 "Zarra" 并断言"列表中没有此人"。
- @4K 五位全 Y（引用了 needle 并作答 `ramen`）。
- ⇒ 2B 档 2K 处引擎 **4/5**、llama **5/5**；4K 处**双方 5/5**。

---

## 4. 多轮对话（会话模式）

**口径**：引擎 `--serve --auto-load` vs `llama-server -c 8192 -t 4 -np 1`；同一对话（首轮植入代号「蓝莓」→ 11 轮 ~740-tok 背景材料 + 「用一句话概括」→ 末轮追问代号），两臂**重放同一批 assistant 回复**；`gen≤64`。

| 轮 | prompt tok（引擎/llama） | prefill 引擎 (ms) | prefill llama (ms) | 引擎/llama | TPOT 引擎 (ms) | TPOT llama (ms) | 引擎/llama |
|---|---|---|---|---|---|---|---|
| 1 | 39 / 39 | 507 | 410 | 1.24× | 48 | 53 | **0.91×** |
| 2 | 754 / 752 | 9 143 | 8 832 | 1.04× | 72 | 79 | **0.91×** |
| 3 | 1 525 / 1 521 | 11 876 | 12 073 | 0.98× | 85 | 106 | **0.80×** |
| 4 | 2 296 / 2 290 | 15 303 | 14 743 | 1.04× | 110 | 146 | **0.75×** |
| 5 | 3 067 / 3 059 | 19 128 | 17 037 | 1.12× | 135 | 198 | **0.68×** |
| 6 | 3 838 / 3 828 | 22 613 | 19 740 | 1.15× | 160 | 232 | **0.69×** |
| 7 | 4 609 / 4 597 | 26 061 | 22 378 | 1.16× | 185 | 278 | **0.67×** |
| 8 | 5 380 / 5 366 | 29 568 | 24 919 | 1.19× | 211 | 310 | **0.68×** |
| 9 | 6 151 / 6 135 | 33 250 | 27 695 | 1.20× | 237 | 344 | **0.69×** |
| 10 | 6 923 / 6 905 | 37 396 | 30 535 | 1.22× | 262 | 385 | **0.68×** |
| 11 | 7 695 / 7 675 | 39 912 | 33 243 | 1.20× | 301 | 414 | **0.73×** |
| 12 | 7 784 / 7 764 | 1 800 | 6 583 | — | 570 | 409 | — |

- **两臂末轮均输出 `蓝莓`** ⇒ **多轮记忆保留正确 @7.8K**（引擎 ✓ / llama ✓）。
- **前缀复用两侧均生效**（引擎 prefill 只按增量；llama `cached_tokens` 逐轮达 7 738）。
- **增量 prefill 速率随 ctx 退化**：引擎每轮增量 ~771 tok，速率 **85 → 19 tok/s**（0.75K→7.7K）。
- **decode**：引擎全程领先 **0.67–0.91×**（即快 1.10–1.49×），与 §2.2 同向。
- **prefill**：引擎 **0.98–1.24×**（长上下文侧略慢）。

---

## 5. 优化档矩阵（稀疏注意力 / 逐层驻留）〔2026-10-10 追加〕

**口径**：CLI `--stream-test --stream-ctx N --bench-seqlen N --stream-n M`（ctx=8192→M=8、ctx=4096→M=32）；t4 / governor=performance；**同一会话内交错**（A 默认在**首、中段各测一次**作漂移锚，符合 `session_local_ab_caliber`）；每臂单次读数，逐 k 档比比值。

- **生效性凭证**（serve 模式 `[OPT]` 回显）：`sparse_attn=1 sparse_k=32 sparse_block=32 sparse_probe=8 sparse_min_ctx=1024 sparse_pf_group=0 sparse_pf_k=0 sparse_pf_min_ctx=3072 l3_evict=0` ⇒ 证明 `--sparse-attn --sparse-k 32` 已解析；`--sparse-attn` **默认只作用于 decode**，稀疏 **prefill** 须 `VLLM_SPARSE_PREFILL=1` **且** ctx≥3072 才启用（[vllm_safetensors.c L18826-L18851](../src/model/vllm_safetensors.c#L18826-L18851)）。
- **档位定义**：A 默认（无稀疏/全驻留）；**B 稀疏 decode** = `--sparse-attn --sparse-k 32`；**C 稀疏 decode+prefill** = B + `VLLM_SPARSE_PREFILL=1`；**F 逐层驻留** = `VLLM_VQF_STREAM=1`（keep=1）；**F16** = `VLLM_VQF_STREAM=16`。

### 5.1 ctx=8192（prompt 8182 tok）

| 档 | prefill | prefill t/s | prefill ATTN 份额 | decode TPOT | rss_end |
|---|---|---|---|---|---|
| A 默认 | 291.8 s | 28.0 | 71.3% | 301 ms | 1400 MB |
| **B 稀疏 decode** | 302.0 s | 27.0 | 71.1% | **148 ms（2.03×）** | 1401 MB |
| **C 稀疏 decode+prefill** | **178.0 s** | **46.0（1.64×）** | **49.9%** | **148 ms（2.03×）** | 1399 MB |
| A 复测（漂移锚） | 301.2 s | 27.0 | 70.9% | 301 ms | 1399 MB |
| **F 逐层驻留 keep=1** | 307.8 s | 27.0 | 70.8% | 348 ms（0.86×） | **669 MB（−52%）** |
| **F16 逐层驻留 keep=16** | 307.2 s | 27.0 | 70.7% | 351 ms（0.86×） | **1073 MB（−23%）** |

### 5.2 ctx=4096（prompt 4086 tok）

| 档 | prefill | prefill t/s | prefill ATTN 份额 | decode TPOT | rss_end |
|---|---|---|---|---|---|
| A 默认 | 93.4 s | 44.0 | 54.2% | 165 ms | 1169 MB |
| **B 稀疏 decode** | 93.8 s | 44.0 | 54.4% | **111 ms（1.49×）** | 1168 MB |
| **C 稀疏 decode+prefill** | **81.9 s** | **50.0（1.14×）** | **46.2%** | **114 ms（1.45×）** | 1171 MB |
| A 复测（漂移锚） | 92.1 s | 44.0 | 53.7% | 165 ms | 1169 MB |
| **F 逐层驻留 keep=1** | 93.8 s | 44.0 | 53.9% | 201 ms（0.82×） | **436 MB（−63%）** |
| **F16 逐层驻留 keep=16** | 94.1 s | 43.0 | 54.3% | 183 ms（0.90×） | **842 MB（−28%）** |

### 5.3 判读

1. **稀疏 decode = 本档最大且最稳的杠杆**：decode **2.03× @8K / 1.49× @4K**（增益随 ctx 增大——注意力份额越大越省）；**prefill 不受影响**（0.96–1.00×，漂移锚 A_rep 同为 0.96–1.00× ⇒ 该 0.96× 是会话漂移而非回归）。decode 是真实在线负载主体，故这是本档最有价值的档位。
2. **稀疏 prefill 只赚 prefill**：**1.64× @8K / 1.14× @4K**，prefill ATTN 份额 **71.3%→49.9%**（8K）；ctx 越大越值（<3072 是净亏，与源码门控设计一致）。B 与 C 的 decode 相同（148 ms）⇒ 稀疏 prefill 不影响 decode 档。
3. **逐层驻留 = 纯代价换 RSS**：keep=1 RSS **−52%（8K）/−63%（4K）**、keep=16 **−23%/−28%**；但 decode **慢 +15.6%…+21.8%（keep=1，0.86×/0.82×）/ +10.9%…+16.6%（keep=16，0.86×/0.90×）**，prefill 基本不变。
   - RSS 降幅**与 ctx 无关**（keep=1：−731/−733 MB；keep=16：−327 MB 两档一致）⇒ 省的是**权重页**而非 KV。
   - keep=1 与 keep=16 的 decode 罚**几乎相同**（348 vs 351）⇒ 罚来自**每 token 的驻留管理（madvise/DONTNEED churn）**，与"留住多少层"弱相关 —— 与 K-L0-004「异步预取无效、驱逐才是决定性动作」同构。
   - ⚠️ **系统级无真实收益**：2B 模型仅 2.33 GB、全驻留 RSS 仅 1.4 GB ≪ 板端 15.9 GB RAM ⇒ 本档**无内存压力可解**，RSS 降低只是权重页搬去页缓存（与 K-L0-004 §9 对 8B 的结论一致）。
4. **质量未门（诚实披露）**：稀疏为**近似注意力**，本节**只测性能**。源码内**已有 2B 实测记录**：`needle@50% / k=32` 下稀疏 prefill 使针块在**写入侧**被污染，模型**答"未指定"**（[vllm_safetensors.c L2498-L2510](../src/model/vllm_safetensors.c#L2498-L2510)）⇒ **稀疏 prefill 上线前必须过质量门（本项未重测）**；稀疏 decode 的质量门同样未在本项覆盖。

---

## 6. 结论

1. **2B 档可在板端从零构建**（板端编译转换器 → 下载 HF → 转 VQF；**须手工放 `config.json`+`vocab.bin`**）。
2. **prefill**：引擎落后 llama **1.07–1.26×**（比 8B 档的 1.14–1.38× **更接近**）。
3. **decode**：引擎**全程领先 1.03–1.38×**，**交叉点 ≤1K**（8B 档为 ~2K）⇒ 档位越小，引擎的 decode 优势越早兑现。
4. **质量**：NIAH 引擎 @2K **4/5**（pos100 真失）/ @4K **5/5**，llama 两侧 5/5；多轮记忆 **@7.8K 双方均正确**。
5. **内存**：引擎 RSS **1.37 GB**（8K 会话）≪ 板端 RAM。
6. ⚠️ **2B 的 VQF `max_seq` 仅 8192** ⇒ 本档 8K 封顶（8B/30B 分别 20480）；多轮只能到 ~7.8K。
7. **优化档（§5，2026-10-10 追加）**：**稀疏 decode** decode **2.03× @8K / 1.49× @4K**（prefill 不变，最大且最稳）；**稀疏 prefill** prefill **1.64× @8K / 1.14× @4K**（ATTN 份额 71.3%→49.9%，但**质量风险已知、未门**）；**逐层驻留** RSS **−52%…−63%（keep=1）** 换 decode **慢 15.6%…21.8%**，且 2B 全驻留仅 1.4 GB ≪ 15.9 GB ⇒ **本档无内存压力、驻留纯负**。

### 未覆盖（诚实登记）

1. **16K+**（VQF 头 `max_seq=8192` 硬限；未验证引擎能否抬 max_seq）。
2. **稀疏档质量门**（§5 只测性能；源码内已有 2B 稀疏 prefill 污染针块的记录，须独立复测 NIAH/SC 才能上线）；**L3 落盘档**（`--l3-evict`）未在 2B 复测；逐层驻留与稀疏的组合档未测。
3. **llama 侧同口径 RSS/VmHWM**。
4. **t8/t1/t2 线程扩展曲线**。
5. **全为单会话读数**；`ctx=8192` 用 gen=8（其余 gen=32），口径略异；§5 各臂单次读数（A 有复测锚）。

---

## 附录 A · 测试中发现（2B 的 no-think 后缀不兼容；**未修复**，登记备查）

- **现象**：2B（Qwen3-VL-2B-Instruct）经 `--serve` 聊天路径，`enable_thinking=false` 时 **`n_tokens=0`（立即 EOS）或复述 prompt**（输出 `<|im_start|>user …`）；追问代号输出错乱。
- **定位（4 组对照）**：

| 档 | 请求 | 输出 |
|---|---|---|
| P1 | `enable_thinking=false` + system | `''`（gen=0） |
| P2 | `enable_thinking=true` + system | `已记住代号：蓝莓。`（gen=8）✓ |
| P3 | `false`，无 system | `''`（gen=0） |
| P4 | `true`，无 system | `已收到，代号为"蓝莓"确认。`（gen=12）✓ |
| P5 | 不传（默认 thinking 开） | `已记住代号：蓝莓。`（gen=8）✓ |

- **机制（判读）**：引擎在 `enable_thinking=false` 时于 `<|im_start|>assistant\n` 后**手写追加** `<think>\n\n</think>\n\n`（[vllm_server.c L191-L194](../src/serve/vllm_server.c#L191-L194)）。该后缀是 **Qwen3 文本族**的渲染约定；**Qwen3-VL-2B-Instruct 不接受**（llama.cpp 的同名开关经其 jinja 模板不产生该串，故正常）。
- **影响面**：仅影响**显式传 `enable_thinking=false`**的 VL-Instruct 请求；**不传（默认）即正常**。
- **本项处置**：K-BENCH-019 全部实测**改为默认（thinking 开）**以取得有效数据；**未改代码**。建议后续把该后缀改为**按模型类型/模板**选择（或提供逃生舱）。
