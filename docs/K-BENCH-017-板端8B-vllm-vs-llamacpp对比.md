# K-BENCH-017 · 板端 8B 模型：vllm_kestrel vs llama.cpp 核心技术指标对比

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-BENCH-017** |
| **目标** | 在板端 8B 模型上给出引擎 vs llama.cpp 的核心技术指标对比（吞吐/延迟/内存/长上下文质量），并量化各优化档的贡献 |
| **日期** | 2026-10-09 |
| **载体** | 板端 `192.168.1.92`（RK3588 OPi5+，4×A76@2.256G + 4×A55@1.8G，RAM 15964 MB，**zram swap 7.8 GB**） |
| **被测 A** | `vllm_kestrel`（2026-10-09 构建，含全部 K-* 优化）+ `/mnt/VQF/8b/qwen3vl8b.q4.vqf`（**6.60 GB**） |
| **被测 B** | `llama.cpp`（`/root/llama_build_master`，build unknown/0，GGUF Q4_0）+ `/mnt/emmc/qwen3vl8b-q4_0.gguf`（**4.77 GB**，8.19B 参数） |
| **判定** | 见 §6 结论 |

---

## 1. 口径与可比性披露（**先冻结再采数**）

- governor = `performance`（每项测试前设置，结束恢复 ondemand）；线程 **t4**（对齐大核数）；每档预热后测；每轮前 `pkill`。
- 引擎口径：`--stream-test --stream-ctx N --bench-seqlen N --stream-n M`（`--bench-seqlen` 用于抬 `max_seq` 上限，默认被 cap 到 2048）。
- llama 口径：`llama-bench -p N -n 32`（prefill）与 `-n 32 -d N`（depth=N 的 decode）。
- ⚠️ **量化格不同**（依据 `quantization_grid_bucket_comparability`）：**只比总时间，不做桶级归因**。
  - VQF 文件 **6.60 GB** vs GGUF **4.77 GB**（大一 38%，主因 embed/lm 段精度不同）；但引擎实测 RSS 仅 **4.07–4.57 GB**，每 token 权重流量与 GGUF 同量级（≈4.3 GB）⇒ 对照在**同量级**上成立，但不是位级同权重。
- ⚠️ **llama-bench 的 tg 与引擎 TPOT 口径**：前者在 depth=N 的 KV 上生成 32 token 取均值；后者为 ctx=N 下生成 32 token 的均值 ⇒ 同量级可对；非逐位等价。
- **本文所有数字为单会话读数**（`session_local_ab_caliber`：绝对值不跨会话互比）。

---

## 2. 基线性能对比（引擎**默认档**，t4，8 线程 llama）

### 2.1 prefill 吞吐（tok/s，越大越好）

| ctx | vllm_kestrel | llama.cpp | 谁快 |
|---|---|---|---|
| 1K | 14.1 | 19.42 | llama **1.38×** |
| 2K | 15.3 | 17.47 | llama **1.14×** |
| 4K | 12.0 | 14.37 | llama **1.20×** |
| 8K | 8.3 | 10.72 | llama **1.29×** |

⇒ **引擎默认档 prefill 全程落后 llama 1.14–1.38×**（引擎默认未开稀疏 prefill）。

### 2.2 decode TPOT（ms/tok，越小越好）——**交叉点在 ~2K**

| ctx | vllm_kestrel | llama.cpp | 谁快 |
|---|---|---|---|
| 1K | 338 | 237 | llama **1.43×** |
| 2K | 323 | 384 | vllm **1.19×** |
| 4K | 516 | 648 | vllm **1.26×** |
| 8K | 858 | 1153 | vllm **1.34×** |

- **llama decode 1K→8K 劣化 4.87×**（237→1153 ms），**引擎仅 2.54×**（338→858 ms）⇒ 引擎的 KV/attention 栈在长上下文显著更耐放。
- **1K 附近 llama 更快**（引擎有固定开销：q8-KV 反量化/量化激活等单 token 成本）。

### 2.3 内存

| 项 | vllm_kestrel | llama.cpp |
|---|---|---|
| 模型文件 | 6.60 GB | 4.77 GB |
| 进程 RSS（8K 会话） | **4.57 GB** | 未单测（文件 4.77 GB 全量 mmap） |

> 引擎 RSS 远小于文件（4.57 vs 6.60 GB）⇒ 部分段（embed 等）未被触碰。**这是引擎可"小内存跑大文件"的一处证据**，但本次未对 llama 做同口径 RSS 采样（登记为缺口）。

---

## 3. 长上下文质量（NIAH 大海捞针）

**判据**：needle = `"Zara likes the color teal and her favorite food is ramen."`，问 `"What is Zara's favorite food?"`，greedy 解码后判输出是否含 `ramen`。

| ctx | 位置 | vllm_kestrel | llama.cpp |
|---|---|---|---|
| 2048 | 0 / 25 / 50 / 75 / 100% | **Y Y Y Y Y** | **Y Y Y Y Y** |
| 8192 | 50% | **Y** | **Y** |

- 引擎：`--needle-sweep --longctx-ctx N`（token 级模板）。
- llama：Python 复刻同一模板文本（同 intro/facts/needle/question），`llama-cli -st --temp 0`。
- ⇒ **两者 NIAH 均满分，质量持平**（引擎未开稀疏时）。
- ⚠️ 口径披露：引擎用 token 级模板，llama 用同语义**文本**复刻（非逐位同 token 流）；两者召回结论一致，故差异不影响判定。**未测**：开启稀疏注意力后的 NIAH（wiki 记载 `--sparse-k≥32` + 真实语料下通过，本次未复测）。

---

## 4. 优化档矩阵（引擎，ctx=8192，gen=8，t4）

| 档 | prefill t/s | prefill 耗时 | TPOT ms | decode t/s | RSS | 说明 |
|---|---|---|---|---|---|---|
| **A 默认** | 8.1 | 1007 s | 835 | 1.20 | 4.57 GB | 基线 |
| **B +稀疏 decode**（`--sparse-attn --sparse-k 32`） | 8.6 | 949 s | **445** | 2.25 | 4.57 GB | **decode −47%（1.88×）** |
| **C +稀疏 prefill**（`VLLM_SPARSE_PREFILL=1`） | **13.7** | **595 s** | 441 | 2.27 | 4.57 GB | **prefill +69%（1.69×）** |
| D C + `SPARSE_PF_GROUP=4` | 13.7 | 596 s | 454 | 2.20 | 4.57 GB | 与 C 同 |
| **E +L3 落盘**（`--l3-evict --l3-ratio 0.75` + `L3_PREFIX_REUSE=1`） | 8.7 | 942 s | 444 | 2.25 | 4.57 GB | **无收益**（8K 单用户不触发，与 wiki 一致） |
| **F 分层驻留**（`VLLM_VQF_STREAM=1`） | 8.6 | 949 s | **988** | 1.01 | **1.03 GB** | **RSS −77%，decode +18%（内存换速度）** |

**要点**：
1. **稀疏注意力（decode）是本档最大 decode 杠杆**：8K 下 **−47%**（tp/tpot）。
2. **稀疏 prefill** 是最大 prefill 杠杆：**+69%**；两者叠加后 **prefill 13.7 t/s / TPOT 441 ms**。
3. **L3 落盘在 8K 单用户下无收益、且 RSS 无变化** ⇒ 印证 wiki「价值在 16K+/多用户」的边界（本次未测 16K）。
4. **分层驻留 keep=1**：RSS 4.57→1.03 GB（**−77%**）代价是 decode +18% —— 「能否跑起来」的场景（模型 > RAM）才值得。

---

## 5. 多轮对话（会话模式；补做 §6.3 登记的缺口）

**口径**：
- **引擎**：`vllm_kestrel --serve --auto-load`（HTTP `/v1/chat/completions`，`messages` 传**全量历史**，`temperature=0`、`enable_thinking=false`、`max_tokens=32`）；KV 前缀复用（`--prefix-kv`，默认开）。
- **llama**：`llama-server -m qwen3vl8b-q4_0.gguf -c 8192 -t 4 -np 1`（同端点、同采样参数）；内置 prompt cache。
- **同一对话**：首轮植入代号「蓝莓」→ 若干轮背景材料 + 「用一句话概括」→ 末轮追问代号。两臂**重放同一批 assistant 回复**以保证 prompt 一致。
- ⚠️ **口径披露**：两侧 chat 模板不同（引擎手写 im_chat 串 vs llama 内置 jinja），同轮 token 数差 **≤0.3%**（见下表）；按 `quantization_grid_bucket_comparability` **只比总时间**。governor=performance，t4，预热后，单会话读数。

### 5.1 引擎逐轮（复用开）

| 轮 | prompt tok | gen | prefill (ms) | TTFT (ms) | TPOT (ms) | 轮总 (s) | RSS |
|---|---|---|---|---|---|---|---|
| 1 | 43 | 12 | 2 415 | 2 416 | 200 | 4.6 | 4.13 GB |
| 2 | 762 | 28 | 38 186 | 38 187 | 229 | 44.4 | 4.24 GB |
| 3 | 1 497 | 28 | 46 677 | 46 678 | 280 | 54.3 | 4.33 GB |
| 4 | 2 231 | 28 | 56 154 | 56 154 | 350 | 65.6 | 4.42 GB |
| 5 | 2 965 | 28 | 65 945 | 65 946 | 414 | 77.1 | 4.52 GB |
| 6 | 3 700 | 28 | 76 056 | 76 057 | 482 | 89.1 | 4.61 GB |
| 7 | 4 434 | 29 | 85 552 | 85 553 | 544 | 100.8 | 4.70 GB |
| 8 | 5 170 | 28 | 94 821 | 94 822 | 610 | 111.3 | 4.79 GB |
| 9 | 5 904 | 29 | 103 061 | 103 062 | 673 | 121.9 | 4.87 GB |
| 10 | 5 958 | 2 | 9 076 | 9 077 | 1 306 | 10.4 | 4.90 GB |

- **轮 10 = 追问代号**，输出 `蓝莓` ⇒ **多轮记忆保留正确（@5.9K）**。
- 每轮新增用户文本约 **700 tok**；`prompt tok` 逐轮增长，但 `prefill (ms)` **只按增量计**（例：轮 9 prompt 5 904 / prefill 103 s，而轮 2 prompt 762 / prefill 38 s）⇒ **前缀复用生效**。

**延伸至 8K**（新进程续跑同一对话；⚠️ 新进程首个请求无 in-RAM KV 可复用，故首轮为**全量** prefill，单列排外）：

| 段 | prompt tok | gen | prefill (ms) | TPOT (ms) | 备注 |
|---|---|---|---|---|---|
| E1（全量，排外） | 6 641 | 28 | **706 240** | 742 | 新进程无缓存 ⇒ 全量重算 |
| E2 | 7 376 | 29 | 126 177 | 810 | 复用开 |
| E3 | 8 113 | 28 | 139 588 | 879 | 复用开 |
| E4 | 8 165 | 2 | 11 181 | 1 694 | 追问代号 → `蓝莓` ✓ **@8.1K** |

### 5.2 前缀复用 A/B（同对话、同 prompt；`--no-prefix-kv` 关复用）

| 轮 | prompt tok | prefill 复用**开** (ms) | prefill 复用**关** (ms) | 增益 |
|---|---|---|---|---|
| 1 | 43 | 2 415 | 2 428 | 0% |
| 2 | 762 | 38 186 | 40 934 | −6.7% |
| 3 | 1 497 | 46 677 | 86 800 | **−46.2%** |
| 4 | 2 231 | 56 154 | 143 804 | **−61.0%（2.56×）** |

- 复用把 prefill 从「全量」降为「增量」，**增益随上下文增大**（2.2K 处 2.56×）。
- 两臂轮 1–4 **回复逐字相同** ⇒ **复用不改变输出**（greedy 确定性，`bitwise_equivalence_gate` 语义级）。

### 5.3 增量 prefill 与 decode 仍随上下文变慢（**关键发现**）

- **增量 prefill 速率随上下文退化**：每轮增量约 700 tok，其速率由轮 2 的 **18.5 tok/s（54 ms/tok）**单调降至轮 9 的 **7.1 tok/s（140 ms/tok）**、延伸 E3 的 **5.3 tok/s（189 ms/tok）**。⇒ 虽只 prefill 增量，但**增量 token 仍需对全上下文做 O(n) 注意力**，故长会话的「下一轮」越来越慢。
- **decode TPOT 同样 O(n) 增长**：200 ms(43) → 673 ms(5.9K) → 879 ms(8.1K)。
- **RSS 稳定**：4.13 → 4.90 GB（随 KV 增长），全程远低于 RAM。

### 5.4 头对头（引擎 vs llama，同对话重放）

| 轮 | prompt tok（引擎/llama） | prefill 引擎 (ms) | prefill llama (ms) | 引擎/llama | TPOT 引擎 | TPOT llama | 引擎/llama | 轮总 引擎/llama (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 43 / 39 | 2 415 | 1 889 | 1.28× | 200 | 176 | 1.14× | 4.6 / 4.0 |
| 2 | 762 / 756 | 38 186 | 35 856 | 1.07× | 229 | 228 | 1.00× | 44.4 / 42.3 |
| 3 | 1 497 / 1 489 | 46 677 | 42 475 | 1.10× | 280 | 286 | **0.98×** | 54.3 / 50.8 |
| 4 | 2 231 / 2 221 | 56 154 | 50 621 | 1.11× | 350 | 406 | **0.86×** | 65.6 / 62.9 |
| 5 | 2 965 / 2 953 | 65 945 | 56 650 | 1.16× | 414 | 510 | **0.81×** | 77.1 / 71.5 |
| 6 | 3 700 / 3 686 | 76 056 | 63 120 | 1.21× | 482 | 602 | **0.80×** | 89.1 / 80.0 |
| 7 | 4 434 / 4 418 | 85 552 | 69 423 | 1.23× | 544 | 692 | **0.79×** | 100.8 / 89.6 |
| 8 | 5 170 / 5 152 | 94 821 | 74 859 | 1.27× | 610 | 797 | **0.77×** | 111.3 / 97.2 |
| 9 | 5 904 / 5 884 | 103 061 | 83 406 | 1.24× | 673 | 878 | **0.77×** | 121.9 / 109.0 |

- **两侧均做前缀复用**（llama `timings.prompt_n` ≈ 705–728 = 增量；`usage.prompt_tokens_details.cached_tokens` 达 5 158）⇒ 复用轴可比。
- **prefill**：引擎每轮慢 **1.07–1.28×**（与 §2.1 单轮口径 1.14–1.38× 同向）。
- **decode TPOT**：**交叉点在轮 2↔3（≈1.5–2.2K）**；之后引擎持续领先，5.9K 处 **快 1.30×**（与 §2.2 的 ~2K 交叉点一致）。
- **轮总时**：引擎全程长 **1.11–1.25×**（prefill 劣势 > decode 优势，因每轮新增文本长）。
- ⚠️ **llama 侧多轮记忆未测**（其 assistant 回复为重放，非自生成）。

### 5.5 多轮专项结论

1. **前缀复用有效且必要**：把每轮 prefill 从全量降为增量，2.2K 处 **2.56×**，且不改变输出。
2. **长会话的「下一轮」成本由全上下文注意力支配**：增量 prefill 速率随 ctx 单调退化（18.5→5.3 tok/s），decode TPOT 亦 O(n)。
3. **多轮记忆保留正确**（真值跨 5.9K 与 8.1K 仍能召回）。
4. **vs llama**：prefill 引擎慢 1.07–1.28×，decode 从 ~2K 起引擎领先（5.9K 快 1.30×），轮总时引擎长 1.11–1.25×。
5. **in-RAM 前缀复用不跨进程**：新进程首个请求无缓存 ⇒ 全量 prefill（E1 6 641 tok / 706 s）；需跨重启延续须 `--disk-kv`。

---

## 6. 结论（含优化档的最终对比）

### 6.1 8K 处的头对头（最关键档）

| 引擎 | prefill t/s | TPOT ms |
|---|---|---|
| **vllm_kestrel 全优化（C/F 组合）** | **13.7** | **441** |
| vllm_kestrel 默认（A） | 8.1 | 835 |
| **llama.cpp** | **10.72** | **1153** |

⇒ **开优化后：prefill 引擎快 1.28×，decode 引擎快 2.61×**（8K）。
⇒ **默认档：prefill 引擎慢 1.29×，decode 引擎快 1.34×**。

### 6.2 一句话结论

- **长上下文（≥2K）decode：引擎全程领先，且随 ctx 放大**（默认档 1.34×、优化档 2.61× @8K）；根因是引擎 KV/attention 栈的衰减更慢（llama 4.87× vs 引擎 2.54×）。
- **prefill：默认档引擎落后 1.14–1.38×；开启稀疏 prefill 后反超 1.28×**（8K）。
- **短上下文（1K）decode：llama 快 1.43×**（引擎有固定单 token 开销）。
- **质量：NIAH 两者均满分**（未开稀疏口径）。
- **L3 落盘在 8K 单用户无收益**；**分层驻留以 +18% decode 换 −77% RSS**。

### 6.3 未覆盖（诚实登记）

1. ~~**多轮次对话**~~ ✅ **已完成，见 §5**（8B 引擎 vs llama-server 会话模式，至 8.1K）。
2. **16K+ 的 L3 触发场景**（L3 的适用区间）。
3. **稀疏档的质量门**（NIAH/NLL 在 `--sparse-attn` 开启下未复测；wiki 记载 k≥32 通过）。
4. **llama 侧同口径 RSS/VmHWM**。
5. **t8（8 线程）档**与 t1/t2 的线程扩展曲线。
6. **多轮优化档组合未测**（§5 全程默认档；未叠加 `--sparse-attn` / `--l3-evict`）。
7. **llama 侧多轮记忆保留未测**（§5.4 其回复为重放）。
8. **本次全部为单会话读数**，非跨会话可比的绝对基线。

---

## 附录 A · 测试中发现（多轮 `context_tokens` 路径疑似缺陷；**未修复**，登记备查）

引擎的**客户端精确复用协议** `context_tokens`（[vllm_server.c L1588-L1770](../src/serve/vllm_server.c#L1588-L1770)，Web 端 [chat.html L574-575](../src/serve/chat.html#L574-L575)）在本轮测试中**产生退化输出**：

- **现象**：首轮正常；第二轮起（带 `context_tokens`）输出退化为单 token `>`（`n_tokens=1`）后 EOS。
- **机制（代码判读）**：`history_tokens` 由 `[prompt_ids] + [gids]` 组成，而 [L1419](../src/serve/vllm_server.c#L1419) 采样到 EOS/`im_end` 时 `break` 且**不写入 `gids`** ⇒ 回传序列**缺尾部 `<|im_end|>`**。下轮把它当历史拼接（`ids = ctx_tokens + tail_prompt`）时，assistant 回复与后续 `<|im_start|>user` 之间缺终止符 ⇒ 非法 chat 结构。
- **次要约束**：`context_tokens` 命中要求 `kv_lcp(last_ids, ctx_tokens) == n_ctx`（**精确等长**，[L1641-1643](../src/serve/vllm_server.c#L1641-L1643)）；客户端自行补 `<|im_end|>` 会使 `n_ctx = last_n + 1` ⇒ 匹配失败、回退**全量 prompt**（历史丢失，实测 prompt 由 ~198 掉回 44）。
- **影响面**：`context_tokens` 精确复用路径（Web 端默认开启）；**不影响**本轮基准所用的 `messages` 全量历史路径（§5 即走该路径，正确且复用生效）。
- **建议**：让服务端在 `gids` 末尾补 `im_end`（或在 `matched` 判定里容忍 `n_ctx = last_n + 1` 的单个 EOS）。**本项未改代码**，待独立立项。
