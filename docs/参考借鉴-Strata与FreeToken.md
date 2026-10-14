# 参考借鉴：Strata 与 FreeToken → vllm_kestrel

- **日期**：2026-10-08
- **来源**：`参考/Strata-main`（分层驻留 MoE 引擎，C++/CUDA/HIP/SYCL）、`参考/FreeToken-main`（Python/CUDA 引擎）
- **我方基线**：纯 CPU + NEON，RK3588（4×A76 + 4×A55）/ 15.6 GB RAM / SD 62–65 MB/s + eMMC 313 MB/s，**无 GPU**
- **用途**：为 K-PF-005（prefill GATEUP）、K-L0-003（热度驻留）、K-L0-001（权重驻留）提供外部参照

---

## 0. 一句话结论

**两个项目与我方最相关的都不是"算法新招"，而是三条工程纪律 + 三个具体手法**：
① 并行任务粒度与核绑定；② 热度/驻留的**度量口径**；③ 大页与 I/O 队列深度。
其余大量机制（CUDA VMM、pinned、PCIe 批量 DMA、Triton 压缩）**均依赖 GPU，不可借**。

---

## 1. 交叉验证：我方**已经做对**的部分（不构成新借鉴）

| 事项 | 我方实测 | Strata / FreeToken 侧 | 判读 |
|---|---|---|---|
| 跨层专家无共性 | 某层 top-16 在其它层平均排名百分位 **0.566 ≈ 随机** | Strata `h_layer` held-out = **0.0456**（仅 4.6% 的 (layer,token) 能让某层十个专家全驻留）⇒ 明确「不能做逐层分组的 kernel」 | **两边独立得同一结论** ⇒ 必须逐层独立掩码，不做跨层共享 |
| 全局共享热集会失效 | —— | Strata 实测：一个全局 `next_free_` 被 48 层共享 ⇒ 命中率仅 **2.97%**；改「每层 8 槽」→ **21.4%**，每层 64 槽 → **70.4%** | 与我方「逐层掩码」结论一致；**"每层配额"是必要设计** |
| KV 侧收益有限 | K-KV-002 三条件里两条不满足 | Strata `RESEARCH_RUNS.md` R3：**"K/V block cache 每 GiB 毫无价值，只有专家能帮 decode"**（cells 4096→65536 只 34.05→34.92 tok/s，而专家槽 2000→5000 让 decode 27.5→34.2） | 两边同向：**优先专家，不是 KV** |
| 专家聚簇分组 GEMM | ARM 批式已是 expert-outer + 归属表（`ex_t/ex_j`） | Strata `moe_fused.cu` + `prefill.cpp`（MMQ_GROUP=16 按专家聚簇，权重只读一次服务多行） | **我方已具备同一思想**（ARM 侧自研版）；差异在实现平台 |
| k 方向 ILP 是 gu 的杠杆 | x86 侧已做（`VLLM_MOE_ILV`），**ARM 未镜像** | FreeToken `expert.cpp`：512-bit 行 dot 一次处理 8 个 64-权重块、scale 预乘成向量、**整行只 reduce 一次**；并注明「256-bit 行 dot 因 compute-bound 只有 ~39 GB/s，机器能流 52」 | **同构优化**，可作为 ARM 实现时的写法参照（K-PF-005 L1） |

---

## 2. 高价值可借鉴（按对我方瓶颈的直接程度排序）

### ★T1-a 并行任务**粒度**——最可能解释"利用率仅 2.6%"

- **Strata 的做法**：CPU pool 的任务划分明确按「**每个参与线程约 3 个任务**」设计（`pool.cpp` `run_split`/`run_split_multi`，`parts_a_ ≈ 3 tasks/thread`），并提供 `--pool-tasks N` 可调（1..4096，被行数封顶）。
- **FreeToken 的做法**：行 store 的实测结论「**pread 在 ~16 线程饱和，再多只增延迟**」。
- **我方现状**：ARM 批式 gu 的任务数 = `ne × ge4 = 128 × 192 = 24,576`，t4 下 **6144 个任务/线程** ⇒ 任务过细，调度与伪共享开销可能吃掉大量时间；这与"带宽利用率 2.6%、算力 ~5–8%、t4 就饱和"**高度吻合**。
- **建议**：把任务按 **m_e（每专家实际 token 数）聚簇后再切分**，并把每线程任务数压到个位数~十几量级做 A/B。
  - 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/src/kernels/cpu/pool.cpp#L700-L757`、`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/kernels/cpu/pool.hpp#L174-L189`、`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/kernel/csrc/row_store/row_store_ext.cpp#L50-L52`

### ★T1-b 按**每专家行数**排序以均衡/减少空转（Strata `resident_sort`）

- **做法**：`mmq_resident_sort.hpp` — 纯 host、可选、不改精度：按**每个专家的行数（count）升序** `stable_sort` 专家顺序，重建 `off/slot/src`，使**每组的 max_rows 贴近该组自身行数**，减少 padding 与空转；遍历顺序与组内行序不变。
- **对我方**：我方 task = (行组 g, 专家 e)，行组主序 ⇒ 分到 token 少的专家在每个行组上仍跑满量 ⇒ 空转。**按 m_e 聚簇/排序**可把稀疏专家压到一起。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/src/prefill/mmq_resident_sort.hpp#L1-L57`、接入点 `file:///d:/项目/NewVLLM/参考/Strata-main/src/prefill/prefill.cpp#L2820-L2827`

### ★T1-c 大页（TLB）——我方**尚未做**的具体优化

- **Strata 的量化**：`pinned.hpp` 明写「**33.97 GB 用 4 KB 页 = 830 万 TLB 项，任何 TLB 都放不下，每个专家的每个 matvec 都吃 TLB miss**」，故 host 专家 arena 强制**页锁定 + 尽量 2 MB 大页**。
- **对我方**：16.85 GB 单文件 mmap，逐层访问跨 11 个区间（我方实测 locality 0.030）⇒ 同样面临 TLB 压力。**可试 `madvise(MADV_HUGEPAGE)`**（或 hugetlbfs / 透明大页调参）并 A/B `VQF-STREAM` 与全驻留两档。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/core/pinned.hpp#L1-L14`

### ★T2-a 只钉**性能核**（对应我方 A76/A55）

- **Strata**：`detect_cpu_topology` 识别 P-core/E-core，`PoolAffinity::PCores` **严格限制在性能核**；`Auto` 优先 P-core→SMT→E-core；并明确「**efficiency cores 多的 CPU 上，更少线程可能更快**」（`--pool-workers` 校准项）。
- **FreeToken**：`physical_core_cpus()` 「**每物理核一线程并 pin**」，理由是「MoE decode 是带宽受限，SMT 兄弟只争同一核的 load 端口、不加带宽」；且「**spin-barrier 在超订时严重退化**」。
- **对我方**：实测 t4→t8 仅 **1.07×**，与「A55 拖后腿」一致。建议把 A/B 从"线程数"改成"**钉在哪几个核**"（A76×4 vs A76×4+A55×4 vs A55×4）。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/kernels/cpu/pool.hpp#L66-L104`、`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/moe/cpu_executor.py#L92-L141`

### ★T2-b 避免 host 自旋抢 worker 的核（Strata 实测 **1.35× 损失**）

- **实测**：pool 在 5 workers 上 36.32 GB/s（= 5/6 × 44.14），但在 host loop 里只有 **26.9 GB/s** —— 未绑定的 host 自旋抢了 worker 的核/其 SMT 兄弟，**1.35× 差距不是 kernel 问题**；修法是让 host 线程也参与 drain（`host_works`），并把自旋的 LOCKED RMW 计数删掉（自我制造的争用）。
- **对我方**：若有一个线程做调度/搬运而不参与计算，值得排查同样的抢核问题。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/kernels/cpu/pool.hpp#L106-L149`、`#L196-L274`

### T3-a WILLNEED **步进造深队列**（优于逐页 fault）

- **Strata 实测**（Gen3 NVMe）：权重/驻留副本用 `madvise/posix_fadvise(WILLNEED)` **128 KiB 步进**提前请求，使盘看到**深队列**而非单页 fault ⇒ ready 从 ~920 s → **70 s**、fill 从 **39 MB/s → 3.2 GB/s**。
- **对我方**：已有 `MADV_WILLNEED`，但**是否按固定步长分批下发**未验证。冷档被 SD 62–65 MB/s 顶格时，队列深度是关键。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/docs/DETAILS.md#L280-L284`

### T3-b `O_DIRECT` 绕过页缓存（我方「文件 > RAM」场景）

- **Strata**：`platform/direct_file.hpp`（Linux `O_DIRECT` preads + 小线程池，offset/length/buffer 须 4096 对齐）；专家侧还有 `set_unbuffered` + 「页缓存能不能装下」的**计时探针**判别（随机 64 KiB 读：缓存 ~10 µs、盘 ~80 µs）。
- **FreeToken**：`drop_page_cache` 用 `fadvise(DONTNEED)` 丢页缓存，避免 banks 与 checkpoint **双重缓存导致 OOM**。
- **对我方**：16.85 GB > 15.6 GB RAM ⇒ 读进来"用完即弃"的数据白占页缓存。可试 O_DIRECT + 对齐读，并用同样的**微探针**（64 KiB 随机读耗时）判定当前是否命中页缓存。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/platform/direct_file.hpp#L1-L9`、`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/core/expert_source.hpp#L559-L570`、`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/models/loader.py#L74-L83`

### T3-c router **lookahead 预热**（预测下一层要用的专家，只 warm 不改结果）

- **Strata**：CPU pool 开始第 l 层时，一个 worker 把**第 l+1 层的 router**（BF16 host 副本）作用在第 l 层的 MoE 输入上（残差逐层变化小），取 top-k，剔除已驻留的，只对源页面发 `WILLNEED`/`PrefetchVirtualMemory`。**"预测只预热页面，从不改变算哪些专家或怎么算"**（正确性安全），深度 `STRATA_IO_PREFETCH_DEPTH` 默认 1（可到 4）。
- **对我方**：已有 `VLLM_EW_PREFETCH_LAYER` 的跨层预读，但**用 router 预测**比"整层预读"精确得多，且可与 `VLLM_EW_HEAT` 掩码互补。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/core/expert_source.hpp#L203-L244`、`#L594-L635`

### T4-a MoE 优先 + KV 保底的**预算二分**（纯算术，可平移）

- **FreeToken `cache_budget.py`**：`net = memory_ratio×baseline_free − weights − fixed`，`(1−memory_ratio)` 留作头寸；`plan_cache_budget` **MoE 优先**——先预留 `kv_reserve_pages`，专家贪婪吃满剩余并钳在 `[floor, min(total, max_slots)]`；`dsv4_solve_num_pages` 用**二分**求"字节精确"塞进预算的最大页数。
- **对我方**：可直接搬到「RK3588 15.6 GB 上 16.85 GB 单文件 + KV」的联合规划（把"GPU 字节"换成"宿主 RAM 字节"）。
- 出处：`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/engine/cache_budget.py#L31-L95`、`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/kvcache/dsv4_cost_model.py#L231-L267`

### T4-b 现成的**热度指标口径**（可直接与我们的 top-H 覆盖对照）

- **FreeToken `decode_routing_stats()`** 直接给出：`working_set_mean/max`、**`experts_for_90pct`**、`oracle_hit_at_slots`（按 `cache_size/num_layers` 槽位的最优命中率**上界**）、`norm_entropy`。
- **对我方**：我们的分析只报了 top-H 覆盖；加 `norm_entropy`（分布"均匀度"标量）+ `oracle_hit_at_slots`（**上界**，避免把实现差距当成算法没空间）会让 K-L0-003 的结论更可比。
- 出处：`file:///d:/项目/NewVLLM/参考/FreeToken-main/python/freetoken/moe/offload_cache.py#L979-L1009`

### T4-c 方法论纪律：**"先不淘汰"以保度量**（很有价值）

- **Strata `ExpertCache` 显式不做淘汰**：`admit()` 从不驱逐，理由是「**淘汰策略是要用实测回答的问题（LFU-decay vs LRU sweep），用占位策略会把下游全部尺寸化的命中率钉死**」。
- **对我方**：正在推进的 EW/自适应热度**同理**——在拿到"命中率-容量曲线"之前，不要先塞一个淘汰策略进去，否则后续所有尺寸化实验都被它污染。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/core/expert_cache.hpp#L19-L22`、`#L142-L145`

### T4-d 别把**排名**当**频率曲线**（Strata 自己踩过的坑）

- Strata：coverage curve 注释写「top 5,805 pairs 声称占 94.9% 的 mass，但实测 **69.2%**；**没有哪个指数能同时拟合两台机器，因为这条曲线是模型路由的属性、不是通用常数**，而 profile 文件只带排名、不带计数」。且 `h_expert` **held-out 0.6447** vs README 公布的 **0.6573（in-sample）**。
- **对我方**：K-L0-003 的 top-H 覆盖率是**单机单 prompt 单次**，且我已知"计数绝对值有 nb 倍放大"——**引用时必须带"in-sample / 单次"口径**，不得外推到其它模型/语料。
- 出处：`file:///d:/项目/NewVLLM/参考/Strata-main/src/program/generate.cpp#L3245-L3268`、`file:///d:/项目/NewVLLM/参考/Strata-main/include/strata/core/expert_cache.hpp#L11-L17`

---

## 3. 明确**不可借鉴**（依赖 GPU / 显存 / PCIe）

| 机制 | 出处 | 原因 |
|---|---|---|
| VRAM 层、expert cache device arena、`vmm.hpp`、`on_device.hpp`、`VRAM_ELASTIC` | Strata `core/*`、`docs/VRAM_ELASTIC.md` | 需 CUDA VMM（HIP 都拒绝） |
| `pinned.hpp` 的 `cudaHostRegister`、PCIe 直取、`peer_experts` | Strata `core/pinned.hpp` | 目的给 copy engine DMA |
| `dma_batch.hpp` / `cudaMemcpyBatchAsync`（提交延迟 −82.7% 但**总量仅 −1.3%**、**未确立 tok/s 提升**） | Strata `docs/BATCHED_DMA.md` | CUDA 13+ 专用；且作者自述收益未确立 |
| MMQ / `mma.sync` / `cp.async` / WMMA(RDNA3) / cuBLAS / hipBLASLt | Strata `prefill/*.cu` | GPU 张量核 |
| Triton 压缩/稀疏 kernel（gated_pool、kpool_compress） | FreeToken `kernel/triton/**` | 需 Triton；但**算法（online-softmax 门控池化）纯 CPU 可实现** |
| `fast_index_copy` / stream memop / io_uring→GPU 前端同步 | FreeToken `kernel/*` | 无 GPU/DMA 引擎 |
| DSV4 的滑窗+Indexer 稀疏架构 | FreeToken `attention/dsv4_*` | **模型架构自带**（DeepSeek-V4），与我方 Qwen3-30B-A3B 无关 |

---

## 4. 建议的落地顺序（与既有立项对齐）

| 序 | 动作 | 挂到 | 预期 |
|---|---|---|---|
| 1 | 查清 ARM gu 的**任务粒度**（24,576 task / 4 线程）+ 按 m_e 聚簇后 A/B | **K-PF-005** | 最可能解释 2.6% 利用率；先做**零风险探针**（只改任务切分，不改算法） |
| 2 | **核绑定 A/B**：A76×4 vs A76×4+A55×4 vs A55×4 | K-PF-005 | 现成 t4/t8 数据的细化，成本极低 |
| 3 | **大页** `MADV_HUGEPAGE` A/B（逐层档 vs 全驻留） | K-L0-001 / K-PF-005 | 我方尚未做过；Strata 有量化依据 |
| 4 | 用 `norm_entropy` + `oracle_hit_at_slots` 重述 K-L0-003 热度结论 | K-L0-003 | 让结论可比、并给出**上界** |
| 5 | WILLNEED **步进队列** + O_DIRECT（冷档） | K-PF-005 / K-L0-001 | 只在冷档有意义，先量"页缓存命中探针" |
| 6 | MoE-first 预算二分 | 新（联合内存规划） | 纯算术，可与 KV 侧合并考虑 |

> 纪律提醒：以上 1–3 都属 **K-PF-005 的 G0 追加探针**（零/低代码改动），应在**进 G1 冻结判据之前**做完——否则可能把"任务粒度问题"误当成"ILP 问题"去改内核。
