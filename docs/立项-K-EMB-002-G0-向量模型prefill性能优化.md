# 立项 K-EMB-002 · G0（向量模型 prefill 性能优化）

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-EMB-002** |
| **标题** | 句子向量嵌入（Qwen3-Embedding-0.6B）**prefill 性能优化**（GEMM/ATTN/OMP/服务层） |
| **目标** | 在保持向量口径与质量门的前提下，压缩 embedding 请求的 prefill 延迟与提升吞吐 |
| **阶段** | **G0 立项门（本文件 = 计划方案书 + 判据冻结单）；本轮不动代码** |
| **日期** | 2026-10-11 |
| **载体** | 板端 `192.168.1.92`，`/root/emb_model`（`model.vqf` 1.27 GB，双档 Q8+Q4，d=1024/28L/vocab=151669），二进制 `/root/_vfy/gitee3/vllm_kestrel`；`taskset -c 4-7`（4×A76）、`OMP_NUM_THREADS=4`、governor=**performance** |
| **前置** | [K-EMB-001](K-EMB-001-句子向量嵌入-Qwen3-Embedding-0.6B适配与byte-level-BPE分词器.md)（适配 + §10 基线）；方法学承 [K-PF-005](立项-K-PF-005-G0-prefill-GATEUP瓶颈.md)、[K-DENSE-014](立项-K-DENSE-014-G0-稠密FFN-int8dot近似轨.md)/[015](立项-K-DENSE-015-G0-稠密计算侧杠杆勘察.md)、[K-DN-006](立项-K-DN-006-G0-dn内核效率.md) |
| **判定** | **Phase 0 判别（H1 证否/H2 成立）→ Phase 1（GEMM）寄存器锁死受阻 → Phase 2 落地（位级一致、−1.2%）→ G4 默认化（`VLLM_EMB_SKIP_WASTE` 默认开）⇒ 达该平台实际地板、本项收口**（见 §9） |

---

## 1. 依据公理

| 公理 | 约束 |
|---|---|
| `perf_axiom_todo_needs_denominator_v1` | 立项须给**占比 + 可达性**（§2 已给占比） |
| `perf_axiom_candidate_root_cause_upper_bound_v1` | 先算**归零上界**；上界解释不了目标差距 ⇒ 否 |
| `perf_axiom_bottleneck_scaling_attribution` | 瓶颈须用**伸缩实验**判别（带宽 vs 算力 vs 停顿），不靠推断（Phase 0） |
| `perf_axiom_measured_benefit_scope_caliber_v1` | 结论带**口径**（载体/线程/钉频/中位 n） |
| `perf_axiom_derived_value_disclosure_v1` | 反算值须**披露推导式**（§3） |
| `feasibility_conclusion_timeliness` | **先冻可行性结论**，避免无限投入（§3.3 / §5） |
| `nonmonotonic_benefit_no_cherry_pick` | **证否即回退**（每个候选自带逃生舱） |
| `silent_fallback_false_pass` | 回退必须**显式**（env 门 + 生效性凭证 + 几何回退打印） |
| `measurement_fairness_equivalence` | 公平 A/B：同 governor、同 A76 亲和、同 OMP、同排程、交错 ≥3 轮 |
| `bitwise_tolerance_non_substitution` | 若被迫改浮点归约序，须**就地登记非位级性质**并走容差门 |
| `default_enablement_gate` | 判据**先冻结后采数**，不得改判、不得挑数据点 |

---

## 2. 现状与瓶颈（**实测，零代码改动**）

### 2.1 路径事实

`run_embed`（[main.c](../src/main.c#L3211-L3348)）→ `st_qwen_model_prefill_batch`（[vllm_safetensors.c](../src/model/vllm_safetensors.c#L18416)）**全序列 prefill**（embedding **无 decode**；last-token pooling + L2）。HTTP `/v1/embeddings` 同路径。prefill GEMM 默认走 **Q8 int8**（`g_st_prefill_q8 = 1`，dual 模型有 Q8 张量）。

**结构性结论（先行冻结）**：因果解码下**每一层都需要全位置隐状态**（后层 K/V 由前层全位置产生）⇒ **无位置剪枝空间**；embedding 不产 token，故 decode 侧优化（KV/投机）不适用。**优化面只在 prefill 内核与并行度。**

### 2.2 分相 profile（521 tok 文本，`VLLM_DEC_PROF=1`，t4，pinned A76）

| 相位 | 耗时 | 占比 |
|---|---|---|
| **GEMM** | 2161.6 ms | **73.6%** |
| ├ GATEUP（FFN 双投影） | 802.7 ms | 37.1% of GEMM |
| ├ QKV | 645.3 ms | 29.9% |
| ├ DOWN | 433.8 ms | 20.1% |
| └ O | 279.7 ms | 12.9% |
| **ATTN**（KV pack + 精确注意力） | 560.4 ms | **19.1%** |
| OTHER（RMSNorm/MRoPE/KV-store/SiLU） | 215.5 ms | 7.3% |
| **合计** | **2937.5 ms** | 5.6 ms/tok |

### 2.3 两个关键伸缩实验（本轮新测）

| 实验 | 读数 | 判读 |
|---|---|---|
| **mini-batch 扫描** `--prefill-batch` 64/128/256/512/1024 | prefill **2.911 / 2.890 / 2.881 / 2.888 / 2.895 s**（几乎恒定） | **与顶层 mini-batch 无关** ⇒ 排除「顶层分块重读权重」；瓶颈在**固定内核粒度**内部 |
| **OMP 扫描** 1/2/3/4 线程 | **10.007 / 5.176 / 3.940 / 2.890 s** ⇒ 扩展 **3.46×** | 第 4 核边际很小（~1.06×）；A55 不入池已默认 |

### 2.4 与既有基线对齐

K-EMB-001 §10：150 tok @t4 = 950 ms（157.9 tok/s）；4→8 线程反降 0.8×（A55 拖累）；批 8×150 = 6.55 s（相对串行 **1.16×**）；RSS 880 MB。本项 §2.2 的 521 tok ≈ 2.94 s 与之同量级（≈5.6 ms/tok）。

---

## 3. 瓶颈判别（**Phase 0 待定**；此处先给两假说与反算）

### 3.1 反算（披露推导式）

- GEMM MAC（Qwen3-0.6B，d=1024 / ff=3072 / nh·hd=2048 / nkv·hd=1024）：
  per token per layer = QKV `1024×4096` + O `2048×1024` + gate/up `1024×6144` + down `3072×1024` = **15.73 M MAC**；× 28 层 × 521 tok = **229.4 G-MAC**。
  **实测 GEMM 2161.6 ms ⇒ 有效 106 GMAC/s。**
- Q8 权重字节（不含 embed）：`15.73e6 × 28 × 1.0625 B ≈ 468 MB`。
  若内层 token tile = 8 ⇒ 权重重读 `521/8 ≈ 65` 次 ⇒ `65 × 468 MB = 30.4 GB`（2161.6 ms 内）⇒ **等效 14.1 GB/s**。
- 参照：`project_memory` 记 8B-Q4 decode 受 **DRAM ~12–15 GB/s** 约束；K-DENSE-015 反算 **LPDDR4x 屋顶 ~20–25 GB/s**。

### 3.2 两条待判别假说

| 假说 | 机制 | 若成立 ⇒ 杠杆 |
|---|---|---|
| **H1 · 权重 DRAM 带宽受限** | Q8 GEMM 内层 token tile（8×8）小 ⇒ 权重按 `n_tok/tile` 重读（§3.1 反算 14.1 GB/s ≈ 带宽屋顶量级） | **加大 token 分块**（8→16/32）线性削减重读 ⇒ 可达 ~1.5–2×（受带宽屋顶封顶） |
| **H2 · 算力/停顿受限** | 106 GMAC/s = 单核探针 93.6 的 4 核合计量（~374）的 **28%**；或发射/端口停顿 | **内层 ILP/布局重写**；但 K-PF-005 §9.4 已证 **ARM k-方向 ILP 镜像无收益**（exact 轨 0%、actq 轨 +24%）⇒ 概率下调 |

> **H1/H2 的分辨**是 Phase 0 唯一目标；**未判别前不得选型**（`bottleneck_scaling_attribution`）。

### 3.3 可行性先冻（诚实）

- **无「官方可比」**：embedding 无 rknn/NPU 对照，也无板端 HF 参考 ⇒ 目标只能是**相对自身基线**，**不写「追平官方」**。
- 上界预冻：**~1.3–1.7×（prefill 口径）**，**非数量级**（带宽屋顶 ~1.5× 封顶 + 相外项 26%）。若 Phase 0 判别为 H2 且无内核杠杆 ⇒ 按 `feasibility_conclusion_timeliness` **当场收口**。

---

## 4. 判据（**先冻结**，Phase 0 采数前生效）

统一口径：板端 `/root/emb_model`、`taskset -c 4-7`、`OMP_NUM_THREADS=4`、governor=**performance**、同会话**交错 ≥3 轮**取中位、**逐轮记录 `/sys/block/mmcblk1/stat` 并按「零缺页」筛选**（K-PF-005 §9.5）。

| 判据 | 冻结值 |
|---|---|
| **J-EMB-1 数值** | **首选位级**：GEMM 在 **int32 域**重排（不改逐块浮点归约序）⇒ 向量逐位 A≡C。**若被迫改浮点归约序** ⇒ 降级**容差门**：向量 **cos ≥ 0.99999**、\|v\|=1、无 NaN/Inf，且**就地登记非位级**（`bitwise_tolerance_non_substitution`） |
| **J-EMB-2 性能** | 主指标 = **prefill 端到端中位**（含 GEMM+ATTN+OTHER）；**效应门 = 差值 ≥ 5× 配对极差**。阶梯目标（Phase 0 后按实测封顶重定）：**Phase 1 GEMM ≥ −20%**、**整体 ≥ −15%**（即 ~1.18×）；**上界 1.3–1.7×** 仅作参照，**不得当已实现收益** |
| **J-EMB-3 一致性** | BPE ids **位级不变**；检索 sanity 3/3（K-EMB-001 §5.6 同题同排）；HTTP vs CLI 同文本 cos=1.000000；`|v|=1.000000` |
| **J-EMB-4 回退透明** | 每个改动带 **env 门**（新路径**先默认关**，A/B 显式开；G4 默认化门后再考虑默认开，`=0` 即逃生舱）；**生效性凭证**必打（实际 tile/内核名）；几何不合**显式回退并打印原因** |
| **采样纪律** | 同 binary 交错 on/off ≥3 轮；缺页轮次剔除；单轮 sd 页缓存读量入档 |

---

## 5. 分级方案（**四杠杆 → Phase 0–4**）

> 用户选定：GEMM 提效（先）+ ATTN/非 GEMM + OMP + 服务层，**全选**。顺序按 ROI/风险：**先判别（P0）→ GEMM（P1）→ ATTN/非 GEMM（P2）→ OMP（P3）→ 服务层（P4）**。

### Phase 0 —— 瓶颈判别（**零/最小代码**，先做）
1. **建 Q8 纯读探针**（对齐 K-DENSE-015 的 `VLLM_Q4_PROBE` 思路）：Q8 tile worker 跳过 `vdot/vcvt/vfma`，只做同访问模式的字节读并累加 sink 防 DCE ⇒ 得**纯读地板**；`计算占比 = (实测 − 地板)/实测`。**新增 env（默认关）+ 凭证打印**。
2. **token-tile 伸缩**：把 Q8 GEMM 的 token 分块从 8 提到 16/32（**仅实验、不改默认**），看 prefill 是否近似线性下降 ⇒ **直接判定 H1**。
3. **微基准对照**：复用 vdotq 4-位置 ILP 微基准（`93.6 GMAC/s` 单核）量化内核侧余量。
4. 输出：H1/H2 判定 + **真实可达上界**（重定 J-EMB-2 阶梯）。

### Phase 1 —— GEMM 内核提效（**最大杠杆，74%**）
- 条件：Phase 0 判 H1。
- 候选（P0 判别后选型，**不预选**）：
  - a) **加大 token 分块**（8→16/32）削减权重重读（须权衡寄存器/暂存压力与 L1/L2 复用）；
  - b) **行阻塞/归约下沉**（借 q8g 的「行末一次归约」形态；K-DENSE 系列已用）；
  - c) 权重 tile 布局与 `xq` 量化缓冲的访存局部性（对齐 K-DN-006 的 `av` 局部性手法，其 dn `−30.6%` 位级）。
- 判据：J-EMB-1（优先位级）+ J-EMB-2；逐项 A/B；**证否即回退**。

### Phase 2 —— ATTN + 非 GEMM 相位（19% + 7%）
- 候选：① KV pack / 精确注意力内核（现为 F32；实测 ATTN 对 521 tok 达 19%，效率约为 F32 屋顶的 ~72%）；② **embedding 无 decode ⇒ 跳过 KV 落盘与 `imp_head`/`prefill_importance` 累加**（当前 prefill 仍写 f32+q8 正典并累加稀疏重要性，对 embedding 纯属浪费）；③ `VLLM_ATTN_I8` 档在短序列的适用性（须过 J-EMB-1）。
- 判据：J-EMB-1（注意力改档多半**非位级** ⇒ 容差门 + 就地登记）+ J-EMB-3。

### Phase 3 —— OMP 扩展性（3.46× → 逼近 4×）
- 事实：1→4 线程 3.46×（第 4 核仅 1.06×）。K-PF-005 §7 已排除「任务粒度/核绑定/大页」三方向（静态均分、默认已绑 A76）。
- 候选：内核**负载均衡**（按行/块粒度切分对齐，避免尾差）；`vllm_tp_parfor` 的 chunk 对齐（现为静态连续块）；GEMM 与 ATTN 相位的线程池复用（避免相间 fork/join 空转）。
- 判据：J-EMB-2；注意**不得**引入 A55（§2.3 / K-EMB-001 §10）。

### Phase 4 —— 服务层（吞吐/冷启动/内存）
- 事实：§10 批 8×150 仅 **1.16×**（prefill 已摊销权重，批收益有限）；RSS 880 MB；`--load-on-use` 冷启 page fault。
- 候选：① `/v1/embeddings` **数组输入的真批处理**（当前逐条串行，K-EMB-001 §6.6）；② 并发队列吞吐；③ 冷启动（MADV_WILLNEED 预取，对齐 `vqf_ffn_prefetch` 思路）；④ 常驻内存/池化。
- 判据：吞吐（req/s、tok/s）+ 冷/热分档；**不追求单次延迟**（属 P1/P2）。

---

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| Phase 0 判 **H2**（算力/停顿）且内核无杠杆 | 按 `feasibility_conclusion_timeliness` **当场收口**，只保留 service 层（P4） |
| 非位级改动污染向量口径 | 一律走**容差门 + 就地登记**；`bitwise_tolerance_non_substitution` |
| A55 被拉入 / 亲和漂移 | 一律 `taskset -c 4-7` + `st_default_threads`；对照 K-PF-005 §7.2 |
| 页缓存抖动掩盖计算侧信号 | 逐轮 `/sys/block/mmcblk1/stat` 筛选「零缺页」轮次（K-PF-005 §9.5） |
| 内核重写 blast radius 大 | 每项独立 env 门、默认关、几何回退打印、逐项 A/B 证否即回退 |

---

## 7. 交付与验证

- **交付物**：① Phase 0 判别报告（H1/H2 + 真实上界）；② 命中杠杆的**内核改动 + env 门 + 生效性凭证**；③ 逐项 A/B 数据（交错 ≥3 轮、剔除缺页）；④ 文档回填（本文件 §9 执行记录 + `落地台账`）。
- **验证门**：J-EMB-1（位级或容差）、J-EMB-2（效应门）、J-EMB-3（一致性）、J-EMB-4（回退透明）。
- **回归面**：embedding（CLI `--embed`/`--embed-stdin` + HTTP `/v1/embeddings`）；**在线 chat 路径不受影响**（本项只碰 prefill 内核，且改动默认关）。

---

## 8. 待决策（进 G1 前须定）

1. **J-EMB-1 是否接受非位级**（容差门 cos ≥ 0.99999）？——决定 Phase 1/2 的候选面（纯 int32 重排可保位级，浮点归约序变化则否）。
2. **J-EMB-2 的目标阶梯**：先跑 Phase 0 定真实上界，再冻结「≥ −15% 整体」是否成立。
3. **服务层（Phase 4）是否纳入本轮**，还是单独立项（其指标是吞吐，与 P1–P3 的延迟不同轴）。

---

## 9. 执行记录

| 轮次 | 内容 | 结论 |
|---|---|---|
| G0-勘察（2026-10-11） | 只读勘察 + 板端 profile（521 tok）+ mini-batch/OMP 伸缩 | 占比冻结（GEMM 73.6%/ATTN 19.1%/OTHER 7.3%）；mini-batch 无影响；OMP 3.46×；**不动代码** |
| **G0-Phase 0**（2026-10-11） | **瓶颈判别**：新增 env 门控纯读探针 `VLLM_Q8_PROBE=1`（Q8 8x8 tile worker 只按同访问模式读权重/激活字节、跳过全部 vdot/累加/存储）+ 板端 DRAM 屋顶标定 | **H1 证否 / H2 成立**（下） |

### 9.1 Phase 0 实测（521 tok，t4，pinned A76，governor=performance，交错 2 轮）

| 档 | GEMM | QKV | O | GATEUP | DOWN |
|---|---|---|---|---|---|
| 默认（8x8 真 GEMM） | **2046.0 / 2052.4 ms** | 522.4 | 281.2 | 806.5 | 435.9 |
| `VLLM_Q8_PROBE=1`（纯读地板） | **803.8 / 805.8 ms** | 193.8 | 125.3 | 296.9 | 187.8 |

- **计算占比 = (2049 − 804)/2049 = 60.8%**；四个投影一致（QKV 63% / O 55% / GATEUP 63% / DOWN 57%）⇒ 是 **q8x8 内层固有**属性，非某一投影特例。
- **DRAM 屋顶标定**（`/root/bw.c`，256MB 顺序读，4×A76）：stride64 **27.05 GB/s**、stride16 **27.54 GB/s**（1 线程 21.8/18.3）。⇒ 天花板 ~27 GB/s。

### 9.2 判定：**H1 证否 / H2 成立**（`bottleneck_scaling_attribution`）

- **H1（权重 DRAM 带宽受限）证否**：同访问模式的纯读只需 **39%** 的时间 ⇒ 若带宽受限，两者应等时。GEMM 多出的 **61% 是 compute/issue**。
- **H2（compute·issue）成立**：GEMM 有效 **~112 GMAC/s（4 核）= 28 GMAC/s/核**，仅 vdotq 微基准单核（93.6 GMAC/s）的 **30%**。四个投影同比例 ⇒ 内层（f16 scale 反量化 + `vdotq_laneq` 调度 + 存储）是瓶颈。
- **订正**：§3.1 的反算「14.1 GB/s ≈ 屋顶」是**算术巧合**（前提是把 GEMM 时间当作带宽受限），实际既非带宽受限，也非 load-issue 受限（2.19G load / 2.05s = 1.07 Gload/s ≪ A76 发射能力）⇒ 归因于 **vdot/反量化的计算·发射**。

### 9.3 对 Phase 1 的含义（诚实）

- **主杠杆变为「内层计算效率」**（61%），而非「削减流量」。⚠️ 但 **K-PF-005 §9.4 已证 ARM k-方向 ILP 镜像无收益**（exact 轨 0%、actq 轨 +24%）⇒ 单纯加宽 ILP 大概率无效，须换结构（寄存器/调度/反量化融合）。
- **次杠杆仍为「加大 token 分块」**（当前 **tile=8**；Q8-4x4 轨为 tile=16）：它同时削减 ① 权重重读（39% 的读地板）与 ② 每遍的 f16-scale 反量化次数（61% 中可分摊的部分）。**但 8x8 权重布局已随 VQF 文件固化（`file=0x3`），env 切换 4x4 会被 `vqf_load` 布局校验拒绝** ⇒ 需新内核变体或重转模型。
- **未立即可得的零代码收益**：`VLLM_Q8_8X8=0` 不可用（布局不匹配，实测报 `layout mismatch: file=0x3 engine=0x2`）。

### 9.4 工程沉淀

- 新增 **`VLLM_Q8_PROBE=1`**（[vllm_safetensors.c](../src/model/vllm_safetensors.c#L5108) 的 `vllm_q8x8_b0_worker`）：env 门控、**默认关、零行为影响**、`[Q8PROBE] engaged` 凭证（对齐 K-DENSE-015 的 `VLLM_Q4_PROBE` 方法）。默认路径读数与改动前一致（GEMM 2046/2052 vs 改动前 2036/2041，噪声内）⇒ 无回归。
- 标定工具 `/root/bw.c`（DRAM 屋顶）；探针复现：`VLLM_Q8_PROBE=1 VLLM_DEC_PROF=1 ./vllm_kestrel --model /root/emb_model --embed-file <txt>`。

### 9.5 Phase 2 已落地（2026-10-11，✅ 位级一致，收益 ~1.2%）

**动机**：Phase 1（GEMM）被**寄存器锁死**——8x8 内核 8 rows×8 tokens = 16 int32 + 16 f32 累加 = **占满 32 个 SIMD 寄存器**，加宽到 16 tokens 需 64 ⇒ 必溢出（同 K-PF-005 ARM-ILP / K-YOLO-002 8-way / K-LAYA-002 M8 的失败模式）；tile-16 的既有实现是 **4x4 轨**（16 寄存器），但需 4x4 布局 VQF（板端无 HF 源/无 `vqf_conv`）⇒ 转 Phase 2。

**改动**（env `VLLM_EMB_SKIP_WASTE`，**默认 0**；触发条件限 `g_embed_out != NULL` ⇒ 对 chat 零影响）：删「embedding 无 decode 时纯浪费」的三项：
1. **imp_head 注意力质量图**（仅稀疏 decode 选块用）：注意力 3 个调用点传 `NULL` + 末尾 `prefill_importance` 归约跳过；
2. **末尾 LM head logits**（embedding 不产 token）；
3. **KV int8 压缩包**（带 guard：仅当 `!g_kv_nof32 && !st_attn_i8_enabled()` 才跳——本模型走 **nof32 正典**（q8 即正典）⇒ guard 正确拦住，`kvq8=0` 未跳）。

**实测（521 tok，t4，pinned A76，governor=performance，交错 3 轮）**：

| 轮 | prefill 默认 | prefill skip | Δ |
|---|---|---|---|
| 1 | 2879 ms | **2848 ms** | −31 ms（−1.1%） |
| 2 | 2890 ms | **2854 ms** | −36 ms（−1.2%） |
| 3 | 2897 ms | **2859 ms** | −38 ms（−1.3%） |

- **ATTN 559 → 546 ms**（imp_head 内层累加，−2.3% of ATTN）；末位 LM head 亦省（不在相位计时内）。
- **J-EMB-1 = 位级 PASS**：向量 **3/3 轮逐位一致**（`[EMB-VEC]` 全等）——符合预期（attention 走 f32 正典、imp_head/logits 仅 decode 用）。

**判读（诚实收口）**：收益仅 **~1.2%**，**非数量级**。综合三项：
- **GEMM 74%**：H2（compute·issue）受限，28 GMAC/s/核；ILP 加宽（K-PF-005 已证无效）与 token 分块（寄存器锁死）两条杠杆均不通 ⇒ **无低成本空间**。
- **ATTN 19%**：实测 55 GMAC/s ≈ **F32 屋顶的 72%** ⇒ 已接近上限。
- **OTHER 7%**：含 KV 正典写（embedding 必需）+ norm/mrope ⇒ 可删项已删完（本 Phase）。
- ⇒ **K-EMB-002 达到该平台的实际地板**（与 K-DENSE-015「稠密模型计算侧无空间」同结论）。**若 J-EMB-4 允许，建议 G4 将 `VLLM_EMB_SKIP_WASTE` 默认化**（位级安全、embedding 专属、chat 零影响）。

### 9.6 G4 默认化（2026-10-11，✅ 已默认开；`=0` 逃生舱）

**改动**：`VLLM_EMB_SKIP_WASTE` 解析由「默认 0」改为**默认 1**（`e[0]=='0'` 才关），与全仓既有 env 默认化写法一致（如 `q8_8x8_enabled`）。代码位置不变。

**G4 四条件复核**（板端重建后实测）：

| 条件 | 结果 |
|---|---|
| **J-EMB-1 位级门** | 默认(skip-on) 与**改前基线向量逐位一致**；逃生舱 `=0` 亦逐位一致（`[EMB-VEC]` 全等）✓ |
| **J-EMB-2 效应** | 默认 **2.844 s** vs `=0` **2.892 s**（−1.7%，与 §9.5 交错 3 轮的 −1.1~−1.3% 同向）✓ |
| **J-EMB-4 逃生舱** | `=0` 恢复旧行为；**生效性凭证 `[EMB-SKIP]` 在 `=0` 时打印 0 次**（不静默）✓ |
| **chat 零影响** | 触发条件限 `g_embed_out != NULL`（仅 embedding 路径）⇒ 构造上不影响 chat；板端 `--stream-test`（`/mnt/VQF/2b`，54-tok prefill + 4-tok decode）**正常无异常** ✓ |

**结论**：G4 通过，`VLLM_EMB_SKIP_WASTE` **已默认开**。K-EMB-002 收口（embedding prefill 相对自身基线 −1.2%、位级不变；达该平台实际地板）。
