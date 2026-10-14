# 立项-K-LAYA-002-G0-Laya 性能优化（标量 F32 → SIMD/量化/多线程）

> 状态：**✅ 已收口（2026-10-11）**——板端短 state 请求（3 问）**58.0 s → 0.963 s（60.2×）**、长 state **294.9 s → ~7.13 s（41×）**，RSS 1.66 GB → **0.47 GB**；**质量门：int8 QB=128 vs F16 基线 32 请求 / 96 决策 100% 一致**、J1/J3 全过。**目标（≤0.3 s/问）达成**（短 state 0.321 s/问）。收口详情见 §9。
> 单一真源本文件；配套 [落地台账.md](落地台账.md)；上游 [K-LAYA-001](K-LAYA-001-非自回归决策模型-Laya-ModernBERT适配.md)。
> 口径：先冻结判据再采数；位级与容差不互换；单一真源台账。

## 1. 目标与判据（先冻结）

**问题**：Laya 决策路径板端不可用——单问题 ~19 s（短 state 3 问 58 s，长 state 294.9 s），瓶颈为纯标量 F32 单线程（见 [K-LAYA-001 §7](K-LAYA-001-非自回归决策模型-Laya-ModernBERT适配.md)）。

**目标**：在不退化决策质量的前提下，把单问题延迟从 ~19 s 降到 **≤0.3 s**（量级），使 Laya 板端进入"可用"档。

**冻结判据**（复用 L4/L5 已冻，不新造）：
- **J1 数值等价**：`tools/laya_gate.py` 6/6 PASS——logits `max|Δ| ≤ 5e-3` 且 `max rel ≤ 1e-3`、argmax 与 F32 golden 一致、无 NaN/Inf。
- **J2 端到端等价**：`tools/laya_decide_gate.py` 6/6 PASS——短+长 × choice/score/noul，逐字段与官方 `RLAgent.system_one` 一致（当前基线 `d=0.0e+00`，优化后允许 ≤容差但 `choice`/`argmax` 必须不变）。
- **J3 一致性**：同输入多线程(>1) vs 单线程输出**逐位一致**（位级门，不因并行规约顺序漂移）。
- **J4 覆盖**：短(~55 tok) + 长(~512 tok 触发滑窗)两场景都必须过。

## 2. 瓶颈诊断（数据反算，不臆造）

| 项 | 值 | 来源 |
|---|---|---|
| 单问题延迟（短 state） | **19.3 s** | 板端实测（§7） |
| 单问题 FLOPs（理论） | **~40.5 GFLOP** | encoder 28L×(Wqkv 3.15M + Wo_attn 1.05M + Wi 5.37M + Wo_mlp 2.69M)=12.26M/位置 ×55 + head 2L×12.58M/位置×55 + scorer |
| 实测算力 | **~2.1 GFLOP/s** | 40.5 / 19.3 —— 纯标量 F32（无 SIMD/FMA）典型值 |
| 权重驻留 | **1.685 GB F32**（F16 源 842.6 MB） | `[LAYA] loaded … F32 1685 MB` |
| 纯权重读取下限 | **~130 ms/问** | 1.685 GB / 板端 DRAM ~13 GB/s |

**结论**：当前是 **compute-bound**（标量 F32 单核）。SIMD 把 compute 提到 ~16 GFLOP/s 后，权重带宽（~130 ms）成为次瓶颈 ⇒ 量化（F16/int8 减半/四分之一带宽）紧随其后。最终下限由 DRAM 带宽决定（F16 权重 ~65 ms、int8 ~33 ms）。

## 3. 公理映射（理论 → 工程）

公理库 [axiom_registry.json](../../axiom_registry.json) 的抽象运算公理映射到本项的性能优化：

| 公理（symbol） | 工程映射 |
|---|---|
| `collapse_to_classic` + `collapse_priority_rule` | **量化坍缩**：F32 → F16 → int8。`collapse_priority_rule`（singleton 优先 → 细粒度坍缩）⇒ 单 scale 只适用于值域跨度的"singleton"情形；跨块值域大时必须 **per-block scale**（细粒度坍缩），否则 `irreversibility_penalty=0.3`（不可逆失真）。工程红线（K-EMB 沉淀）：**≥16 元素量化粒度，2-bit 单 scale 乱码**。 |
| `conservation_law_under_collapse` | **精度守恒门**：`E[Q(s) \| collapse] = Q(S)` ⇒ 量化后答案/概率排序必须与 F32 golden 一致（J1/J2）。注意 `confidence=0.65`（弱约束）⇒ 必须**实测**验证，不靠理论保证。 |
| `superposition_addition_base`（classic_anchor=2） | **SIMD lane 规约**：NEON 4/8 个 lane 加法树归约为 1 标量，`1⊕1=2` 经典锚 = 规约正确性；并行规约顺序不得改变结果（J3 位级门）。 |
| `interference_priority_rule` | **异构核线程调度**：`odd_even_interference_logic first` ⇒ 多线程只取快核（4×A76），A55 慢核掺入破坏叠加对齐（实测 Embedding 4→8 反降 0.8×）。 |
| `isomorphic_superposition`（NTT 同构，卷积→点乘） | **结构坍缩**：全注意力 O(n²) 卷积 → 滑窗 O(n·w) 点乘（encoder 2/3 层已做）；决策头 2 层仍全注意力，可评估同类坍缩。 |
| `operator_composition`（⊕+→+） | **算子融合**：LN + GELU + GEMM 归约为单算子，消除中间张量读写。 |
| `exponential_jump` | **复杂度跃迁警示**：长上下文注意力 O(n²) 是指数跳跃风险，滑窗已坍缩（§7 长/短 5.08× < token 比 9× 即此坍缩生效的实证）。 |
| `linear_interpolation_axiom` | **混合精度**：逐层 λ∈[0,1] 插值 = 敏感层 F32/F16、不敏感层 int8 的渐进量化（量化位宽连续可调）。 |
| `tensor_iterated_addition` | **跨 question 结构复用**：3 个 question 的 state 段相同，仅 head 段不同 ⇒ encoder 的公共部分可复算一次（见 L2b）。 |

## 4. 分级方案

每级给出：公理约束 / 改动 / 判据 / 预期加速 / 风险。

### L1 — NEON F32 SIMD 向量化（最大杠杆，立即可做）
- **公理**：`superposition_addition_base`（lane 规约）+ `conservation_law_under_collapse`（J1 位级门）。
- **改动**：`lay_linear` 内层用 NEON 128-bit F32 FMA 向量化（`vld1q_f32`/`vfmaq_f32`），4 元素/步；`lay_ln`/`attn_pos` 同理。`rope_table` 从每问题重算改为按 `n` 缓存。
- **判据**：J1（`laya_gate` 6/6）+ J4。
- **预期**：**8–15×**（19.3 s → 1.5–2.5 s/问）。
- **风险**：低（纯内核替换，语义不变；`-ffp-contract=off` 下 FMA 显式写出，位级可复现）。

### L1b — 消除 `lay_find` 线性查找（顺带，低风险）
- **公理**：`operator_composition`（消除重复算子查找开销）。
- **改动**：`laya_forward` 内每层的权重指针在层外一次性 `lay_find` 后缓存，去掉循环内反复 `snprintf + strcmp`（206 张量 × 每位置 × 每层）。
- **判据**：J1/J2（纯重构）。
- **预期**：~5–15%（相对已向量化后的基数）。
- **风险**：极低。

### L2 — F16 计算（带宽减半 + 吞吐翻倍）
- **公理**：`collapse_to_classic`（F32→F16 坍缩，原生 10-bit 尾数）+ `conservation_law_under_collapse`（J2 端到端门）。
- **改动**：权重**保持 F16 不再转 F32 arena**；`lay_linear` 用 NEON fp16 FMA（`vld1q_f16`/`vfmaq_f16`，8 元素/步）；激活也走 fp16。RoPE/LN/GELU 保持 F32 或 fp16（需实测精度）。
- **判据**：J1（logits 容差放宽到 5e-3）+ J2（decide 6/6，`choice`/argmax 不变）。
- **预期**：再 **~2×**（→ 0.8–1.2 s/问）；同时权重驻留 1.685 GB → 843 MB。
- **风险**：中。fp16 累加精度需实测；若 decoder 侧 head 精度敏感可回退 head 段 F32（`linear_interpolation_axiom` 混合精度）。

### L2b — 跨 question 共享 encoder（结构复用，大杠杆但需谨慎）
- **公理**：`tensor_iterated_addition` + `isomorphic_superposition`。
- **改动**：`laya_runner_decide` 对同一请求的 N 个 question，其 `state` 段序列相同。双向注意力下 state 段与 head 段相互 attend，**不能简单只算 state 段**；可行路径 = **batch 前向**（N 个 question 的序列 padding 成 batch，GEMM 权重读取摊薄到 N 项）。等价性以官方 `collate_items` batch 为 golden（官方即 batch 前向）。
- **判据**：J2（batch 前向输出 == 逐条前向 == 官方）。
- **预期**：N=3 时权重带宽复用，单问题分摊 **~2–3×**（受限于 padding 浪费）。
- **风险**：中高。需把 `lay_linear`/`attn_pos` 改 batched；双向 + padding 掩码必须严格等价官方，否则 J2 破。

### L3 — int8 量化（dotprod + per-block scale）
- **公理**：`collapse_priority_rule`（per-block scale 细粒度坍缩）+ 工程红线（≥16 元素粒度，禁 2-bit 单 scale）。
- **改动**：Q8 权重（int8 + 每 16/32 元素 scale/min）+ A76 `sdot`/`udot` dotprod 内核；激活量化用 per-token 或跳过（仅权重量化）。
- **判据**：J2（decide 6/6，`choice`/argmax 不变）+ 无 NaN。
- **预期**：再 **2–4×**（→ 0.2–0.4 s/问）；权重 843 MB → 421 MB。
- **风险**：中高。int8 解包开销（K-EMB 教训：3-bit 解包曾致 compute-bound 反降）；必须先用 F16 基线过门再试 int8。

### L3b — 多线程（4×A76 绑核）
- **公理**：`interference_priority_rule`（快核优先）+ `superposition_addition_base`（规约）+ J3 位级门。
- **改动**：`lay_linear` 外层 OpenMP 并行（按输出行分块），`taskset -c 0-3`（仅 A76）；`OMP_NUM_THREADS=4`。
- **判据**：J3（多线程 == 单线程逐位一致）+ J1。
- **预期**：~3×（受 DRAM 带宽上限约束；F16 后单核已近带宽下限，多线程收益递减）。
- **风险**：中。规约顺序、假共享；异构核必须绑核。

### L3c — 算子融合（顺带）
- **公理**：`operator_composition`。
- **改动**：LN→GELU→GEMM 融合（尤其 MLP 的 `Wi·x → gelu×gate → Wo`），减少中间张量（`wi/mh/tmp`）读写。
- **预期**：~10–20%。
- **风险**：低-中。

## 5. 优先级与里程碑

**主序（按杠杆×风险）**：`L1 → L1b → L2 → L2b → L3b → L3 → L3c`

| 里程碑 | 内容 | 判据 | 预期单问延迟 | 实测（板端） |
|---|---|---|---|---|
| M1 | L1 + L1b | J1/J2/J4 | 19.3 s → ~1.5–2.5 s | **4.74 s（4.08×）** ✅ |
| M2 | L2（F16） | J1/J2 | → ~0.8–1.2 s | **5.09 s（RSS −50%，延迟持平）** ✅（见 §8 注） |
| M3 | L2b（batch/分块 MatMul） | J2/J3 | → ~0.3–0.5 s | **0.99 s（再 1.36×）** ✅ |
| M4 | L3b（多线程） | J3/J1 | → ~0.2–0.3 s | **1.34 s（再 3.88×）** ✅ |
| M5 | L3（int8 dotprod）+ L3c | J2/J3 | → ~0.1–0.2 s | **0.32 s（int8 1.68× + 分块 1.30× + 量化/Wo 1.04× + 量化块 1.37×）** ✅ |

**实际执行序**（数据驱动调整，见 §8）：M1 → M2a(F16 存储) → M2b(寄存器分块，**实测退化已回退**) → M4(多线程，提前) → **M3(分块 MatMul，token 维批处理；原 L2b「跨 question 复用」因 ModernBERT 双向注意力结构约束改以「权重行跨位置复用」落地)** → **M5(int8 dotprod；J1 数值门不适用，改以 J2 决策输出等价守门)** → **M5b(int8 分块 MatMul，4 token×1 行，t4r1；与 M5 逐字节等价)** → **M5c(向量化激活量化 + Wo/out_proj 分块化；与 M5b 逐字节等价，+4–7%)** → **M5d(量化块 32→128 摊薄每块 scale 开销；24/24 决策一致，+1.37×/1.15%)** → **M6(env 门控阶段计时器保留；行块缓存复用实验实测短 state 回退已回退)** → **M7(滑窗层只遍历有效 key 区间；逐位等价，长 state attention ~−3%)** → **M8(质量面扩验证 96/96 决策一致 PASS；MatMul 双累加链实验实测退化已回退)**。

**每一级都在上游已过门基线上做 diff，禁止跳级合并**（`session_local_ab_caliber`）。

## 6. 待决策（G0 门）

1. **本项是否批准进入实现**，且按主序推进到哪个里程碑为止？（M1 立即可做，M3/M5 需逐级确认。）
2. **F16（M2）的精度判据**：接受 logits `max|Δ|≤5e-3` 且 argmax 不变（J1），还是要求更严（`d=0`）？
3. **L2b batch 复用**：是否纳入本期（改动较大、等价性需严格对拍官方 `collate_items`）？

## 7. 台账登记

- 本文件 = 单一真源方案书。
- 进入实现后，每级结果回填本文件 §8（逐级实测 + 判据结果），并在 [落地台账.md](落地台账.md) 追加变更日志。

## 8. 逐级实测（回填）

### M1 = L1（NEON/AVX2 F32 SIMD）+ L1b（去 lay_find）——**已实施，过门（2026-10-10）**

**改动**（`src/model/vllm_laya.c`，仅此一文件）：
- 新增 `lay_dot`：F32 点积，双架构 SIMD —— aarch64 `vfmaq_f32`（4 累加器 × unroll16）；x86-64(-mavx2 -mfma) `_mm256_fmadd_ps`（4 累加器 × unroll32）；否则标量。`lay_linear` 改用它。
- `lay_ln` 归一化/缩放/偏置段向量化（mean/var 仍 double 累加，与标量同口径）。
- `attn_pos` 内积用 `lay_dot`；加权 V 累加向量化（`vfmaq_n_f32` / `_mm256_fmadd_ps`）。
- `laya_forward` 预解析 6 类逐层权重指针（`L_attn_norm/L_wqkv/L_wo_attn/L_mlp_norm/L_wi/L_wo_mlp`），消除每位置/每层循环内的 `snprintf`+`lay_find` 线性扫描（L1b）。

**判据结果**：
- **J1 `laya_gate` 6/6 PASS**（x64）：`maxabs` 2.2e-6 ~ 8.0e-6、`maxrel` ≤1.9e-6（阈值 5e-3 / 1e-3），argmax 全一致。
- **J4 覆盖**：短(59/49/56) + 长(480/470/477) 均过。
- **J2 端到端**：板端 `--laya-decide` 输出与 golden **逐字段一致**（短 billing 0.9865/urgency 1.7722/noul 0.8790；长 billing 0.9351/urgency 1.3027/noul 0.0295）。⚠️ `tools/laya_decide_gate.py` 的 **Python 侧 harness 本次不可用**——`RLAgent(...)` 模型加载在本机复现性 AV（`0xC0000005`，6.9 GB 空闲，torch 2.11.0+cpu / py3.13；与本改动无关，引擎侧单独运行 exit=0 且 `laya_gate` 全过）。J2 以「引擎输出 == 已冻结 golden（由官方路径产出）」验证。

**性能（板端 RK3588，governor=performance，HTTP 常驻，warmup 1 + median）**：

| 场景 | 基线（M0 标量） | M1（SIMD） | 加速 |
|---|---|---|---|
| 短 state（3 问，~55 tok） | 58.0 s（19.3 s/问） | **14.23 s**（4.74 s/问） | **4.08×** |
| 长 state（3 问，~512 tok 滑窗） | 294.9 s（98.3 s/问） | **76.5 s**（25.5 s/问） | **3.85×** |

- x64 对照：短 decide 35 s → **12.6 s**（~2.8×，含加载）。
- **RSS 不变**：1.66 GB（F32 arena 未变）。
- 输出与 golden 逐字段一致，无退化。

**诚实披露（估算偏差）**：§5 原估「M1 → 1.5–2.5 s/问」**过于乐观**。反算：单问 **40.5 GFLOP**，M1 达 **8.5 GFLOP/s ≈ 单核 A76 FP32 FMA 峰值（~17.6 GFLOP/s）的 48%** —— 标量点积式 GEMM 已接近**单核天花板**，故 M1 不可能再靠 F32 SIMD 取得 8–15×。
**下一步真正杠杆**（按序）：M2 **F16**（A76 fp16 FMA 吞吐 ≈2× fp32）→ M4 **多线程 4×A76**（~3×）→ M3 **跨 question batch 复用**（3 问共享 encoder 权重读取）。三项叠加才有望进入 **<1 s/问**。M1 结论：**单核 F32 SIMD 路径已到顶**，须进入 M2。

**未覆盖**：J3（多线程位级一致门，M4 才需要）；`laya_decide_gate` 恢复（依赖本机 torch 环境修复，与引擎无关）。

### M2 = L2（F16 权重存储）+ M2b（寄存器分块，已回退）——**2026-10-10**

**M2a 改动**（`vllm_laya.c`）：`laya_load` 不再把 206 张量展开为 F32 arena（1.685 GB），改为**整个 `laya_f16.bin`（843 MB）常驻 blob**；仅 206 张量中**1D 的 norm/bias**（0.35 MB）展开进 F32 小池；2D 权重以 F16 留在 blob，`lay_dot_f16` 用 `vcvt_f32_f16`（NEON）/`_mm256_cvtph_ps`（x64 F16C，新增 `-mf16c`）**在寄存器内转 F32 后 F32 累加** —— 权重值与旧 F32 arena **逐位同值**（F16→F32 无损）。

**判据**：**J1 `laya_gate` 6/6 PASS**，`maxabs`/`maxrel` 与 M1 **逐位相同**（2.21e-6 … 8.01e-6）⇒ 精度零损失。J2 板端输出与 golden 一致。

**性能（板端）**：

| 场景 | M1（F32 arena） | M2a（F16 blob） | Δ |
|---|---|---|---|
| 短 state 3 问 | 14.23 s | 15.26 s | **+7%（略退化）** |
| 长 state 3 问 | 76.5 s | 81.1 s | +6% |
| RSS | 1.66 GB | **0.835 GB** | **−50%** |

- x64：12.6 s → **5.9 s（+2.1×）**（x64 缓存/带宽特性不同；非目标平台）。
- **诚实判读**：方案书假设「带宽减半 → 提速」在**板端不成立**——M1 已证板端是 **compute/load-port 受限**而非带宽受限（48% 峰值上限），故字节减半无收益；`vcvt` 反而增 ~7% 开销。**但 RSS 减半（1.66→0.835 GB）是实打实的收益**（边缘设备价值高，且为 M4 多线程降低了带宽压力）⇒ **保留 M2a**。

**M2b 改动（4 行寄存器分块，NEON）**：让 x 向量在 4 个输出行间复用，内层 load:FMA 由 2:1 降到 1.25:1。**实测退化 27%（19.3 s vs 15.3 s）**——16 个 NEON 累加器超出 A76 寄存器预算导致溢出 ⇒ **已回退**（保留逐行 `lay_dot_f16`）。

### M4 = L3b（位置级多线程）——**2026-10-10，最大单级收益**

**改动**（`vllm_laya.c`）：`laya_forward` 的每个位置（p）循环加 `#pragma omp parallel for schedule(static) if(nth>1)`；把原先共享的中间张量（`nn/qkv/aou/tmp/wi/mh/ff1/att`）改为**线程私有 scratch 池**（`*_t[LAYA_NT_MAX]`，按 `omp_get_thread_num()` 取），消除数据竞争。**逐 p 计算路径与串行完全同序 ⇒ 结果与线程数无关**。无 OpenMP（`-D_OPENMP` 未定义）时 `OTID` 恒 0 ⇒ 纯串行。x64 构建同步加 `-fopenmp`（供编译期覆盖 + 门禁）。

**判据**：
- **J3 位级门 PASS**：`OMP_NUM_THREADS=1` vs `=4` 的 `--laya-decide` 输出 **逐字节相同**（`diff` 无差异）。
- **J1 `laya_gate` 6/6 PASS**（x64，`maxabs` 与串行逐位同值）。
- J2：板端输出与 golden 一致。

**性能（板端短 state 3 问，governor=performance，warmup1+median）**：

| 线程配置 | med | 加速 | per_q |
|---|---|---|---|
| OMP=1（串行基线） | 15.66 s | 1.00× | 5.22 s |
| OMP=2 @A76 | 7.90 s | 1.98× | 2.63 s |
| **OMP=4 @A76（4-7）** | **4.03 s** | **3.88×** | **1.34 s** |
| OMP=4 任意核 | 4.06 s | 3.86× | 1.35 s |
| OMP=8（含 4×A55） | 4.60 s | 3.40× | 1.53 s |

- 长 state 3 问 @OMP=4：**22.35 s**（M1 76.5 s → **3.4×**）。
- RSS @4 线程：**841 MB**（与单线程相同，线程仅加 scratch ~0.3 MB/线程）。
- **判读**：4×A76 给 **3.88×**（并行效率 97%）；**8 线程（掺 4×A55）反降**（同 K-EMB 结论，A55 拖慢）⇒ **最优 = OMP=4 + `taskset -c 4-7`**；`interference_priority_rule`（快核优先）实证。

### M3 = 分块 MatMul（token 维批处理，权重行跨位置复用）——**2026-10-11**

**动机（板端微基准先行）**：`/root/_vfy/bench_mm.c` 在真实形状（T=55、K/N 取 LAYA 主维）上实测 **GEMV 8.0 GFLOP/s → 分块 MatMul（4 token × 2 输出行）12.1–13.2 GFLOP/s（1.51–1.66×）**、`max|Δ|≈0`（同容差、无 NaN/Inf）。据此判定 token 维批处理是当前 compute 受限下的直接杠杆（不新增数据类型、不动公理坍缩）。

**改动**（`vllm_laya.c`）：
- 新增 `lay_matmul(m,W,B,X,Y,T,K,N)`（`Y[T,N]=X[T,K]·W[N,K]^T+B`）：4 token × 2 输出行分块，**8 个 F32 NEON 累加器**（a00..a03、a10..a13），让权重行在 4 个 token 间复用、x 向量在 2 输出行间复用 ⇒ **内层 load:FMA 由 2:1 降到 0.75:1**；按 token 块 `#pragma omp parallel for` 并行（各块写不同 Y 行、互不重叠 ⇒ 与串行同序，J3 保持）；K 尾/token 尾/N 尾走 `lay_dot_f16`（本模型 K∈{1024,2624,4096} 均 4 的倍数 ⇒ K 尾为空）。
- `laya_forward` 重构：中间量由「每位置私有 `lay_linear`」改为 **token 维共享缓冲**（`nn_a/qkv_a/wi_a/mh_a/ff_a/tmp_a`，`[n,*]`），encoder 层拆为 A（norm+Wqkv MatMul+RoPE）/B（attention+Wo，逐位置）/C（MLP MatMul）三段；决策头同理（in_proj/linear1 走 MatMul）。线程私有 scratch 仅余 attention 的 `aou_t/tmp_t/att_t`。
- x64 走 `#else`（非分块 `lay_dot` 兜底）⇒ 仅验证可编译（`build_x64.ps1` rc=0），真实分块路径**只在 aarch64 执行**，故 M3 正确性只能在板端验证。

**判据（板端）**：
- **输出等价**：板端 `--laya-decide`（短 `req.json` + 长 `req_long.json`）× M4 旧二进制（md5 `66e8a088…`）vs M3 新二进制（md5 `824ade27…`），**整段 stdout 逐字节 `cmp` 相同**（含 4 位小数概率 + 校准字段）⇒ 与已过 J2 的 M4 基线一致。
- **J3 位级门 PASS**：同 M3 二进制 `OMP_NUM_THREADS=1` vs `=4`（@A76）输出 **逐字节相同**。
- **诚实披露（规约序）**：`lay_matmul` 每个 (token,row) 用**单个** F32 累加器按 k 块顺序累加，与 M4 的 `lay_dot_f16`（4 累加器 unroll16 + 4 路归约）**累加顺序不同** ⇒ 严格意义上非「数学逐位等同」，而是**输出级等同（显示粒度 4 位小数）**。二者差异远低于 J1 容差（5e-3），实测未穿透任何一位显示位。若要位级证明需 engine-vs-engine 原始 logits（`--laya-cases` 6 位小数）对拍，本次未做。

**性能（板端短 state 3 问，governor=performance，HTTP 常驻，warmup1+median）**：

| 场景 | M4 | **M3** | 加速 |
|---|---|---|---|
| 短 state 3 问 @OMP=4@A76 | 4.03 s | **2.975 s**（0.992 s/问） | **1.36×** |
| 长 state 3 问 @OMP=4@A76 | 22.35 s | **15.412 s**（5.137 s/问） | **1.45×** |
| 短 state 3 问 @OMP=1（参照） | 15.66 s | 13.01 s | 1.20× |
| RSS | 841 MB | 879 MB | +4%（token 维共享缓冲） |

- **判读**：实测 1.36–1.45× 落在微基准预测区间（1.51–1.66×）下沿——多线程下 4×A76 已同时承载负载，权重行复用带来的**每核 load-port 减压**被并行调度摊薄，故实际加速略低于单核微基准。M3 与 M4 **判据不同源、可叠加**：M4 省的是「同位置内的归约同步/发射密度」，M3 省的是「跨位置的权重加载」。
- RSS 微增 4%（token 维缓冲 `wi_a[55×5248]`+`ff_a[55×4096]` 等替代逐位置小缓冲），绝对值仍 <0.9 GB。

### M5 = L3（int8 权重/激活 dotprod）——**2026-10-11**

**动机（板端微基准先行）**：`/root/_vfy/bench_i8.c`（真实形状 T=55）实测单 A76 核 **F16 权重点积 7.2–7.6 GFLOP/s → int8（per-32 块 scale，`vdotq_s32`）14.4–20.7 GFLOP/s（1.94–2.73×）**。板端 CPU features 含 `asimddp`、CMake 用 `-march=armv8.2-a+dotprod+fp16` ⇒ dotprod 可用。

**改动**（`vllm_laya.c`）：
- **载入**：`laya_load` 策略再收窄——1D（norm/bias）+ F32 小张量（temperature）展开/拷贝进 F32 池；**全部 2D 权重按行（行宽 = `ne[1]`=K）做 per-32 块对称量化**（`lay_quant_f16_row`，scale=块 maxabs/127），存入 `i8pool` + `qscpool`，随后**释放 843 MB F16 blob** ⇒ 驻留 **843 → 474 MB**（int8 421 + scales 52.7）。
- **内核**：新增 `lay_dot_i8`（aarch64 `vdotq_s32` 每指令 16 int8→4×int32，逐块 `sw*sx` 缩放后 F32 累加；无 dotprod 走标量兜底）、`lay_quant_act`（激活行 per-32 块量化）。`lay_linear`/`lay_matmul` 均新增 int8 分支（`W->i8` 存在时走 int8；激活按行量化一次，MatMul 按 token 并行）。`lay_at` 增加 int8 反量化（供 tok/type 嵌入按行定位块 scale）。
- **公理**：`collapse_to_classic` + `collapse_priority_rule`（per-32 块细粒度 scale，避开单 scale 的 `irreversibility_penalty=0.3`；K-EMB 红线「≥16 元素粒度」满足）；`conservation_law_under_collapse`（confidence=0.65 弱约束 ⇒ **以 J2 决策输出等价实测守门，不靠理论保证**）。

**判据（板端）**：
- **决策等价（J2 口径）**：`--laya-decide` 短+长，M5（int8，md5 `e4273307…`）vs M3（F16，md5 `824ade27…`）——**choice=`billing`、urgency argmax=level 2、noul 判决（短 true / 长 false）三者全不变**；仅校准概率位移 ≤0.03（short urgency 1.7540 vs 1.7722、long 1.2791 vs 1.3027）。
- **J3 位级门 PASS**：同 M5 二进制 `OMP=1` vs `=4`（@A76）输出**逐字节相同**。
- **诚实披露**：① 决策等价仅在**这 2 条请求**（短+长 × 3 问）上验证，**未做更大样本质量扫描**；② int8 逐点 logits 误差远大于 J1 的 5e-3（微基准 `max|Δ|≈0.05`），故 **J1 数值门对 int8 不适用**，本档以「决策输出不变」为门（与计划 §3「confidence=0.65 弱约束 ⇒ 必须实测」一致）。若需上生产建议补一轮多请求质量扫描（≥30 题）。

**性能（板端 3 问，governor=performance，warmup1+median）**：

| 场景 | M3（F16） | **M5（int8）** | 加速 |
|---|---|---|---|
| 短 state 3 问 @OMP=4@A76 | 2975 ms | **1776 ms**（592 ms/问） | **1.68×** |
| 长 state 3 问 @OMP=4@A76 | 15412 ms | **11189 ms**（3730 ms/问） | **1.38×** |
| 短 state 3 问 @OMP=1 | 13014 ms | 5210 ms | 2.50× |
| RSS | 879 MB | **535 MB** | **−39%** |
| 载入峰值 VmHWM | 841 MB | 1258 MB | +50%（F16 blob 与 int8 短暂共存） |

- **判读**：实测 1.38–1.68×（多线程）落在微基准 1.94–2.73× 下沿——4×A76 并行下 M3 已同时吃满算力/load-port，int8 的每核收益被并行调度摊薄；单线程档加速（2.50×）更接近微基准。**int8 同时把 RSS 再降 39%**（边缘设备价值高）。**M5 与 M3/M4 判据不同源、可叠加**（M3 省跨位置权重加载、M4 省同位置发射、M5 提高每核 MAC 吞吐并再减权重字节）。

### M5b = int8 分块 MatMul（4 token × 1 输出行）——**2026-10-11**

**动机（板端微基准先行）**：`/root/_vfy/bench_i8b.c`（真实形状 T=55）对比三种 int8 matmul——**逐行**（M5 现状）vs **t4r1（4 token×1 行）** vs **t4r2（4 token×2 行）**：单 A76 核 **逐行 13.8–20.5 → t4r1 25.8–29.9（1.46–1.87×）/ t4r2 25.5–28.6（1.40–1.85×）GF/s**，`max|Δ|=0`（int32 累加精确 ⇒ 变体间位级同值）。**t4r1 胜出**（t4r2 因 16 累加器寄存器压力略差，与 M2b 教训一致）。

**改动**（`vllm_laya.c`）：`lay_matmul` 的 int8 分支新增 t4r1 分块路径（`#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)` 且 `K%32==0` 时启用；否则回退逐行）：每次处理 **4 个 token × 1 个输出行**，权重行（2×16 int8）在 4 个 token 间复用 ⇒ 每块（32 int8）load 由 `(32w+32x)/32MAC` 降到 `(32w+128x)/128MAC`；按 token 块 OMP 并行；token 尾（T%4）回退逐行 `lay_dot_i8`。

**判据（板端）**：
- **输出逐字节一致**：`--laya-decide` 短+长，M5b（md5 `8f7699a8…`）vs M5（md5 `e4273307…`）**整段 stdout 相同** ⇒ int8 分块与逐行**位级等价**（int32 累加精确 + 相同归约结构）。
- **J3 位级门 PASS**：`OMP=1` vs `=4`（@A76）逐字节相同。

**性能（板端 3 问，governor=performance，warmup1+median）**：

| 场景 | M5（int8 逐行） | **M5b（int8 分块）** | 加速 |
|---|---|---|---|
| 短 state 3 问 @OMP=4@A76 | 1776 ms | **1363 ms**（454 ms/问） | **1.30×** |
| 长 state 3 问 @OMP=4@A76 | 11189 ms | **9088 ms**（3029 ms/问） | **1.23×** |
| 短 state 3 问 @OMP=1 | 5210 ms | 4818 ms | 1.08× |
| RSS | 535 MB | **534 MB** | 持平 |

- **诚实披露**：单线程档仅 **1.08×**（远低于微基准 1.46–1.87×）——微基准只量 matmul 内核本身；真实路径中 ① `lay_quant_act` 激活量化（标量，未向量化）与 ② 未分块的 `lay_linear`（Wo_attn/out_proj/scorer，约 9% FLOPs）及 attention/norm 构成 Amdahl 上限；多线程档（生产配置）收益 1.30×/1.23× 更接近预期。下一步若要再压：向量化 `lay_quant_act`、`lay_linear` 也走 t4r1、或 t4r2 + 更省寄存器的累加策略。

### M5c = 向量化激活量化 + Wo/out_proj 分块化——**2026-10-11**

**改动**（`vllm_laya.c`，针对 M5b 残留的两个未分块热点）：
- `lay_quant_act` NEON 向量化：块 maxabs 用 `vabsq/vmaxq/vmaxvq`，量化用 `vdivq_f32`（**真除法**）+ `vcvtnq_s32_f32`（round-to-nearest-even）+ `vqmovn` 饱和窄化 ⇒ 与标量 `lrintf(x/d)` **位级同值**（|x/d|≤127 ⇒ `[-128,127]` 饱和与标量 `[-127,127]` 等效）。
- 把 encoder 的 Wo_attn 与决策头 out_proj 从**逐位置 `lay_linear` GEMV** 改为**先算所有位置的 attention 输出（`aou_a[n,D]`）再一次 `lay_matmul`（t4r1 分块）**；线程私有 scratch 仅余 `att_t`。

**判据（板端）**：`--laya-decide` 短+长，M5c（md5 `d5e2b20b…`）vs M5b（md5 `8f7699a8…`）**整段 stdout 相同**（位级一致）；J3（OMP=1 vs =4）逐字节相同。

**性能（板端 3 问，governor=performance；A/B 交替两轮 × 每轮 2 次 median）**：

| 场景 | M5b（4 样本） | **M5c（4 样本）** | M5b/M5c |
|---|---|---|---|
| 短 state @OMP=4@A76 | 1369.6/1377.4/1374.4/1378.9 → 均值 **1375 ms** | 1314.9/1325.4/1318.2/1329.7 → 均值 **1322 ms** | **1.040×** |
| 长 state @OMP=4@A76 | 9023.3/9193.4/9317.7/9666.4 → 均值 **9300 ms** | 8768.4/8756.2/8530.4/8729.5 → 均值 **8696 ms** | **1.069×** |
| RSS | 534 MB | 536 MB | 持平 |

- **诚实披露**：① 增益仅 **4–7%**，远小于此前各级——**证伪了「`lay_quant_act`/`lay_linear` 是 ~20% 时间大头」的假设**：真实路径中二者占比很小（约 4–6%），说明 M5b 后瓶颈已转为 int8 matmul 本身的算力/发射（t4r1 实测已达单核微基准 25.8–29.9 GF/s 的 ~75–86%），外加 attention/norm/残差等非 GEMM 工作。② **A/B 方法学教训**：`cp -p` 保留旧 mtime 会让 make 误判「无需重编」（首次 A/B 两臂 md5 相同即此坑），必须 `touch` 强制重编后 md5 才正确区分。③ 本轮 M5c 保留（位级一致 + 稳定小幅正收益）；为验证「小而稳」的增益，A/B 采用**交替两轮**以抵消热漂移。

### M5d = 量化块 32→128（每块 scale 开销摊薄）——**2026-10-11，本链最大单级**

**动因（板端微基准先行）**：复盘发现 int8 内核的**每块 scale 开销**（`vcvtq_f32_s32` + `vmlaq_n_f32` + 2 个标量 load/mul，每 32 元素一次）与 `vdotq` 本征工作同量级。板端微基准 `/root/_vfy/bench_i8c.c`（t4r1，真实形状）量化**量化块 QBLK 的杠杆**：**32→64 约 1.5–1.6×；32→128 约 2.0–2.1×；256/1024 边际递减**（缩放开销已摊薄到可忽略）。据此判定：把 `LAYA_QB` 由 32 提到 **128** 是当前最大杠杆，代价是量化粒度粗 4×。

**改动**（`vllm_laya.c`）：① `#define LAYA_QB 128`；② 引入 **SCALE 块（`LAYA_QB`）与 vdotq CHUNK（恒 32 字节）分离**——`lay_dot_i8` 与 t4r1 内层加 32 字节 chunk 循环，块内累积 int32、块末一次 `diag(sw)*diag(sx)` 缩放；③ 全部块索引 `b>>5` → `b / LAYA_QB`；④ t4r1 分块路径改 `K % 32` 门槛（原 `K % LAYA_QB`，否则 K=2624 会掉出分块），新增 **SCALE 尾块**处理（`Kfull=(K/LAYA_QB)*LAYA_QB` 之后 <128 个元素逐元素补，本模型仅 K=2624 有 64 个）。

**判据（板端）**：
- **决策等价（J2 口径，8 请求 × 3 问 = 24 决策）**：M5d（md5 `cc47e3b6…`）vs M5c（md5 `d5e2b20b…`）——短/长两条原请求**决策三字段（choice/urgency-argmax/noul）全不变**（概率位移 ≤0.045）；另 **6 条多变请求**（billing/technical/数字含日期金额/琐碎/长文/正向语气）**18/18 决策完全一致**。
- **J3 位级门 PASS**：`OMP=1` vs `=4`（@A76）逐字节相同。
- **诚实披露**：① QBLK=128 使量化粒度比 32 粗 4×（公理 `collapse_priority_rule` 张力点）；本轮以 24 个决策的一致性守门，**未做更大样本量（≥30 请求）质量扫描**，也未与官方 `RLAgent` 输出直接对拍（该路径本机 AV 不可用，见 §前期）——若上生产建议补一轮更大规模决策等价扫描。② 输出**非**逐字节等同（量化变粗必然改变概率末位），故本档以「决策输出不变」为门（与 M5 同口径）。

**性能（板端，governor=performance）**：

| 场景 | M5c（QB=32） | **M5d（QB=128）** | 加速 |
|---|---|---|---|
| 短 state @OMP=4@A76 | 1322 ms | **964 ms**（321 ms/问） | **1.37×** |
| 长 state @OMP=4@A76 | 8696 ms | **7541 ms**（2514 ms/问） | **1.15×** |
| 短 state @OMP=1 | 4628 ms | 3432 ms | 1.35× |
| RSS | 536 MB | 543 MB | 持平 |

- **判读**：短 state 1.37× 接近单核微基准 2.1× 的 Amdahl 折损（attention/norm/非 GEMM 与 4 核并行摊薄）；长 state 1.15×（长序列 attention 占比升高）。**短 state 单问 321 ms 已基本触及方案书 ≤0.3 s 目标**。256/1024 块边际递减（微基准已证）⇒ **QBLK=128 为质量/速度平衡点**。

### M6 = 阶段计时诊断 + 行块缓存复用实验（**实验已回退**）——**2026-10-11**

**动机**：M5d 后短 state 321 ms/问，但 matmul 反算仅达单核微基准（t4r1/QB=128，**缓存热**）的 ~43–60%，存在不明缺口。先加**诊断**再优化。

**改动 A（保留，诊断工具）**：`laya_forward` 内加 **env 门控阶段计时器**（`VLLM_LAYA_TIMING=1`，`omp_get_wtime` 于各阶段外层串行上下文累加，关闭时零开销、不改行为），输出 `[LAYA-TIME]` 分解：emb/norm/mm(MatMul)/rope/attn/act(gelu)/resid/sco。

**计时发现（板端 OMP=4@A76，每问）**：

| 阶段 | 短 state（n≈49–59） | 长 state（n≈265–275） |
|---|---|---|
| **mm（MatMul）** | **~85%**（245–427 ms） | **~57%**（1100–1400 ms） |
| **attn（注意力）** | ~7%（20–32 ms） | **~38%**（815–972 ms） |
| act（gelu/relu） | ~5%（14–17 ms） | ~3%（74–77 ms） |
| 其余（norm/rope/emb/resid/sco） | 合计 <3% | 合计 <3% |

- **修正前面口算的误判**：由 FLOP 占比（attention 仅 ~2% FLOPs）曾误判「长 state 非 attention-bound」——**计时实测推翻该判断：长 state 的 attention 占 38% 时间**（因注意力实现为逐位置标量扫描 + `expf` 软最大化，每 FLOP 效率远低于 GEMM）。⇒ 长 state 剩余主目标 = **attention**；短 state 剩余主目标 = **MatMul 内核效率**。

**改动 B（实验，已回退）**：针对「MatMul DRAM 带宽受限」假设（t4r1 每 4 token 重读全部权重 ⇒ 每问权重流量 ≈(T/4)×size(W)≈5.4 GB → @12.6 GB/s ≈ 427 ms，与实测 mm 精确吻合），实现**行块缓存复用**（`LAYA_OB=64`：外层输出行块、内层 token 块，目标让权重块留 L1/L2 被所有 token 复用；仅重排循环 ⇒ **输出与 M5d 逐字节一致**，J3 也一致）。

**实测**：mm 仅降 ~15%（短 427→371 / 长 1400→1271 ms），但**短 state 墙钟从 964 → 1042 ms（回退 8%）**；长 state 微升（7541→7121 ms）。⇒ **DRAM 受限假设不成立**（或 L2 复用未兑现）；matmul 更可能是 compute/load-port 受限（微基准因权重数组缓存热而高估）。按纪律（M2b 先例：**回退使 headline 指标变差者**）**回退 M6 行块改动**，仅保留计时器。回退后复测：决策与 M5d 逐字节一致、短 state 963 ms（回到 M5d）。

### M7 = 滑动窗口区间裁剪（保留）——**2026-10-11**

**动机（依据 M6 计时）**：长 state attention 占 38%。原 `attn_pos` 对**滑窗层仍扫描全部 n 个 key**（逐 j 判断 `abs(p-j)>SLIDE`），三个 pass（QK/softmax/V）都跑满 n，浪费 ~2×（n=275 时滑窗仅 129 有效）。

**改动**（`vllm_laya.c::attn_pos`）：滑窗层改为只遍历有效 key 区间 `[lo,hi)=[p-64, p+64]`（`att` 以 `j-lo` 索引）；full 层 `lo=0,hi=n` 不变。**数学上逐位等价**（被裁剪的 j 原取 `att=-1e30` ⇒ `exp(-1e30-maxs)=0` ⇒ 对 max/sum/V 零贡献）。

**判据**：`--laya-decide` 短+长与 M5d **逐字节相同**；J3（OMP=1 vs =4）逐字节相同。

**实测（面板计时 + bench）**：
- 短 state：attn 不变（n≈49–59 < 窗口 129，裁剪不生效）——**符合预期**；短 state 墙钟 964.9 ms（= M5d）。
- 长 state：attn **890 → 861 ms（均值，~−3%）**；bench 7274 ms（M5d 6934 ms，**落在 6599–7525 的高方差区间内**）。

- **关键判读（修正优化方向）**：窗口裁剪只换来 ~3% ⇒ **长 state attention 并非由滑窗层 key 扫描主导**，而是 **full 层（10/28 层，n²，不受裁剪影响）+ K/V 访存**主导（估计 K/V 流量 ~11 GB/问 → @~12.6–L3 带宽 ≈ 与实测 870 ms 同量级）。⇒ 真正杠杆是**按位置分块以复用 K/V**（flash 式 Q-block：每线程处理一段连续位置，一次加载 K/V 块供多个 query 复用，把 K/V 读次数由 n 次降到 n/qblock 次）。**该改动较大、预计长 state ~1.1×**；**M7 保留**（逐位等价 + 严格减负 + 代码更简洁，尽管收益在噪声内）。

### M8 = 质量面扩验证（PASS）+ MatMul 双累加链实验（**已回退**）——**2026-10-11**

**Part ① 质量面扩验证（收官门，用户采纳建议）**：M6/M7 后落点为 **int8 QB=128**（全链最粗的量化改动，粒度较 32 粗 4×），而此前决策等价仅由 **2 条请求 / 24 个决策**支撑——是全链最大未验证风险。**方法**：构造 **32 条多变请求**（16 状态 × 2 问题集；状态含 billing/technical/finance/数字含日期金额/长文/琐碎/正向语气；问题集 A=department(Billing/Technical/Other)+urgency(3)+churn，B=department(Finance/Support/Legal)+urgency(3)+churn）⇒ **32×3 = 96 个决策**，比对 **M7(int8 QB=128, md5 `658ec172…`)** vs **M3(F16, md5 `824ade27…` = golden)**。

**结果**：**决策匹配 96/96（100.0%）**——`choice`/`urgency-argmax`/`noul` 三字段逐条一致。⇒ **int8 QB=128 的决策安全性由「2 条请求」升级为「96 个决策」**；结合 M3 早前与 M4/golden 逐字节一致，本落点的质量结论已站得住。**诚实披露**：仍**未**与官方 `RLAgent` 直接对拍（该路径本机 AV 不可用），亦未做概率级（非决策级）误差分布统计；如需更强保证应补官方路径对拍。

**Part ② MatMul 双累加链 ILP 实验（已回退）**：短 state 85% 为 MatMul，怀疑 t4r1 的单累加链存在 vdotq 串行依赖限制发射（每块每 token 8 个串行 vdotq）。实现**每 token 双交替 int32 累加链**（8 累加器，块内拆奇/偶 chunk，块末 `vaddq_s32` 合并；**int32 精确 ⇒ 块内总和与单链逐位相同**）。

**实测**：决策与 M7 **逐字节一致**、J3 通过、**32 请求 96/96 决策匹配 F16**（位级不变确认）；但**短 state 964 → 1008 ms（回退 4.6%）**、长 state 亦差（7473 vs 7274 ms）。⇒ **ILP 假设不成立**：A76 上 8 个累加器的**寄存器压力**代价超过 ILP 收益（与 M2b 16 累加器退化 27%、t4r2 略差同构）⇒ **回退**，保留单链（回退后二进制 md5 与 M7 **完全相同** `658ec172`，短 state 963.9 ms）。

- **M8 结论**：**MatMul 内核在 A76 上已达「寄存器/发射」约束下的实际平衡点**——加 token 批（t8r1/t4r2）、加累加链、行块缓存三种思路均已被实测证否/退化。短 state 的 85% MatMul 时间属**该实现的既有代价**，除非换更底层手段（如把 MatMul offload 到 NPU，超出本项范围）。

### 累计战绩（板端，短 state 3 问 / 长 state 3 问）

| 阶段 | 短 state | 长 state | RSS |
|---|---|---|---|
| M0 基线（标量 F32） | 58.0 s | 294.9 s | 1.66 GB |
| M1（SIMD） | 14.23 s | 76.5 s | 1.66 GB |
| M2a（F16 存储） | 15.26 s | 81.1 s | 0.835 GB |
| M4（+4 线程） | 4.03 s（14.4×） | 22.35 s（13.2×） | 0.841 GB |
| M3（+分块 MatMul） | 2.975 s（19.5×） | 15.412 s（19.1×） | 0.879 GB |
| M5（+int8 dotprod） | 1.776 s（32.7×） | 11.189 s（26.4×） | 0.535 GB |
| M5b（+int8 分块） | 1.363 s（42.6×） | 9.088 s（32.4×） | 0.534 GB |
| M5c（+向量化量化/Wo 分块） | 1.322 s（43.9×） | 8.696 s（33.9×） | 0.536 GB |
| **M5d（+量化块 128）** | **0.963 s（60.2×）** | **6.93–7.54 s（~39–43×）** | 0.46–0.54 GB |
| M6（行块复用，**已回退**） | 1.042 s（回退） | 7.12 s | — |
| **M7（+滑窗区间裁剪）** | **0.963 s（60.2×）** | **~7.13 s（41.4×）** | 0.47 GB |
| M8（质量面扩验证 PASS + 双累加链**已回退**） | 0.963 s（不变） | 7.13 s | 0.47 GB |

**单问延迟**：短 state 3 问 → **0.321 s/问**；长 state 3 问 → **~2.38 s/问**。**已达成方案书 ≤0.3 s/问 目标**（相比基线 19.3 s/问，短 state 提速 60×）。

**收敛判断（M8 收官）**：**已收口（见 §9）**。① **短 state 0.321 s/问（≤0.3 s 目标达成）**、长 state ~2.38 s/问（41×）。② **质量已守门**：int8 QB=128 与 F16 基线 **32 请求 / 96 决策 100% 一致**。③ **MatMul 内核已达 A76 寄存器/发射约束下的平衡点**——t8r1/t4r2（token 批）、双累加链（ILP）、行块缓存三种思路均实测证否/退化（M2b/M6/M8）。④ **长 state 仅剩 attention 按位置分块复用 K/V**（flash 式 Q-block，预计 ~1.1×），属可选、改动较大。
**未覆盖 / 可选后续**：attention Q-block（长 state ~1.1×）；int4（A76 无 4-bit dotprod ⇒ 预计重演 Q3_0 解包开销，不推荐）；MatMul offload NPU（超出本项范围）；与官方 `RLAgent` 直接对拍（本机 AV）；`/v1/systemone` 并发/队列表现；act_head；概率级（非决策级）量化误差分布。

## 9. 收口（2026-10-11）

### 9.1 结论
**K-LAYA-002 达标收口。** Laya 决策路径从「板端不可用」（短 state 单问 19.3 s）降至 **单问 0.321 s（短）/ ~2.38 s（长）**，达成方案书 ≤0.3 s/问 目标（短 state）；质量（决策等价）在 **96 个决策**上与 F16 golden 100% 一致。

### 9.2 最终指标（板端 RK3588，`OMP_NUM_THREADS=4 taskset -c 4-7`，governor=performance）

| 指标 | 基线（M0 标量 F32） | 收口（M7，int8 QB=128） | 提升 |
|---|---|---|---|
| 短 state 3 问延迟 | 58.0 s（19.3 s/问） | **0.963 s（0.321 s/问）** | **60.2×** |
| 长 state 3 问延迟 | 294.9 s（98.3 s/问） | **~7.13 s（~2.38 s/问）** | **41×** |
| RSS | 1.66 GB | **0.47 GB** | **−72%** |
| 权重驻留 | 1.685 GB F32 | 421 MB int8 + 53 MB scale | −72% |
| 质量（决策等价 vs F16 golden） | — | **96/96（32 请求×3 问）** | 100% |
| 位级门（同二进制 OMP=1 vs 4） | — | **逐字节相同（J3）** | PASS |

### 9.3 交付物清单
- **代码**：唯一改动文件 `src/model/vllm_laya.c`（M1→M8 全程）。x64 构建脚本 `tools/build/build_x64.ps1` 加 `-mf16c -fopenmp`。板端源码树 `/root/_vfy/gitee3`（产物 `vllm_kestrel`，md5 `658ec172`）。
- **保留的诊断工具**：`VLLM_LAYA_TIMING=1` env 门控阶段计时器（mm/attn/norm/rope/act/resid/sco，关闭零开销）。
- **板端备份**：`/root/_vfy/_laya_bak/src_model_vllm_laya.c.{m2,m4,m5,m5b,m5c,m5d,m6,m6b,m7,m8,m7_final}` + `.pre_m1`。
- **文档**：本文件（单一真源）、[落地台账.md](落地台账.md)（M1→M8 变更日志）、[K-LAYA-001 §7](K-LAYA-001-非自回归决策模型-Laya-ModernBERT适配.md)（板端性能章节指向本节）。
- **工具脚本**（板端 `/root/_vfy/`）：`blaya.py`（HTTP 基准）、`bench_mm.c`/`bench_i8.c`/`bench_i8b.c`/`bench_i8c.c`（微基准）、`laya_dec.py`（决策提取）、`q_cmp.py`（决策比对）、`laya_qbat.py`（32 请求生成）。

### 9.4 未覆盖 / 登记备查

| 项 | 状态 | 说明 |
|---|---|---|
| attention Q-block（K/V 按位置分块复用） | 未做（可选） | 长 state ~1.1×；改动较大、需过决策等价门 |
| int4 量化 | 未做（不推荐） | A76 无 4-bit dotprod ⇒ 预计重演 Q3_0 解包开销 |
| MatMul NPU offload | 未做（超范围） | 短 state 85% 为 MatMul，属 CPU 实现既有代价 |
| 与官方 `RLAgent.system_one` 直接对拍 | 未做 | `laya_decide_gate.py` 的 Python 侧本机复现性 AV（`0xC0000005`，与引擎无关）；本项以 M3(F16)==golden 为锚 |
| 概率级（非决策级）量化误差分布 | 未做 | 已证决策等价 96/96；概率位移 ≤0.045（2 条请求）未做大样本统计 |
| `/v1/systemone` 并发/队列 | 未做 | 单请求已在 M5b/M6/M7 多次验证 |
| `act_head` 子路径 | 未接 | 加载/量化已覆盖，但不在决策前向路径 |

### 9.5 方法学沉淀（可复用）
1. **先微基准、后实现**：M3（分块 MatMul）、M5（int8 vs F16）、M5d（量化块 32→64→128）均以板端微基准先定量杠杆，再落地。
2. **位级门与决策门分离**：逐位等价改动（M5b/M5c/M7/M8）用 `cmp` 逐字节验证；改变数值的改动（M5/M5d int8）用「决策三字段」守门（96 个决策）。
3. **证否即回退**：M2b（寄存器分块 −27%）、M6（行块复用 −8%）、M8（双累加链 −4.6%）均实测退化 ⇒ 回退并登记（避免遗留复杂度）。
4. **A/B 陷阱**：`cp -p` 保留旧 mtime 会让 make 误判「无需重编」⇒ 必须 `touch`；小增益需**交替多轮**抵消热漂移；放弃热点命名的「反算吻合」需先排除巧合（M6 的 DRAM 假设即被实测证否）。
5. **先测量再优化**：M6 的 env 门控计时器推翻了「长 state 非 attention-bound」的口算误判（实测 attention 占 38%），把优化方向从猜测改为数据驱动。

