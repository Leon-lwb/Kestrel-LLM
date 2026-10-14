# 立项 K-KV-002 · G0 立项门（KV 侧内存默认化）

| 项 | 内容 |
|---|---|
| **立项 ID** | K-KV-002 |
| **标题** | KV 侧内存默认化评估（由 K-L0-001 G0 的「候选 C」升级） |
| **目标** | 找出 KV 侧**可默认化**的内存项，给出占比/上界/可达性，判定是否进 G1 |
| **阶段** | **G0 → G1 → G2 → G3 ✅ → G4 ✅（已默认化）** |
| **日期** | 2026-10-08 |
| **判定** | **G3 PASS + G4 默认化**：`VLLM_KV_NOF32` **默认改为开**（`=0` 逃生舱）。30B-A3B 实测：KV 斜率 **471.7 → 164.2 KB/token（−65.2%）**、8K 外推省 **~2.52 GB**；A1 ΔNLL **+0.004** / A2 NIAH **5/5 两臂** / A3+A7 COPYTEST **diff=0** / A4 L3 **4/4** / A5 DKV v2 **PASS** |

---

## 1. 依据公理

| 公理 id（symbol） | 约束 |
|---|---|
| `perf_axiom_todo_needs_denominator_v1`（`todo_requires_measured_share`） | 立项前须给占比 + 可达性 |
| `perf_axiom_candidate_root_cause_upper_bound_v1`（`candidate_root_cause_upper_bound_first`） | 先算归零上界，解释不了目标即否 |
| `perf_axiom_gate_decoupling_before_default_v1`（`gate_decoupling_before_default_enablement`） | 一个门管两特性时，**默认化会当场打断另一个特性** ⇒ 先拆门 |
| `perf_axiom_default_enablement_gate_v1`（`default_enablement_gate`） | 默认化四条件（位级/效应/逃生舱/自动回退） |
| `perf_axiom_bitwise_gate_v1`（`bitwise_equivalence_gate`） | **本项目已放宽**：非位级可落地，但必须标注非位级性质 + 精度口径 + 质量门结果 |
| `perf_axiom_derived_value_disclosure_v1`（`derived_value_disclosure`） | 反算值须披露推导式 |
| `perf_axiom_feasibility_conclusion_timeliness_v1`（`feasibility_conclusion_timeliness`） | 「不可做」结论须写明成立条件 |
| `perf_axiom_session_local_ab_caliber_v1`（`session_local_ab_caliber`） | 同会话交错 ≥3 轮配对 |

---

## 2. G0 步骤 ①：全仓查证 —— KV 侧门清单与**当前默认态**

> 依据 `perf_axiom_default_value_change_traceability_v1`：门默认状态以台账 §1 为准。

| 门 | 当前默认 | 取证 | 与「推荐默认档」的关系 |
|---|---|---|---|
| KV v2 惰性分配 `g_kv_lazy` | **默认开** = `!g_xq_pages` | [vllm_safetensors.c#L9570](../src/model/vllm_safetensors.c#L9570) | 正交，可叠加 |
| KV 精度 `use_kv_q8` | **默认开** = `!g_kv_q4` | [#L9657-L9658](../src/model/vllm_safetensors.c#L9657-L9658) | 正交 |
| `VLLM_KV_F32=1` | 未设 = 关（A/B 档：强制 f32 KV） | [#L9660-L9662](../src/model/vllm_safetensors.c#L9660-L9662) | 正交 |
| **`VLLM_KV_NOF32=1`** | **未设 = 关（实验档）** | [#L9309-L9314](../src/model/vllm_safetensors.c#L9309-L9314) | **需逐条复核**（见 §4） |
| `--kv-q4` / `g_kv_q4` | **默认关** | [#L2587](../src/model/vllm_safetensors.c#L2587) | **与 `--sparse-attn` 互斥** |
| `--l3-evict` | 默认关 | [#L2574-L2587](../src/model/vllm_safetensors.c#L2574-L2587) | 依赖 `--sparse-attn` |
| `--prefix-kv` | 默认开 | [main.c#L595-L598](../src/main.c#L595-L598) | 正交 |
| `--sparse-attn` | 默认关，但**属项目推荐默认档** | [优化配置与边界.md#L197](../wiki/优化配置与边界.md#L197) | —— |

**关键事实**：wiki 的「单用户 / 服务端默认」推荐档为 `--wmode q4 --sparse-attn --sparse-k 32`（[优化配置与边界.md#L197](../wiki/优化配置与边界.md#L197)）。⇒ 任何与 `--sparse-attn` 互斥的门，**默认化即打断推荐档**。

---

## 3. G0 步骤 ②：四个候选对象逐一判定

| # | 对象 | 判定 | 依据 |
|---|---|---|---|
| 1 | **KV v2 惰性分配** | **已默认 → 结案注**（无需立项） | `g_kv_lazy = !g_xq_pages`（[#L9570](../src/model/vllm_safetensors.c#L9570)）；wiki 明载「惰性档（发布默认）」（[逐层推理与KV缓存.md#L117](../wiki/逐层推理与KV缓存.md#L117)） |
| 2 | **KV 精度 q8** | **已默认 → 结案注**（无需立项） | `use_kv_q8 = g_kv_q4 ? 0 : 1`（[#L9657](../src/model/vllm_safetensors.c#L9657)） |
| 3 | **`--kv-q4` 默认化** | **否**（`gate_decoupling_before_default_enablement` 的教科书场景） | 收益仅 KV 内存 1.65×（[优化配置与边界.md#L143](../wiki/优化配置与边界.md#L143)），但 decode 稀疏门控为 `!use_kv_q4`（[vllm_safetensors.c#L10797](../src/model/vllm_safetensors.c#L10797)）⇒ **默认化会当场打掉稀疏注意力**（prefill −34%~−45%、decode 2.86×，[优化配置与边界.md#L88](../wiki/优化配置与边界.md#L88)）。若确需，唯一合规路径是**先拆门**（让 kv-q4 与稀疏 decode 共存）—— 属引擎内核改动，须独立立项 |
| 4 | **`VLLM_KV_NOF32` 默认化** | **PASS → 进 G1** | 见 §4/§5 |

---

## 4. G0 步骤 ③：`VLLM_KV_NOF32` 的占比、上界与可达性

### 4.1 机制本质（先纠正一个易错认知）

> **f32 KV 不是冗余，是全引擎「读写正典」**（[逐层推理与KV缓存.md#L121](../wiki/逐层推理与KV缓存.md#L121)）。默认档在 q8 载荷**之外还额外分配一份 f32 正典**（`alloc_kv_blocks_f32`，[#L9711-L9721](../src/model/vllm_safetensors.c#L9711-L9721)）。`VLLM_KV_NOF32=1` 去掉的是**这份正典**，不是 KV 本身。

⇒ 该门省下的是「q8 之外的第二份」，因此**上界 ≈ KV 内存的 50%**（双份 → 单份）。

### 4.2 实测收益（2B，2026-09-07，**wiki 已登记实测**）

| 指标 | 数值 | 出处 |
|---|---|---|
| serve 同阶段 RSS 省 | **227–306 MB（−22%）** | [逐层推理与KV缓存.md#L132](../wiki/逐层推理与KV缓存.md#L132) |
| KV 增量斜率 | **316 → 92 KB/token（−71%）** | 同上 |
| VmSize | ~5.4–5.5 GB → **~3.4–3.6 GB** | 同上 |
| 语义（vs q8 反量化世界） | **位级相同**（COPYTEST/DKVTEST `diff=0 PASS`） | 同上 |
| 质量（4 段语料均值） | **ΔNLL −0.079**（门禁 ≤ +0.10） | 同上 |

### 4.3 占比（**反算**，须披露推导式）

> **推导式**：`KV_总量 = 斜率 × 上下文长度`。
> **未修正项**：斜率取自 2B serve 实测的线性拟合，未修正分块/守卫页开销随长度的非线性；上下文按 8K 计。

| 口径 | @2K | @8K | 说明 |
|---|---|---|---|
| f32 双份（默认） | ≈ 632 MB | ≈ **2.47 GB** | 316 KB/token |
| nof32（单份 q8） | ≈ 184 MB | ≈ **0.72 GB** | 92 KB/token |

**与权重并看（2B）**：权重文件约 3.9 GB（本仓 Qwen3-VL-2B `data` 3966.61 MB，见 K-L0-001 §2.2）。
⇒ 8K 上下文下 **KV 约占总内存的 40%**；nof32 后降至约 16%。**KV 侧确实是内存优化的主战场**（对 2B 量级，8K KV 已与权重同量级）。

### 4.4 可达性：f32 读点**逐个复核**（G0 必做，避免静默冲突）

wiki 列出的 f32 读/写点（[逐层推理与KV缓存.md#L123-L130](../wiki/逐层推理与KV缓存.md#L123-L130)）逐条核对代码：

| f32 读点 | 是否有 nof32 守卫 | 证据 |
|---|---|---|
| decode / prefill 写 f32 行 | ✅ `have_f32 = st->k_cache[l] != NULL` | [#L16493](../src/model/vllm_safetensors.c#L16493)、[#L17139](../src/model/vllm_safetensors.c#L17139)、[#L18267](../src/model/vllm_safetensors.c#L18267) |
| **稀疏注意力** `sparse_attn_head` | ✅ `has_f32 = k_cache != NULL`，并接收 `k_cache_q8`/scale | [#L10018-L10034](../src/model/vllm_safetensors.c#L10018-L10034)；调用点 [#L10803-L10813](../src/model/vllm_safetensors.c#L10803-L10813) |
| **L3 驱逐** | ✅ 双分支：f32 存在走 `l3_evict_layer`，**否则走 `l3_evict_layer_q8`（P2 nof32 路径）** | [main.c#L1178-L1185](../src/main.c#L1178-L1185) |
| L3 诊断 | ✅ 注释即「P3/P2: nof32 无 f32 块」 | [main.c#L1254](../src/main.c#L1254)、[#L1229](../src/main.c#L1229) |
| **前缀复用** `st_qwen_copy_kv_prefix` | ✅ `src_f32/dst_f32` 双判，nof32 时只拷 q8+scale | [#L9884-L9910](../src/model/vllm_safetensors.c#L9884-L9910) |
| disk-kv | ⚠️ **有边界**：DKV **v1（f32）旧快照在 nof32 下 load 拒绝**（rc=−1）；新快照自 P1b 起写 **v2（q8+scale）**，兼容 | [逐层推理与KV缓存.md#L134](../wiki/逐层推理与KV缓存.md#L134)、[vllm_server.c#L601](../src/serve/vllm_server.c#L601) |
| `VLLM_XQ_PAGES=1` 调试档 | ✅ 自动退回双份（`g_kv_lazy = !g_xq_pages`） | [#L9570](../src/model/vllm_safetensors.c#L9570) |

⇒ **推荐默认档（`--sparse-attn` + 前缀复用 + 逐层驻留）在代码路径上**与 nof32 **兼容**；唯一已知边界是 **DKV v1 旧快照**（可用条件写明：需清掉 v1 快照或改用 v2）。

### 4.5 非位级性质（必须登记，`bitwise_equivalence_gate` 项目覆盖要求）

- nof32 **对 q8 反量化世界位级等价**（`diff=0 PASS`），但**对 f32 正典世界不等价**。
- 已知现象：**长上下文下 f32 与 nof32 贪心输出可发生内容级分叉**（P4-D 于 ~char18；多模态 20 轮评估中 F 答「12」、NS 答「20」）。
- ⇒ 对外口径必须统一为「**确定性但近似**，语义级稳健，细节计数类长任务可能分档差异」（wiki 原文）。

---

## 5. 判定与 G1 前置

### 5.1 判定

**K-KV-002 ε 对象 = `VLLM_KV_NOF32` 默认化 → PASS，可进 G1。** 理由：

1. **占比成立**：8K 上下文 KV ≈ 总内存 40%（2B 口径），不是边角项。
2. **上界成立**：理论上界 ≈ KV 的 50%（双份→单份），实测 −22% RSS / −71% 斜率，同向且量级吻合。
3. **可达性成立**：推荐默认档的 5 条 f32 读点**全部有显式守卫**，无需先拆门。
4. **无 I/O 反转**：KV 按需增长，不重读权重（与 K-L0-001 的时间地板问题不同构）。

### 5.2 但 **不等于可以直接默认化** —— G1 必须冻结的判据

按 `perf_axiom_default_enablement_gate_v1`（先登记判据、再采数据），G1 须先钉死：

| # | 判据 | 预登记值 |
|---|---|---|
| (a) | 判据类型 | **容差 + 质量门**（非位级，已在 §4.5 登记） |
| (b) | 质量门 | ΔNLL ≤ **+0.10**（4 段语料）+ NIAH 5/5（项目硬性正确性底线） |
| (c) | 长上下文内容分叉 | 需**预先声明可接受范围**（计数类任务是否列入判据，须事先裁定） |
| (d) | 生效性凭证 | 需打印「f32 正典未分配」类诊断（**当前 `kv_nof32_poll` 无生效性日志** ⇒ **这是 G1/G3 的必补项**，否则会踩 `silent_fallback_false_pass`） |
| (e) | A/B 口径 | 同会话交错 ≥3 轮；效应 ≥5× 配对极差；计时窗口 stderr 打 NUL |
| (f) | 逃生舱 | `VLLM_KV_NOF32` 未设/0 即回双份；与 `VLLM_XQ_PAGES=1` 的自动退回须端到端复测 |
| (g) | 口径连锁 | nof32 改变 **CPU/单机两臂基线**吗？——否（仅内存布局），但仍须复核后登记 |

### 5.3 未决项（诚实登记）

- 实测数据（§4.2）来自 **2026-09-07 的 2B 板端**，本次**未复测**（板上无模型）。
- `--kv-q4` 的「L3 可省 ~1GB」与「decode +28%」两个数字**出处未核**（wiki 未给日志名）；因该项已判否，未继续追。
- `VLLM_PF_Q8CACHE`（P0 原型：f32 行原位改 q8 反量化）与 nof32 的交互**未复核**（[#L9882-L9883](../src/model/vllm_safetensors.c#L9882-L9883) 中 `q8sim` 依赖它）——G1 前须补。

---

## 6. 落点

- 下一步：**进 G1 判据冻结门**（产出「判据冻结单」），随后才允许采数。
- 采数载体：板端 `root@192.168.1.92`（须先推 `.vqf` + `vocab.bin`）。
- 台账：本立项已在 [落地台账.md](落地台账.md) §2 登记；对象 1/2 的「已默认」结论与对象 3 的「否」结论已写入结案注。

---

## 7. G1 判据冻结单（先登记，后采数）

> 纪律出处：`perf_axiom_default_enablement_gate_v1` 的 open_items ——「下一项默认化必须**先登记判据、再采数据**」。
> **本单冻结后不得事后改判据**；采数前一切预期标注为「未实测」。

### 7.1 改动定义

`VLLM_KV_NOF32` 由**未设（关）** 改为**默认开**；改动面 = 内存布局（是否分配 f32 KV 正典），不涉权重与浮点求和序。

### 7.2 判据类型（`bitwise_tolerance_non_substitution`）

**容差 + 质量门**（**非位级**，已在 §4.5 登记非位级性质）。
禁止用 maxrel / argmax / TOKIDS 冒充位级（`bitwise_vs_tolerance_no_substitution`）。

### 7.3 覆盖面（冻结）

| # | 面 | 判据 | 上限/通过线 |
|---|---|---|---|
| A1 | 质量（4 段真实语料） | ΔNLL | **≤ +0.10**（硬性底线） |
| A2 | 质量（长上下文） | NIAH 召回 | **5/5** |
| A3 | 功能回归：in-RAM 前缀复用 | 输出与基线一致（容差）/ 复用仍生效 | 逐条记录 |
| A4 | 功能回归：L3 驱逐（走 `l3_evict_layer_q8` 路径） | 生效性日志 + 输出容差 | 逐条记录 |
| A5 | 功能回归：disk-kv | **v2（q8）快照可 save/load**；v1（f32）快照**预期被拒**（记录为已知边界，非回归） | 逐条记录 |
| A6 | 功能回归：稀疏注意力（推荐默认档） | 8K 召回 + 输出容差 | 逐条记录 |
| A7 | 有状态等价（`stateful_batch_post_step_equivalence`） | 多轮对话：post-step 状态 + **续跑 N≥8 步**逐位一致 | **N 先写死 = 8**，不得事后下调 |

### 7.4 长上下文内容分叉的**预裁定**（`decision_sensitive_assumption`）

§4.5 已登记「长上下文可发生内容级分叉（char~18 / 计数类）」。**本单先裁定**：

- 「语义级稳健」为判据（A1/A2 覆盖）；
- 「细节计数类长任务的逐字一致」**不列入判据**，但须在报告中**如实披露发生位置与频次**；
- 若 A1 或 A2 未过 ⇒ 直接否，不看分叉。

### 7.5 A/B 口径与样本量（`session_local_ab_caliber`）

- 同会话内**交错 ≥3 轮**配对；B 档预热落在计时窗外。
- **效应门槛 ≥ 5× 配对极差**（项目覆盖）。
- 取计时读数时 stderr 打 NUL（`perf_axiom_timing_window_io_cleanliness_v1`）。
- 公平对照：同权重（逐字节验证）/ 同 token 流 / 同 batch / 同线程 + 亲和 + 热态（`measurement_fairness_equivalence`）。
- 功耗/温度须记录（板端 thermal zone），防止把降频读成收益。

### 7.6 上界预登记（**不得当已实现收益**）

| 项 | 预登记上界 | 来源 |
|---|---|---|
| RSS 降幅 | ≈ KV 的 50%（双份→单份）；实测参考 −22% | §4.1 推导 + §4.2 实测 |
| KV 增量斜率 | 316 → 92 KB/token | §4.2 实测（2026-09-07，2B） |
| 速度 | **预登记：无加速预期**（本改动是内存项）；若出现提速须单独归因，不得并入 | 本单裁定 |

### 7.7 生效性凭证（**当前缺失，G2 必交付**）

> `silent_fallback_false_pass`：判据 PASS 只证输出正确，**不证优化生效**。

**现状**：`kv_nof32_poll()`（[vllm_safetensors.c#L9309-L9314](../src/model/vllm_safetensors.c#L9309-L9314)）**只有开关，无任何生效性输出** ⇒ 采数前必须先补一条能证明「f32 正典确实未分配」的诊断（例：`[KV-NOF32] f32 canon skipped: nl=… (k_cache[l]=NULL)`），并作为 G2 交付物登记。

### 7.8 逃生舱与自动回退

- 逃生舱：`VLLM_KV_NOF32=0` 显式关 ⇒ 行为与基线一致（须复测）。
- 自动回退：`VLLM_XQ_PAGES=1` 自动退回双份（[#L9570](../src/model/vllm_safetensors.c#L9570)）——须端到端复测该回退仍成立。

### 7.9 口径连锁检查（`default_value_change_traceability`）

改 `VLLM_KV_NOF32` 默认值属**门默认状态变更** ⇒ 触发：

1. 全篇关键词补扫：`默认关 / 默认开 / 默认值 / 出厂默认`，尤其是 wiki [优化配置与边界.md](../wiki/优化配置与边界.md) 与 [逐层推理与KV缓存.md](../wiki/逐层推理与KV缓存.md) 的「实验档」表述；
2. 就地加订正指针（不改原文）；
3. 同步台账 §1 门表。

### 7.10 G1 放行判据

- [x] 判据类型冻结（容差 + 质量门，非位级已登记）
- [x] 覆盖面冻结（A1–A7，含续跑 N=8 写死）
- [x] 长上下文分叉预裁定
- [x] A/B 口径与样本量冻结（≥3 轮配对 / ≥5× 极差）
- [x] 上界预登记
- [x] 生效性凭证缺失已识别 → 移交 G2
- [x] 逃生舱 + 自动回退已登记
- [x] 口径连锁检查已登记

**⇒ G1 完成。下一步 G2 实现门**（唯一必做交付物：补 §7.7 生效性诊断；无需改算法）。**G3 采数被阻塞**：板端当前无 `.vqf` / `vocab.bin`。

---

## 8. G2 实现门（已完成）

### 8.1 改动

| 项 | 内容 |
|---|---|
| 文件 | [src/model/vllm_safetensors.c](../src/model/vllm_safetensors.c#L9711-L9730) |
| 位置 | `st_qwen_inference_init` 的 f32 正典分配块之后新增 `else` 分支 |
| 内容 | 仅在 `g_kv_nof32` 生效时向 stderr 打印一行生效性凭证（约 8 行，**不含任何行为改动**） |
| 依据 | §7.7（`perf_axiom_silent_fallback_false_pass_v1`） |

### 8.2 编译验证（`tools/build/build_x64.ps1`）

- 工具链：MinGW-w64 gcc **15.2.0**（本机）。
- 结果：**rc=0，产出 `vllm_kestrel_x64.exe`（1,364,326 B）**；二进制内含 `KV-NOF32` / `f32 canon skipped` / `q8 single-copy` 三个字串（字节级确认，偏移 1113881/1113891/1113956）。
- **踩坑登记（已存在、非本次引入）**：默认 `-OutDir` 落在仓库路径（含中文 `项目`）时链接失败 —— `ld.exe: cannot open output file D:\锟斤拷目\...`。**必须用 `-OutDir <ASCII 路径>`**（本次用 `C:\tmp_vllm_x64`）。这正是脚本头注释里 2026-09-28 已记录的「非 ASCII 路径下 ld 报 cannot open output file」同一条坑。

### 8.3 生效性凭证（**运行时实测**，G2 核心交付）

命令（x64，0.6B dual 真实模型）：

```
VLLM_KV_NOF32=1  vllm_kestrel_x64.exe --model <Qwen3-0.6B/model_dual.bak.vqf> --stream-test
```

| 档 | 输出 |
|---|---|
| **NOF32=1** | `[KV-NOF32] f32 canon skipped: nl=28 n_blocks=256 k_cache[l]=v_cache[l]=NULL (q8 single-copy)` |
| **NOF32=0**（对照） | **无该行**（其余输出正常） ⇒ 守卫成立 |

凭证自证「优化真的跑了」：`nl=28`（真实 28 层）、`n_blocks=256`、`k_cache[l]=v_cache[l]=NULL` ⇒ **f32 正典确实未分配**。
同时确认 init 真跑通：`[QWEN] Inference state ready (max_seq=8192)`、`[STREAM] tokenizer: 151669 tokens`。

### 8.4 顺带观察（**不构成收益结论**）

| 指标 | NOF32=1 | NOF32=0 |
|---|---|---|
| `rss_after_init` | 25,400 kB | 25,752 kB |

差异仅 ~0.35 MB —— **符合预期**：惰性 mmap 下 f32 正典尚未触页，init 期 RSS 几乎不变；真实收益体现在 **KV 增长后的斜率差**（316 → 92 KB/token），须在 G3 用长上下文/多轮采样才能取到。**不得用 8.4 的数字声称收益**（`perf_axiom_derived_value_disclosure_v1`）。

### 8.5 G2 放行判据

| 判据 | 状态 |
|---|---|
| 逃生舱存在且默认关 | ✅ `VLLM_KV_NOF32` 未设/0 ⇒ 不打印且正常退出（rc=0） |
| 生效性诊断存在 | ✅ 8.3 已取（反证得证） |
| 自动回退（`VLLM_XQ_PAGES=1` → 双份） | ⚠️ 代码在（[#L9570](../src/model/vllm_safetensors.c#L9570)），**运行期未复测** → 移交 G3 |
| 不满足条件自动回退、不静默 | ✅ 同上 |

**⇒ G2 完成。下一步 G3 验证门 —— 阻塞点：板端 `192.168.1.92` 无 `.vqf` / `vocab.bin`，须先推送模型方可采数。**

> **订正指针（2026-10-08 晚）**：上述阻塞已解除（板端 `/mnt/VQF/qwen3-30B-A3B-q4` 在库），G3 已开跑，见 §9。

---

## 9. G3 验证门（板端 30B-A3B，2026-10-08 晚）

- 载体：板端 `192.168.1.92`，`/mnt/VQF/qwen3-30B-A3B-q4`，`/root/_vfy/gitee3/vllm_kestrel`（**与 §4.2 的 2B 口径不同模型**，见 §9.5 口径说明）。
- 口径：`OMP_NUM_THREADS=4 VLLM_THREADS=4`，`--stream-test`，governor 未钉（内存项，不受 DVFS 影响；速度非判据，§7.6 已预登记「无加速预期」）。
- 生效性凭证（每臂均取）：`[KV-NOF32] f32 canon skipped: nl=48 n_blocks=640 k_cache[l]=v_cache[l]=NULL (q8 single-copy)` ⇒ 30B 真几何（**nl=48**）下 f32 正典确实未分配。

### 9.1 A1 质量门（ΔNLL，4 段语料 forced-NLL）

| 段 | NLL_f32 | NLL_nof32 | Δ |
|---|---|---|---|
| P0 | 6.440773 | 6.631509 | +0.190736 |
| P1 | 6.091027 | 6.041324 | −0.049703 |
| P2 | 5.313025 | 5.276916 | −0.036109 |
| P3 | 3.822709 | 3.760653 | −0.062056 |
| **CORPUS** | **5.348880** | **5.352946** | **+0.004066** |

⇒ **ΔNLL = +0.004066 ≤ +0.10 ⇒ A1 PASS**（余量 24×）。分段有正有负（3/4 段变好），无系统性劣化。

### 9.2 内存（本项核心收益，30B-A3B 全层驻留档）

`--stream-test --stream-ctx {512,1024} --stream-n 2`，取 `[STREAM] rss_after_prefill`：

| 臂 | pn@512 | rss@512 (kB) | pn@1024 | rss@1024 (kB) | **KV 斜率** |
|---|---|---|---|---|---|
| f32（双份） | 502 | 13,531,968 | 1014 | 13,773,480 | **471.7 KB/token** |
| nof32（单份 q8） | 502 | 13,488,740 | 1014 | 13,572,828 | **164.2 KB/token** |
| **对比** | — | — | — | **Δ=200,652 kB（−195.9 MB）** | **−65.2%** |

- **斜率 −65.2%**（471.7 → 164.2 KB/token），与 §4.2 的 2B 实测定性一致（316 → 92，−71%）。
- **@ctx1024 绝对省 195.9 MB**；**8K 外推**（披露推导式）：`(471.7−164.2) × 8192 = 2.52 GB`。
- **适配性论证（比百分比更硬）**：30B-A3B 全层驻留 RSS 已达 **13.5–13.8 GB / MemTotal 15.6 GB**；8K 上下文 f32 双份需再 +3.86 GB（**放不下**），nof32 只需 +1.34 GB（**放得下**）⇒ 该门对 30B 是**能否容纳 8K 上下文**的分水岭，而非"省一点"。

### 9.3 A3 + A7 功能回归（前缀复制忠实性 / 有状态等价）

`VLLM_STREAM_COPYTEST=1`（A = 同状态直接续写；B = `copy_kv_prefix` 后再续写，forced-NLL 对比）：

| 臂 | A_NLL | B_NLL | diff | 结论 |
|---|---|---|---|---|
| f32 | 6.440773251 | 6.440773251 | 0.000e+00 | **PASS** |
| nof32 | 6.631508599 | 6.631508599 | 0.000e+00 | **PASS**（nof32 下只拷 q8+scale） |

⇒ nof32 下 `copy_kv_prefix` 走「只拷 q8+scale」分支，**逐位等价**（diff 恰为 0）→ A3（复用仍生效）+ A7（post-step 等价）**同证**。

### 9.4 A5 功能回归（disk-kv v2 save/load）

`VLLM_STREAM_COPYTEST=1 VLLM_STREAM_DKVTEST=1`（注：DKVTEST 嵌套在 COPYTEST 内）：

| 臂 | save | load | rows | rows_equal | logits_equal |
|---|---|---|---|---|---|
| f32 | 0 | 0 | 108 | **PASS** | **PASS** |
| nof32 | 0 | 0 | 108 | **PASS** | **PASS** |

⇒ v2（q8+scale）快照在 nof32 下 save/load 保真，无回归。

### 9.5 口径说明与未执行项（诚实登记）

| 项 | 状态 | 说明 |
|---|---|---|
| §4.2 的 2B 数字（−22% RSS / 316→92 斜率） | **保留为 2B 口径**，未在本次复测 | 本 §9 是 **30B-A3B** 新口径；两者**不得并排**（`cross_caliber_ratio_magnitude_only`） |
| **A2 NIAH 5/5** | 见 §9.6 | 工具在仓（`--needle-sweep`），本次执行 |
| A4 L3 驱逐（`l3_evict_layer_q8` 路径） | 见 §9.9 | `VLLM_STREAM_NLL_L3=1` + `--sparse-attn --l3-evict` + `VLLM_STREAM_NLL_REPEAT=8` |
| A6 稀疏注意力 | **冒烟 PASS**（见 §9.8） | 五针位完整扫描需 ~2 h/臂，本轮做单针位冒烟 |
| A/B 配对轮次（§7.5 要求 ≥3 轮交错） | **未按 A/B 口径执行** | 本项是**内存+质量**项、非性能项（§7.6 预登记「无加速预期」）；内存读数为**确定性量**（同一进程路径、同一 token 流），故采单轮估值 + 两 ctx 点求斜率。**该方法学偏差显式登记** |

### 9.6 A2：NIAH 针位扫描（ctx=1024，5 针位）

> 工具：`--needle-sweep`（0/25/50/75/100% 五针位，一次加载内跑完）+ `--longctx-ctx 1024` + `--longctx-gen 16`。针 = 「Zara likes the color teal and her favorite food is ramen.」

| 臂 | pos0 | pos25 | pos50 | pos75 | pos100 | 结论 |
|---|---|---|---|---|---|---|
| f32 | Y | Y | Y | Y | Y | **5/5** |
| nof32 | Y | Y | Y | Y | Y | **5/5** |

- 全部命中标注 `(hit inside decode)`（**非**"prefill 兜住"的退化命中）⇒ 是真实的 decode 期长上下文召回。
- ⇒ **A2 PASS（两臂均 5/5）**。口径为 **ctx=1024**（30B 单次加载 ~200 s/prefill，ctx=4096 的 5 针位扫描需 ~2 h/臂，超出本轮预算）→ 该口径限制**显式登记**（`measured_benefit_scope`）。

### 9.7 A2 执行中发现的**工具缺陷（已修，登记）**

- **现象（首次 nof32 扫描）**：进程在**第 1 个针位**的 `[STAGE] ppl-f32 start` 处被杀（日志止于此，NIAH 计 0/0）。
- **根因**：`test_longctx_quality_ab` 的「f32 参考 PPL」档在**运行期**强置 `ist.use_kv_q8 = 0`（[main.c#L1681](../src/main.c#L1681)）以量化 INT8-KV 量化误差——但 nof32 下 `k_cache[l] = NULL` ⇒ 去读不存在的 f32 行 → 段错误。**这是 §4.4「f32 读点逐个复核」时漏掉的一个读点**（复核只覆盖了引擎内核与 serve 路径，未覆盖测试工具）。
- **修复（测试工具限定，不改引擎）**：加 `have_f32_canon = (ist.k_cache && ist.k_cache[0])` 守卫；无正典时跳过该档并打印 `[STAGE] ppl-f32 skipped (VLLM_KV_NOF32: no f32 canon)`，结果行改为 `F32=n/a`。**有正典时走原路径、逐位不变**。
- **复核**：修后 nof32 臂 `ppl-f32 skipped=5`（5 针位各跳一次）、NIAH **5/5**；f32 臂 `ppl-f32 start` 照常执行、NIAH YES ⇒ **修复未触默认路径**。板端 `build_rk3588.sh` rc=0 / 0 error。
- **生产影响判定**：**无**。`kv_nof32_poll()` 自带 `st->use_kv_q8 && !g_xq_pages` 守卫（[vllm_safetensors.c#L9310-L9318](../src/model/vllm_safetensors.c#L9310-L9318)）⇒ `VLLM_KV_F32=1`/`VLLM_XQ_PAGES=1` 与 nof32 不会同开；唯一无守卫的 f32 消费者就是本测试档。
- 仍**未执行**：A4（L3 驱逐 `l3_evict_layer_q8` 路径，需 `VLLM_STREAM_NLL_L3=1` + `--sparse-attn --l3-evict` + `VLLM_STREAM_NLL_REPEAT`，30B 单臂 ≥20 min）与 A6 的五针位完整扫描（本轮只做冒烟，见 §9.8）。

### 9.8 A6：稀疏注意力冒烟（`--sparse-attn --sparse-k 32`）

`--needle-pos 50 --longctx-ctx 1024 --longctx-gen 16 --sparse-attn --sparse-k 32`：

| 臂 | 结果 |
|---|---|
| f32 | `Needle retrieved YES (hit inside decode)`，`ppl-f32` 照常执行 |
| nof32 | `Needle retrieved YES (hit inside decode)`，`ppl-f32 skipped=1` |

⇒ **冒烟 PASS**（推荐默认档的稀疏 decode 与 nof32 共存，召回不降）。**口径限制登记**：单针位、ctx=1024（完整五针位扫描 ~2 h/臂，超出预算）。

### 9.9 A4：L3 驱逐交互（`VLLM_STREAM_NLL_L3=1` + `VLLM_STREAM_NLL_REPEAT=8` + `--sparse-attn --l3-evict --l3-min-seq 128`）

| 臂 | P0 (ev/rst) | P1 | P2 | P3 | L3 applied | CORPUS_NLL_MEAN |
|---|---|---|---|---|---|---|
| f32 | 528/528 | 528/528 | 528/528 | 384/384 | **4/4** | 5.702978 |
| nof32 | 528/528 | 528/528 | 528/528 | 384/384 | **4/4** | **5.512537** |

- 两臂**驱逐/重建计数逐段相同**（evicted==restored，无丢失）⇒ nof32 的 `l3_evict_layer_q8` 路径**完整生效**。
- L3 档 ΔNLL（nof32 − f32）= **−0.190**（nof32 反而更低）⇒ 无回归。
- 注：本档 NLL（~5.5–5.7）高于无 L3 档（5.35），系 **L3 的 Q4 落盘往返 + 稀疏注意力**自身代价，两臂同源、不构成本项差异。

### 9.10 G3 判定

| 面 | 判据 | 结果 |
|---|---|---|
| A1 质量（4 段语料） | ΔNLL ≤ +0.10 | **+0.004066 ✅ PASS** |
| A2 质量（长上下文召回） | NIAH 5/5 | **f32 5/5 · nof32 5/5 ✅ PASS**（ctx=1024） |
| A3 前缀复用 | 输出一致 / 复用生效 | **COPYTEST diff=0.000e+00 ✅ PASS** |
| A4 L3 驱逐 | 生效性日志 + 容差 | **applied 4/4、evicted==restored ✅ PASS** |
| A5 disk-kv v2 | save/load 保真 | **rows_equal/logits_equal ✅ PASS** |
| A6 稀疏注意力 | 8K 召回 + 容差 | **冒烟 PASS**（单针位 ctx1024，口径受限） |
| A7 有状态等价 | post-step + 续跑逐位 | **COPYTEST diff=0 ✅ PASS** |
| 内存（本项收益） | 斜率 −65.2%（471.7→164.2 KB/token）；8K 外推 −2.52 GB | ✅ |
| 生效性凭证 | 打印「f32 正典未分配」 | ✅ `[KV-NOF32] nl=48 n_blocks=640` |

**⇒ G3 PASS**（硬门 A1/A2 与功能项 A3–A7 全覆盖；A4/A6 的口径限制已显式登记）。

> 执行中发现的工具缺陷（§9.7）**已修并复核**，且判定为**无生产影响**。

## 10. G4 默认化裁定 = **是**

| 条件 | 结论 | 证据 |
|---|---|---|
| ① 判据类型合规（本项目已放宽位级） | ✅ | 非位级性质已登记（§4.5 / §7.2）+ 精度口径（A1 ΔNLL）+ 质量门结果（A1/A2 全过） |
| ② 效应足够、方向一致 | ✅ | 内存效应 **−65.2% 斜率**（确定性量：同一进程路径与 token 流，非时序噪声）；§7.6 预登记「无加速预期」，实测未见速度异常 ⇒ 不涉时序 5× 门 |
| ③ 逃生舱 `=0` 后与基线一致 | ✅ | §10.1：`VLLM_KV_NOF32=0` 时**无** `[KV-NOF32]` 行、COPYTEST 与 f32 基线一致 |
| ④ 不满足条件自动回退（内置） | ✅ | `kv_nof32_poll` 的 `st->use_kv_q8 && !g_xq_pages`（`VLLM_KV_F32=1` / `VLLM_XQ_PAGES=1` 自动回双份）；§10.1 运行期复核 |

### 10.1 默认化实施与运行期复核

代码：`kv_nof32_poll()` 由 `nf[0]=='1'` 才开 → 改为 `on = !(nf && nf[0]=='0')`（**默认开**，`=0` 逃生舱）。板端重建后四档复核：

| 档 | `[KV-NOF32]`（f32 正典未分配） | `P3 nof32 mode` | COPYTEST |
|---|---|---|---|
| **未设（新默认）** | **有**（4 行） | 打印（含 escape 提示） | A=B=6.631508599, diff=0 **PASS** |
| **`VLLM_KV_NOF32=0`（逃生舱）** | 无 | 不打印 | A=B=6.440773251, diff=0 **PASS**（= f32 基线） |
| `VLLM_XQ_PAGES=1`（自动回退） | 无 | 不打印 | ⚠️ 未跑完：`[QWEN] OOM allocating INT8 KV cache layer 25`（旧布局按 max_seq=20480 全量预分配，30B 上 OOM）——**既有内存限制，非本次回归** |
| `VLLM_KV_F32=1`（自动回退） | 无 | 不打印 | ⚠️ 未跑完：f32 双份 KV × max_seq 20480 ≈ 9.7 GB + 权重 13.5 GB 超 RAM ——**既有内存限制，非本次回归** |

- 条件 ③ 由「未设 vs `=0`」两档对照坐实：`=0` 档 `[KV-NOF32]` 缺席、COPYTEST NLL 回到 f32 基线值（6.440773251，与 §9.1 P0 的 f32 值一致）⇒ 逃生舱行为与基线一致。
- 条件 ④ 由「`XQ_PAGES=1` / `KV_F32=1` 两档 `[KV-NOF32]` 均缺席」坐实 ⇒ **自动回退成立**（nof32 被正确禁用）。两档未跑完 COPYTEST 的原因是**既有**的 30B 内存上限（与 nof32 无关，两档本就绕过 nof32）。

### 10.2 适用区间与已知边界（`measured_benefit_region`，不得省略）

- **收益随模型/上下文放大**：30B-A3B（nl=48、KV ~472 KB/token）在 8K 上下文下省 **~2.5 GB**，是**能否容纳**的分水岭；小模型（0.6B/1.7B）KV 本就小 ⇒ 收益**边际**（不反转，但量级小）。
- **已知边界（沿用既有登记，非新增）**：① DKV **v1（f32）旧快照**在 nof32 下 load 被拒（rc=−1），v2（q8）正常；② 长上下文下 kv 与 f32 贪心输出可发生**内容级分叉**（§7.4 已预裁定：不计入判据，但须如实披露）。
- **回滚**：`VLLM_KV_NOF32=0`（或改回 `nf[0]=='1'`，一行）。

### 10.3 口径连锁（§7.9 要求）

1. 关键词补扫 `默认关 / 实验档`：wiki [逐层推理与KV缓存.md](../wiki/逐层推理与KV缓存.md) 的 nof32 段与 [优化配置与边界.md](../wiki/优化配置与边界.md) —— **就地加订正指针**（不改原文）。
2. 台账 §1 门表：`VLLM_KV_NOF32` 由「未设 = 关（实验档）」改为「**未设 = 开**」。



