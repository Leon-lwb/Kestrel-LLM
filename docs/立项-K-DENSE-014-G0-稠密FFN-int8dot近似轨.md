# 立项 K-DENSE-014 · G0/G1（稠密 FFN int8-dot 近似轨）

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-DENSE-014** |
| **标题** | 稠密模型 FFN（gate/up + down）的 **Q4×f32 精确轨 → int8-dot 近似轨** 移植 |
| **目标** | 判定「MoE 上已默认化的 actq（激活 int8 量化 + `vdotq_s32`）技术」能否移植到**稠密** FFN，且在质量不劣化下进默认路径 |
| **阶段** | **G0 立项（占比/上界/可达性）+ G1 判据冻结；G2–G4 待执行** |
| **日期** | 2026-10-08 |
| **载体** | 板端 `192.168.1.92`，稠密模型 `/mnt/VQF/8b/qwen3vl8b.q4.vqf`（Qwen3-VL-8B，text 侧 dense **36 层**，ffn=12288） |
| **前置** | [K-ACTQ-013](立项-K-ACTQ-013-G0-decode-actq默认化.md)（MoE decode actq 默认化，**同源技术**）；[K-ACTQ-013 §14](立项-K-ACTQ-013-G0-decode-actq默认化.md)（**作用域边界实测**：确认 MoE actq 对稠密零影响 ⇒ 稠密是独立空白） |
| **判定** | 待 G2–G4 |

---

> ## 📌 就地订正（2026-10-09，G0 复核）—— 判定改为「**否：目标优化已存在**」
>
> **订正对象**：上表「判定」行；§3「技术可移植性」表稠密列；§4/§5/§6 全部随前提失效。
>
> **结论**：§3 断言"稠密 FFN 走 `st_q4_row_dot`（Q4×f32 精确轨）"**不成立**。
> - `st_q4_row_dot` 的**唯一定义**在 [vllm_safetensors.c L449](../src/model/vllm_safetensors.c#L449) 的 `#if ST_ARCH_X86 && !ST_HAVE_NEON` 块（L286–L2288）内 —— 它是 **x86 标量回落轨，ARM 上根本不参与编译**；
> - 同名函数 `dyn_matvec_q4_q8_fused_gate_up_batched_neon` 在 x86（[L1688](../src/model/vllm_safetensors.c#L1688)）与 ARM（[L6846](../src/model/vllm_safetensors.c#L6846)）各有一份，§3 误取了 **x86 那份**的 `st_q4_row_dot`；
> - **ARM 稠密 FFN 的真实实现**（[L6846](../src/model/vllm_safetensors.c#L6846) / [L6926](../src/model/vllm_safetensors.c#L6926)）第一步即 `quantize_row_q8_0_act`（激活 → int8），worker（[vllm_q4_gu_b_worker L6805](../src/model/vllm_safetensors.c#L6805) / [vllm_q4_dn_b_worker L6892](../src/model/vllm_safetensors.c#L6892)）用 `i8x16_dot_s32`（= `vdotq_s32`）—— **与 MoE actq 同形态的 int8-dot 近似轨**。
>
> **板端取证（2026-10-09，`/mnt/VQF/8b/qwen3vl8b.q4.vqf`，`VLLM_DENSE_PROBE=1`，t4）**：
> ```
> [DENSE-PROBE] proj=gate_up phase=prefill path=arm-neon-int8 repack=1 n_batch=236 rows=12288 cols=4096  act=int8 vdotq_s32=1 m4f=0
> [DENSE-PROBE] proj=down    phase=prefill path=arm-neon-int8 repack=1 n_batch=236 rows=4096  cols=12288 act=int8 vdotq_s32=1 m4f=0
> [DENSE-PROBE] proj=gate_up phase=decode  path=arm-neon-int8 repack=1 n_batch=1   rows=12288 cols=4096  act=int8 vdotq_s32=1 m4f=0
> [DENSE-PROBE] proj=down    phase=decode  path=arm-neon-int8 repack=1 n_batch=1   rows=4096  cols=12288 act=int8 vdotq_s32=1 m4f=0
> ```
> prefill（n_batch=236）与 decode（n_batch=1）**双阶段**均落在 int8-dot；`m4f=0` 证明未走 f32 的 M4f GEMV 快路（`VLLM_SPEC_M4F_DECODE` 默认关）。
>
> **连带作废**：§4 可达上界（1.40× / 2.2×）、§5 可达性（"改动面 = 新增内核"）、§6 判据（ΔNLL/NIAH/code + 逃生舱 `VLLM_DENSE_ACTQ`）——前提不成立，**G2–G5 取消，本次不产生任何生产代码改动**。
>
> **对 K-ACTQ-013 §14 的再解释**："MoE actq 对稠密零影响"**不是因为稠密走精确轨**，而是**稠密 FFN 本来就在 int8 轨上**——两个结论不冲突，此前的因果归因需按本订正理解（见 [落地台账](./落地台账.md)）。
>
> **可复现凭证**：`[DENSE-PROBE]` 探针已常驻（`VLLM_DENSE_PROBE=1`，默认关，缓存的 env 检查、零生产影响），作为本判定的取证钩子。

---

## 1. 依据公理

| 公理 | 约束 |
|---|---|
| `perf_axiom_todo_needs_denominator_v1` | 立项须给占比 + 可达性 |
| `perf_axiom_candidate_root_cause_upper_bound_v1` | 先算归零上界 |
| `perf_axiom_bitwise_tolerance_non_substitution_v1` | **非位级** ⇒ 质量门（沿用 actq 族 +0.10/NIAH） |
| `perf_axiom_measured_benefit_scope_v1` | 收益倍率必须带口径；**MoE 倍率不可外借** |
| `perf_axiom_external_borrow_requires_own_breakeven_v1` | 外部/跨架构经验须自算盈亏平衡 |

---

## 2. 对象与占比（板端实测，2026-10-08）

稠密 Qwen3-VL-8B（`qwen3vl8b.q4.vqf`，text 侧 dense 36 层），t4，governor=performance：

| 段 | 读数 | FFN 占比 |
|---|---|---|
| **decode**（ctx256，`[DECODE-KERNELS]`） | 187.39 ms/tok；GATEUP 47.8% + **DOWN 24.4%** | **FFN = 72.2%**（QKV 12.2% / O 8.2% / LM 7.5%） |
| **prefill**（ctx246，`[PREFILL-KERNELS]`） | 12394.6 ms；GATEUP 49.3% + **DOWN 26.9%** | **FFN = 76.2%** |

⇒ **FFN 是稠密的绝对瓶颈**（decode 72% / prefill 76%），且当前走**精确轨**（见 §3）。

---

## 3. 技术可移植性（读实现，关键结论）

| 项 | MoE（已默认化 actq） | **稠密（本项）** |
|---|---|---|
| FFN 内核 | `st_moe_ffn_sparse_q4` → actq 用 int8 激活 + `vdotq_s32` | `dyn_matvec_q4_q8_fused_gate_up_batched_neon` → [st_q4_row_dot](../src/model/vllm_safetensors.c#L449-L482)（**Q4 权重 nibble × f32 激活，标量**） |
| decode 门 | `actq_dec_on()` | **无**（稠密链上无任何 actq） |
| 同源性 | — | 同为「Q4 权重 nibble × 激活」⇒ **技术同源可移植** |

> ⚠️ **本表稠密列已于 2026-10-09 被证伪**（见文首「就地订正」）：稠密 ARM FFN 实际走 `quantize_row_q8_0_act` + `i8x16_dot_s32`（int8-dot），并非 `st_q4_row_dot`（x86 专有）。下表保留原文以留痕。

> 命名澄清：稠密内核名里带 `q4_q8` 是**历史命名**——`st_q4_row_dot` 的 `x` 是 `const float *`（**f32 激活精确轨**），不是 int8 点积。⇒ **稠密 FFN 目前没有 int8-dot 近似轨，这正是本项的空间。**

---

## 4. 可达上界（**反算，须披露推导式；MoE 倍率仅作参照**）

| 档 | 推导式 | 预计 |
|---|---|---|
| **decode** | FFN 72.2% 走 actq；若达 MoE 同源 decode **1.65×** ⇒ `1 / (0.278 + 0.722/1.65)` | **≈ 1.40×** |
| **prefill** | FFN 76.2% 走 actq；若达 MoE 同源 gu **3.6×** ⇒ `1 / (0.238 + 0.762/3.6)` | **≈ 2.2×** |

> ⚠️ **口径披露（必读）**：
> - 上表是**反算上界**，MoE 倍率是**参照值、不可外借**（`measured_benefit_scope`）——稠密 FFN 是**单一 dense GEMM**（ffn=12288、d=4096），几何与 MoE（每专家 ef=768）不同，**真实倍率须稠密内实测**；
> - 稠密 decode 是 **n_batch=1 的窄矩阵**（1×4096 × 4096×12288），MoE decode 是 top-k 8 专家宽 GEMM——**窄矩阵对 int8-dot 的收益可能更低**（延迟/带宽受限更甚）；
> - 故 §4 的 1.40×/2.2× 是**乐观上界**，实际可能显著低于此。

---

## 5. 可达性（改动面）

| 项 | 内容 |
|---|---|
| 性质 | **算法替换型**（新增 int8-dot 内核），非"消瓶颈型"开关 |
| 改动面 | 新增稠密 FFN 的 int8 激活量化 + vdot 内核（gate/up + down，ARM NEON）；**比 MoE actq（已有内核、仅拆门）改动面大** |
| 门 | 拟新增 `VLLM_DENSE_ACTQ`（G2 默认关，`=0` 逃生舱回退精确轨） |
| 参考 | MoE actq 的量化/解包内核（`quantize_row_q8_0_act` + `actq_tab`）可复用思想 |

---

## 6. G1 判据冻结单（**先登记后采数**）

| 项 | 冻结值 |
|---|---|
| **判据类型** | **非位级（容差 + 质量门）**——算法替换，非任务映射 |
| **质量门（主）** | `ΔNLL ≤ +0.10`（4 段语料 forced-NLL，`VLLM_STREAM_NLL=1`；沿用项目常数） |
| **质量门（必做）** | **NIAH 5/5**（`--needle-sweep`） |
| **质量门（必做）** | **code 正确性冒烟**（同 K-ACTQ-013 §13：8 题 Python 函数，本地真实 exec） |
| **性能** | decode tpot + prefill 分段：差值 ≥ **5× 配对极差**（t4，performance，交错 ≥3 轮，零缺页） |
| **生效性凭证** | `[DENSE-ACTQ] … ON/OFF (escape: VLLM_DENSE_ACTQ=0)` + 真执行过 `ENGAGED` |
| **逃生舱** | `VLLM_DENSE_ACTQ`（G2 默认关；`=0` 回退精确轨 `st_q4_row_dot`） |
| **自动回退** | 几何不满足（如 `cols%32!=0`、`mode!=1` 4x4 布局、非 aarch64 dotprod）⇒ 回退精确轨并打印 |
| **上界预登记** | decode **≈1.40×**、prefill **≈2.2×**（**反算，不得当已实现**） |

---

## 7. 判定敏感项

| 假设 | 被否后结论如何变 |
|---|---|
| 稠密 decode 窄矩阵 int8-dot 收益接近 MoE 1.65× | 若窄矩阵收益低（延迟受限）⇒ decode 收益大幅缩水，可能只赚 prefill |
| 稠密 FFN 是**带宽受限**（权重 16.45GB 逐 token 读） | 若带宽受限，int8-dot **不省权重读** ⇒ 收益主要来自 prefill 的激活复用，decode 可能**零收益**（与 K-SPEC-011"专家并集"同构） |
| 质量门可过 | actq 族在 MoE 上 ΔNLL +0.032/+0.061；稠密 FFN 无 MoE router 误差 ⇒ **预期同量级**，但须实测 |

> ⚠️ **最可能的翻转点（先登记）**：稠密 decode 是**带宽受限的窄矩阵**，int8-dot 只省**激活侧**的计算、**不省权重读取**——若权重读是瓶颈，**decode 可能零收益甚至负收益**（MoE decode 的 1.65× 得益于"top-8 专家只读 6% 权重"的稀疏性，稠密没有这个优势）。**这就是必须稠密内实测、不可外借倍率的根因。**

---

## 8. 待办（G2 之后）

1. **G2**：新增稠密 FFN int8-dot 内核（gate/up + down，ARM NEON vdot）+ `VLLM_DENSE_ACTQ` 门 + 生效性 + 自动回退；
2. **G3**：质量门（ΔNLL / NIAH / code）+ 性能（decode tpot + prefill 分段，5× 门）+ 逃生舱位级；
3. **G4**：默认化四条件 → 裁定；
4. **G5**：台账/计划书/教程闭环。
