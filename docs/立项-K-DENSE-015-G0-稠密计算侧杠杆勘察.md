# 立项 K-DENSE-015 · G0（稠密计算侧杠杆勘察）

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-DENSE-015** |
| **标题** | 稠密模型（Qwen3-VL-8B）**计算侧**是否还有可落地杠杆——带宽 vs 算力判定 |
| **目标** | 在 [K-DENSE-014](立项-K-DENSE-014-G0-稠密FFN-int8dot近似轨.md) 订正（**稠密 FFN 已 int8-dot**）之后，回答"稠密 FFN 占 decode 72.2% / prefill 76.2%，还能不能优化" |
| **阶段** | **G0 勘察（零代码，用内建探针）⇒ 判定 否** |
| **日期** | 2026-10-09 |
| **载体** | 板端 `192.168.1.92`，`/mnt/VQF/8b/qwen3vl8b.q4.vqf`（text 侧 dense 36 层，d=4096、ffn=12288），t4，governor=performance |
| **前置** | [K-DENSE-014 订正块](立项-K-DENSE-014-G0-稠密FFN-int8dot近似轨.md)（FFN 已 int8-dot、`st_q4_row_dot` 系 x86 专有） |
| **判定** | **否：稠密两阶段均受 DRAM 带宽约束，计算侧无空间** |

---

## 1. 依据公理

| 公理 | 约束 |
|---|---|
| `perf_axiom_todo_needs_denominator_v1` | 立项须给占比 + 可达性 |
| `perf_axiom_candidate_root_cause_upper_bound_v1` | 先算"归零上界"；上界解释不了差距 ⇒ 否掉 |
| `perf_axiom_counter_microbench_for_subresolution_v1` | 读数与被测同量级时，用"计数 × 微基准" |
| `perf_axiom_measured_benefit_scope_v1` | 收益倍率必须带口径 |

---

## 2. 方法与工具（零代码，全部为**已内建**探针）

| 工具 | 作用 | 依据 |
|---|---|---|
| `VLLM_Q4_PROBE=1` | C tile worker **纯读**（跳过 vdot/vcvt/vfma，读到的字节累加进 sink 防 DCE）＝同访问模式的**纯内存读地板** | [vllm_safetensors.c L4441](../src/model/vllm_safetensors.c#L4441) |
| `VLLM_Q4_CTILE=1` | `q4x4_gemm_batched` **强制走 C tile**（跳过 asm 快路） | [L4404](../src/model/vllm_safetensors.c#L4404) |

> 关键点：decode（n_batch=1）**本来就落在 C tile 路径**（asm 快路要求 `n_batch>3`）⇒ `VLLM_Q4_PROBE=1` 直接给出 decode 的纯读地板，**计算占比 = (实测 − 地板)/实测**。prefill（n_batch=236）走 asm，probe 对其无效（实测已证：见下表 A ≈ B 的 prefill），故 prefill 用 `CTILE` 档做对照。

---

## 3. 实测（ctx=246，n=8，t4，performance，预热后单轮）

| 档位 | prefill (ms) | decode (ms/tok) | sd | 凭证 |
|---|---|---|---|---|
| **A** 默认 | **11602.7** | **191** | 0 | — |
| **B** `Q4_PROBE=1`（decode 纯读） | 11706.6 | **188** | 1792 | `[Q4PROBE] engaged` |
| **C** `Q4_CTILE=1`（全 C tile） | 36410.6 | 201 | 0 | `[Q4CTILE] engaged` |
| **D** `Q4_CTILE=1 Q4_PROBE=1`（全纯读） | 23205.2 | 180 | 2120 | 两凭证 |

**判读**：
- **decode**：`A 191` vs `B 188` ⇒ **计算占比 = (191−188)/191 = +1.6%** ⇒ **decode 是纯带宽地板**——把 vdot/vcvt/vfma 全部删掉只省 1.6%。
- **prefill**：`A 11602.7`（生产 asm）vs `C 36410.6`（强制 C tile）= **asm 快 3.14×** ⇒ **生产已在最快已知内核上**。
- **内建对照自证**：`A` 与 `B` 的 prefill 几乎相同（11602.7 vs 11706.6）⇒ probe 确实只作用于 C-tile（decode），prefill 走 asm 未被扰动，实验装置有效。

---

## 4. 反算（**须披露推导式；标注假设**）

| 量 | 推导 | 结果 |
|---|---|---|
| prefill FFN 权重字节 | 36 层 × (2×12288×4096 + 4096×12288) × 4.5 bit/8 | ≈ **3.06 GB** |
| prefill FFN 权重流量 | 真 GEMM 按 token tile 重读 ⇒ ⌈236/4⌉ ≈ **59 次**（**假设 tile=4**） | ≈ **180 GB** |
| prefill 有效带宽 | 180 GB / (11.60 s × 0.762 FFN 占比 ≈ 8.84 s) | ≈ **20.5 GB/s** |

⇒ prefill FFN 权重流量已逼近 **RK3588 LPDDR4x 可用带宽屋顶**（~20–25 GB/s 量级）⇒ **prefill 亦以带宽为主约束**（与 decode 同构）。
⚠️ 该行为**反算**（依赖 tile=4 假设），非直测；但方向与 decode 的 1.6% 直测**一致**。

---

## 5. 归零上界与可达性

| 阶段 | 归零上界 | 可达性 | 结论 |
|---|---|---|---|
| **decode** | 计算侧全部归零 = 191/188 = **1.016×**（≈0） | 无计算可去 | **否** |
| **prefill** | 生产已用 asm（3.14× 于 C tile） | 无更快已知内核 | **否（低概率待证）** |

---

## 6. 判定与边界

**判定：否——稠密 8B 的 decode 与 prefill 均受 DRAM 带宽约束，计算侧（int8-dot / ILP / 任务序）无可落地空间。**

- 与项目既有认知一致：`project_memory` 记「8B Q4 受限于 DRAM 带宽（~12–15 GB/s），FFN 占比约 72%」——本项在**新载体（Qwen3-VL-8B）上以直测复现该结论**。
- 这也解释了为何 MoE 侧的 5 项优化对稠密零影响（K-ACTQ-013 §14）后，**稠密侧一直没有等价杠杆**：MoE 的收益来自"只读 6% 专家权重"的**稀疏性省字节**，稠密**没有这个结构性优势**（每 token 必读全部权重）。

**未覆盖（登记，非本项结论）**：
1. **IO / 字节轴**：更激进量化（Q4→更低 bit）受**质量红线**约束（`project_memory`：Q2_0 在 32 元素单 scale 下乱码）⇒ 属质量门下的独立立项，非计算侧；
2. **分层驻留 / 流式**：属 **K-L0 系列**（每层有效块动态加载 / 热度驻留），是"减少重复读"的轴，非本项；
3. **prefill kernel 级余量**：本项只证明"已用最快已知内核"，**未定量**其与 int8 屋顶的差距（如需可用 asm 版纯读探针，需新增插桩）。

**口径**：板端 Qwen3-VL-8B（`qwen3vl8b.q4.vqf`）、ctx=246、n=8、t4、governor=performance、单轮读数（无 sd 缺页）。
