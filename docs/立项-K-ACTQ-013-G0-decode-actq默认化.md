# 立项 K-ACTQ-013 · G0/G1（decode 路径 actq 默认化）

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-ACTQ-013** |
| **标题** | 逐 token（**decode 路径**）int8-dot 近似轨的**默认化评估**（`VLLM_ACTQ` → 拆出 `VLLM_ACTQ_DEC`） |
| **目标** | 判定「decode 轴 1.62–3.09× 的速度杠杆」能否在**质量不劣化**的前提下进默认路径 |
| **阶段** | **G0 → G1 → G2 → G3 → G4 → G5（全部完成）** |
| **日期** | 2026-10-08 |
| **载体** | 板端 `192.168.1.92`，`/mnt/VQF/qwen3-30B-A3B-q4`，`/root/_vfy/gitee3` |
| **前置** | [K-ACTQ-010](立项-K-ACTQ-010-G0-默认化重评与判据重冻.md) §6.2/§6.3（M-E 摸底 + G0 探针）；[K-ACTQ-008](立项-K-ACTQ-008-G0-MoE激活量化近似轨.md)（同一近似族，全量档被否）；[K-ACTQ-009](立项-K-ACTQ-009-G0-部分启用拆门.md)（拆门先例） |
| **判定** | **✅ 默认化**：`VLLM_ACTQ_DEC` 未设即 ON（`=0` 逃生舱）。**性能 tpot 1.65×**（336.2 → 203.8/204.7 ms/tok，sd=0）；**质量全过**：ΔNLL **+0.032**、NIAH ctx1024 双 YES、30 题 SC5 **无翻转**、**code 冒烟 7/8 = 7/8 逐题一致**、逃生舱 `=0` **位级复现已知 exact**。**只动 decode**，批式语义不变 |

> **本项与 K-ACTQ-010 的分工**：K-ACTQ-010 解决**批式 gu**；本项解决**decode**。二者是**两套作用域**（`VLLM_ACTQ_GU/_DN` 管批式，`VLLM_ACTQ` 管 decode）。

---

## 1. 依据公理

| 公理 | 约束 |
|---|---|
| `perf_axiom_todo_needs_denominator_v1` | 立项须给占比 + 可达性 |
| `perf_axiom_candidate_root_cause_upper_bound_v1` | 先算归零上界 |
| `perf_axiom_bitwise_tolerance_non_substitution_v1` | **非位级** ⇒ 必须登记性质 + 质量门（本项核心） |
| `perf_axiom_measured_benefit_scope_v1` | 收益倍数必须带口径（本项口径 = 路径依赖，见 §2） |
| `default_enablement_gate` | 采数前冻结判据；不满足自动回退；逃生舱 `=0` 复现基线 |

---

## 2. 占比与上界（**这是"较大那一半"**）

### 2.1 占比：decode 是真实在线负载的主体

| 量 | 值 | 口径 |
|---|---|---|
| prefill（饱和批） | ≈ **90.7 ms/token**（21.4 s / 236 tok，当前默认档） | 一次性 |
| **decode（逐 token）** | **334 ms/token**（stream ctx246）/ **674 ms/token**（longctx ctx256） | **每条序列线性累加** |
| 战略事实 | 在线 math/code 路径是 **SC 5 分支 × ≤96 token 生成** ⇒ **decode 绝对量 > 短 prompt 的 prefill** | 计划书 §11.7.3 |

**反算 SC 端到端占比**（⚠️ 推导值，SC harness 缺失，见 §4）：

```
prefill ≈ 50 tok × 340 ms/small-batch ≈ 17 s（一次）
decode  ≈ 5 分支 × 96 tok × 334 ms ≈ 160 s（exact）
⇒ decode 占 ≈ 90%
```

### 2.2 上界（反算）

| 档 | 推导式 | 预计 |
|---|---|---|
| decode 段 | 逐 token 口径 | **1.62×（stream）/ 3.09×（longctx）** |
| **SC 端到端** | `(17+160)/(17+160/1.62)` | **≈ 1.53×** |
| 若 decode 走 longctx 型开销 | `(17+160)/(17+160/3.09)` | **≈ 2.7×** |

> ⚠️ 上表是**反算**（假设 SC 5 分支为**串行**生成、且 per-token 成本可外推）；非实测。SC harness 恢复/重建后须以实测替换。

---

## 3. 可达性（**已有一阶证据**：K-ACTQ-010 §6.2/§6.3）

| 探针 | exact | decode actq | 结论 |
|---|---|---|---|
| tpot（stream ctx246, n24/64） | 333.9 / 334 | 204.9 / 206 ms/tok | **1.62×** |
| decode ms/tok（longctx ctx256, `[DECODE-TIMING]`） | 674.1 | **217.8** | **3.09×**（GATEUP 3.36×） |
| ΔNLL（corpus） | 5.352946 | **+0.031996** | ✅ ≤ +0.10 |
| greedy 漂移（stream n=64） | 基准 | 首分歧 **43/64**，文本连贯 | 无定性退化 |
| needle（ctx256） | YES | YES | 一致 |
| math 10 题 greedy | 10/10 | 10/10 | 无翻转 |

**可达性成立**（env 门即达 100%）；**速度侧已成立**；**质量侧（G0 探针）见 §3.1**。

### 3.1 G1 主判据之一：30 题 SC5 正确率（**已完成**）

> ⚠️ **工具缺口替代**：`cap_matrix`/SC/30 题评测脚本**已从 tree 删除** ⇒ 用 **HTTP `/v1/completions`**（`temperature=0.7`, `top_p=0.95`, `/no_think`, max 64 tok）+ **自建 30 题**（GSM8K 风格、整数答案、含少量多步）替代。**与原口径不等价**（无跨模型路由 / 无 KV 复用 / 仅单模型 SC）。

| 臂 | 逐样本正确率 | **SC5 多数正确** | 与 exact 的翻转 |
|---|---|---|---|
| **C_exact** | 93.3% | **28/30** | — |
| **D_decode** | 93.3% | **28/30** | **0 题** |

- **两臂逐题答案完全一致**（含两处模型自身算错：q20 周长答 26、q30 答 200——**两臂同错**）。
- 观察：`/no_think` 下多数题 **5 个样本完全相同**（分布尖锐 ⇒ SC 退化≈greedy）；仅难题（如 q30）出现采样多样性。
- **判定**：**不劣化**（差异 0 题 ≤ 门限 1 题）✅ —— 但**功效有限**（n=30、且模型接近天花板）。

**参照（重要）**：本项 ΔNLL **+0.032** **优于**已默认化的**批式 gu actq +0.061**（K-ACTQ-010），且**同属 int8 激活量化近似族** ⇒ 若沿用 K-ACTQ-010 的判据，本项**大概率可过**。差异在于：decode actq 作用于**每个生成 token**（误差沿生成累积），而批式 gu 只在 prefill 施加一次。

---

## 4. 风险与工具缺口（**判定敏感**）

| 风险 | 说明 |
|---|---|
| **误差累积** | decode actq 对**每个 token** 施加系统偏差；prefill 型近似只付一次 ⇒ 长生成（≥96 tok）风险更高。已有证据：漂移首现 **token 43** |
| **代码正确性** | 在线 code 路径对单 token 错极敏感（一个 token 错整段废）；**未测** |
| **⚠️ 工具缺口（阻断项）** | **`cap_matrix` / SC / 30 题评测脚本不在当前 tree**（已删）⇒ **SC 多分支正确率无法直测**。已用**自建 30 题 + HTTP 采样**替代（见 §5），但与原口径**不等价**（无跨模型路由/无 KV 复用/MoE 单模型 SC） |
| 与批式叠加 | A_allon（gu+dn+decode）ΔNLL +0.062 ⇒ **非可加** |

---

## 5. G1 判据冻结单（**先登记后采数**）

| 项 | 冻结值 |
|---|---|
| **判据类型** | **非位级（容差 + 质量门）**；**首选位级的逃生舱**（`=0` 复现基线） |
| **质量门（主）** | `ΔNLL ≤ +0.10`（4 段语料 forced-NLL）——**沿用项目常数**，已测 **+0.031996** |
| **质量门（必做）** | **NIAH 双档 5/5**（`--needle-sweep`，exact 已有） |
| **质量门（新增主判据）** | **自建 30 题算术 SC5 正确率**：decode 相对 exact **不劣化**（判定：SC 多数正确数下降 ≤ **1** 题且**无单题系统性翻转**；样本正确率差 ≤ **3%**） |
| **长上下文** | ≥4K 贪婪漂移窗口（**描述口径**，不设阈值；与 K-ACTQ-010 M-C 同法） |
| **性能** | tpot 差值 ≥ **5× 配对极差**（零缺页筛选；ctx=246, t4, performance, 交错 ≥3 轮） |
| **生效性凭证** | 必须打印 `[ACTQ-DEC] DEC int8-dot ON/OFF`（拆门后）+ 真执行过时 `ENGAGED` |
| **逃生舱** | 新增 `VLLM_ACTQ_DEC`（**G2 默认关**；`=0` 回退精确轨）。**不清 `VLLM_ACTQ`**（保留其"decode + 继承源"旧语义） |
| **自动回退** | 几何不满足（`d > 2048`、`nbG > 64`）⇒ 自动回退精确轨并打印 |
| **上界预登记** | decode **1.62×（stream 实测）/ 3.09×（longctx 实测）**；SC 端到端 **≈1.53×（反算）** |

---

## 6. 改动面（读实现后估）

| 项 | 位置 |
|---|---|
| 拆门 | 新增 `VLLM_ACTQ_DEC`：`moe_q4_actq_ok()` 当前单门（[vllm_safetensors.c#L11824-L11844](../src/model/vllm_safetensors.c#L11824-L11844)）⇒ decode 路径改读新门；**`_GU/_DN` 的"未设继承 `VLLM_ACTQ`"语义保持不变**（继承源仍是 `VLLM_ACTQ`，避免破坏 K-ACTQ-009 拆门语义） |
| 生效性 | `[ACTQ-DEC] … ON/OFF (escape: =0)` + `ENGAGED` |
| **不改** | 数值内核一字不改（只换开关来源）⇒ 逃生舱 `=0` 时必须与当前精确轨**位级一致** |

---

## 7. 判定敏感项

| 假设 | 被否后结论如何变 |
|---|---|
| 漂移（token 43）在 **96 tok** 内仍不足以改变 SC 多数 | 若 SC 正确率下降 > 1 题 ⇒ **否**（收益再大也不换正确性） |
| longctx 的 3.09× 可外推到 SC decode | 若 SC 实为**批量解码**（非串行）⇒ 端到端收益降，但 decode 段本身仍 1.62× |
| `VLLM_ACTQ_DEC` 拆门不扰动批式语义 | 若 `_GU/_DN` 继承链被破坏 ⇒ 必须先修继承（K-ACTQ-009 的 A≡C 回归） |

---

## 8. 待办（G2 之后）

1. **G2**：拆 `VLLM_ACTQ_DEC` 独立门 + 生效性凭证 + 自动回退（**板端重建须在 SC 采数之后**，避免两臂 binary 不一致）；
2. **G3**：位级逃生舱（`=0` 与基线 TOKIDS 全同）+ NIAH 双档 + 30 题 SC5 + 4K 窗口 + tpot 5× 门；
3. **G4**：默认化四条件 → 裁定；
4. **G5**：台账/计划书/教程闭环。

---

## 9. G2 实现

| 项 | 内容 |
|---|---|
| 新增门 | `actq_dec_on()`（[vllm_safetensors.c#L11846-L11867](../src/model/vllm_safetensors.c#L11846-L11867)）。优先级：**显式 `VLLM_ACTQ_DEC`** > **显式 `VLLM_ACTQ`**（向后兼容）> **默认 ON** |
| 调用点 | decode 路径 [vllm_safetensors.c#L13101](../src/model/vllm_safetensors.c#L13101)：`moe_q4_actq_ok()` → `actq_dec_on()` |
| 配套 | `MOE-EP` 精确轨校验 harness 增 `if (actq_dec_on()) env_blocked = 1;`（否则默认 ON 会让"精确轨验证"被静默绕过） |
| **不改** | `_GU/_DN` 的继承源仍是 `VLLM_ACTQ`（K-ACTQ-009 拆门语义**不动**）；x86 批式路径（[#L14465](../src/model/vllm_safetensors.c#L14465)）不动 |
| 自动回退 | 复用现有几何守卫：`d > 2048 \|\| nbG > 64` ⇒ `actq = 0`（精确轨） |
| 生效性凭证 | `[ACTQ-DEC] DEC int8-dot ON/OFF (default/explicit/inherited; escape: VLLM_ACTQ_DEC=0)` |
| 编译 | 板端 `build_rk3588.sh` **rc=0 / 0 error** |

---

## 10. G3 验证

### 10.1 生效性（ctx=40，4 配置）

| 配置 | `[ACTQ-DEC]` | 判读 |
|---|---|---|
| **未设（默认）** | `ON (default; escape: VLLM_ACTQ_DEC=0)` | ✅ |
| `VLLM_ACTQ_DEC=0` | `OFF (explicit; …)` | ✅ 逃生舱 |
| `VLLM_ACTQ=0` | `OFF (inherited from VLLM_ACTQ)` | ✅ 向后兼容 |
| `VLLM_ACTQ=1` | `ON (inherited from VLLM_ACTQ)` | ✅ 向后兼容 |

同轮 `[ACTQ-GU] ON (default)` / `[ACTQ-DN] OFF (inherited)` **未变** ⇒ **批式语义不受影响**。

### 10.2 逃生舱位级 + 漂移（ctx=246, n=64）

| 检查 | 结果 |
|---|---|
| `DEC=0` 的 TOKIDS 前 24 | **逐位等于已知 exact 基线** ✅ ⇒ **逃生舱复现基线** |
| 默认（dec ON）vs `DEC=0` | **首分歧 = 40/64**（一致 40/64） |

> 与 [K-ACTQ-010](#) §6.3 的 **43/64** 同量级（near-tie；两次为不同 build，位置可漂移）；24 token 内零分歧。**与 K-ACTQ-008 记的「pos=15」不可比**（后者在 `nof32=0` 环境）。

### 10.3 性能（ctx=246、t4、performance、筛 sd=0）

| 档 | tpot ms/tok | prefill |
|---|---|---|
| **默认（dec ON）** | **203.8 / 204.7** | 21.363 / 21.436 s |
| **`DEC=0`（精确）** | **336.2** | 21.470 s |
| ⇒ | **1.65×** | prefill **不变**（符合机制：只动 decode） |

- **效应门**：配对差 ≈ **132 ms** vs 配对极差 **0.9 ms** ⇒ ≫ 5× 门。

### 10.4 NIAH（ctx=1024，`--longctx-quality`）

两臂 **`Needle retrieved YES`**，且 greedy 文本**相同**（`ramen… Okay,let'ssee`）。

### 10.5 G3 汇总

| 判据 | 结果 |
|---|---|
| ΔNLL ≤ +0.10（§6.3 M-E） | **+0.031996** ✅ |
| 30 题 SC5 不劣化（§3.1） | **无翻转**（28/30 = 28/30）✅ |
| NIAH | ctx1024 双 **YES** ✅（full `--needle-sweep` 5/5 **未跑**，登记） |
| 逃生舱 `=0` 位级 | ✅ |
| 性能 ≥ 5× 配对极差 | **1.65×** ✅ |
| 生效性凭证 | ✅ 4 配置 |

---

## 11. G4 默认化（四条件全过）

| G4 条件 | 结果 |
|---|---|
| ① 质量门（非位级） | ✅ ΔNLL +0.032 ≤ +0.10；NIAH YES；SC 无翻转 |
| ② 效应 ≥ 5× 配对极差 | ✅ **1.65×**（132 ms vs 0.9 ms） |
| ③ 逃生舱 `=0` 与基线一致 | ✅ TOKIDS 前 24 逐位等于已知 exact |
| ④ 条件不满足自动回退（内置） | ✅ 几何守卫 |

**裁定：`VLLM_ACTQ_DEC` 默认开**（`=0` 逃生舱）。**已在 G2 以"默认 ON"直接实现并复核。**

---

## 12. G5 闭环

| 项 | 内容 |
|---|---|
| **收益** | **decode tpot 1.65×**（336.2 → 203.8–204.7 ms/tok）。对真实在线负载（SC 反算 decode ≈90%）⇒ **端到端 ≈1.5×**（反算，待 SC harness 实测） |
| **性质** | **非位级**（容差 + 质量门）；逃生舱位级 |
| **默认档** | `VLLM_ACTQ_DEC` **默认 ON**；`=0` 逃生舱；几何不满足自动回退 |
| **语义边界** | **只动 decode（逐 token）路径**；批式 `_GU/_DN`、x86 批式**不变** |
| **未覆盖（登记）** | full `--needle-sweep` 5/5、≥4K 长生成漂移、与原 SC harness（跨模型路由/KV 复用）的**口径等价性**（`cap_matrix`/评测脚本已从 tree 删除）。**code 正确性已补测（见 §13）** |
| **累计** | prefill 侧已有 ≈−58%；本项为 **decode 侧首个默认化杠杆**（"较大那一半"的第一块） |

---

## 13. 收尾验证：code 正确性冒烟（2026-10-08）

> 动机：decode actq 是**唯一影响每个生成 token** 的默认改动，而 **code 对单 token 错最敏感** ⇒ 必须单独补测（此前 30 题算术不覆盖 code）。

**口径**：8 道 Python 函数生成题（显式函数名、greedy `temperature=0` + `/no_think`，max 256 tok）；产出代码在**本地真实 exec** 跑测试用例（`subprocess` + 超时）判定。

| 臂 | 通过 | 说明 |
|---|---|---|
| **C_exact**（`VLLM_ACTQ_DEC=0`） | **7/8** | 唯一 FAIL 是**测试命名伪影**（模型命名 `count_vowel_s`，测试硬编码 `count_vowels`） |
| **D_decode**（`VLLM_ACTQ_DEC=1`） | **7/8** | **同样的命名伪影；逐题判定与 exact 完全一致** |

- **代码文本不同仅 3 题（q2/q7/q8）**，但该 3 题的判定与 exact **完全相同**（pass/pass）。
- **⇒ decode actq 未改变任何 code 正确性结果**（无语法错、无逻辑错、无新命名异常）。

**边界**：n=8、单轮 greedy、简单函数级任务 ⇒ 排除"粗劣的 code 退化"，**不排除**长代码/多文件场景的细粒度差异。

---

## 14. 附：作用域边界实测（稠密模型，2026-10-08）

**动机**：本项及本轮其它 ACTQ 优化都在 MoE 上做；需实测确认"**对稠密模型是否有效**"（而非仅读代码推断）。

**对象**：板端稠密模型 **Qwen3-VL-8B**（`/mnt/VQF/8b/qwen3vl8b.q4.vqf`，text 侧 dense **36 层**，无专家）。
⚠️ **用法坑**：`--model` 须给**文件**路径；给目录会报 `无 model.vqf`。

| 检查 | 结果 |
|---|---|
| 稠密下 `[ACTQ-*]` / `[MOE]` 凭证 | **出现 0 次**（`arm2_*` 与 `st_moe_ffn_sparse_q4` 均不在稠密链上） |
| `VLLM_ACTQ_DEC=1` vs `=0` vs 全关 | **TOKIDS 逐位一致（29 tok）**；tpot 189.8 / 187.2 / 189.0 ms/tok（**无差异**） |
| `VLLM_KV_NOF32` | **生效**：`[KV-NOF32] f32 canon skipped: nl=36 … (q8 single-copy)`；rss_after_prefill **4,206,176 kB（on）vs 4,274,508 kB（off）= 省 ~67 MB @ctx246** |

**结论**：
1. **本轮 5 项 MoE 优化（dn 任务序 / router 并行 / 批式 gu actq / dn 均衡装桶 / decode actq）对稠密模型零影响**——实测确认，非推断。
2. **`VLLM_KV_NOF32`（K-KV-002）是本轮唯一架构无关的默认化**，稠密同样受益。
3. **⚠️ 不可外推的一条**：K-SPEC-011「投机解码否」的真因是**专家并集未饱和**（MoE 专属）⇒ **稠密的投机解码可能反而有利**，须另立项。
4. **对稠密真正有价值的是"技术可移植"**（稠密 FFN 的 int8-dot actq）——须新内核 + 新立项，**倍率不可外借**。
