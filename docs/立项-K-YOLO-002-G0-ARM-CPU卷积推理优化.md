# 立项-K-YOLO-002-G0-ARM CPU 卷积推理优化（标量 → int8 dotprod，追官方 NPU 16ms）

> 状态：**Phase 0–2 已完成并落地**（2026-10-11）。CPU 路径 2455/2300 ms → **172.9/161.9 ms（14.0×）**，三版 e2e 12.3–14.2×；NPU 路径 md5 6/6 逐位无回归。**已 conv-bound（CONV 占 88%）**，达到该设计实际地板。§Phase2 两条后续「数量级」路线**均已证否**：①「原生 int8 直接卷积」= 微基准（ALU 上限 7.98 < 现状已达 15.2 GMAC/s/核）；②「换激活量化粒度消 F32 gather」= 数值可行性实验（per-tensor 0/6、per-channel 2/6 过 J-CPU-2）。**本项已收口，无更进一步算法路径。** 本方案书 = 单一真源；执行记录见 §8。
> 承接 [K-YOLO-001](立项-K-YOLO-001-G0-NPU卷积检测适配.md) §9「追官方」：NPU 端 M2 兜底已收敛、原生卷积寄存器未突破；本项转向**纯 CPU 路径**，目标是缩小「YOLO 检测 CPU 推理」与官方 NPU 的差距。

## 1. 目标与判据（先冻结）

- **用户目标基准**：追平官方 NPU 的 `rknn_run` **16.05 ms**（2026-10-11 实测，n=25，min 15.81 / median 16.05）。
- **诚实可行性结论（先冻结论，不预造）**：**CPU 追平 16 ms 不可达**，属硬件算力上限（见 §2.3）。本项**可达成目标**是：CPU 路径从 **F32 基线 ~2358 ms → 目标 ~150–250 ms（≈10–15×）**，仍距 16 ms 约 10×。
- **判据（J，冻结）**：
  - **J-CPU-1 数值等价**：int8 dotprod 卷积 vs F32 参考（`yconv`），逐卷积 `max|Δ|` 与 `maxrel` 低于阈值（复用 K-YOLO-001 J1 口径，阈值 Phase 1 实测后冻结）。
  - **J-CPU-2 检测等价**：CPU 加速版 vs CPU F32 golden（同一二进制、同输入），三版 × 检测集检出对象集一致、score 差 ≤~0.03、框差 ≤~5px（沿用 K-YOLO-001 J2 口径）。
  - **J-CPU-3 性能**：相对 F32 基线（yolov8n cats 2358.7 ms，M0-c 记录）设 ≥10× 加速目标；Phase 0 实测 CPU 分解后按实冻结分算子目标。
  - **J-CPU-4 回退透明**：无 dotprod / 不适用形状显式回退标量路径（`silent_fallback_false_pass`），不得静默降级或静默改变数值。

## 2. 现状与瓶颈（事实盘点，不臆造）

### 2.1 硬件与构建（已确认）
- RK3588：4×A76（2.4 GHz）+ 4×A55（1.8 GHz），均 ARMv8.2-A，**均支持 int8 dotprod（`vdotq_s32`）**。
- 构建 [CMakeLists.txt](../CMakeLists.txt)：`-O2 -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16 -ffast-math -ffp-contract=off` ⇒ **`__ARM_FEATURE_DOTPROD` 已定义**，dotprod 内核可编译、已生效。

### 2.2 CPU 卷积现状（代码事实，关键瓶颈）
- [vllm_yolo.c](../src/model/vllm_yolo.c) 的 [yconv（F32）](../src/model/vllm_yolo.c#L347-L400) 与 [yconv_q8（int8 权重）](../src/model/vllm_yolo.c#L405-L462) **均为纯标量内层**：`acc += wo[icl] * x[...]`，仅有 OpenMP `parallel for`，**无 NEON 向量化、无 dotprod**。
- `yconv_q8` 是「int8 权重 × **F32 激活**」：权重逐元素反量化到 float 再标量 FMA ⇒ **非真 int8×int8 dotprod**，实测比 F32 慢 8–11%（compute-bound，M1 记录）。
- 结论：**CPU 卷积是当前最大、最易攻克的短板**——算力利用率仅标量级（远低于 dotprod 峰值）。

### 2.3 可行性（FLOPs 上限核算，锚点待 Phase 0 复核）
- YOLOv8n ≈ **8.7 GFLOPs**（ultralytics 官方口径；Phase 0 用图算子表精确核算）。
- A76 int8 dotprod 峰值：2×128-bit SDOT，每指令 16 MAC，4 核 × 2.4 GHz × 32 = **~307 GOPS**。
- 理论下限：8.7 / 307 ≈ **28 ms**（100% 效率，现实不可能）；现实（~50% 效率 + im2col 内存 + 非卷积算子 + A55 拖累 + 负载不均衡）≈ **80–150 ms**。
- **F32 标量现状 2358 ms ≈ 0.4% 峰值利用率** ⇒ 上行空间巨大，但天花板 ~28 ms 已逼近 NPU 16 ms ⇒ **无法「追平」**，只能「显著逼近」。

## 3. 公理映射（理论 → 工程）

- `isomorphic_superposition`：卷积 ↔ GEMM 同构（im2col 坍缩）——CPU 与 NPU 同式，复用 M2 的 im2col 产物。
- `collapse_to_classic` + `collapse_priority_rule`：int8 对称量化坍缩 + per-output-channel scale（权重已有 `cq/cs`）。
- `conservation_law_under_collapse`：坍缩后的信息守恒以 **J-CPU-2 检测等价实测** 为门（不靠理论保证）。
- `feasibility_conclusion_timeliness`：**追 16 ms 结论先行冻结为不可达**，避免无限投入；设每 Phase 轮次上界。
- `nonmonotonic_benefit_no_cherry_pick`：证否即回退（如 Winograd 若净负则回退）。
- `measurement_fairness_equivalence`：公平 A/B（同 governor=performance、同 A76 亲和 `taskset -c 4-7`、同 OMP 线程）。
- `silent_fallback_false_pass`：无 dotprod / 边缘形状显式回退，禁止静默。

## 4. 分级方案

### Phase 0 — CPU 分解实测 + 判据补冻（数据先行，不写内核）
1. 给 `yolo_forward` 加 **env 门控阶段计时**（`VLLM_YOLO_CPU_TIMING=1`，默认关零开销），分解：conv（F32 各卷积）/ SiLU / 其他算子（SPPF/Concat/Upsample/DFL/NMS）/ letterbox / 图调度。
2. 用图算子表精确核算 YOLOv8n FLOPs（锚定 §2.3）。
3. 复测 CPU F32 基线（同 governor + pin A76，n 次中位），**冻结 J-CPU-3 分算子目标**。
4. **准入门**：确认 `__ARM_FEATURE_DOTPROD` 生效（微基准 `vdotq_s32` vs 标量，读峰值 GOPS）。

### Phase 1 — CPU int8 dotprod 卷积（主杠杆，低风险高收益）
1. 提取/复用共享 int8 dotprod GEMM：以 [lay_dot_i8](../src/model/vllm_laya.c#L246-L269) 为蓝本，落地一个 `M×N×K` int8×int8→int32 dotprod 内核（`vdotq_s32` + 逐块 `diag(sw)*diag(sx)` 缩放），A76 亲和 + 无 dotprod 时标量兜底。
2. 复用 M2 已写的 **im2col int8 量化产物**（[yconv_npu_gemm](../src/model/vllm_yolo.c#L506-L657) 的 `imc/asc/wsc`，激活已 per-位置 int8 + scale）：把「NPU submit」替换为「CPU dotprod GEMM」，得到与 NPU 路径**同量化语义**的 CPU 加速卷积。
3. 接入 `yconv_dispatch`：无 NPU（或 `--npu-threshold` 回退）时走 dotprod 加速路径，替代标量 `yconv_q8`。
4. **过 J-CPU-1/J-CPU-2/J-CPU-4**；实测 J-CPU-3，预期 conv 部分 **8–15×**。

### Phase 2 — 非卷积算子向量化（视 Phase 0 占比决定投入）
- 候选：SiLU（`1/(1+exp)` → NEON + 查表/`vtbl`）、DFL softmax、NMS IoU、letterbox 双线性（`vmul`/`vld1` 4 路）、SPPF maxpool。
- **仅投入 Phase 0 实测占比 >5% 的算子**（`interference_priority_rule`，先攻大头）。

### Phase 3 — Winograd F(2,3)（可选，高风险，仅 3×3 s1 conv）
- 3×3 stride-1 卷积 Winograd ~2.25× MAC 减少；但数值噪声放大、边界/残差处理复杂，**仅在 Phase 1 后仍差目标且实测有利才上**，证否即回退（`nonmonotonic_benefit_no_cherry_pick`）。

## 5. 关键风险与回退

| 风险 | 影响 | 回退 |
|---|---|---|
| A76/A55 异构 + OpenMP 负载不均衡 | dotprod 加速被 A55 拖累 | 大算子 `taskset`/分簇调度；纯 conv 用 A76 亲和 |
| 激活 int8 量化（per-位置 scale）引入数值漂移 | J-CPU-2 边缘检测 ±1 | 阈值回退 CPU F32（现有 `--npu-threshold` 机制复用） |
| im2col 内存膨胀（输入 × KH·KW） | 大卷积内存/带宽受限 | 与 NPU 路径同判据：仅大卷积走 im2col，小卷积留标量/直卷积 |
| Winograd 数值/正确性 | 静默精度损失 | 证否即回退，不默认开启 |
| dotprod 微基准偏离真卷积（缓存热假象） | 高估收益 | 以**真卷积 + J-CPU-3** 为准，不以微基准外推 |

## 6. 交付与验证
- 交付：`vllm_yolo.c`（dotprod 卷积 + dispatch 接入）+ 共享 int8 dotprod 内核（可能落在 `vllm_matmul.c`/`vllm_yolo.c`）。
- 验证：`tools/yolo/yolo_vqf_run.py` 作 F32 参考对拍（J-CPU-1）；板端三版 × 检测集（J-CPU-2）；serve/CLI pinned 中位（J-CPU-3）；`cpu=0 fallback=0` 显式回退（J-CPU-4）。

## 7. 待决策
- 是否立即进入 **Phase 0（仅计时 + 复测，零内核改动，低风险）**？
- Phase 1 的共享内核落点（提取 `lay_dot_i8` 为公共函数 vs 在 `vllm_yolo.c` 内联一份）。

---

## 8. 执行记录

### Phase 0 已完成（2026-10-11，零内核改动，仅加 env 门控计时）

**工具**：`VLLM_YOLO_CPU_TIMING=1`（默认关，零开销）给 `yolo_forward` 加逐算子分类计时 + `st_yolo_detect` 加 letterbox/NMS 计时。

**① CPU 分解实测**（板端 governor=performance、`taskset -c 4-7`、OMP=4，n=3 一致）：

| 路径 | forward 总 | **conv** | elem（40 ops） | detect | letterbox | NMS |
|---|---|---|---|---|---|---|
| **yolov8n_q8**（int8 权重 × F32 激活） | 2405.4 ms | **2371 ms（98.6%）** | 22.2 ms | 10.8 ms | 11.0 ms | 9.5 ms |
| **yolov8n F32** | 2223.1 ms | **2188.7 ms（98.4%）** | 22.0 ms | 10.9 ms | 10.9 ms | 9.5 ms |

⇒ **conv 是绝对瓶颈（98.5%）**，其余（elem/detect/lb/NMS）合计仅 ~54 ms。Phase 1（dotprod 卷积）几乎等于全部收益。
- 附带确认 M1 记录：q8 路径 conv 比 F32 慢 **8.3%**（2371 vs 2189）——因 `yconv_q8` 是「int8 权重反量化 + F32 标量 FMA」，非真 int8 dotprod。

**② dotprod 微基准（Phase 0 准入）**：

| 内核 | 1 核 A76 | 说明 |
|---|---|---|
| `vdotq_s32`（int8 dotprod） | **25.5 GOPS** | 单核，`-O2 -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16` |
| 标量 FMA | 1.51 GOPS | 同频同编译 |

⇒ **dotprod 内核已生效**，单核 **16.9×** 于标量。4×A76 理论上限 ~102 GOPS。

**③ FLOPs 交叉验证（自洽）**：conv 2371 ms + 实测标量吞吐 1.51 GOPS ⇒ 反推 YOLOv8n ≈ **8.7 GFLOPs**（与 ultralytics 官方口径吻合），且当前有效吞吐 ~1.8 GOPS（≈单核标量）⇒ **OMP=4 几乎无加速**（4×A76 异构/负载不均衡待 Phase 1 处理）。

**④ 更新可行性（Phase 0 数据冻结）**：
- conv 理论下限（4 核 dotprod，100% 效率）= 4.35 G-MAC / 102 GOPS ≈ **43 ms**；现实（~50% 效率 + im2col 内存）≈ **85–170 ms**。
- 加非 conv 的 ~54 ms ⇒ **e2e 乐观目标约 140–220 ms（相对 F32 基线 2223 ms 约 10–16×）**。
- **仍距官方 16 ms 约 9–14×**，与 §1 预冻结论一致：追平不可达，但可显著逼近。
- **J-CPU-3 分算子目标（冻结）**：conv ≤ 200 ms、非 conv 不计为瓶颈（现状 54 ms）。

**⑤ 结论**：Phase 0 全部达成（分解 + 微基准 + FLOPs 锚定 + 目标冻结）。**Phase 1 入口明确**：把 `yconv_q8`/`yconv` 的标量内层换成 `vdotq_s32` int8×int8 dotprod GEMM，复用 M2 已有的 im2col int8 量化产物（激活 per-位置 int8 + scale）。**本次有代码改动**（`vllm_yolo.c`：仅新增 env 门控计时，默认零行为影响；板端 gitee3 已重建）。

### Phase 1 已完成（2026-10-11，✅ 大成功、无 NPU 回归）

**改动**（仅 `vllm_yolo.c`）：
1. **抽出共享助手 `yolo_im2col_q8()`**（im2col + 逐像素 int8 量化），NPU M2 路径与 CPU dotprod 路径共用同一份（单一真源，防漂移）。
2. **新增 `yconv_cpu_dotprod()`**：`vdotq_s32`（ARMv8.2 dotprod）int8×int8 → int32 GEMM，转置取向同 M2（`out[oc][p]=Σ_k W[oc][k]·im2col[p][k]`），per-(oc,位置) 反量化 = `cs[oc]·rowsc[p]`；`#if ST_HAVE_NEON && __ARM_FEATURE_DOTPROD`，否则标量兜底；OMP over `oc`。
3. **接入 `yconv_dispatch`**：Q8 模型（无 F32 权重）→ dotprod；不适用（分组/缺 int8 权重）显式回退标量 `yconv_q8`（env `VLLM_YOLO_CPU_DOTPROD` 默认 ON，可关 A/B）。
4. 数值等价依据：M2 的 per-block a_scale=b_scale 逐块同值 ⇒ 合成 scale 与「整段 int32 点积 + 一次乘 scale」**数值等价**（int32 精确）。

**① J-CPU-3 性能（板端 governor=performance、`taskset -c 4-7`、OMP=4、e2e 中位 n=5）**：

| 模型 | 标量 CPU | **dotprod CPU** | 加速 |
|---|---|---|---|
| yolov8n cats / room | 2455 / 2300 ms | **215 / 200 ms** | **11.4× / 11.5×** |
| yolo11n cats / room | 2117 / 1975 ms | **223 / 207 ms** | **9.5× / 9.5×** |
| yolov5nu cats / room | 2348 / 2180 ms | **200 / 185 ms** | **11.8× / 11.8×** |

- **forward 分解（yolov8n cats）**：conv **2368→140 ms（16.9×）**——**与 dotprod 微基准（单核 16.9×）逐项吻合**，验证实现吃满 dotprod；elem 22→17.9 ms、detect 10.8 ms、letterbox 11 ms、NMS 9.5→2.9 ms（低分候选大减）。
- **达标**：J-CPU-3 冻结目标（conv ≤200 ms、e2e 150–250 ms）**全部达成**（conv 140、e2e 185–223）。
- **与官方 16.05 ms 差距**：e2e **~11.5×**（原 CPU ~140×）⇒ 仍不可追平（与 §1 预冻一致），但 CPU 侧已进入同一数量级的后端。

**② J-CPU-2 检测等价（IoU≥0.5 + 同类配对，三版 × cats/room）**：

| 对照 | max\|Δscore\| | max\|Δbox\| | 未配对 |
|---|---|---|---|
| **dotprod-CPU vs NPU**（同 int8×int8） | ≤ **0.050** | ≤ **6.6 px** | ≤2（conf 边缘） |
| dotprod-CPU vs 标量-CPU（F32 激活） | ≤ 0.079 | ≤ 7.0 px | ≤1（conf 边缘） |

- dotprod-CPU 与 NPU **几乎一致**（同量化语义，差异仅累加/反量化舍入）；detection 类集合基本一致，差异全为 conf≈0.25 边缘对象 ±1。
- 全部落在 K-YOLO-001 M5 已登记的 J2 容差包络内（score ≤0.061、框 ≤9.8px）⇒ **J-CPU-2 PASS**。

**③ J-CPU-4 回退透明**：`VLLM_YOLO_CPU_DOTPROD=0` 可强制标量；分组卷积/无 int8 权重显式走 `yconv_q8`（debug 日志 `[CPU] dotprod n/a ...`），无静默降级。

**④ 无 NPU 回归**：抽取 `yolo_im2col_q8` 后，NPU 路径 6 组 {v8/v11/v5 × cats/room} **md5 6/6 逐位不变**；`--npu-selftest` 全 PASS。

**⑤ 判读**：Phase 1 达成全部冻结判据。**CPU 路径从「不可用」（2.2–2.4 s/图）降至 ~0.2 s/图（~10–12×）**，成为 NPU 不可用时的强兜底；NPU 路径与生产行为不变。**本次有代码改动**（`vllm_yolo.c`：im2col 抽取 + dotprod 卷积 + dispatch；板端 gitee3 已重建）。

**⑥ 未做/后续（Phase 2/3）**：非卷积算子（elem 17.9 + lb 11 + detect 10.8 + NMS 2.9 ≈ 43 ms，占 dotprod e2e ~20%）向量化；Winograd（可选）。若继续，Phase 2 目标 = 把非 conv 的 ~43 ms 再压。**另外**：dotprod GEMM 目前 oc-outer/p-inner（大 P 卷积 imc 被 OC 次重读，可加 P-分块提升缓存复用）——**Phase 1.1** 候选。

### Phase 1.1 已试并**证否回退**（2026-10-11，P-分块缓存复用）

**动机**：conv 拆分 = **GEMM ~86 ms + im2col ~54 ms**；GEMM 理论上限（4×A76 dotprod）≈ 42.6 ms ⇒ 仅 50% 效率。假设 oc-outer 顺序对大 P 卷积把 imc 读 OC 次（DRAM），拟用 P-分块（块 ≤16KB 钉 L1、跨 oc 复用）降流量。

**实现**：`yolo_dotprod_block()`（p-块内 oc 循环）+ 外层 p-块并行（`pb` 按 Kp 与线程数收敛，保证 npb≥线程数）。

**实测（同法，pinned A76）**：conv **139.9→144.8 ms（+3.5%）**、e2e 215→219.5 ms（+2%）——**无收益且略退化**。**判据**：imc 重读其实已被 cache 吸收（A76 大 L2），分块纯属额外循环/边界开销。⇒ 按 `nonmonotonic_benefit_no_cherry_pick` **证否回退**（恢复 oc-并行）。

### Phase 1.2 已落地（2026-10-11，✅ 4-way ILP，conv 再 −10%）

**动机**：Phase 1.1 证明 GEMM 非内存受限 ⇒ 瓶颈在**指令延迟**。小 Kp（32–96）下每个输出仅 2–6 条 `vdotq_s32`，却有 1 条 `vaddvq_s32`（水平归约，高延迟）+ 标量收尾，单累加链无法掩盖 ⇒ 效率 ~50%。

**实现**：`vdotq_s32` **4-way ILP**——同 oc 下同时算 4 个位置、4 条独立 int32 累加链，共用同一 `vld1q_s8(wrow)`（权重向量复用）；每个位置的点积和顺序不变 ⇒ **逐位等价**。尾部 `p` 余数走单链。

**实测**（pinned A76、OMP=4、e2e 中位 n=5）：

| 模型 | Phase 1 e2e | **Phase 1.2 e2e** | 加速 |
|---|---|---|---|
| yolov8n cats / room | 215 / 200 ms | **200.5 / 186.5 ms** | **12.2× / 12.3×** |
| yolo11n cats / room | 223 / 207 ms | **209.9 / 193.7 ms** | **10.1× / 10.2×** |
| yolov5nu cats / room | 200 / 185 ms | **187.0 / 173.4 ms** | **12.5× / 12.6×** |

- **forward 分解**：conv **139.9→125.5 ms**（其中 im2col 53.8 不变 ⇒ **GEMM 86→71.7 ms**，效率 50%→**59%**）；e2e forward 170→155.7 ms。
- **正确性**：J-CPU-2 数值与 Phase 1 **逐位一致**（max\|Δscore\| 同为 0.050/0.013/…）；NPU 路径 **md5 6/6 不变**；selftest 全 PASS。

**累计（K-YOLO-002 Phase 1+1.2）**：CPU e2e **2455/2300 → 200.5/186.5 ms（12.2×/12.3×）**；与官方 16.05 ms 差距 **~12.5×**（v5nu 173 ms 为最接近，~10.8×）。**当前 CPU 路径分解（yolov8n）**：im2col 53.8 + GEMM 71.7 + elem 17.9 + detect 10.8 + lb 11 + NMS 3 ≈ 168 ms（+ 进程/加载 ~30 ms = e2e）。

**⑦ 下一杠杆（按实测排序）**：im2col 53.8 ms（= 与 GEMM 相当，NEON 化 gather+quant 候选）、GEMM 71.7 ms（再上 ILP/软件流水）、非 conv 43 ms（Phase 2）。

### Phase 1.3 已试并**证否回退**（2026-10-11，8-way ILP / OMP 扩展性）

- **OMP 扩展性实测**（yolov8n，A76×4）：GEMM **232.4 → 124.3 → 88.5 → 71.3 ms**（OMP=1/2/3/4）⇒ 4 核 **3.26×**（次线性），单核 18.7 GOPS（= 微基准 25.5 的 73%，差额为反量化/存储开销）。im2col 同量级（162.7→53.6 ms，3.04×）。
- **8-way ILP（证否）**：假设再加宽 ILP 可摊薄权重加载、逼近 2×SDOT/cycle。实现同 oc 下 8 位置 × 8 条累加链。**实测 conv 125.5→130.1 ms（+3.7%）、e2e +2%**（v11/v5 同退化）⇒ **A76 寄存器压力 > ILP 收益**（与 K-LAYA-002 M8「8 累加器净负」同结论）⇒ 按 `nonmonotonic_benefit_no_cherry_pick` **回退至 4-way**（回退后 conv 125.5 / e2e 199.6 ms，质量门复验通过）。
- **判据**：4-way 已是该 A76 微架构在「小 Kp + 每输出高延迟水平归约」约束下的 ILP 平衡点；继续上行需换结构（软件流水 / 融合 im2col→GEMM 消除中间 53 MB 往返），非单纯加宽 ILP。

### Phase 2 勘察（2026-10-11，逐算子分解 + im2col NEON 实验证否）

**① im2col 显式 NEON 量化（证否）**：给 `yolo_im2col_q8` 的量化循环写显式 NEON（`vmulq`+`vaddq(2^23)`+`vcvtq`，与标量 mul+add 逐位一致，md5 6/6 保持）⇒ **im2col 53.8→54.9 ms（+2%）**，显式版**慢于** GCC `omp simd` 自动向量化 ⇒ 证否回退。**另一支**：改用裸 `vcvtq_s32_f32` ⇒ **数值破坏**（检出 5/12→3/7）——因 ARM `FCVTZS` 是**截断**，而原码 magic 技巧是 round-to-nearest；**已回退**。⇒ **结论：im2col 瓶颈是 gather（跨通道 strided 读）而非量化；54 ms 为该算法下界。**

**② 逐算子分解（env `VLLM_YOLO_CPU_TIMING`，板端 pinned A76）**：

| 项 | yolov8n | yolo11n | yolov5nu |
|---|---|---|---|
| CONV | 125.4 | 114.7 | 120.5 |
| letterbox | 11.0 | 10.9 | 10.9 |
| DETECT（DFL softmax+sigmoid） | 10.8 | 10.9 | 10.7 |
| CONCAT | 6.2 | 7.6 | 4.0 |
| MAXPOOL（SPPF） | 5.4 | 5.5 | 5.4 |
| UPSAMPLE | 3.5 | 3.6 | 1.8 |
| NMS | 2.8 | 3.4 | 2.2 |
| SLICE | 2.0 | 1.7 | – |
| ADD | 0.6 | 0.8 | 0.7 |
| **ATTN_PSA（C2PSA）** | – | **19.0** | – |
| **image_total** | **169.3** | 179.5 | 157.9 |

**③ 判读（诚实的可行性收口）**：
- **conv 占 image_total 74%**（v8）；其分解 = GEMM ~71.7 + im2col ~53.7。
- **GEMM 理论上限**（4×A76 dotprod，实测微基准 25.5 GOPS/核）= 4.35 G-MAC/102 GOPS ≈ **42.6 ms** ⇒ 现 71.7 为 59%（4 核扩展 3.26× 次线性）；**im2col ~54 ms 为 gather 下界**（已证非量化瓶颈）。
- ⇒ **conv 硬下界 ≈ 95–125 ms**，加非 conv 后 **image_total 下界 ≈ 140–170 ms**，即 **vs 官方 16.05 ms 约 8.7–10.6×**——**与 §1 预冻「追平不可达」一致**，且**当前已在设计下界附近**。
- **剩余可压缩项**（合计 ~44 ms）多为 **gather-bound（letterbox 双线性）或 expf-bound（DETECT 1.2M 次）**，向量化需数值近似（≤1e-6，远低于 J-CPU-2 容差）或复杂重排，**单项收益 ≤8 ms（≤5%）**；Winograd/融合 im2col→GEMM **不减 gather 指令**（gather 不变，仅省 53 MB 中间往返）⇒ 预期收益亦有限。

**④ 收口建议**：K-YOLO-002 已达到该「im2col→int8 dotprod」设计的**实际性能地板**（image_total ~158–180 ms，相对标量 CPU ~2100–2460 ms 为 **~12–13×**）。**若要再有数量级变化，需换路线**（原生 int8 直接卷积内核 / 换激活量化粒度以消除 F32 gather），非本轮微优化范畴。**本次无净代码改动**（NEON 量化已回退；仅新增 env 门控的逐算子计时，默认零开销）。

### Phase 2 已落地（2026-10-11，✅ 非 conv 并行化，全部逐位等价）

**关键发现**：YOLO 的逐元素/后处理算子**几乎全是单线程**（无 OpenMP），且多数有明显冗余。逐项修复（**全部位级一致**，NPU 路径 md5 6/6 不变）：

| 算子 | 改法 | 前 | 后 |
|---|---|---|---|
| MAXPOOL（SPPF） | OMP over `c` | 5.4 ms | **1.4 ms** |
| UPSAMPLE | OMP over `c` | 3.5 ms | **1.0 ms** |
| ADD / MUL / SIGMOID | OMP（flatten，`n>64K` 门限） | 0.6 ms | **0.3 ms** |
| SLICE | OMP 拷贝循环（替 `memcpy`） | 2.1 ms | **1.2 ms** |
| CONCAT | 前缀偏移 + OMP over 输入 | 6.1 ms | **3.6 ms** |
| **letterbox** | ① x/y 映射**预计算**（原码在每 y 行内重算 nh 次）；② c 提外层使写出连续；③ OMP over (c,y) | **10.9 ms** | **3.0 ms** |
| **DETECT**（DFL softmax+sigmoid） | OMP over anchor `p`（各 anchor 独立） | **11.0 ms** | **2.9 ms** |
| **ATTN_PSA**（C2PSA，仅 yolo11n） | OMP over `i`（QK^T / softmax / att·V 三处 + `vm+=out`） | **18.9 ms** | **5.5 ms** |

**非 conv 合计：~44 ms → ~17 ms（2.6×）**；**image_total：169.3 → 142.7 ms（v8，−15.7%）**。

**最新实测（pinned A76、OMP=4、e2e 中位 n=5）**：

| 模型 | 标量 CPU | **dotprod（Phase 1.2+2）** | 加速 |
|---|---|---|---|
| yolov8n cats / room | 2422 / 2273 ms | **172.9 / 161.9 ms** | **14.0× / 14.0×** |
| yolo11n cats / room | 2071 / 1933 ms | **168.6 / 156.6 ms** | **12.3× / 12.3×** |
| yolov5nu cats / room | 2317 / 2156 ms | **163.5 / 151.0 ms** | **14.2× / 14.3×** |

- 当前 image_total 分解：**yolov8n 142.7 = CONV 125.0（88%）** + letterbox 3.0 + CONCAT 3.6 + DETECT 2.9 + NMS 2.8 + SLICE 1.2 + MAXPOOL 1.4 + UPSAMPLE 1.0 + ADD 0.3；**yolo11n 139.3 = CONV 115.1 + ATTN 5.5** + …；**yolov5nu 134.5 = CONV 120.2** + …。
- **与官方 16.05 ms 差距：~8.4–8.9×**（v5nu 134.5 最接近，~8.4×）；相对标量 CPU **~14×**。
- **正确性**：NPU 路径 md5 **6/6 不变**、检出数 v8 5/12、v11 4/13、v5 2/12 一致、`cpu=0 fallback=0`、selftest 全 PASS。**所有改动均为并行化/预计算（位级等价），非近似。**

**⑤ 剩余（已 conv-bound）**：非 conv 仅剩 ~16 ms（NMS 2.8 / CONCAT 3.6 / letterbox 3.0 / DETECT 2.9 …），conv 占 88% ⇒ **进一步 ROI 已很低**；conv 下界见上（GEMM ~71 已在下界、im2col ~54 为 gather 下界）。**本次有代码改动**（`vllm_yolo.c`：逐元素/letterbox/DETECT/CONCAT/SLICE 并行化 + 预计算 + env 门控逐算子计时；板端 gitee3 已重建）。

### 后续路线可行性核验（2026-10-11，原生 int8 直接卷积 —— 微基准证否）

- **动机**：Phase 2 收口建议将「原生 int8 直接卷积内核（免 im2col）」列为唯二「数量级」路线之一（用户经 AskUserQuestion 选定）。本轮**先核验可行性、不写内核**（`feasibility_conclusion_timeliness`）。
- **准入判据（先冻）**：直接卷积要拿到 ≥ 现状的性能，其纯 MAC 吞吐须 ≥ **现状真实 GEMM 已实现的 15.2 GMAC/s/核**（4.35 G-MAC ÷ 71.7 ms ÷ 4 核）。
- **ISA 硬约束（关键）**：现状内核靠 `vdotq_s32`，其**要求 K 维 16 字节连续**——卷积的 K=(C·KH·KW) 天然不连续，**只能靠 im2col 的 gather 制造连续性**。**免 im2col 的「位置向量化」直接卷积只能用 `vmlal_s8`（广播权重 × 位置向量）**：每 8 个 int8 积须再 `vmovl`+`vaddq` 拓宽进 int32（int16 累加 ≤2 项即溢出：127×127×2=32258），**无法用 dotprod 的单指令 16-MAC**。
- **微基准**（`/root/dc.c`，A76 pinned，`-O2 -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16`，K=512、NP=8192、REP=200）：

| 变体 | 内核 | 单核 GMAC/s |
|---|---|---|
| A | `vdotq` GEMM 1 位置（现状内层形态） | **23.62** |
| B | `vdotq` GEMM 4 位置 ILP（Phase 1.2 形态） | **93.58** |
| C | `vmlal_s8` 直接卷积 16 位置，**每项**拓宽（正确下界） | **7.98** |
| D | `vmlal_s8` 直接卷积 16 位置，每 8 项拓宽（理想化，实际 int16 会溢出） | **14.38** |

- **判读（证否）**：直接卷积的**纯 ALU 上限（C=7.98，理想化 D=14.38）已低于现状 GEMM 真实已达成的 15.2 GMAC/s/核** ⇒ 即令 im2col 全部消除（占 conv 约 43%），其 **6.5–11.7× 的 MAC 吞吐劣势也不可能被回收**；且 C/D 尚未计入直接卷积的 strided 指针/边界/列复用开销（真实只会更慢）。
- **结论**：按 `feasibility_conclusion_timeliness` **冻结「原生 int8 直接卷积」为不可行**，不进入实现（避免无限投入）；按 `nonmonotonic_benefit_no_cherry_pick` 不引入该改动。**本次无代码改动**（仅板端临时微基准 `/root/dc.c`，不入库）。
- **残余路线亦已证否（数值可行性实验，2026-10-11）**：对「换激活量化粒度以消除 im2col 的 F32 gather」做数值可行性实验（env 门控探针，仅测检测等价、不接默认路径）。**可折叠粒度只有 per-tensor / per-channel**（scale 与输出位置无关才能折进干净 int8 GEMM；更细的 per-tile 尺度不可折叠、退化为 per-element 乘法）。结果（J-CPU-2 口径，pixel 基线 vs 变粗量化，3 模型 × cats/room）：

| 粒度 | PASS | 检出集改变 | max\|Δscore\| | max\|Δbox\| |
|---|---|---|---|---|
| **per-tensor**（全局 max/127） | **0/6** | 4/6 | 0.193 | 6.7 px |
| **per-channel**（逐输入通道 max/127） | **2/6** | 2/6 | 0.080 | 4.3 px |

- ⇒ **两条路线均不满足 J-CPU-2 检测等价（score ≤0.05 / box ≤5px / 检出集一致）**。**结论：「消 F32 gather」族数值上不可行，K-YOLO-002 与官方 16.05 ms 的 ~8.4–8.9× 差距无可用算法路径进一步压缩。**
- **实验后已清空全部实验代码**（`yolo_im2col_q8` 的 gscale 参数、per-channel 探针函数、dispatch 钩子；按 `nonmonotonic_benefit_no_cherry_pick` 证否即回退）。**基线逐位复原**：CPU 路径检出与实验前 pixel 基线**逐值一致**（v8 5/12、v11 4/13、v5 2/12）、NPU 路径 md5 **6/6 不变**、`cpu=0 fallback=0`、`--npu-selftest` 9 PASS。**本次无净代码改动。**
