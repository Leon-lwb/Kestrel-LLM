# 立项-K-YOLO-001-G0-YOLO 目标检测适配（CPU 基线 → RK3588 NPU 卷积推理）

> 状态：**✅ 达标关闭（2026-10-11）**——K-YOLO-001 全里程碑完成：**M0**（自有格式 VQF v4 转换器 → C 引擎加载/前向 → CLI → serve `POST /v1/detect` → 板端部署 → CPU 基线）、**M1**（int8 权重：质量不退化、体积 −75%）、**M2**（im2col→GEMM 上 NPU + 最小提交调度 + im2col 单遍化 + C2PSA 卷积下推）、**M3**（Phase-1 ✅ 打通 DIRECT 原生卷积 `max|Δ|=0`；Phase-2 证否、保守关闭）、**M5**（三版 × 9 真图全量验证）。**判据 J1–J5 全部通过**：J1 数值（转换门 ≤5.1e-6 / C 引擎 1.9–3.5e-6 / NPU 逐卷积 maxabs 0.02–0.9）；J2 检测等价（golden v8 66/68、v11 67/71；板端 CPU vs NPU 242/247 配对，score 差 ≤0.061、框差 ≤9.8px，仅 conf 边缘 ±1）；J3 三版 × 9 图；J4 回退显式无静默降级；**J5 加速 v8 6.16–7.37×、v5 5.74–6.63×、v11 5.04–6.49×（均 ≥3×）**。**⚠️ 方法学**：NPU 计时须先固定 `devfreq=performance`（否则严重偏低）。**v5 = YOLOv5u（用户已决策，见 §8）**。详见 §7 M5。另完成**官方 RKNN 路线对照**（§7「可选 A/B」：同模型同尺寸，官方 `rknn_run` 15.9 ms vs 本 DIRECT 端到端 365–387 ms，信息性、不进交付链）。
> 单一真源本文件；配套 [落地台账.md](落地台账.md)；上游 [K-BENCH-020](K-BENCH-020-板端多模态推理-Qwen3VL-2B与8B本引擎.md)（视觉/NPU 现状）、[vllm_npu.h](../include/npu/vllm_npu.h)（DIRECT 驱动契约）。
> 口径：先冻结判据再采数；位级与容差不互换；单一真源台账；NPU 失败**显式**回退，不静默降级。

---

## 1. 目标与判据（先冻结）

**问题**：引擎目前无目标检测能力，且 NPU 只标定了 int8/int4 **GEMM**（1×1 卷积 FC 模式）数据通路；YOLO 是**卷积**网络，其 3×3/stride/pad 卷积数据通路在 DIRECT 驱动上**完全未标定**。

**目标**：
1. **功能**：YOLOv5 / YOLOv8 / YOLO11 三版目标检测在引擎 **serve 端点**可用（输入图片 → 输出检测框 JSON）。
2. **加速**：卷积推理走**自研 DIRECT 驱动**上 RK3588 NPU（保持"零第三方依赖 + 不转 `.rknn` 模型"的既有红线，不引入 `librknnrt`）。
3. **质量**：检测结果（类别 / 置信度 / 框坐标）与 CPU golden **等价**，不因量化坍缩或 NPU 数据通路差异而退化。
4. **格式红线（用户约束）**：**部署/运行时格式必须是引擎自有格式（VQF 家族）**——**不引入 ONNX / `.rknn` 作为模型载体**。图结构 + 权重统一装入 VQF v4（检测扩展，见附录 A）。`.pt` 官方原始权重是唯一数据源，由离线工具**直接解析**为自有格式；`ultralytics`/`torch` 仅作**离线解析与 golden 参考**，不进入部署链。

**冻结判据**（复用 K-LAYA-002 / K-BENCH-020 已冻口径，不新造）：

| 判据 | 内容 | 门限 |
|---|---|---|
| **J1 算子数值等价** | 单算子（conv / maxpool / silu / concat / resize / DFL）NPU vs CPU 参考 | F32 位级或紧容差；int8 `max\|Δ\| ≤ tol`（tol 依量化粒度定，M2 冻结） |
| **J2 检测等价** | 同图同权重，输出框（cls/conf/xyxy）vs CPU golden | 框 IoU 匹配 + conf `\|Δ\| ≤ tol`，无 NaN/Inf |
| **J3 三版覆盖** | v5 / v8 / v11 各 ≥5 张真实图 | 全部过 J2 |
| **J4 回退透明** | NPU 不可用/失败时 | 显式回退 CPU，日志可辨（`[NPU] fallback: <reason>`），**绝不静默**（公理 `silent_fallback_false_pass`） |
| **J5 性能目标** | 单图端到端延迟（板端 RK3588，serve 常驻，t4@A76） | **已冻结（2026-10-11）**：CPU 基线中位 **1.82–2.36 s/图**（见 §7 M0-c）；目标 = **NPU（M2/M3）后 ≤ 基线的 1/3（≥3×）**，且 J2 不退化 |

> 判据冻结原则：J1–J4 立即冻结；J5 性能目标依赖 M0 实测基线（`todo_requires_measured_share` 公理——无分母不下结论）。

---

## 2. 现状与瓶颈（事实盘点，不臆造）

### 2.1 板端环境（2026-10-11 实测）

| 项 | 值 |
|---|---|
| SoC | RK3588（OPi5+），8 核（4×A76@2.256G + 4×A55@1.8G），RAM 15.9 GB |
| NPU 内核驱动 | **rknpu v0.9.6**（DRM 节点 `/dev/dri/renderD129`；**无** `/dev/rknpu` misc 节点） |
| NPU 硬件 | 3 core @1.0 GHz（`fdab0000.npu`）；NPU3 为最快单核 |
| `librknnrt.so` | 1.4.0（2022-09）——**本项不采用**（自研 DIRECT 路线不依赖它） |
| 板端 python | 3.10.12 + numpy 1.21.5；**无** rknn-toolkit / rknnlite / cv2 |

### 2.2 引擎 NPU 现状（`src/npu/`）

- [vllm_npu_direct.c](../src/npu/rk3588/vllm_npu_direct.c)：in-tree 零依赖 rknpu 驱动，已标定 **int8/int4 group-wise GEMM**（CNA `conv_mode=0` 即 FC/1×1 卷积；CNA+DPU+CORE+DPU_RDMA regcmd 生成器 `npu_direct_gen_matmul_regs`，特征布局 NC1HWC2/C2=16，weight N-group/K-group 打包）。
- 校准契约：`VLLM_NPU_REGS_VERIFIED`（2026-08-20，rknpu 0.9.6 DRM），`--npu-calib` / `--npu-selftest` 对 CPU 参考 PASS/FAIL。
- [vllm_npu.h](../include/npu/vllm_npu.h)：DIRECT/RKNN 双后端抽象，`vllm_npu_gemm_gw` 入口（M≥4 才 offload，M==1 走 CPU）。
- **结论**：DIRECT 驱动**只懂 GEMM，不懂真卷积**（kernel>1 / stride / pad 的 CNA `conv_mode≠0` 字段未标定）。

### 2.3 视觉/检测现状

- 视觉代码 [vllm_vision.c](../src/model/vllm_vision.c) 是 Qwen3-VL 的 ViT；其 patch_embed 是**手写 patch 提取 + matvec**，非通用卷积。
- serve 已有 `image_url` base64 解码（stb_image）；**无** YOLO / 通用卷积 / 检测后处理代码。

### 2.4 YOLO 三版算子共性（领域事实，M0 阶段以官方 ONNX graph 为准核对）

三版 YOLO 的检测网络骨干是同一族算子，**卷积占绝对主导**：

| 算子 | 说明 | NPU 落点 |
|---|---|---|
| **Conv(k,s,p) + BN + SiLU** | 3×3/1×1 卷积为主 | **核心攻坚**：CNA conv 模式（或 im2col→GEMM 兜底） |
| **C2f / C3 / C3k2** | 多 bottleneck 分支 + concat | 由多个 Conv + Concat 组成 |
| **SPPF** | 3×maxpool 级联 | DPU/CPU |
| **Upsample / Resize** | nearest | CPU |
| **Concat** | 通道拼接 | CPU（拷贝） |
| **Detect head** | 1×1 conv + DFL + anchor(±) decode | 1×1 conv 可复用 GEMM；DFL/decode CPU |

> 差异点（须按版分别适配）：v5 anchor-based / v8 anchor-free+DFL / v11 有 C2PSA 注意力。**结构以官方权重经 ONNX 导出的 graph 为准**，不凭记忆硬编码。

### 2.5 瓶颈诊断

- **功能瓶颈**：引擎缺 YOLO 前向 + 权重加载 + 检测后处理整条链。
- **加速瓶颈**：卷积数据通路（CNA `conv_mode≠0`）未标定，是自研 DIRECT 路线的**唯一未验证面**，也是最大不确定性。
- **已知锚点**：现有 GEMM 标定已证 CNA/DPU 的 int8 数据通路可用；卷积可**降级为 im2col→GEMM** 复用该锚点。

---

## 3. 公理映射（理论 → 工程）

公理库 [axiom_registry.json](../../axiom_registry.json) + 项目专属 [kestrel_prefill_axioms.json](../../kestrel_prefill_axioms.json) 映射：

| 公理（symbol） | 工程映射 |
|---|---|
| `isomorphic_superposition`（convolution collapses to pointwise multiplication） | **卷积↔GEMM 同构**：NPU CNA 的 `conv_mode=0`（FC/1×1）已证可用；kernel>1 卷积可经 **im2col 坍缩为 GEMM**（同构坍缩），作为真卷积标定前的**兜底路径**。真卷积标定成功 = 恢复 CNA 原生卷积模式，省去 im2col 展开开销。 |
| `collapse_to_classic` + `collapse_priority_rule` | **int8 量化坍缩**：YOLO 权重/激活 int8 化。`collapse_priority_rule`（singleton 优先 → 细粒度坍缩）⇒ 跨通道值域大时**必须 per-channel scale**，否则 `irreversibility_penalty=0.3`。工程红线（K-EMB/K-LAYA 沉淀）：≥16 元素量化粒度。 |
| `conservation_law_under_collapse`（`confidence=0.65` 弱约束） | **精度守恒门**：`E[Q(s)\|collapse]=Q(S)` ⇒ int8 后检测排序必须与 F32 golden 一致（J2）。弱约束 ⇒ **必须实测**，不靠理论保证。 |
| `MI_fusion_rule`（blas_precision_efficiency_tradeoff） | **int8 带宽收益**：权重流 4× 缩小（fp32→int8），误差界受控；对应 J1 容差门。 |
| `superposition_addition_base`（classic_anchor=2） | **DPU/CNA 规约正确性**：多 lane 部分和归约；规约顺序不得改变结果（J1 位级门）。 |
| `interference_priority_rule` | **异构核调度**：NPU 只取快核（NPU3）；CPU 侧多线程只取 4×A76。 |
| `operator_composition`（⊕+→+） | **算子融合**：Conv+BN 融合为单卷积（BN 折入 weight/bias），减少中间张量读写。 |
| `exponential_jump` | **复杂度跃迁警示**：im2col 内存放大（3×3 conv 展开 ≈9× 特征）、SPPF/注意力 O(n²) 风险，需显式评估。 |
| `block_matrix_assoc_natural` | **分块自然同构**：im2col 后的大 GEMM 按 block 分片，复用已标定的 int8 GEMM 分块路径。 |
| `linear_interpolation_axiom` | **混合精度**：敏感层（detect head、第一层）F32/int8 细粒度，不敏感层（中间 conv）int8。 |
| `measurement_fairness_equivalence`（项目专属） | **公平 A/B**：NPU vs CPU 对照须同权重、同图、同后处理、同热态。 |
| `bitwise_equivalence_gate` + `bitwise_tolerance_non_substitution`（项目专属） | **位级与容差不互换**：F32 路径追求位级；int8 走容差门，但**不得**用容差糊弄位级回归。 |
| `silent_fallback_false_pass`（项目专属） | **回退透明**：NPU offload 失败必须显式日志 + 回退，禁止"看似过了实为静默降级"（J4）。 |
| `feasibility_conclusion_timeliness`（项目专属） | **可行性别拖**：卷积标定设**时间/尝试上界**，超界即回退 im2col 方案（§5 回退）。 |
| `single_source_ledger_with_inline_pointers`（项目专属） | **单一真源台账**：本文件 + 落地台账，不另开新账。 |

---

## 4. 分级方案

每级给出：公理约束 / 改动 / 判据 / 预期 / 风险。

### M0 — 自有格式转换 + CPU 端到端基线（功能闭环，不碰 NPU）

- **公理**：`conservation_law_under_collapse`（先立 golden）、`collapse_priority_rule`（int8 化的前提是 F32 golden 先对）、`single_source_ledger_with_inline_pointers`（格式单一真源）。
- **改动**：
  1. **权重获取**：官方 ultralytics `.pt`（已就位，v5/v8/v11 三版 n/s/m/l/x）。
  2. **自有格式转换（离线，python）**：`tools/yolo/yolo_to_vqf.py` —— 用 `torch`/`ultralytics` **读取 `.pt` 的模块图与权重**（注意：这是**离线解析工具**，非部署依赖），按**附录 A 规格**直接写出 **VQF v4（检测）自有格式**：图结构（算子序列 + 超参）编成 graph 张量，权重按 qtype（F32/Q8_0/F16）编成张量目录。**不经过 ONNX / `.rknn`**。
  3. **C 侧加载器 + 前向**：引擎 `vqf.c` 扩展 v4 分支读取检测模型；新增 `vllm_yolo.c` 实现 Conv/BN/SiLU/C3(C2f/C3k2)/SPPF/Concat/Upsample/Detect + DFL + anchor(±) decode + NMS。
  4. **serve 端点**：`vllm_server.c` 新增 `POST /v1/detect`（image 输入 → 检测框 JSON）。
- **判据**：J2（CPU 输出 vs **torch golden**）、J3 三版可跑。
- **预期**：功能闭环；给出 **CPU 基线延迟**（供 J5 冻结）。
- **风险**：低（纯 CPU，无 NPU 不确定性）；主要工作是图解析与逐算子对齐。

### M1 — CPU 基线采数 + 判据补冻 + 权重 int8 化（纯 CPU）

- **公理**：`measurement_fairness_equivalence`、`collapse_priority_rule`（per-channel scale）。
- **改动**：M0 采 CPU 延迟基线 → **冻结 J5 性能目标**；权重 F32→int8（per-channel）量化，CPU int8 内核验证量化坍缩不坏 J2。
- **判据**：J2 在 int8 权重下仍过；J5 目标冻结。
- **预期**：得到 NPU 对照的 int8 golden。
- **风险**：低。

### M1 int8 权重量化 + J2 守门（2026-10-11，已达成）

**改动**：
- 转换器 `--q8`：per-output-channel 对称 int8（`quant_q8`，`collapse_priority_rule` 细粒度）——权重存 `c%04d_w`(QT_Q8) + `c%04d_s`(F32 scale[oc])，bias 恒 F32；
- C 引擎 [vllm_yolo.c](../src/model/vllm_yolo.c)：新增 `yconv_q8`（int8 权重 × F32 激活，逐 oc 一次 scale）+ 加载期 cid→(int8, scale) 直查；`ATTN_PSA` 的 qkv/proj/pe 经 `conv_w_f32` 反量化（注意力占比小）。

**产物体积**：yolov8n 12.04 → **3.07 MB（−75%）**；yolo11n 10.00 → **2.58 MB（−74%）**。

**J2（int8 vs ultralytics `predict`，同阈值）**：

| 模型 | 匹配 | 检测集 | maxIoU | score 差 | 框差 |
|---|---|---|---|---|---|
| yolov8n_q8 | **18/18** | 与 F32 完全相同（5/13） | ≥0.959 | ≤0.073 | ≤2.2px |
| yolo11n_q8 | **17/17** | 与 F32 完全相同（4/13） | ≥0.937 | ≤0.033 | ≤2.5px |

**板端（governor=performance，t4@A76，serve 常驻，n=5 中位）——F32 vs Q8**：

| 模型 | F32 cats/room | Q8 cats/room | 变化 | RSS |
|---|---|---|---|---|
| yolov8n | 2352.7 / 2140.6 ms | 2537.9 / 2306.5 ms | **+7.9% / +7.8%（变慢）** | 538 → 529 MB |
| yolo11n | 2020.9 / 1816.0 ms | 2253.7 / 2003.4 ms | **+11.5% / +10.3%（变慢）** | 299.5 → 291.4 MB |

**判读（诚实披露，公理 `nonmonotonic_benefit_no_cherry_pick`）**：
- **质量面**：int8 量化**不改变检测集**（板端 5/13、4/13 逐框一致；x64 J2 全匹配）⇒ 量化坍缩在 YOLO 检测上是安全的。
- **CPU 速度面**：**int8 反而慢 8–11%** —— CPU 路径是 compute-bound（非带宽），scalar int8 多出 int8→float 转换开销，且无 dotprod int8 内核 ⇒ **int8 在 CPU 上无收益**（仅省内存 −2% RSS）。
- **结论**：**int8 的价值在 NPU（M2 的 int8 GEMM）**，不在 CPU。CPU int8 内核（dotprod）**不投入**（超越本项目标；NPU 才是加速面）。

---

### M2 im2col→GEMM 上 NPU（2026-10-11，已实测：功能达成、性能不达 J5）

**改动**（[vllm_yolo.c](../src/model/vllm_yolo.c) / [vllm_yolo.h](../include/model/vllm_yolo.h) / [main.c](../src/main.c)）：
- 新增 `yconv_npu()`：kernel>1 卷积经 **im2col** 展开为 int8 GEMM，调用**已标定**的 `vllm_npu_gemm_gw`（DIRECT，int8）；不适用/失败 **显式回退 CPU**（J4，`[NPU] fallback conv ...` + 计数，`silent_fallback_false_pass`）。
- `st_yolo_set_default_npu()`：main 在 `vllm_npu_init` 成功后注册句柄；`--npu` 即启用（无 `--npu` → 纯 CPU，行为不变）。
- **诊断门（env）**：`VLLM_YOLO_NPU_CHECK=1`（逐卷积 NPU vs CPU 对拍，[YOLO-J1]）、`VLLM_YOLO_NPU_DEBUG=1`（回退原因）。
- **阈值**：`main.c` 在**检测模型 + `--npu` 且未显式给 `--npu-threshold`** 时**自动取 1e6**（避免"带 --npu 反而更慢"的静默退化；LLM 路径默认阈值 1e8 不变）。

**关键设计（转置取向）**：DIRECT gemm_gw 的 CNA 几何只标定**小 M**（特征行数；`CNA_DATA_SIZE0` 11-bit、`CNA_CONV_CON2` 10-bit）**× 大 K（32 块）× 大 N（Nt 分块）**。把 im2col 的 M=空间位置（数千）当 GEMM 的 M 会**溢出 M 编码 / 撑爆 CBUF**（首版即此错，J1 maxrel 达 1e9）。故卷积**转置**为：
`out[oc][p] = Σ_k Wconv[oc][k]·im2col[p][k]` ⇒ **M=OC（16..512，%4）**、**N=P=OH·OW**（driver 原生 N 分块）、K=C·KH·KW（补 Kp=32 倍数）。
“特征 A”= 卷积权重（`nw[cid]=[OC][Kp]`，per-channel scale 复制到各 K 块，语义与 CPU `yconv_q8` 等价）；“权重 W”= im2col（`[P][Kp]`，逐像素对称 int8 scale）；**out 布局 = 引擎槽位 [oc][OH][OW]，无需转置**。`wkey=0`（im2col 每调用变化，禁权重缓存）。

**J1（逐卷积 NPU vs CPU，env `VLLM_YOLO_NPU_CHECK`）**：`maxabs` 0.02–0.9（SiLU 后量级）；`maxrel` 大是因 SiLU 后近零分母，**以 maxabs 为准**。43–63 个卷积全部上 NPU（阈值相关）。

**J2（CPU vs NPU，同二进制）**：**检出对象集一致**（yolov8n cats 5/5、room 13→12；yolo11n cats 4/4、room 13/13），score 差 ≤ ~0.03、框差 ≤ ~5px（<1%）。**注意**：yolov8n/room 有 **1 个边缘检测**（conf 0.263）在 NPU int8 激活量化后跌破 0.25 阈值——int8 激活量化的透明代价，登记。

**J5（板端 serve 常驻，t4@A76，governor=performance，n=5 中位）**：

| 模型/图 | CPU（Q8 基线） | NPU（默认阈值 1e8） | NPU（`--npu-threshold 1e6`，全卷积上 NPU） |
|---|---|---|---|
| yolov8n cats | 2489.1 ms | 2547.5 ms（+2.3%） | **2091.1 ms（1.19×）** |
| yolov8n room | 2319.8 ms | 2393.2 ms（+3.2%） | **1932.8 ms（1.20×）** |
| yolo11n cats | 2228.8 ms | 2231.9 ms（+0.1%） | **1682.5 ms（1.33×）** |
| yolo11n room | 2022.0 ms | 2057.5 ms（+1.8%） | **1640.0 ms（1.23×）** |

**瓶颈定位（实测分解，yolov8n cats：im2col=0.308s / npu_call=1.443s；其余非卷积 CPU ≈0.34s）**：
- DIRECT gemm_gw 的 CNA 几何是**为「少量大 GEMM」标定**的（LLM prefill：M≤204、K 4096、N 12288）；卷积经 im2col 变成**数千次小提交**（n_tiles×n_blocks，如 cid=0 P=102400 → 100 子块），且“权重”角色（im2col）在 driver 内**每调用重打包 O(N·K)** ⇒ **提交/staging 开销**吃掉 NPU 算力。默认阈值（1e8）下 20 个亚阈值小卷积反而留 CPU（更慢），故**必须 `--npu-threshold 1e6`** 才见增益。
- 非卷积算子（SPPF/Concat/Upsample/DFL/NMS）留 CPU ≈0.34s，为 M4 才有机会下推。

**判读（`nonmonotonic_benefit_no_cherry_pick`）**：**M2 兜底路径功能达成、质量合格（J1/J2 通过），但性能仅 1.2–1.33×，远未达 J5 的 ≥3×**。与 §5 预判风险一致：**im2col 开销 + 小提交 + 非卷积算子留 CPU 共同抵消 NPU 收益**。⇒ 按 §5 **登记 M2 为「保底可用、非达标」**；J5 目标须靠 **M3（原生 CNA 卷积，单卷积单提交）** 才有望达成（M3 高风险，设尝试上界）。

#### M2 优化（2026-10-11 续）：最小提交调度 → 达标

**动机（参考取证）**：FOSS 参考（OpenNPU, MIT）实测该 NPU **每次 submit ≈0.5 ms**、DMA 受限 ~1 GB/s ⇒ 卷积-as-GEMM 的墙钟由**提交数**（`ceil(N/Nt)·K/G`）主导，而非 FLOPs。核算 M2 首版：`npu_call` 1.44 s ≈ **~2900 次提交 × 0.5 ms**（完全吻合）⇒ **减少提交数**是低风险高价值杠杆（无需新寄存器语义）。

**改动**：`vllm_yolo.c` 新增 `yolo_pick_tile()` —— 每卷积在 CBUF 约束下（12×32768 B 同时装特征 `M·Kb` 与权重 `Np_tile·Kb`，int8 特征 +1 bank slack；Nt∈[64,1024] 且 32 对齐）**搜索最小提交数的 (Kb, Nt)**；`vllm_npu_gemm_gw_ex(..., nt_hint)` 新增每调用 Nt 提示（`vllm_npu.h/.c`、`vllm_npu_direct.h/.c`；LLM 路径走 `nt_hint=0` 保持默认，**selftest 全 PASS 无回归**）。scale 由 per-channel 复制到 `n_blocks=Kp/Kb` 块（语义不变）。**续优化**：im2col **单遍化**（per-thread 行缓冲 gather+求 max，再量化；x 只读一遍 vs 原两遍）＋**内部像素无边界快路径**（3×3 padding 只影响外圈）；`yattn_psa` 的三个卷积改走统一派发 `yconv_dispatch` → **C2PSA 的 1×1 qkv/proj 也上 NPU**（depthwise `pe` 仍 CPU）。

**实测（板端，同口径，serve 常驻 n=5 中位）**：

| 指标 | 首版（G=32,Nt=1024） | 优化后 |
|---|---|---|
| `npu_call`（yolov8n cats） | 1.443 s | **0.41–0.45 s（~3.4×↓）** |
| `im2col`（yolov8n cats） | 0.308 s | **0.26 s** |

| 模型/图 | CPU | NPU | 加速 | J5（≥3×） |
|---|---|---|---|---|
| yolov5nu cats | 2483.8 ms | **757.3 ms** | **3.28×** | ✅ |
| yolov5nu room | 2191.5 ms | **705.1 ms** | **3.11×** | ✅ |
| yolov8n cats | 2548.1 ms | **761.3 ms** | **3.35×** | ✅ |
| yolov8n room | 2375.2 ms | **722.5 ms** | **3.29×** | ✅ |
| yolo11n cats | 2207.2 ms | 773.4 ms | 2.85× | ✗ |
| yolo11n room | 2098.0 ms | 736.7 ms | 2.85× | ✗ |

**J2（三版 CPU vs NPU，同二进制）**：主检测对象集一致；score 差 ≤~0.03、框差 ≤~5px。**登记**：conf 0.25 阈值附近的**边缘检测**在 int8 激活量化后可能 ±1（v8/room 丢 1 个 conf 0.263、v5/cats 丢 1 个 0.26、v5/room 多 1 个 0.253）——int8 激活量化的透明代价，非结构性退化。

**判读（`nonmonotonic_benefit_no_cherry_pick`）**：**v5 / v8 达标 J5（3.1–3.33×）；yolo11n 2.88–2.90×**。剩余瓶颈 = `npu_call`（0.36–0.45 s，受 HW 单次提交成本 × 提交数下限 `Σceil(P/1024)` 制约；Nt>1024 会 wedge 驱动，不冒进）+ 非卷积 CPU（SPPF/Concat/Upsample/DFL/NMS，M4 才有机会下推）+ yolo11n 的 depthwise `pe` 与注意力残留在 CPU。⇒ **M2 兜底路径收敛**：三版中两版达标、一版 2.9×；M3（原生卷积，单提交）仍是全线上探上限的选项。

#### M2 追加实验：成本模型调度（已证否回退，2026-10-11 续）

**动机**：实测 v11 共 **346 次提交**、`npu_call`≈1.27 ms/提交；首层 cid=0（3→16）P=71680 → **70 次提交**（占总提交 20%）而 FLOPs 仅 ~62 MFLOP，算术强度极低 ⇒ 猜想「该卷积 CPU 更快」。

**改动**：`yconv_npu` 加成本模型（npu_us = submits×1270；cpu_us = FLOPs/4000，即假设 CPU 4 GFLOP/s），预测 CPU 更快时显式回退 CPU；`VLLM_YOLO_NPU_DEBUG` 打印每卷积 submits。

**实测（板端同口径）**：**显著退化**——v8 cats **761→1003 ms**、v11 room 737→893 ms（15 个卷积被路由到 CPU，合计仅 ~236 MFLOP 却多耗 ~240 ms ⇒ **CPU 卷积对小通道形状的有效吞吐仅 ~1.0 GFLOP/s**，远低于 4 的假设）。⇒ 按 `nonmonotonic_benefit_no_cherry_pick` **回退成本模型**（保留 submits 统计与 `[NPU-SCHED]` 诊断）；回退后复测与基线一致（v8 757/718、v11 766/723 ms）。**沉淀（方法学）**：小通道卷积（C≤16）的 CPU 标量 int8 路径有效吞吐 ~1 GFLOP/s，**不可**按大矩阵口径外推。

#### M3 参考取得（2026-10-11，重要资产）

FOSS 取证确认 **Mesa `rocket` gallium 驱动（RK3588 NPU）已实现经厂商流对拍的卷积 regcmd 生成**——`iav/mesa` 分支 `rk3568-test-session-20260820` 的 `src/gallium/drivers/rocket/{rkt_regcmd.c,registers.xml,rkt_task.c,rkt_coefs.c}`（MIT）含：`CONV_CON1.CONV_MODE`（**0=Direct 常规卷积 / 3=Depthwise**）、`CONV_CON2`（RK3588 厂商把 `KERNEL_GROUP` 恒置 0，只写 `FEATURE_GRAINS` = `stride_y+kh+stride_y*((Wout<28)+(Wout<14))`）、`CONV_CON3`（X/Y stride）、`CONV_CON4.RGB_BYTELENGTH`、`DATA_SIZE0/1/2/3`（输入 W/H/C、输出 W、atomics）、`WEIGHT_SIZE0/1/2`（kernel W/H/C/N）、`PAD_CON0`（上半=zero_point-0x80，低半=PAD_LEFT|PAD_TOP）、`DMA_CON0=0x00171c07`（bit20 为 XML RESERVED，漏写则 MAC 拿不到数据）、`CBUF_CON1`（上半=input_width，低=input_data_entries）、特征 NC1HWC2（`FEATURE_ATOMIC_SIZE=16`，`rkt_surf_px(px)=px>1?align(px,4):px`）、权重 cube 打包（`rkt_coefs.c`）、CBUF bank 规划（`rkt_task.c`）。**⇒ M3 从「无参考盲猜」变为「有权威参考的大工程移植」**（注意参考为 RK3568 分支，RK3588 的 bank 数/步长单位需按本驱动已有标定复核）。**风险**：板端驱动支持软复位（dmesg 高频 `RKNPU: soft reset, num:6`），wedge 非必然需冷启动。

#### M3 Phase-1 实施规格（2026-10-11 冻结）→ **✅ 已达成（2026-10-11）**

**结果**：`--npu-calib-conv` **PASS，`max|Δ|=0`（逐位精确）**——`conv C=16 H=8 W=8 3x3 s=1 p=1 OC=16 int8->int32`；`--npu-selftest` 8/8 gemm_gw **全 PASS（无回归）**。⇒ **RK3588 DIRECT 驱动已能原生执行真卷积**（此前只懂 1×1/GEMM）。新增自包含函数 `npu_pack_feature_conv/npu_pack_weight_conv/npu_direct_gen_conv_regs/npu_conv_submit_once/vllm_npu_direct_conv_calib`（[vllm_npu_direct.c](../src/npu/rk3588/vllm_npu_direct.c)，matmul 路径未改）+ `--npu-calib-conv`（[main.c](../src/main.c)）。

**实际采用值**：CNA `CONV_CON1=0`、`CONV_CON2=FEATURE_GRAINS(4)<<4`（=stride+kh）、`CONV_CON3=9`、`CONV_CON4=W*H*C`、`DATA_SIZE0=(W<<16)|H`、`DATA_SIZE1=((C-1)<<16)|C`、`DATA_SIZE2=OW`、`DATA_SIZE3=OW*OH`、`WEIGHT_SIZE0/1/2`、`CBUF_CON0=0xA2`、`CBUF_CON1=(W<<16)|entries`、**`CVT_CON0=0xB`（沿用有符号域，Mesa 的 0xa/u8-offset 域不适用）**、`PAD_CON0=0x11`、**`DMA_CON0=0x00171c07`**、**`DMA_CON1=W*4`、`DMA_CON2=4*(W*4)*(H/4-1)`（本驱动 RK3588 单位，非 Mesa 的 W/`rkt_surf_px`）**；CORE `DATAOUT_SIZE_0`、DPU cube 几何按 conv 改写、`DST_SURF_STRIDE=(OW*OH)<<4`、`SURFACE_ADD=(OW*OH*8)<<4`；DPU_RDMA 沿用 matmul。

**关键发现（RK3588 特有，写入标定表）**：**DPU 写出的行数由 `CORE_DATAOUT_SIZE_0` 的高度字段控制，且该字段在 RK3588 上以「两行输出」为单位**——`OH-1` 只写 `OH/2` 行（症状=每 surface 只写一半、后半是 sentinel）；须写 **`2*OH-1`** 才写满。改 `DPU_DATA_CUBE_WIDTH/HEIGHT/WDMA_SIZE_1/DST_SURF_STRIDE` 均**无效**。**风险点消解记录**：① 有符号域 → 沿用 `CVT_CON0=0xB` ✓；② 步长单位 → 用本驱动 matmul 的 RK3588 单位 ✓；③ 输出 cube `[OC/4][OH*OW][4] int32` 反交织 ✓（一次命中）；④ bank 划分 → `0xA2` 可用。

**未做（Phase-2）**：把原生卷积接入 YOLO `yconv_npu`（kernel>1 走原生卷积、单卷积单提交），含 stride-2/1×1 分支、C>CBUF 的通道分组、YOLO 槽位 `[C][H][W]`→特征打包与 `[oc][OH][OW]` 读回、bank 规划。**预期收益**：`npu_call` 由「提交数 ∝ ceil(P/1024)·K/G」降为 O(卷积数) ⇒ J5 有望全线 >3×。

#### J5 复测与重大方法学修正：**NPU DVFS 未固定导致此前所有 NPU 计时偏低**

**发现**：板端 NPU 有独立 devfreq（`/sys/class/devfreq/fdab0000.npu`，0.3–1.0 GHz，默认 governor=`rknpu_ondemand`）。此前 M2/M3 的所有 NPU 计时都在**按需降频**下取得，故偏低。**按 `measurement_fairness_equivalence` 固定 `governor=performance`**（与既定 CPU 基线口径一致）后复测：

| 模型/图 | CPU（governor=performance） | NPU（**NPU governor=performance**） | 加速比 |
|---|---|---|---|
| yolov5nu cats / room | 2317.9 / 2147.5 ms | **393.9 / 375.0 ms** | **5.88× / 5.73×** |
| yolov8n cats / room | 2421.7 / 2268.0 ms | **388.3 / 365.0 ms** | **6.24× / 6.21×** |
| yolo11n cats / room | 2093.6 / 1943.2 ms | **404.8 / 381.0 ms** | **5.17× / 5.10×** |

（两次独立复测稳定，差异 <1%。）**⇒ J5（≥3×，且 J2 不退化）三版全线达标且余量巨大**（v8 6.2×、v5 5.7–5.9×、v11 5.1×）。**注意**：之前记录的 v5 3.08–3.21× / v8 3.31–3.33× / v11 2.88–2.90× 均为 **NPU 降频下**的读数，**作废**（保留以示方法学教训：NPU 计时必须先固定 devfreq）。J2 复测仍一致（检出集同 CPU，score 差 ≤0.03，conf 阈值边缘 ±1）。**部署口径**：运行时须 `echo performance > /sys/class/devfreq/fdab0000.npu/governor`（重启后需重设）。


#### M3 Phase-2 实施与**证否**结论（2026-10-11）——保守保留，未启用

**已实现**（[vllm_npu_direct.c](../src/npu/rk3588/vllm_npu_direct.c) / [vllm_yolo.c](../src/model/vllm_yolo.c)）：
- 驱动：`vllm_npu_direct_conv_i32()`（复用 Phase-1 几何，单提交返回 int32，scratch BO 复用）；`CONV_CON3` 改为按几何写 stride；新增 `--npu-calib-conv` 的**形状包络扫描**（env `VLLM_NPU_CONV_SWEEP`/`VLLM_NPU_CONV_SHAPE` + `VLLM_NPU_CONV_{GRAINS,LINE_STRIDE,SURF_STRIDE,...}` 寄存器旋钮）。
- YOLO：`yconv_npu` 改为在 **M2（im2col→GEMM）** 与 **M3 原生卷积**间按**提交数**择优（`yolo_pick_native` 按 CBUF 约束搜 (OC_band, rb)），原生用逐张量对称 int8 激活 + 现有 per-channel `cs[oc]` 反量化，失败**显式回退** M2。**默认关闭**（需 `VLLM_YOLO_NPU_NATIVE=1` 且限于已标定极小形状）。

**证否（板端实测，[NPU-CONV-SWEEP]，env 旋钮逐一试）**：**Phase-1 的寄存器配方不能推广到 8×8 以外的形状**——
| 形状 (C,H,W,K,s,p,OC) | 结果 |
|---|---|
| 16,8,8,3,3,1,1,16 | **PASS**（唯一稳定通过） |
| 16,24,8,3,3,1,1,16（OW=8） | FAIL（DPU 超时→soft reset；另一次曾 PASS → **run-to-run 不稳定**） |
| 16,16,16 / 16,32,32 / 16,40,40 s1 | FAIL（超时）或 max\|D\| 巨大（乱码） |
| 16,32,32 s4（强凑 OW=8） | FAIL（乱码） |
| 32,16,16 OC=32；64,20,20 OC=64 | rc=0 但 max\|D\| 巨大（乱码） |

- **失败现象**：多数 = `rc=-2` DPU 作业超时（dmesg `RKNPU: job timeout ... raw status 0xc0010005, require mask 0x30` → `soft reset`）；部分 = 提交成功但输出乱码（DMA 读到错误地址）。
- **已排除假设**：① 表面读步长 `DMA_CON2`（8×8 对 32/64/256 均 PASS，16×16 对 256/960/1024 均 FAIL ⇒ 非主因）；② `FEATURE_GRAINS`（16×16 试 4/5/6/8 均 FAIL）；③ 输出侧 `CORE_W/CUBE_W/WDMA_W` 单改（16×16 试 ×2 均 FAIL）。**非** OW≤8 的简单边界（OW=8 的 24 行、s4 均失败）。
- **判读**：Mesa 权威参考为 **RK3568** 分支，其 output/DMA/bank 语义在 **RK3588** 上需重新标定而**未完成**；仅 C=16/8×8/3×3/s1/p1 的 Phase-1 特例可用。原生卷积 **不作为 YOLO 前向的可用路径**。

**保守落地（本项采用的最终状态）**：`yconv_npu` **默认走 M2**（原生需显式 env 且限于该极小形状，YOLO 形状永不满足 ⇒ 等价于不启用）；**不引入任何 wedge 风险**。回归验证：
- `--npu-selftest` **8/8 PASS**、`--npu-calib-conv` **PASS max\|Δ\|=0**（无回归）。
- **J2**（`m2_final_j2.sh`）：三版 CPU vs NPU 检出集一致，仅 conf 0.25 边缘 ±1（与既有 M2 登记一致）。
- **J5**（taskset c4-7，n=5 中位）：`native=0`（纯 M2 路径），`submits=371`、`im2col≈0.25s`/`npu_call≈0.44s` 与 M2 档案一致。**加速比以紧随其后的「J5 复测」为准**（NPU devfreq 固定 performance 后：**v5 5.73–5.88×、v8 6.21–6.24×、v11 5.10–5.17×**）；文中另记的 4.17×/3.57× 等读数仍在 **NPU ondemand 未固定**状态下取得，仅作过程记录。

**（原始冻结规格如下，保留备查）**
**目标**：在**不动**已标定 matmul 路径的前提下，新增自包含的原生卷积通路 + 自检 `--npu-calib-conv`；验收 = 单个常规 3×3 卷积（如 C=16,H=W=8,s1,p1,OC=16,int8）**NPU 结果 vs CPU oracle 相对判据 <1e-3**，且 `--npu-selftest` 仍全 PASS（无回归）。**用户口径：只优化 NPU，CPU 仅验证不优化。**

**提取出的寄存器事实（源：Mesa `iav/mesa` @ `rk3568-test-session-20260820`，MIT；已对厂商流对拍）**：
- `CONV_CON1(0x100C)`：`CONV_MODE[3:0]`（**0=Direct 常规 / 3=Depthwise**）、`IN_PRECISION[6:4]`、`PROC_PRECISION[9:7]`、`NONALIGN_DMA[30]`、`GROUP_LINE_OFF[29]`。
- `CONV_CON2(0x1010)`：`FEATURE_GRAINS[13:4]`（= `stride_y+kh+stride_y*((Wout<28)+(Wout<14))`，FC/1×1 取 1）、`KERNEL_GROUP[23:16]`（**RK3588 厂商恒 0**）。
- `CONV_CON3(0x1014)`：`CONV_X_STRIDE[2:0]`、`CONV_Y_STRIDE[5:3]`。
- `CONV_CON4(0x1018)`：`RGB_BYTELENGTH[23:0]`（非 ARGB 可 0，但**非 0 才放行 CNA OP_ENABLE**）。
- `DATA_SIZE0(0x1020)`：`DATAIN_HEIGHT[10:0]`、`DATAIN_WIDTH[26:16]`；`DATA_SIZE1(0x1024)`：`DATAIN_CHANNEL[15:0]`、`DATAIN_CHANNEL_REAL[29:16]`（=C-1）；`DATA_SIZE2(0x1028)`：`DATAOUT_WIDTH[10:0]`；`DATA_SIZE3(0x102C)`：`DATAOUT_ATOMICS[21:0]`（=OW*OH）、`SURF_MODE[23:22]`。
- `WEIGHT_SIZE0(0x1030)`=`KH*KW*C*N`；`WEIGHT_SIZE1(0x1034)`=`KH*KW*C`；`WEIGHT_SIZE2(0x1038)`：`WEIGHT_KERNELS[13:0]`、`WEIGHT_HEIGHT[20:16]`、`WEIGHT_WIDTH[28:24]`。
- `CBUF_CON0(0x1040)`：`DATA_BANK[3:0]`、`WEIGHT_BANK[7:4]`；`CBUF_CON1(0x1044)`：`DATA_ENTRIES[13:0]`（Mesa 另把 `input_width` 写进 RESERVED 上半）。
- `PAD_CON0(0x1068)`：`PAD_TOP[3:0]`、`PAD_LEFT[7:4]`（pad **值**放 RESERVED 上半 = `zero_point-0x80`）。
- `DMA_CON0(0x1078)`=**`0x00171c07`**（`FETCH_PIXEL_LEN=28`,`DATA/WEIGHT_BURST_LEN=7`,bit20=1；**bit20 漏写则 MAC 拿不到数据**）；`DMA_CON1(0x107C)`=`LINE_STRIDE[27:0]`；`DMA_CON2(0x1080)`=`SURF_STRIDE[27:0]`。
- `FC_DATA_SIZE0(0x1084)`：`DMA_HEIGHT[10:0]`、`DMA_WIDTH[29:16]`；`FC_DATA_SIZE1(0x1088)`：`DMA_CHANNEL[15:0]`。
- 特征布局 **NC1HWC2，C2=16**：`[C/16][H][W][16]`；`surf` 以 4 像素对齐（`rkt_surf_px(px)=px>1?align(px,4):px`）。
- 权重 cube（常规卷积，`icb=align(C,16)`）：`pos(oc,ky,kx,ic)=(oc/16)*KH*KW*16*icb+(ky*KW+kx)*16*icb+(oc%16)*icb+ic`（Mesa 存 `u8=w+128`；本引擎走**有符号 int8 + CVT bypass**域，故按本驱动的符号域打包，不加 0x80 偏移——**待板端验证**）。
- `CORE`：`MISC_CFG(0x3010)`、`DATAOUT_SIZE_0(0x3014)=((Hout-1)<<16)|(Wout-1)`、`DATAOUT_SIZE_1(0x3018)=Cout-1`（本驱动偏移已按 RK3588 标定，沿用）。
- `DPU`：`FEATURE_MODE_CFG` OUTPUT_MODE=4/burst=2；`DATA_CUBE_WIDTH=Wout-1`、`HEIGHT=Hout-1`、`CHANNEL=align(Cout,16)-1`；`DST_SURF_STRIDE` 与 `SURFACE_ADD` 按输出 surface 计。

**已知风险点（需板端逐条消解）**：① 有无符号域（CVT bypass vs Mesa 的 0xa + u8 偏移）；② RK3588 的 `LINE_STRIDE`/`SURF_STRIDE` 单位与 `rkt_surf_px` 对齐规则；③ DPU 输出 NC1HWC2(C2=4,int32) → `[oc][oh][ow]` 的反交织；④ CBUF bank 划分（RK3588 banks=12 vs 参考 8）。

---

### M2 — im2col→GEMM 上 NPU（复用已标定 GEMM，兜底路径）

- **公理**：`isomorphic_superposition`（卷积→GEMM 同构坍缩）、`block_matrix_assoc_natural`、`MI_fusion_rule`。
- **改动**：kernel>1 卷积经 **im2col 展开为 GEMM**，调用**已标定**的 `vllm_npu_gemm_gw`（int8）上 NPU；BN 已融合；非卷积算子（SPPF/Concat/Upsample/DFL/NMS）留 CPU。im2col 内存放大按 `exponential_jump` 显式评估（3×3 → 9× 特征量）。
  > **实现修订（见 §7 M2，2026-10-11）**：必须用**转置取向**（M=OC、N=像素数 P、K=IC·KH·KW）——gemm_gw 的 CNA 几何只支持小 M；若把 im2col 的 M=空间位置当 GEMM 的 M，会溢出 M 编码并撑爆 CBUF（首版即此错）。
- **判据**：J1（im2col-GEMM 卷积 vs CPU 参考）、J2/J3/J4。
- **预期**：端到端卷积上 NPU；加速受 im2col 开销 + 非卷积算子留在 CPU 制约（**收益不预判，以 J5 实测为准**）。
- **实测结论（2026-10-11）**：J1/J2 通过；**首版 J5 仅 1.2–1.33×**（瓶颈=小提交）；**经最小提交调度优化后 v5/v8 达 3.1–3.35×、v11 2.85×**（详见 §7 M2）。
- **风险**：中。im2col 内存/带宽可能抵消 NPU 收益；若实测退化则回退并登记（`nonmonotonic_benefit_no_cherry_pick`）。**→ 已证实并登记**。

### M3 — DIRECT 真卷积标定（核心攻坚，可回退）

- **公理**：`isomorphic_superposition`（原生卷积恢复）、`superposition_addition_base`、`feasibility_conclusion_timeliness`（设上界）。
- **改动**：扩展 `vllm_npu_direct.c` 的 regcmd 生成器，标定 CNA `conv_mode≠0` 的 kernel/stride/pad/dilation 字段 + NCHW 特征布局（vs 现有 NC1HWC2）。新增 `--npu-calib-conv` 对 CPU 参考校准，复用 `VLLM_NPU_REGS_VERIFIED` 契约。
- **判据**：J1（真卷积 vs CPU 参考，**先 int8 后 F32**）、J2/J3/J4。
- **预期**：省去 im2col 展开开销，卷积原生上 NPU。
- **风险**：**高**。卷积 regcmd 字段是逆向标定，无公开文档（仅 RK3588 TRM 第 36 章 + Mesa rkt_regcmd FOSS 参考）；**设尝试上界，超界即回退 M2**（§5）。

### M4 — 非卷积算子下推 + 混合精度（可选，视 M2/M3 结果）

- **公理**：`operator_composition`（融合）、`linear_interpolation_axiom`（敏感层细分）。
- **改动**：DPU eltwise 下推 SiLU/Concat/add；detect head 1×1 conv 复用 GEMM；敏感层保留高精度。
- **判据**：J1–J5。
- **预期**：进一步收敛延迟。
- **风险**：中。仅在前序已达标且收益可量化时做。

### M5 — 三版全量验证 + 收口

- **公理**：`measurement_fairness_equivalence`、`single_source_ledger_with_inline_pointers`。
- **改动**：三版 YOLO × ≥5 真实图全量 J2/J3/J5 复测；回填台账；收口。
- **判据**：J1–J5 全过。
- **预期**：立项达标关闭。

---

## 5. 关键风险与回退

| 风险 | 概率 | 回退/对策 |
|---|---|---|
| **卷积 regcmd 标定长期攻不下**（M3） | 高 | 回退 **M2 im2col→GEMM**（已复用标定锚点），满足"卷积上 NPU"目标但收益打折；登记未覆盖 |
| **im2col 内存/带宽抵消 NPU 收益**（M2） | 中 | 实测证否即回退，登记；或限大卷积 im2col、小卷积 CPU |
| **int8 量化坍缩破坏检测**（M1/M2） | 低 | per-channel scale（`collapse_priority_rule`）+ 敏感层 F32（`linear_interpolation_axiom`）；J2 守门 |
| **NPU 静默降级** | — | J4 强制：失败显式日志 + 回退 CPU，禁止静默（`silent_fallback_false_pass`） |
| **官方权重版本漂移** | 低 | 锁定 `.pt` 版本号（记录 SHA256）+ 格式 version 字段，台账登记 |

---

## 6. 未覆盖 / 登记备查（初始）

- 分割/姿态/旋转框类 YOLO 变体（本项仅**目标检测** head）。
- 视频流/多帧批处理、`/v1/detect` 并发/队列。
- 模型量化标定集（calibration dataset）的选取口径（M1 冻结）。
- NPU 多核（NPU0/1/2）并行调度（本项先单核 NPU3）。
- ✅ **已完成**（2026-10-11）：与 `librknnrt` RKNN 路线的对照（信息性，不进交付链）——见 §7「可选 A/B：官方 RKNN 路线对照」。

---

## 7. 执行记录

### M0-a 自有格式转换器（2026-10-11，已达成）

**产物**：
- [tools/yolo/yolo_to_vqf.py](../tools/yolo/yolo_to_vqf.py) —— `.pt` → VQF v4 离线转换器（torch/ultralytics 仅作离线解析；**不经过 ONNX/.rknn**）。
- [tools/yolo/yolo_vqf_run.py](../tools/yolo/yolo_vqf_run.py) —— VQF v4 读回执行器（numpy 执行 graph，对比 torch golden）。

**实现要点**：
- 图展平到原语：`CONV / ADD / CONCAT / MAXPOOL / UPSAMPLE / SIGMOID / MUL / SLICE / SOFTMAX / ATTN_PSA / DETECT`；
- **BN 折叠进卷积**（含注意力内 qkv/proj/pe 三卷积）；C2f/C3f/C3k2/C3k/C2PSA/SPPF/Bottleneck 全部递归展开为原语；
- 三版 Detect 共用 **anchor-free DFL 头**（本 Ultralytics 版的 yolov5 亦为 anchor-free，`reg_max=16`）。

**数值门（J1/J2 前置，随机输入 640×640，相对判据 <1e-4）**：

| 模型 | family | ops | convs | graph | tensors | VQF | max\|Δ\| | rel | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| yolov5n | v5 | 101 | 76 | 5260 B | 153 | 10.12 MB | 3.34e-3 | **5.1e-6** | **PASS** |
| yolov8n | v8 | 104 | 64 | 5068 B | 129 | 12.04 MB | 1.18e-3 | **1.8e-6** | **PASS** |
| yolo11n | v11 | 139 | 88 | 6708 B | 177 | 10.00 MB | 1.11e-3 | **1.7e-6** | **PASS** |

> 逐层对拍（v11 含 C2PSA）全主干/颈部 `max|Δ| ≤ 2e-5`；最终输出差异为 float32 累积误差量级（相对 ~1e-6）。

**踩坑记录（不重犯）**：
1. Ultralytics `decode_bboxes` 默认 **`xywh=True`**（中心+宽高），不是 xyxy —— 首版实现按 xyxy 导致 Detect 输出差 ~600×。
2. **注意力内三卷积必须折叠 BN** —— 只取 `.conv.weight` 会漏 BN（bias 亦缺），v11 单层偏差 ~2.2。
3. `ultralytics` 加载 `yolov5n.pt` 时会**自动下载** `yolov5nu.pt` 到模型目录（副作用已清理）。

**未做（M0 剩余）**：板端 CPU 基线延迟采数（J5）。

### M0-b C 引擎侧（2026-10-11，已达成）

**产物**：
- [include/model/vllm_yolo.h](../include/model/vllm_yolo.h) + [src/model/vllm_yolo.c](../src/model/vllm_yolo.c) —— 自有格式 VQF v4 加载器（结构 + FNV 自校验）+ 图执行器（CONV/ADD/CONCAT/MAXPOOL/UPSAMPLE/SLICE/ATTN_PSA/DETECT）+ letterbox(rect) + class-wise NMS；
- [include/model/vqf_format.h](../include/model/vqf_format.h) 与 [vqf_convert/src/vqf_format.h](../vqf_convert/src/vqf_format.h) 增 v4 常量（`VQF_VERSION_DET`/`VQFArchDet`/`VQF_QT_BLOB`）；[vqf.c](../src/model/vqf.c) 对 v4 **显式拒绝**（A.6）；
- CLI：`--yolo-model <vqf>` + `--yolo-image`（JSON 输出）/ `--yolo-raw`+`--yolo-raw-out`（数值门）；
- serve：`POST /v1/detect`（[vllm_server.c](../src/serve/vllm_server.c)；`ctx.yolo_model`，未带 `--yolo-model` → 404）；
- 工具：[tools/yolo/yolo_c_gate.py](../tools/yolo/yolo_c_gate.py)（C 数值门）、[tools/yolo/yolo_j2_check.py](../tools/yolo/yolo_j2_check.py)（J2 检测等价门）。

**J1（C vs torch golden，随机 640×640，相对判据 <1e-4）**：

| 模型 | max\|Δ\| | rel | 判定 |
|---|---|---|---|
| yolov5n（**见下 v5 说明**） | 4.7e-6 | — | PASS |
| yolov8n | 1.22e-3 | **1.9e-6** | **PASS** |
| yolo11n（含 C2PSA） | 2.23e-3 | **3.5e-6** | **PASS** |

**J2（C 引擎 vs ultralytics `predict`，COCO val2017 真图 + 截图，conf=0.25/iou=0.7）**：

| 模型 | 图 | C 检测数 | 匹配 | maxIoU | score 差 | 框差 |
|---|---|---|---|---|---|---|
| yolov8n | cats/room/android(3) | 5+13+1=19 | **19/19** | ≥0.989 | ≤0.005 | ≤1.6px |
| yolo11n | cats/room(2) | 4+13=17 | **17/17** | ≥0.991 | ≤0.004 | ≤0.3px |

**serve 冒烟（x64，`--serve --port 8123 --yolo-model yolov8n.vqf`）**：`POST /v1/detect` 二进制体（`Content-Type: image/jpeg`）→ 200 且 5 检测（cat×2/remote×2/couch）；JSON base64 data URL → 200 且 13 检测（tv/chair×4/person…）；空体 → **400**；未带 `--yolo-model` → **404**。

**踩坑记录（不重犯）**：
1. **`qsort` 比较器写反**（`(a<b)-(a>b)` 实为降序）⇒ NMS 从最低分开始，把高分同框全部抑制（v8 输出 0.335 而非 0.537）——比较器应为 `(a->s > b->s) - (a->s < b->s)`（升序，末尾为最高分）。
2. **`letterbox` 必须用 `auto=True`（rect）语义**：ultralytics `predict` 把图补到 **stride(32) 倍数**（如 2939×2499 → 输入 544×640，不补成正方形）；用 `auto=False`（640×640 补边）会导致 score 0.537↔0.397、框偏 ~32px。已改为 rect。
3. **变尺寸输入**：rect letterbox 后 `yolo_forward` 的 slot0 必须按**实际** (iw,ih) 分配/拷贝，否则越界（首版按 640×640 读 → 崩溃）。
4. **类别名接口**：names blob 为 `'\n'` 分隔，直接 `%s` 会打印整个剩余 blob ⇒ 改为 `st_yolo_class_name(m, idx, buf, len)` 按行截断。
5. **注意力内 qkv/proj/pe 必须折叠 BN**（同 M0-a 坑 2）。

**⚠️ v5 现状（须决策，见 §8）**：用户的 `modl/Ultralytics--yolov5/yolov5n.pt` 是**旧版 ultralytics/yolov5 checkpoint**（pickle 引用 `models.yolo`，本版 ultralytics **无法加载**：`ModuleNotFoundError: No module named 'models'`）。`YOLO('yolov5n.pt')` 会经 `check_yolov5u_filename()` **静默替换**为 `yolov5nu.pt`（并**自动下载**）。即上表 “v5” 实为 **YOLOv5u**（ultralytics 原生 arch，与 v8 同构的 anchor-free DFL 头）；**用户原始 `yolov5n.pt` 尚未被真正支持**。

**未做（M0 剩余）**：板端 CPU 基线延迟采数（J5 冻结）。

### M0-c 板端部署 + CPU 基线（J5 冻结）（2026-10-11，已达成）

**部署**：源码同步 `/root/_vfy/gitee3`（8 文件：`vqf_format.h`×2 / `vllm_yolo.h/.c` / `vqf.c` / `main.c` / `vllm_server.c/.h`；备份 `/root/_vfy/_yolo_bak/`），CMakeLists 加 `src/model/vllm_yolo.c` + `include/model/vllm_yolo.h`；`./build_rk3588.sh` **rc=0**，产物 aarch64 ELF 1,137,992 B。模型包 `/mnt/VQF/yolo/{yolov8n,yolo11n,yolov5nu}.vqf` + `cats.jpg/room.jpg`（COCO val2017）。

**跨架构一致性**：板端 CLI 与 x64 输出**逐检测一致**（cats 5 框 score 完全相同：0.8878/0.8250/0.7806/0.3603/0.3241；room v11 13 框 score 差 ≤0.001）。

**CPU 基线（J5 冻结依据；governor=performance，`OMP_NUM_THREADS=4 taskset -c 4-7`，serve 常驻，warmup 1 + 5 次取中位）**：

| 模型 | cats.jpg | room.jpg | 峰值 RSS |
|---|---|---|---|
| yolov8n | **2358.7 ms**（min 2323.4） | **2142.1 ms**（min 2131.2） | 538 MB |
| yolo11n | **2042.8 ms**（min 1990.6） | **1817.7 ms**（min 1815.6） | 299 MB |
| yolov5nu | **2263.7 ms**（min 2232.7） | **2019.8 ms**（min 2012.7） | 411 MB |

⇒ **J5 冻结**：基线中位 **1.82–2.36 s/图**；目标 = NPU（M2/M3）后 **≤ 基线 1/3（≥3×）** 且 J2 不退化。基准脚本：`/root/_vfy/bench_yolo.py`（serve 常驻计时，隔离模型加载）。

---

### M5 三版全量验证 + 收口（2026-10-11，✅ 达标关闭）

**口径**：三版 YOLO（yolov5nu / yolov8n / yolo11n，均为自有格式 VQF v4 + int8 权重）× **9 张真实图**（COCO val2017 `000000000139/285/632/724/1000/2157/5001` + `cats/room`）；板端 `OMP_NUM_THREADS=4 taskset -c 4-7`、**CPU 与 NPU governor 均 performance**、serve 常驻、warmup 1 + 5 次取中位。

**J3（三版覆盖）**：三版 × 9 图全部跑通（要求 ≥5 图）✅

**J2（检测等价）**：
- **golden 门（x64：C 引擎 CPU vs ultralytics `predict`，conf=0.25/iou=0.7，7 张 COCO 图）**：**yolov5nu 56/59、yolov8n 66/68、yolo11n 67/71**（C 检出被配对率；未配对者均为 conf 边缘 0.25–0.47 的单个目标，系 int8 量化漂移，非结构性）。
- **NPU 门（板端 CPU vs NPU，三版 × 9 图 = 27 组）**：`[M5-J2]` **CPU 检出 247 / NPU 248 / 按 cls+IoU≥0.5 配对 242**；score 差 **≤0.061**、框差 **≤9.8 px**（<1.6%）；差异全部为 **conf=0.25 阈值边缘 ±1**（每个方向均有），与已登记的「int8 激活量化」代价一致。

**J5（性能，27 组）**：

| 模型 | 加速比范围（9 图） | 最小 | 判据（≥3×） |
|---|---|---|---|
| yolov5nu | **5.74–6.63×** | **5.74×** | ✅ |
| yolov8n | **6.16–7.37×** | **6.16×** | ✅ |
| yolo11n | **5.04–6.49×** | **5.04×** | ✅ |

（NPU 端 365–497 ms/图；CPU 端 1941–3142 ms/图。）

**J1（算子数值等价）**：转换门（三版 rel ≤5.1e-6 vs torch golden，且与 numpy 参考执行器一致）；C 引擎数值门 v8 rel 1.9e-6 / v11 3.5e-6；NPU 逐卷积 `maxabs` 0.02–0.9；原生卷积标定 `max|Δ|=0`。✅

**J4（回退透明）**：NPU 不适用/失败一律显式日志 + 回退 CPU（`[NPU] fallback ...` + 计数；原生卷积路径默认关闭），无静默降级。✅

**⇒ J1–J5 全部通过，本立项达标关闭。**

**收官指标**：单图端到端（板端 serve 常驻，NPU devfreq 固定 performance）——**yolov8n 365–390 ms（6.2–7.4× vs CPU）**、**yolov5nu 368–481 ms（5.7–6.6×）**、**yolo11n 376–497 ms（5.0–6.5×）**；检测质量与 CPU golden 等价（J2）。

**交付物**：`tools/yolo/yolo_to_vqf.py`（`.pt`→VQF v4 自有格式，不经过 ONNX/.rknn）、`tools/yolo/yolo_vqf_run.py`、`tools/yolo/yolo_c_gate.py`、`tools/yolo/yolo_j2_check.py`；引擎 `include/model/vllm_yolo.h` + `src/model/vllm_yolo.c`（VQF v4 加载 + 图执行 + letterbox + NMS + **M2 im2col→GEMM 最小提交调度**）、`include/model/vqf_format.h`/`vqf.c`（v4 扩展与显式拒绝）、`src/npu/vllm_npu.h/.c`（`vllm_npu_gemm_gw_ex`）、`src/npu/rk3588/vllm_npu_direct.h/.c`（`nt_hint` + **M3 原生卷积通路与 `--npu-calib-conv`**）、`src/main.c`（YOLO CLI/serve/阈值/`--npu-calib-conv`）、`src/serve/vllm_server.c`（`POST /v1/detect`）。

**未覆盖 / 登记**：M3 原生卷积**未推广**（仅 8×8 标定形状，默认关闭，Phase-2 证否，见 §7）；M4（非卷积算子下推/混合精度）**未做**（用户口径为只优化 NPU、不优化 CPU；且 J5 已达标）→ J5 余量充足，M4 不再必要；旧版 `yolov5n.pt`（非 `u`）arch 不支持；未做并发/多帧、分割/姿态变体；`librknnrt` 路线对照**已完成**（信息性，见下「可选 A/B」）；**部署须固定 NPU devfreq 为 performance**（方法学教训，见 §7 J5 复测）。

### 可选 A/B：官方 RKNN 路线对照（2026-10-11，信息性，不进交付链）

**动机**：兑现 §6 登记的「与 `librknnrt` RKNN 路线对照」——量化「官方 NPU 方案 vs 本自研 DIRECT」的差距。

**口径**：同模型（YOLOv8n）、同输入（640×640）、同板（RK3588）、同 governor（CPU/NPU 均 `performance`）。**官方侧 = 纯 NPU 计算（`rknn_run`）；本侧 = serve 端到端（含 letterbox/前处理/NMS/后处理/解码）——非严格同类**，见下「可比性局限」。

**环境取证（关键坑）**：
- 板载 `librknnrt.so` = **1.4.0（2022-09）**，连它自带的 `/usr/share/rknn_demo/mobilenet_ssd.rknn`（2023-05）都拒载（`Invalid RKNN format`）——**镜像内部版本不自洽**。
- 从本机下载 **librknnrt 2.3.2（2025-04）** 推板，用 `LD_PRELOAD` **非侵入**加载（**不改系统库**）。
- 官方预转模型：`rknn_model_zoo` 的 **`yolov8.rknn`**（自 Radxa `rknn-model-zoo-rk3588_1.6.0-1.deb` 取出，5.68 MB，1×640×640×3 输入、9 输出）。demo 的 `mobilenet_ssd.rknn` 在 2.3.2 下仍 `Verify ModelBuffer failed`（文件疑损坏/格式不符），弃用。
- 基准程序：自写最小 `rknn_bench.c`（仅 `rknn_init/query/inputs_set/run`；无 toolkit / ONNX / cv2）。

**官方 `yolov8.rknn` 实测**（n=20，warmup 后）：
- `rknn_init` rc=0；`sdk api=2.3.2`、`drv=0.9.6`。
- `rknn_run`：**min 15.11 / median 15.92 / max 15.93 ms**；lib 自报 `run_duration=15918 µs`；`inputs_set` 0.10 ms。
- core mask 扫描 0/7/1/2/4 → **15.9–16.3 ms（无并行收益）**；mask=8（NPU3）→ rc=-13（该平台 `max core mask=7`）。

**本自研 DIRECT 路线实测**（同 governor，live 复测）：
- `yolov8n_q8.vqf` serve 端到端：**cats 387.4 / room 365.0 ms**（CPU 2418.8 / 2267.3 ms）——与 M5 收官 365–390 ms 一致。
- 单图分解（CLI，`VLLM_YOLO_NPU_DEBUG=1`）：**63 卷积、native=0、submits=371、im2col=0.069 s、npu_call=0.249 s**。

**差距（同模型同尺寸）**：

| 口径 | 官方 RKNN | 本 DIRECT | 倍差 |
|---|---|---|---|
| NPU 计算（conv） | 15.9 ms | **249 ms** | **~15.7×** |
| 端到端 | ~16 ms + 前后处理 | 365–387 ms | ~23×（非同口径） |
| 有效算力（YOLOv8n ≈8.7 GFLOPs） | ~547 GFLOP/s | ~35 GFLOP/s | ~15.7× |

**根因**（与 M2/M3 结论一致）：官方把整图编成 **NPU 原生图、单次 `rknn_run`、direct conv**；本路线因 **M3 原生卷积 Phase-2 证否（native=0）**，退化为 **im2col→GEMM 的 371 次微提交**（≈0.67 ms/提交）+ im2col CPU（69 ms）+ 非卷积算子留 CPU。

**可比性局限（诚实声明）**：① 官方=纯 NPU 计算（无 letterbox/NMS/解码），本=serve 端到端；② 官方模型为 toolkit 标定 int8，本为自研 int8 VQF；③ 官方 `.rknn` 属**信息性对照，不进本项交付链**（格式红线：部署只用 VQF v4）。⇒ 本 A/B **仅用于定位效率差距，不作为回归判据**。

**判读**：本自研 DIRECT 路线的 NPU 有效算力约为官方 RKNN 的 **1/16**，差距几乎全部来自「无原生卷积 → 微提交 + im2col + CPU 侧算子」。要对齐官方效率，唯一路径是补完 **M3 Phase-2（RK3588 原生卷积 output/DMA/bank 逆向标定）**；否则维持 M2 兜底（已过 J1–J5，本项已达标关闭）。

---

## 8. 待决策：YOLOv5 版本路线（2026-10-11）

**事实**（已实测确认）：
- 用户提供的 `modl/Ultralytics--yolov5/yolov5n.pt`（及 n6/s/s6…）是**旧版 ultralytics/yolov5** checkpoint：`torch.load` 报 `ModuleNotFoundError: No module named 'models'`（pickle 引用 `models.yolo`/`models.common`），本版 `ultralytics==8.4.175` **无法加载**（`detect.py` 亦明确报 "NOT forwards compatible with YOLOv8"）。
- `YOLO('yolov5n.pt')` 经 `ultralytics.utils.checks.check_yolov5u_filename()` **静默改名**为 `yolov5nu.pt` 并**自动下载**（`ultralytics/assets` release）——即之前所有 “v5” 结果（VQF、J1/J2）实为 **YOLOv5u**（ultralytics 原生 arch，与 v8 同构的 anchor-free DFL Detect），**不是用户的原始文件**。

**候选路线**：

| 方案 | 内容 | 代价 |
|---|---|---|
| **A. 用 YOLOv5u（推荐）** | 正式采用 `yolov5nu.pt`（官方 ultralytics 血统），删除/忽略旧版 checkpoint | 需用户确认；旧版 checkpoint 不参与 |
| **B. 适配旧版 yolov5 arch** | 自研 legacy 加载器：`models.common`(Conv6x6/C3/SPPF) + `models.yolo.Detect`（**anchor-based + grid decode**，与本项 anchor-free 路径不同） | 需实现 legacy 图与 decode，工作量中等，且需 legacy 权重可达 |
| **C. 换来源** | 用 LLM/自建脚本从旧 repo（`ultralytics/yolov5`）导出 ONNX/权重——**违反“自有格式、不走 ONNX”红线** | 不建议 |

**影响**：v5 的 J1/J2 已按 YOLOv5u 通过；若选 B，需新增 legacy 算子（anchor decode）并重跑 J1/J2。

**✅ 决策（2026-10-11，用户）**：**采用方案 A —— 用 YOLOv5u**（官方 ultralytics 血统，与 v8 同构的 anchor-free DFL 头，现有管线直接可用）。用户提供的旧版 `yolov5n.pt` 等旧 checkpoint **不参与**本项。后续模型来源：`yolov5nu.pt`（及 s/m/l/x 的 `u` 变体），经 `tools/yolo/yolo_to_vqf.py` 转自有格式 VQF v4。

---

## 9. 追官方优化方案（原生寄存器未收敛·回退 M2；改攻 CPU 路径已执行第 5–9 轮 · 2026-10-11）

> 状态：**寄存器逆向 3 轮（第 1–3 轮）+ 第 4 轮诊断**用尽未收敛 ⇒ 按 `feasibility_conclusion_timeliness` 上界**回退 M2**（原生卷积登记待突破）；**第 6 轮差距重定性**后转攻 CPU 路径，**第 7–9 轮已落地**（im2col/wstg/scratch/BO 复用；第 9 轮 `memset(out)` **证否回退**），e2e 逐位无回归地持续下降。详见下各轮执行记录。
> ⚠️ **轮次编号说明（台账审计 2026-10-11 补）**：本轮分节为「第 1 轮 / 第 2–3 轮 / 第 5–9 轮」——**第 4 轮为诊断轮**（未单独成节，结论已并入本状态行）；下方小节标题「第 5–8 轮」为旧文，**实际已执行至第 9 轮**。

**差距事实（同模型同尺寸，governor=performance）**：官方 `rknn_run` **15.9 ms** vs 本 DIRECT 端到端 **365–387 ms**（NPU 计算 **249 ms**）。**唯一根因** = 本驱动只标定了 GEMM（`conv_mode=0`），真卷积（`conv_mode=1/3` + stride/pad）的 **RK3588 寄存器语义未逆向完成**（M3 Phase-2 证否）。硬件同源（同 rknpu 0.9.6 驱动、同 NPU），**官方能跑 15.9ms ⇒ 硬件做得到，差距纯在逆向**，无捷径。

**公理映射（公理 → 动作）**：
| 公理 | 语义 | 动作 |
|---|---|---|
| `isomorphic_superposition` | 卷积坍缩为逐点乘（cost 1.6） | 认识：M2 im2col→GEMM 坍缩**数学等价但工程有代价（提交爆炸）**；原生=不坍缩 |
| `collapse_priority_rule` | singleton 优先坍缩 | 每形状已标定唯一正确寄存器配方者直接复用，不盲目扫描 |
| `conservation_law_under_collapse` | 坍缩须守恒检测正确性 | J2 检测等价门 |
| `interference_priority_rule` | 高优先级先试 | 先攻「提交数大头」的大 P 卷积上原生 |
| `linear_interpolation_axiom` | 离散态间连续插值 | native 与 M2-GEMM **混合调度** |
| `operator_composition` | 算子融合 | M4 非卷积下推（次要） |
| `feasibility_conclusion_timeliness`（项目专属） | 设上界、超界回退 | 逆向 3 轮 oracle 辅助 |
| `nonmonotonic_benefit_no_cherry_pick`（项目专属） | 证否即回退 | 任一形状 native 不如 M2 即该形状回退 |
| `measurement_fairness_equivalence`（项目专属） | 公平 A/B | 同权重/图/后处理/governor=performance |
| `silent_fallback_false_pass`（项目专属） | 失败显式、绝不静默 | native 默认 off，失败显式回退 |

**分阶段**：
- **Phase 0（前置，零风险）— RK3588 寄存器 golden oracle 对拍台**：本 A/B 已取得关键资产——板上 **librknnrt 2.3.2 + 官方 `yolov8.rknn` 能正确跑原生卷积**，即 RK3588 权威 regcmd 生成器。取证三选一（按成本）：① 读开源 rknpu 0.9.6 内核驱动 RK3588 的 output/DMA/bank 语义，与 Mesa RK3568 参考**差分**；② 内核 debugfs/MMIO 快照 `rknn_run` 前后 CNA/DPU/CORE 块；③ 最小单卷积 `.rknn` + 寄存器 trace。判据：已知-good `16×8×8` 形状 oracle 与 Phase-1 配方逐位一致 ⇒ oracle 可信。
- **Phase 1（高风险，设上界）— 逆向 output/DMA/bank 三件套**：对每个 FAIL 形状（16×16/16×32×32/16×40×40/16×24×8/OC32/OC64/s4）取 oracle 权威寄存器值，定位差异字段；编码进 `npu_direct_gen_conv_regs`，重跑 `--npu-calib-conv`。上界 3 轮。
- **Phase 2 — 原生卷积接入 YOLO（重做 M3 Phase-2）**：单卷积单提交、stride-2/1×1 分支、C>CBUF 通道分组、槽位打包读回；优先大 P 卷积。
- **Phase 3 — 混合调度**：标定好走 native、其余走 M2 GEMM，按提交数择优。
- **Phase 4 — 验证 + 收口**：J1（max|Δ|=0）/ J2（242/247 配对级）/ J5（目标逼近 15.9ms，至少 <249ms）。

**诚实边界**：唯一追平路径 = 原生卷积，无捷径，收益不预判（J5 实测为准）；若 3 轮逆向不收敛 → 回退 M2（已过 J1–J5、本项已达标），追官方登记为「待 RK3588 逆向突破」，不阻塞。

### 第 1 轮执行记录（2026-10-11，Phase 0 ✅ / Phase 1 ⚠️ 未收敛）

**Phase 0 ✅ 建成真正的 RK3588 卷积寄存器 oracle**（关键突破，价值最高）：
- 参考源盘点：Mesa `rocket`（我们手上是 **RK3568** 分支；mainline 连 RK3568 卷积都未跑通，见 dri-devel 2026-07）→ **不可用作 conv oracle**；OpenNPU v1.0（MIT，`poad42/opennpu_rk3588`）**明确未实现 conv2d**（其文档自述 conv 属未解 RE），只提供 1×1 CNA 描述符 → 仅可作 wire-format/1×1 参考。**唯一权威 = 官方 librknnrt 本身**（已证能跑 `yolov8.rknn` = 15.9 ms）。
- ✅ **实测发现官方 `.rknn` 文件内嵌完整 regcmd**（日志：`subgraph0 regcmdbuffer size: 2229696, taskbuffer size: 42840` = **1071 tasks**），**可直接解析**，无需自建 ioctl tracer。解码格式（实测确认）：**8 字节条目 = `reg(u16)@0 | val(u32)@2 | op(u16)@6`**，op 块码 `0x1001=DPU / 0x2001=DPU_RDMA / 0x0201=CNA / 0x0801=CORE / 0x0101=PC`。
- 工具沉淀：[tools/yolo/rknn_regcmd_dump.py](../tools/yolo/rknn_regcmd_dump.py)（从 `.rknn` 提取 CNA/CORE/DPU 块 + CNA 几何汇总表）。yolov8.rknn 内 **364 个真实卷积 CNA 块**。

**Phase 1 ⚠️ 未收敛**（第 1 轮，未达 3 轮上界但如实登记）：
- **实测包线与（新发现）**：现有 `vllm_npu_direct_conv_i32` 仅 **OH=OW=8** 稳定 PASS（`16×8×8 s1` 与 `16×16 s2→OH=OW=8` 均 PASS）；`OH/OW>8` → DPU 超时 `rc=-2`（dmesg `job timeout`/`soft reset`）；`OC≥32` 或 `C≥64` → 乱码 `rc=0`；**边缘形状run-to-run 非确定**（`16×5×8` 出现 PASS/FAIL/FAIL）。⇒ 故障**纯在输出几何**（同输入 16×16 的 s1/s2 一 FAIL 一 PASS，故输入 DMA 无辜）。
- **单字段旋钮穷举（全暴露的输出几何 env）**：`CORE_H/W`、`CUBE_H/W`、`WDMA_H/W`、`DST_SURF_STRIDE`、`SURFACE_ADD`、`ATOMICS`、`GRAINS`、`LINE_STRIDE`、`SURF_STRIDE`、`CVT0`、`OP_EN_VAL`、`NO_RDMA` —— **对 16×16 全部 FAIL**（8×8 下这些旋钮能证出正确值：`DST_SURF_STRIDE=OW*OH*16`、`ATOMICS=OW*OH`、`LINE_STRIDE=W*4`）。
- **oracle 差分（官方 vs 本引擎，同字段）**——已新增 env 旋钮并逐个采用官方值复测，**16×16 仍 FAIL**：

| 寄存器 | 官方 RK3588（真实卷积样本） | 本引擎（8×8 可用值） | 采用官方值复测 |
|---|---|---|---|
| CNA_CONV_CON1 (0x100c) | `0x6000a000` | `0x1b0` | FAIL |
| CNA_CBUF_CON0 (0x1040) | `0x1b`（wt_bk=1, dt_bk=11，12 bank 全用） | `0xa2`（wt_bk=10, dt_bk=2） | FAIL |
| CNA_CBUF_CON1 (0x1044) | `W*H`（640×34→0x5500=21760） | `(W<<16)\|entries` | FAIL |
| CNA_DATA_SIZE1 (0x1024) | `(C-1)<<16 \| C_al`（Creal=C-1） | `(C_al-1)<<16\|C_al` | — |
| CNA_DMA_CON0 (0x1078) | `0x000f000f` | `0x00171c07` | FAIL（单改） |
| CNA_DMA_CON1 (0x107c) | `W` | `W*4` | FAIL（单改） |
| CNA_DMA_CON2 (0x1080) | `0x63d80`（依赖官方自有特征布局） | `W*4*(H/4-1)` | 未单测 |
| CORE_MISC (0x3010) | `0x1` | `0x300` | FAIL |
| CORE_DATAOUT_SIZE_0 (0x3014) | `(OH-1)<<16 \| (OW-1)`（17×320→`0x0010013f`） | `(2*OH-1)<<16 \| (OW-1)` | FAIL（`CORE_H=15`） |
| CORE_DATAOUT_SIZE_1 (0x3018) | `0x1f`（OC=16 却报 32-1） | `OCp-1`=`0xf` | FAIL |

- **阻塞判读**：差异**不是单字段**，且 `DMA_CON1/2` 依赖**官方自有特征内存布局**（本引擎 NC1HWC2 C2=16 自打包，布局不同，不能直接照搬步长）。⇒ 需**按 oracle 逐字段重建**（尤其特征布局/DMA 步长与 DPU 输出宣告），非 env 微调可解。
- **8×8 无回归**：所有对照组 8×8 仍 PASS；本引擎默认值未改（新旋钮仅 env 覆盖），**生产行为不变**。

**下一轮（第 2 轮）入口**：以 `tools/yolo/rknn_regcmd_dump.py` 的 364 个官方 CNA 块 + 对应 CORE/DPU 块为 golden，逐字段重写 `npu_direct_gen_conv_regs`（先对齐特征布局与 DMA 步长，再对齐 CORE/DPU 输出宣告），用 `--npu-calib-conv` 形状包线回归；上界仍为 3 轮。

### 第 2–3 轮执行记录（2026-10-11，⚠️ 三轮用尽仍未收敛 ⇒ 按计划回退 M2）

**第 2 轮——从 oracle 语料拟合并差分字段规则**（全部在板端实测）：
- **官方普通卷积（stride-1）样本**（1×1 W=160 H=70 Cin=32 OC=32；3×3 W=160 H=140/22 Cin=16 OC=16 等）逐寄存器译码后，**确认本引擎两条公式与官方一致**：`CNA_DMA_CON1 = 4*W`、`CNA_DMA_CON2 = DMA_CON1*(H_full/4-1)`（本引擎无带状 H=H_full ⇒ 不用改）；`DATA_SIZE0/2/3`、`WEIGHT_SIZE*`、`CVT_CON0=0xb`、`PAD_CON0`、`DPU_FEATURE_MODE_CFG=0x1e4`、`DST_SURF_STRIDE=OW*OH*16` 均与官方一致。
- **确认官方差异字段**（已逐个/组合采用官方值复测，**16×16 仍全 FAIL**）：`CONV_CON1`（官方普通层=`0x0`、ARGB 首层=`0x6000a000`；本=`0x1b0`）、`CBUF_CON0`（官方 `0x1b`=wt1/dt11；本 `0xa2`=wt10/dt2）、`CBUF_CON1`（官方≈`W*H` 量级；本 `(W<<16)\|entries`）、`CONV_CON2`(FEATURE_GRAINS)（官方实测值域 **9/10/14/22/26**，远超本引擎 `stride+KH`；本引擎扫描 4–34 **全 FAIL**）、`CORE_MISC`(0x3010，官方=1；本=0x300)、`CORE_DATAOUT_SIZE_0`（官方=`(OH-1)<<16\|(OW-1)`；本 `(2*OH-1)<<16`）、`CORE_DATAOUT_SIZE_1`（官方=`align(OC,32)-1`；本 `OCp-1`）、`SURFACE_ADD`（官方=`2*DST_SURF_STRIDE`；本=`OW*OH*128`）、`DMA_CON0`（官方=`0x000f000f`；本=`0x00171c07`）。
- **结构性差异（不可直接照搬）**：官方路径含 **输入带状（band）分块**（H=34/35 等）、**ARGB 打包**（首层 `CONV_CON1=0x6000a000`）、**DMA/CORE 基于「全张量」而非 band** 的步长/尺寸——与本引擎「单发 + NC1HWC2 C2=16 自打包」布局不同，故官方步长不能 1:1 移植。
- **确定性复核**：大形状 `16×16/16×32×32/16×8×24/16×24×8` **6/6 确定 FAIL**、`16×8×8` **6/6 确定 PASS** ⇒ 非随机/IOVA，而是**特定字段值**问题。

**第 3 轮——协调官方 DPU profile 复测（最后一击）**：
- 新增 DPU 阶段旋钮（`DPU_FMT/BS/BN/EW` 等），采用官方样本值（`DATA_FORMAT=0x2ce0`、`BS_CFG=0x20140`、`BN_CFG=0x20040`、`EW_CFG=0x302`）单改 + 协调组合 + 全 profile（叠加 CORE/CBUF/DMA/CONV1/GRAINS 官方值）——**16×16 仍全 FAIL**，且 **DPU profile 反使 8×8 由 PASS 变 FAIL**（官方 requant 语义与本引擎 identity 通路不兼容）⇒ 证否「照搬官方 DPU 值」路线。
- **对照线索**（未解）：matmul 生成器注释记载过**同症状**（"大 N 每块 job timeout、小 N 通过"，根因 WDMA 字段截断），本引擎 conv 的对应字段（`WDMA_SIZE_0` 通道位）已按同法置 0、非截断；conv 大形状超时的**确切断点仍未定位**。

**第 4 轮（补充诊断，仍不收敛）**：
- **matmul↔conv 生成器差分**（本引擎 matmul 大尺寸可用、conv 仅 8×8 可用，二者差异即嫌疑）：matmul 用 `CORE_DATAOUT_SIZE_0=(M-1)<<16|0`、`CUBE_WIDTH=0`、`FEATURE_GRAINS=M+1`（大值可用）；**实测 8×8 用 `CORE_H=7`(plain) FAIL、`=15`(2*OH-1) PASS ⇒ conv 的「两行单位」是真的**（matmul 因每 M 恰一行而用 plain，不矛盾）。
- **官方 `CBUF_CON1` = `data_entries`（高 16 位=0）**（非-ARGB 样本：W=160,Cin=32 → 80 = `W*ceil(Cin/16)/4`）；本引擎写 `(W<<16)|entries`（高=W）。**补测官方式 `CBUF_CON1=entries`（16×16 → 4）及组合仍 FAIL**。
- ⇒ 单/双字段假设已穷尽（`CONV_CON1/CBUF_CON0/CBUF_CON1/GRAINS/CORE_*/DPU_*/DMA0/DST_SS/SURF_ADD/ATOMICS` 及协调组合），**大形状 16×16 恒 FAIL、8×8 恒 PASS**。判定：**断点非任一单字段，需要复刻官方「带状分块 + ARGB 打包 + 自有特征布局」的整体结构，属大工程**（等于重写 conv 数据通路，超出本项 scope）。

**⇒ 三轮上界用尽（+第 4 轮补充诊断），判「未收敛」**（公理 `feasibility_conclusion_timeliness`）：**回退 M2**（已过 J1–J5、本项已达标关闭），「追官方」登记为**待 RK3588 原生卷积寄存器语义（或 TRM ch.36 / 官方特征布局）突破**，不阻塞。**本引擎默认值未改、生产行为（M2）不变；8×8 基线仍 PASS**。新增 env 旋钮保留（默认关，零行为影响），供后续 round-4 复用。

### 第 5 轮：改攻「M2 提交开销」——**证否**（2026-10-11）

**动机**（用户选择）：不追原生卷积，改为削减 M2 的 371 次微提交以缩小与官方的差距。

**排查与实测（板端，governor=performance）**：
- **前提已部分满足**：`vllm_npu_direct.c` 的 **多核并行提交（`VLLM_NPU_MC`，默认 ON，NB=3，每核一任务、无链式）已内置**；单核链式 `task_number>1` 已知 **DEAD**（0.9.6 实测 + rocket 同结论，代码注释有载）。
- **Nt 上限实验（唯一剩余杠杆）**：`Nt` 受 **HW N≤2048 会 wedge** 约束（当前安全值 1024）。加 `VLLM_NPU_NT_MAX` 试提到 2048 → `yolo_pick_tile` 确选 Nt=2048（首卷积 submits 100→38），但**运行 wedge**（进程未完成、dmesg `soft reset`）⇒ **Nt 不可提，提交数杠杆死**（HW 限制）。
- **决定性对照（证否前提）**：同一 yolov8n/cats，`VLLM_NPU_MC=1`（3 核）npu_call **0.261s** vs `VLLM_NPU_MC=0`（单核）**0.233s** —— **多核零收益（单核反而略快）** ⇒ **npu_call 不由提交/ioctl 开销主导**。
- **量化**：im2col 总量 `Σ_conv P*Kp = 53.1 MB`；连同 conv 权重（~MB 级）与输出读写（Σ P·OC·4 ≈ 数十 MB），合计 ~108 MB DMA / 251 ms ≈ **0.43 GB/s** —— 落在此 NPU「DMA 受限（~1 GB/s）」量级，且**三核无加速** ⇒ **npu_call 受 NPU 侧数据搬运（im2col 流量 + 输出）主导**，非提交数。

**判读**：**「削减提交开销」方向证否**——M2 的多核/最小提交调度已是本设计的工程下限；残余差距的根因是 **im2col 的固有 DMA 膨胀（输入 × KH·KW）**，而这**只能由原生卷积（免 im2col）消除**——即又回到第 1–4 轮未突破的原生卷积语义。⇒ 按 `nonmonotonic_benefit_no_cherry_pick` **不引入该改动**（无收益、有 wedge 风险），代码已回退、默认不变。

**结论收敛**：M2 路径**已到结构上限**；追官方的唯一路径仍是原生卷积（已登记待突破）。本项维持 **✅ 达标关闭**。

### 第 6 轮：差距**重定性**（2026-10-11，`VLLM_NPU_TIMING` 实测）——**有办法缩小，且空间大**

**方法**：用引擎内建 `VLLM_NPU_TIMING=1` 采集 63 次 GEMM pass 的分解（板端，governor=performance）。

**实测（yolov8n/cats，63 pass 合计）**：

| 项 | 时间 | 占 npu_call |
|---|---|---|
| **`t_sub`＝真正的 NPU（ioctl+执行）** | **35.8 ms** | 16% |
| `hide`＝未命名残余（im2col→cube 重排 + 每调用 BO 分配等） | **90.0 ms** | 40% |
| `t_acc`＝CPU 回读+累加（其中 memcpy 32.1 / 转置 3.9） | 44.9 ms | 20% |
| `t_pack`＝regcmd+每 tile BO memcpy | 27.5 ms | 12% |
| `setup` | 26.1 ms | 12% |
| gen/sb/aff | 1.3 ms | — |
| **npu_call 合计** | **225.6 ms** | |

端到端 ≈ **im2col(CPU) 138 ms + npu_call 256 ms = 391 ms**。

**重定性（关键）**：**本引擎 NPU 实际只花 ~36 ms，vs 官方 16 ms ⇒ 仅 2.2×**（不是端到端对比的 23×）。端到端 **~91% 是 CPU 路径**（im2col 138 + GEMM 记账 ~190）。⇒ **与官方的差距不是「NPU 内核慢」，而是「本路径把大量工作留在 CPU」**（官方 conv 任务直接产出 requant 结果，无 im2col、无 CPU 回读/累加/重排）。

**⇒ 缩小差距**：**方向明确、空间大**，但杠杆是**削减/搬走 CPU 路径**，而非优化 NPU 内核。按实测从大到小：

| # | 目标 | 实测 | 手段 |
|---|---|---|---|
| 1 | `hide` 90 ms（未命名） | 40% | 先加细粒度计时定位（im2col→cube 重排 `wkey=0` 每次重跑、每调用 3× BO alloc/mmap）；再融合/并行/缓存 |
| 2 | im2col 构建 138 ms | e2e 35% | 已单遍+OMP；可缓存分块转置/向量化量化 |
| 3 | `t_acc` 回读+累加 45 ms | 12% | 已 NEON/L2 分块/绑 A76；可 P1 流水线重叠 |
| 4 | `t_pack` 27 + `setup` 26 ms | 19% | BO 复用（按尺寸缓存）替代每调用 alloc/free |

**诚实边界**：这些都是**CPU 侧**优化（此前用户口径为「只优化 NPU」）；现已按用户选择（多管齐下含 ⑥ 优化 CPU im2col）开放。**理论天花板**：若 CPU 路径削减 ~3×，e2e 391→~150 ms；叠加 NPU 2.2×（原生卷积）→ 可望逼近官方同数量级。**当前未做任何改动、默认不变**。

### 第 7 轮：CPU 路径优化**已落地**（2026-10-11，✅ 有效、无质量回归）

**定位**：给 `hide`（90 ms）加细粒度计时 ⇒ 主体是 **`wstg`（权重 staging + cube 打包）= 83 ms**，且**串行**。

**改动**（`vllm_npu_direct.c` + `vllm_yolo.c`）：
1. **`CALIB-DIAG` 输出哨兵 memset 改为 env 门控**（`VLLM_NPU_SENTINEL`，默认关）——该 0x7F 图案引擎从不读取，纯诊断，写的是 uncached BO。
2. **跳过冗余权重 staging**：`nb==1 && Np==N` 时 staging 缓冲与 `Wq` 逐字节相同 ⇒ 直接从 `Wq` 打包（省一次全量 memcpy）。
3. **并行化 `npu_pack_weight_int8`**（OMP over `ng`，各 ng 写不相交 dst 区）。
4. **`im2col` 量化去掉逐元素 `lrintf`**：改用 `(int)(f + 2^23) − 2^23`（FE_TONEAREST 下与 `lrintf` **逐位一致**，|f|≤127）+ `omp simd`。
5. 新增 `outz`/`wstg` 细粒度计时（`VLLM_NPU_TIMING`，默认关）。

**实测（yolov8n，板端 governor=performance）**：

| 指标 | 优化前 | 优化后 | 增益 |
|---|---|---|---|
| im2col 构建（CPU） | 138 ms | **78 ms** | 1.77× |
| `wstg`（staging+pack） | 83 ms | **36 ms** | 2.3× |
| `npu_call` | 248 ms | **~200 ms** | 1.24× |
| **serve 端到端 cats** | **395.8 ms** | **312.6 ms** | **1.27×** |
| **serve 端到端 room** | **366.2 ms** | **290.8 ms** | **1.26×** |

**质量门（无回归）**：cats 检测（5 框 cls/score/box）与优化前 **逐位一致**；三版 × cats/room 检出数（v8 5/12、v11 4/13、v5 2/12）与登记一致；`cpu=0 fallback=0`；`--npu-selftest` 全 PASS。**均为算法等价变换（量化逐位一致 + 布局保持并行化 + 字节相同跳步），非近似。**

**剩余目标**（按实测）：im2col 78、`t_acc` 46.6、`wstg` 36、`t_pack` 29、`setup` 25（BO 每调用 alloc/free）。⇒ e2e 仍有下行空间。

**第 7 轮追加（同日）**：
- **另加**：把 3 个每调用 scratch（`apad/tbuf/tmps`，均 >128 KB）改为**每设备常驻**（按需增长；glibc 对 >128 KB 走 mmap/munmap）。GEMM 汇总 wall **178→167.9 ms**（−10 ms，63 pass 聚合，较稳）；但 serve 端到端仍在噪声内（313.8/290.9 ms）。
- **免改代码 env 实验（证否）**：`VLLM_OUT_UC=0` / `VLLM_IN_UC=0`（BO 走 cacheable+sync）**均无收益**（0.205–0.208 s，且 CLI 单跑 npu_call 有 ±10% 抖动）⇒ 现有 uncached 默认已是较优。
- **噪声提示（方法学）**：CLI 单跑 npu_call 抖动 ±10%，**结论须以 serve 多次中位 + 63-pass 聚合为准**。

**第 7 轮累计（可信口径，serve 5 次中位）**：cats **395.8→~313 ms**、room **366.2→~291 ms（−21%，≈1.27×）**；质量为**逐位一致**。**与官方 16 ms 的端到端比 ~19.5×（原 ~24×）**；同口径 NPU 计算 ~33 vs 16 ms（2.1×）。

### 第 8 轮：per-call GEMM BO 复用**已落地**（2026-10-11，✅ 有效、逐位无回归）

**目标**：削 `setup` 的 BO alloc 主体（第 7 轮后 `setup` 仍占 23.6 ms / 63 pass）。M2 路径每次 conv 分配 3 个 BO（`bo_in/bo_w/bo_out`），每个 = `MEM_CREATE + MEM_MAP` ioctl + `mmap`（`vllm_npu_direct_bo_alloc_ex`），全 forward ≈189 次创建/销毁纯开销。

**改动**（`vllm_npu_direct.c`）：
1. `vllm_npu_direct_t` 增 3 个常驻 BO 缓存槽 + 容量 + flags：`cbo_in/cbo_w/cbo_out`（`cbo_*_cap`、`cbo_*_fl`）。
2. 新增 `npu_bo_acquire()/npu_bo_release()`（**移动语义**）：acquire 在「`cache->mapped && cap >= need && flags 匹配`」时把缓存 BO **移出**给调用方，否则放掉旧缓存并新建；release 只在成功路径把 BO **移回**缓存。移动语义下，12 处早退错误路径若持有局部 BO 则天然释放，**缓存永不留悬空映射**（`d` 由 `calloc` 零初始化）。flags 参与匹配（`VLLM_IN_UC`/`VLLM_OUT_UC` 切换时自动失效重建）。
3. 3 个分配点改 `npu_bo_acquire`（flags：`bo_in`=`in_uc?0:NPU_MEM_CACHEABLE`、`bo_w`=0、`bo_out`=`out_uc?0:NPU_MEM_CACHEABLE`）；尾部 3 个 `bo_free` 改 `npu_bo_release`（`bo_w` 别名到持久权重 BO 时 `mapped=0` 直接跳过，不入缓存）；`vllm_npu_direct_close` 释放 3 个缓存槽。

**实测（yolov8n_q8，板端 governor=performance，63-pass 聚合）**：

| 指标 | 优化前 | 优化后 | 增益 |
|---|---|---|---|
| `setup`（BO alloc+malloc+哨兵） | 23.6 ms | **4.9 ms** | 4.8× |
| `npu_call` 汇总 wall | 167.9 ms | **157.8 ms** | −10.1 ms |

**serve 端到端 A/B（同一 harness `bench_ab.py`，n=20，取 median；两版各重建一次）**：

| 指标 | OLD（第 7 轮二进制） | NEW（第 8 轮） | 增益 |
|---|---|---|---|
| cats.jpg median | 509.8 ms | **451.7 ms** | **−11.4%** |
| room.jpg median | 466.5 ms | **411.6 ms** | **−11.8%** |

**质量门（逐位一致，非近似）**：用**同一源码 A/B**（OLD=第 7 轮文件，NEW=＋BO 复用）对 6 组 {yolov8n/yolo11n/yolov5nu × cats/room} 全量 stdout 取 **md5** ⇒ **6/6 全等**；检出数 v8 5/12、v11 4/13、v5 2/12 与登记一致；`cpu=0 fallback=0`；`--npu-selftest` 全 PASS。判读：BO 复用是**纯所有权/生命周期重构**，输出字节不变；缓存复用不依赖零初始化（accumulate 仅读 DPU 已写段的全部字节，任一 submit 失败即返回非零走 CPU 兜底）。

**剩余目标**（第 8 轮后，按实测）：`t_acc` 47.4、`wstg` 34.8、`t_sub` 33.5、`t_pack` 27.9、im2col 78（CPU，独立计时）、`setup` 4.9。⇒ `t_acc`/`wstg` 仍是 CPU 侧大头。

**第 8 轮 env 速探（证否，零代码）**——BO 复用后对既有 env 旋钮做 63-pass 聚合 A/B（`env_ab2.sh`，governor=performance）：`VLLM_NPU_WCACHE=64`（158.6 vs baseline 154.6 ms）、`VLLM_P1=1/2`（155.7/155.0）、`VLLM_IN_UC=0`（156.6）、`VLLM_OUT_UC=0`（160.7）**全部无收益或更差** ⇒ 现有默认已是被测最优。**`VLLM_NPU_MC=0`（单核）在 CLI 63-pass 聚合上 wall 147 vs 156 ms（−6%，3 轮交替复测稳定）看似有效**，但 **serve 端到端 A/B（n=15，`bench_mc.py`）证实无净收益**（cats 449.9/448.7、room 411.2/433.5，MC=1 甚至更好）⇒ 该 wall 差为**计时口径伪影**（MC=1 并行核故 `t_sub` 更低，但端到端被其它项抵消），**不改默认**（多核对 LLM G256 prefill 实测 t_sub/3.5，须保留）。**权重缓存不适用于 M2 conv**：本路径 `Wq=im2col`（每调用变化）⇒ `wkey=0` 是正确设计（`Aq=卷积权重` 走 feature 打包，无持久缓存机制）；官方共享权重缓存的收益只在 LLM decode（同层重复）成立，YOLO 每图 63 conv 各一次不成立。

**第 8 轮 per-conv 分解（`perconv.py`，定位剩余大头）**——63 conv 中最大者 **M=16 N=76800 K=32 G=32（wall 13.1 ms，Nt=1024 ⇒ 75 tiles）**：`sub` 4.7 / `acc` 4.3 / `outz` 1.7 / `wstg` 1.2 ms —— `acc`/`outz` 均受**uncached BO 回读 + fp32 输出 memset**的 DRAM 带宽约束（~1.2 GB/s）。次大组 M=64/80 N=4800 K=576：`pack` 3.0–3.9 / `wstg` 2.6–2.9 / `acc` 2.3–3.1 ms。⇒ 剩余大头是**uncached 搬运**（结构性），非可轻易削减的算力。

### 第 9 轮：消冗余 `memset(out)` —— **证否回退**（2026-10-11）

**动机**：`outz`（`memset(out, 0, M*N*4)`）占 8.6 ms/forward（第 8 轮后实测）；`out` 是逐 K-block 累加目标，本可让**每 tile 首个 block（`g==0`）改用赋值**替代「先整体清零再累加」，省掉该 memset。

**实现**（`vllm_npu_direct.c`）：删 `memset(out)`；4 处累加点（prec0/prec1 × NEON/标量）改 `orow = (g==0 ? 0.0f : orow) + …`。**首次实现 bug**：误用 `g0==0`（首个 block-**组**含 `bcur` 个 block，逐 block 覆盖 ⇒ 只留最后一块）⇒ 质量门当场判死（**检出 dets=0、selftest PASS 仅 1**），修正为 `g==0`（首个 block）后 **md5 6/6 逐位全等**。

**判据（同源码 A/B，交替 3 轮，63-pass 聚合）**：R8 wall 中位 **156.4**（`acc` 46.0、`outz` 8.6）vs R9 wall **154.5**（`acc` **53.5**、`outz` 0）⇒ 省下的 8.6 ms memset **被 `t_acc` +7.5 ms 抵消**，净仅 ~1.9 ms（1.2%，近噪声）。再试**无分支掩码**（`vandq` 清零首块）反使 `acc` 升至 **56.6**（更差）。**判读**：`memset(out)` 并非纯浪费——它以**顺序写**把 `out` 全部 line **写分配进 cache**，等价于替累加路径**预热**；删掉后累加的首读变成 cold miss，总代价几乎不变（memset ≈ 预热成本）。⇒ **证否，按 `nonmonotonic_benefit_no_cherry_pick` 回退**（净增益在噪声内且引入热循环分支/掩码复杂度）。**已回退**：源码与板端二进制还原为第 8 轮（md5 `74eaae0c…`），质量门复验 6/6 一致、`cpu=0 fallback=0`、selftest 全 PASS。**方法学沉淀**：`memset(out)` 类"清零"在写分配 cache 上常是**免费预热**，删除前须以 A/B 验证净收益（勿只看被删计时项）。

---

## 附录 A · 自有格式规格（VQF v4：检测扩展）

> 本附录 = YOLO 自有格式的**单一真源**。部署链**只认本格式**，不引入 ONNX / `.rknn`。
> 复用既有 VQF 容器（[vqf_format.h](../include/model/vqf_format.h)）：4096B 头 + `VQFTensor` 目录（64B/个）+ 64B 对齐数据区 + 尾部 FNV/HMAC，签名/加密机制原样继承。

### A.1 容器布局（与 VQF v2/v3 完全一致，仅 version/arch 语义不同）

```
[0..448)   VQFHeader（magic="VQFW", version=4, flags, n_tensors, arch=VQFArchDet, data_offset, file_len, enc_iv, sig）
[448..)    VQFTensor 目录（n_tensors × 64B：name/qtype/rows/cols/offset/bytes）
[data_offset..file_len)  数据区（64B 对齐，per-tensor qtype 编码）
[file_len..)  尾部：明文 FNV-1a(8B) / ENC 时 HMAC-SM3 tag(32B)
```

**头偏移不变**（`VQFArchDet` 控制在 ≤180B，目录仍起于 448B），因此签名/加密/HMAC 代码路径**零改动**复用。

### A.2 `VQFArchDet`（占用原 `arch` 槽；检测模型专用）

```c
#define VQF_VERSION_DET    4
#define VQF_FLAG_DET       (1u<<10)   /* 检测模型族（VQF v4） */
#define VQF_DET_TASK_DETECT 0
#define VQF_DET_TASK_SEG    1
#define VQF_DET_TASK_POSE   2
#define VQF_DET_FAM_V5      0
#define VQF_DET_FAM_V8      1
#define VQF_DET_FAM_V11     2

typedef struct {
    uint32_t task;        /* VQF_DET_TASK_*（本项先只做 DETECT） */
    uint32_t family;      /* VQF_DET_FAM_* */
    uint32_t input_w, input_h;   /* 网络输入（v5/v8/v11 默认 640×640） */
    uint32_t n_classes;
    uint32_t strides[3];  /* 8/16/32 */
    uint32_t head;        /* 0=anchor-based(v5) 1=anchor-free+DFL(v8/v11) */
    uint32_t reg_max;     /* DFL bins（v8/v11=16；v5=0） */
    uint32_t n_ops;       /* graph 算子数（= graph blob 内 op 计数） */
    uint32_t n_anchors;   /* v5 anchors/尺度=3；v8/v11=0 */
    float    conf_thres, iou_thres;   /* 转换时固化的默认阈值 */
    uint32_t graph_bytes; /* "graph" 张量字节数 */
    uint32_t rsvd[8];
} VQFArchDet;    /* 92B ≤ 180B，头偏移不变 */
```

### A.3 张量命名与 qtype

新增 qtype：`#define VQF_QT_BLOB 0x06`（原始字节，非数值张量）。

| 张量名 | qtype | 含义 |
|---|---|---|
| `graph` | `VQF_QT_BLOB` (0x06) | 图结构（A.4 二进制算子表） |
| `anchors` | `VQF_QT_BLOB` (0x06) | v5 anchors 原始 fp32（v8/v11 可缺省） |
| `names` | `VQF_QT_BLOB` (0x06) | 类别名（换行分隔 UTF-8，诊断/JSON 用） |
| `c%04d_w` | F32(0x01)/Q8(0x02)/F16(0x05) | 第 i 个卷积权重（含 BN 已折叠；row-major [oc,ic,kh,kw]） |
| `c%04d_b` | F32(0x01) | 第 i 个卷积 bias（BN 折叠后；无则缺省） |

> **BN 折叠在转换期完成**（`operator_composition` 公理：Conv+BN 融合为单卷积）——引擎前向无需 BN 算子，减少中间张量读写。

### A.4 graph 二进制算子表（M0 冻结）

紧凑小端，自描述：

```
u32 magic='YGRP'(0x50524759) | u32 version=1 | u32 n_ops
每 op:  u32 opcode | u32 n_in | u32 n_param | u32 out_slot | u32 in_slot[n_in] | u32 param[n_param]
```

- `out_slot`/`in_slot` 为**张量槽位号**（0..S-1），槽位是前向时的中间缓冲索引；槽位大小由 shape 推导（M0 冻结推导规则）。
- opcode 表在 **M0 冻结**（含 golden 逐算子对齐），初版覆盖：
  `CONV / ADD / CONCAT / MAXPOOL / UPSAMPLE / SIGMOID / MUL / SLICE / RESIZE / DETECT_v5 / DETECT_v8 / DETECT_v11`。
- 权重 op 的 `param` 携带 `c%04d` 序号 + stride/pad/groups/act，与 A.3 张量名对应。

### A.5 校验与门禁

- **格式自洽**：加载器按 `(magic, version, data_offset, file_len, n_tensors, 目录偏移)` 自检；与 VQF v2/v3 同规（FNV/HMAC/签名）。
- **图-权一致**：`n_ops`/`graph_bytes` 与 `graph` 张量长度、`c%04d` 张量存在性 MUST 一致，否则拒绝加载（不静默）。
- **数值门**：M0 用 torch golden 对拍（J2）；M1 int8 化后仍过 J2。

### A.6 与 LLM 侧 VQF 的关系

同一 magic、同一头布局、同一签名/加密机制；`version` 区分：`2`=稠密 LLM、`3`=MoE LLM、**`4`=YOLO 检测**。加载器按 version 分支——**v4 不含 LLM arch 字段语义，LLM 加载器遇到 v4 须显式拒绝**（反之亦然）。
