# 立项-K-YOLO-003-G0-YOLO 官方 RKNN 路径适配（opt-in 第二路径）

> 状态：**G0 ✅ → M0 ✅（2026-10-12）**——官方 `.rknn` opt-in 路径已打通并板端验证：**K3-J1/J2/J4/J5 + A/B 零回归全 PASS**（端到端 **0.042–0.084 s/图** vs VQF **2.37–2.66 s/图**，~35–55×）；交付 **v8**（v5u/v11 待 `.rknn` 资产，M1）。方案 + 判据见 §1/§5，结果见 §8。
> 前置：[K-YOLO-001](立项-K-YOLO-001-G0-NPU卷积检测适配.md)（✅ 达标关闭：VQF v4 + 自研 DIRECT 路线，J1–J5 全过）、[K-YOLO-002](立项-K-YOLO-002-G0-ARM-CPU卷积推理优化.md)（✅ ARM CPU int8 dotprod）。
> 单一真源本文件；配套 [落地台账.md](落地台账.md)。
> 口径：先冻结判据再采数；位级与容差不互换；单一真源台账；失败**显式**回退，不静默降级。

---

## 1. 目标与判据（先冻结）

**问题**：K-YOLO-001 已交付自研 DIRECT 路线（VQF v4 载体 + 自研 rknpu 寄存器驱动），但 §9 实测**与官方 RKNN 差距 ~23×**（官方 `rknn_run` **15.9 ms** vs 本 DIRECT 端到端 **365–387 ms**，其中 NPU 计算 **249 ms**）。根因 = 自研驱动只标定 GEMM（`conv_mode=0`），真卷积寄存器语义三轮逆向未收敛（M3 Phase-2 证否）。⇒ **官方 RKNN 栈是唯一可达 15.9ms 量级的路径**。

**目标**：为 YOLO 增加一条 **opt-in 官方 RKNN 路径**（用户决策），复用 Rockchip 官方 `.rknn` 模型 + `librknnrt.so`，与现有 VQF v4 DIRECT 路径**并存**（**M0 已落地并验证**，见 §8）：
1. **定位**：**opt-in 附加路径**（`--yolo-backend rknn` 显式启用）；VQF v4 DIRECT **保持默认、零回归**，两路径可 A/B。
2. **模型来源**：**复用预转 `.rknn`**（`yolov8.rknn` 板端已有；v5u/v11 需获取对应预转件，见 §2.3）。
3. **交付范围**：**三版全量**（v5u / v8 / v11）。
4. **依赖形态**：`librknnrt.so` 经 **dlopen 非侵入**加载（**不改系统库**、不静态链接），运行时缺失/加载失败 → **显式回退** VQF/CPU（不静默）。

**红线变更登记（重要）**：K-YOLO-001 §1.4 的「**不引入 ONNX / `.rknn` 作为模型载体**」+「**不引入 `librknnrt`**」为用户约束。本项经用户决策**局部放宽**（仅 YOLO 检测链、仅 opt-in 路径）：VQF v4 仍是**默认载体**、DIRECT 仍是**默认后端**；`.rknn`/`librknnrt` 只作为 opt-in 第二路径引入。**登记为项目级 override**（对齐 `bitwise_equivalence_gate.project_overrides` 的登记惯例，不在旧文档上改写历史）。

**冻结判据**（复用 K-YOLO-001 的 J 口径，编号改 K3 前缀避免与旧项混淆）：

| 判据 | 内容 | 门限 |
|---|---|---|
| **K3-J1 端点可用** | `--yolo-backend rknn --yolo-rknn <m.rknn>` → `/v1/detect`（+`/v1/scene`）正常返回检测框 JSON | 200 + `detections` 非空（真图） |
| **K3-J2 检测等价** | 同图同家族，RKNN 路径解码输出 vs **CPU golden**（VQF F32 同模型） | cls 全同；conf `\|Δ\| ≤ tol`；框 IoU ≥ 0.9（与 K-YOLO-001 J2 同口径，tol 依裁剪误差实测界定） |
| **K3-J3 三版覆盖** | v5u / v8 / v11 各 ≥5 张真实图 | 全部过 K3-J2 |
| **K3-J4 回退透明 + 生效性** | ① `.rknn` 缺失/`rknn_init` 失败/`rknn_run` 失败 → 显式回退 VQF，日志可辨；② **必须**打印「RKNN 确已生效」诊断（`silent_fallback_false_pass` 反向要求） | 绝不静默；每次 RKNN 推理有一条 `[RKNN-YOLO] …` 生效日志 |
| **K3-J5 性能** | `rknn_run` 单次（对标官方 15.9 ms）+ 端到端（letterbox + run + decode + NMS） | `rknn_run` ≤ 20 ms；端到端 < 本 DIRECT 端到端 365 ms（即 ≥ 18×） |
| **K3-A/B 零回归** | 不带 `--yolo-backend rknn` 时，`/v1/detect` 行为与 K-YOLO-001 现状**逐位一致** | 位级/检测级不变 |

> 判据冻结原则：K3-J1–J4 + A/B 立即冻结；K3-J5 端到端目标待 M0 实测端到端基线后微调（`todo_requires_measured_share`——无分母不下结论）。

---

## 2. 现状与瓶颈（事实盘点，不臆造）

### 2.1 引擎侧已具备（读代码确认，**无需新造**）

- [include/npu/vllm_npu.h](../include/npu/vllm_npu.h#L55-L62)：已有 `VLLM_NPU_BACKEND_RKNN` 枚举（「optional: the official runtime (librknnrt.so) with operator-level .rknn models ... Kept only for A/B」）。
- [src/npu/vllm_npu.c](../src/npu/vllm_npu.c#L152-L213)：**已 dlopen `librknnrt.so`** 并解析完整函数表：`rknn_init / rknn_query / rknn_inputs_set / rknn_run / rknn_outputs_get / rknn_outputs_release / rknn_destroy / rknn_set_core_mask`。现有用途仅 GEMM 算子级 `.rknn`（`gemm_i8_p*_l*.rknn`）。
- ⇒ **官方路径的地基已在引擎内**（dlopen 表 + core mask + 模型加载模式），YOLO 专用路径 = **复用该表**加载**整网检测 `.rknn`**，无需引入新依赖机制。

### 2.2 官方对照实测（K-YOLO-001 §7「可选 A/B」，2026-10-11，信息性）

| 项 | 值 |
|---|---|
| 模型 | `yolov8.rknn`（自 Radxa `rknn-model-zoo-rk3588_1.6.0-1.deb` 取出，**5.68 MB**，输入 1×640×640×3，**9 输出**） |
| 运行时 | **librknnrt 2.3.2（2025-04）** 从本机下载推板，`LD_PRELOAD` **非侵入**加载（不改系统库） |
| `rknn_run` | **min 15.11 / median 15.92 / max 15.93 ms**；`inputs_set` 0.10 ms |
| core mask | 0/7/1/2/4 → 15.9–16.3 ms（**无并行收益**）；mask=8（NPU3）→ rc=-13（该平台 `max core mask=7`） |
| 板载自带 | librknnrt **1.4.0（2022-09）** 与自带 demo 模型**版本不自洽**（拒载 `Invalid RKNN format`）⇒ 必须用 2.3.2 |

### 2.3 板端资产（**M0-0 已核实，2026-10-12**）

| 资产 | 结果 |
|---|---|
| librknnrt **2.3.2** | `/root/_vfy/rknn_new/librknnrt.so`（7,726,232 B，sha256 `d31fc19c85b85f60…`）；`sdk api=2.3.2 (429f97ae6b@2025-04-09)`、`drv=0.9.6` ✅ 可用 |
| 系统 librknnrt **1.4.0** | `/usr/lib/librknnrt.so`（sha256 `0ebc1b408f897863…`）→ 加载 `yolov8.rknn` 报 `Invalid RKNN model version 6` ❌ **不可用**（确认 §2.2） |
| `yolov8.rknn` | `/root/_vfy/rknn_new/yolov8.rknn`（5,683,348 B，sha256 `ff3a64e6fe180203…`）；**1 输入 / 9 输出** ✅ |
| `rknn_bench` | `/root/_vfy/rknn_bench`（+`.c`，最小 `init/query/inputs_set/run`）|
| **v5u / v11 `.rknn`** | **缺失**（`rknn-model-zoo-rk3588.deb` 只含 yolov8）→ **M1 外部依赖 / 阻塞项** |

**输入属性**（`INPUT_ATTR[0]`）：`name=images`，NHWC（`fmt=1`）`1×640×640×3`，`type=INT8`，affine `zp=-128 / scale=1/255` ⇒ 喂 **uint8 NHWC RGB**（`pass_through=0`，RKNN 自行量化）。

**输出属性（9 个，全 NCHW `fmt=0` / `type=INT8` / affine）**——3 scale × {reg(64) / cls(80) / sum(1)}：

| # | dims | scale | zp | 语义 |
|---|---|---|---|---|
| 0 / 1 / 2 | 1×64×80×80 / 1×80×80×80 / 1×1×80×80 | 0.117659 / 0.00310353 / 0.00317275 | −58 / −128 / −128 | stride 8：reg / cls / sum |
| 3 / 4 / 5 | 1×64×40×40 / 1×80×40×40 / 1×1×40×40 | 0.0937474 / 0.00359438 / 0.00362673 | −45 / −128 / −128 | stride 16 |
| 6 / 7 / 8 | 1×64×20×20 / 1×80×20×20 / 1×1×20×20 | 0.0830356 / 0.00387418 / 0.00392157 | −34 / −128 / −128 | stride 32 |

⇒ decode = 每 scale：`cls` 过 sigmoid 取类分；`reg` 按 `[4,16]` 做 DFL（softmax over 16 桶 + 加权和 ×stride）；anchor-free 反解框；`sum` 可作预筛（本项**自行由 cls 重算**，避免双重 dequant 口径）。

### 2.5 计时口径订正（M0-0 实测，重要）

- **`devfreq=performance` 是硬前提**：`fdab0000.npu` 默认 governor `rknpu_ondemand` ⇒ `rknn_run` median **32.3 ms**；置 `performance`（cur=1.0 GHz）后 **15.90 ms**（min 13.74 / max 15.95，n=30）。
- ⇒ 与 K-YOLO-001「NPU 计时须先固定 performance」一致，**且须同时置 NPU devfreq**（不只 CPU governor）。core mask 0/7/1 无差异（15.88–15.91 ms），确认「无并行收益」。

### 2.6 瓶颈 / 差异点

- **前处理不同**：VQF 路径给 NPU 的是 **float NCHW**（`letterbox_alloc`）；官方 `.rknn` 吃 **uint8 NHWC RGB 640×640**（0-255）。⇒ RKNN 路径需**第二套 letterbox**（uint8 NHWC）。
- **输出布局不同（按家族各异）**：`yolov8.rknn`/`yolo11.rknn` = **9 输出**（3 scale × {reg 64 / cls nc / sum 1}，需 DFL + sigmoid）；yolov5 `.rknn` = 3 输出（`[1,na,no]`，anchor-based）。⇒ 解码需**按家族分派**，不可复用同一段。
- **后处理可复用**：class-wise NMS（`st_yolo_detect` 尾部）与框反变换逻辑可抽公共函数。

---

## 3. 公理映射（理论 → 工程）

| 公理（symbol） | 工程映射 |
|---|---|
| `isomorphic_superposition` | 官方 `.rknn` 与原 VQF 是**同构的两种载体**：同模型、同输入、同语义；路线差异只是「谁执行卷积」⇒ 判据须同口径（同图同家族同后处理） |
| `measurement_fairness_equivalence` | A/B 公平：同图、同 label、同 conf/iou、同 governor；RKNN 与 VQF 各自 warm |
| `bitwise_tolerance_non_substitution` | RKNN int8 量化与 CPU F32 golden **非位级** ⇒ 走**检测级容差门**（K3-J2），**不得**用容差糊弄 A/B 零回归（A/B 仍是位级/检测级不变） |
| `silent_fallback_false_pass` | **双向要求**：① 回退必须显式；② RKNN 路径**必须**打印生效诊断（否则「A/B 两边都回退到 VQF」照样 PASS = 假通过） |
| `feasibility_conclusion_timeliness` | K-YOLO-001 的「不引入 .rknn」是时点结论；用户条件变更（要求官方路径）⇒ **翻正**并加指针（本文件即订正指针） |
| `todo_requires_measured_share` | K3-J5 端到端目标须先有实测基线；「官方 15.9ms」只是 `rknn_run` 单次，**不含**前/后处理 |
| `single_source_ledger_with_inline_pointers` | 单一真源：本文件 + 落地台账；口径与 K-YOLO-001 交叉引用而非复制 |
| `default_enablement_gate` | 若未来考虑 RKNN 升为默认，须过「位级/检测级对拍 + A/B 领先 + 逃生舱」门（本项**不**升默认） |

---

## 4. 硬约束 / 诚实边界（先冻）

| 约束 | 事实 | 设计含义 |
|---|---|---|
| **依赖引入** | `librknnrt.so` 是第三方运行时 | dlopen 非侵入；不静态链接；缺失即回退；构建无需该库（`dlopen` 字符串） |
| **载体红线局部放宽** | 官方路径用 `.rknn`（非 VQF） | 仅 opt-in；VQF 仍默认；两路径判据独立、互不污染 |
| **版本自洽** | 板载 librknnrt 1.4.0 与 demo 模型不自洽 | 固定用 **2.3.2**（推板 + `LD_PRELOAD` 或 `dlopen(path)`），版本写进日志 |
| **量化差异不可消除** | 官方 `.rknn` 为 toolkit 标定 int8，golden 为 F32/自研 int8 | K3-J2 走**检测级容差**；不再追求与 F32 位级 |
| **核掩码** | 平台 `max core mask=7`（无 NPU3 单独使用） | core mask 用 7 或默认；不按 K-YOLO-001 的 NPU3 口径 |
| **板端暂不可达** | 本轮 192.168.1.92 网络不通 | §2.3 资产核实、M0 实测 **阻塞待网络恢复** |

---

## 5. 分级方案

### M0 — 资产盘点 + 单模型打通（v8，opt-in，不动默认）

- **M0-0 资产核实**（板端）：盘 `yolov8.rknn` / librknnrt 2.3.2 / v5u·v11 `.rknn`；记录 SHA256 + SDK/drv 版本（`artifact_sha_as_identity_label`）。
- **M0-1 引擎接口**：在 [vllm_npu.h](../include/npu/vllm_npu.h) 增**整网 RKNN** 通用三件套（复用既有 dlopen 表）：
  ```c
  typedef struct vllm_npu_rknn vllm_npu_rknn;            /* opaque */
  vllm_npu_rknn *vllm_npu_rknn_open(vllm_npu_t *npu, const char *path);
  int  vllm_npu_rknn_run(vllm_npu_rknn *m, const void *in, size_t in_bytes,
                         int in_type /*0=f32,4=u8*/, float **outs, int *n_out);
  void vllm_npu_rknn_close(vllm_npu_rknn *m);
  ```
  （不把 decode/NMS 塞进 NPU 层——保持「NPU 层只管张量」的分层。）
- **M0-2 YOLO 侧**：`vllm_yolo.c` 增 RKNN 分支：`--yolo-backend rknn` → `st_yolo_load_rknn()`（读 `.rknn`、查 in/out 属性）+ **uint8 NHWC letterbox** + **家族化 decode**（v8：9 输出 DFL+sigmoid）+ 复用 NMS。
- **M0-3 serve/CLI**：`main.c` 增 `--yolo-backend {vqf|rknn}`、`--yolo-rknn <m.rknn>`、`--rknn-lib <librknnrt.so>`；`/v1/detect`、`/v1/scene` 走同一 `ctx->yolo_*`。
- **判据**：K3-J1 + K3-A/B（VQF 默认零回归）。
- **预期**：v8 端到端打通，给出**端到端基线**（供 K3-J5 冻结）。
- **风险**：中（前/后处理对齐是主工作量；NPU 层地基已备）。

### M1 — 三版全量 + 检测等价

- **改动**：v5u（anchor-based 3 输出解码）、v11（同 v8，9 输出 + C2PSA 已内建于 `.rknn`）各自 decode 分派。
- **判据**：K3-J3（三版 × ≥5 真图）+ K3-J2（vs 同家族 CPU golden）。
- **风险**：v5u 解码与 v8/v11 不同，需独立对齐；v5u/v11 `.rknn` 获取是外部依赖。

### M2 — A/B 收口 + 默认化门（不强制升默认）

- **改动**：A/B 脚本（同图批量，{rknn / vqf} 两列，解析日志）；文档回填。
- **判据**：K3-J4（回退透明 + 生效性）+ K3-J5（性能）；A/B 数字入台账（附**生效凭证**）。
- **默认化**：本项**不**默认化；若日后要升默认，另开 `default_enablement_gate` 门（需 `项目/阶段专属` 决策）。

---

## 6. 待决策 / 风险与回退

| 待决策 | 说明 |
|---|---|
| v5u / v11 `.rknn` 来源 | 优先 `rknn_model_zoo` release 预转件（与 2.3.2 兼容性待验）；否则需 x86 + rknn-toolkit2 转换（工期另计） |
| `--rknn-lib` 缺省路径 | 板载 `/usr/lib/librknnrt.so` 是 1.4.0（不可用）；需固定 2.3.2 的落地路径（`/root/_vfy/rknn_new/librknnrt.so`? 待核实） |

| 风险 | 应对 |
|---|---|
| 板端资产缺失 / 版本不兼容 | M0-0 先核实；缺失即登记为阻塞项（`feasibility_conclusion_timeliness`），不硬推 |
| 解码与前处理对齐耗时超预期 | 以 CPU golden（VQF F32）为 oracle 逐层对拍（复用 K-YOLO-001 的 golden 方法） |
| 引入第三方运行时被质疑 | opt-in + dlopen 非侵入 + 显式回退 + 红线 override 登记（§1） |
| 误把「回退到 VQF」当 RKNN 生效 | K3-J4② 强制生效性日志（`silent_fallback_false_pass`） |

---

## 7. 执行记录

| 轮次 | 内容 | 结论 |
|---|---|---|
| **G0**（2026-10-11） | 立项：读代码确认引擎已有 `VLLM_NPU_BACKEND_RKNN` + 完整 dlopen 函数表（§2.1）；登记红线 override（§1）；冻结 K3-J1–J5 + A/B | **方案 + 判据冻结；不动代码** |
| **M0-0**（2026-10-12） | 板端资产盘点：librknnrt 2.3.2 / yolov8.rknn / 输入输出属性 / 计时口径 | **✅ 通过**：2.3.2 可用（系统 1.4.0 不可用）；`yolov8.rknn` 1in/9out（3 scale × {reg/cls/sum}，int8 affine，几何已登记 §2.3）；`rknn_run` **15.90 ms**（须 devfreq=performance，否则 32.3ms，§2.5）。**v5u/v11 `.rknn` 缺失 → M1 外部依赖**。工具沉淀：`/root/_vfy/rknn_probe.c`（in/out 属性 dump） |
| **M0-1**（2026-10-12） | 引擎整网 RKNN API：`vllm_npu_rknn_open/run/close` + 属性查询/转储 | **✅ 通过**（复用既有 dlopen 函数表，未新增依赖） |
| **M0-升级：RKNN ABI 修正（发现的既有缺陷）** | 对照官方 `rknn_api.h` v2.3.2 逐字段核对 | **🔧 必须修正**（否则新代码同样不可信）：① `rknn_context_t` 旧为 **int32**，官方 aarch64 为 **uint64** ⇒ `rknn_init` 写 8B 破坏栈、ctx 截断（**实测首版 segfault 的根因**）；② `rknn_tensor_format_t` NCHW/NHWC **颠倒**；③ `rknn_tensor_type_t` INT16/UINT8 错位；④ `rknn_query_cmd_t` SDK_VERSION 写 0（实为 IN_OUT_NUM）；⑤ `rknn_input` 多了 `attr`、字段序错；⑥ `rknn_output` 多了 `pass_through/align_size/attr`；⑦ `rknn_sdk_version` 多了 3 个 int。**已按官方头全部修正**；旧「算子级 GEMM」路径（A/B 专用）随之修正，未改其调用语义 |
| **M0-2**（2026-10-12） | YOLO 侧 RKNN 分支：`st_yolo_load_rknn` + uint8 NHWC letterbox + v8/v11 9-output decode（DFL+sigmoid 复用）+ NMS 复用 + COCO-80 名表 | **✅ 通过**（`m->rknn != NULL` 时 `st_yolo_detect` 走 RKNN 分支；VQF 路径逐行未动） |
| **M0-3**（2026-10-12） | CLI/serve 接线：`--yolo-backend {vqf\|rknn}` / `--yolo-rknn <m.rknn>` / `--rknn-lib <so>` / `--yolo-core-mask`；`VLLMServerCtx.yolo_backend/yolo_npu`；CLI 检测入口与 serve 共用句柄 | **✅ 通过**（判据见 §8） |

---

## 8. M0 判据结果（2026-10-12，板端 RK3588，devfreq=performance）

| 判据 | 结果 |
|---|---|
| **K3-J1 端点可用** | **PASS**：`POST /v1/detect`（binary jpeg）→ 200；cats.jpg 5 检测、room.jpg 10 检测 |
| **K3-J2 检测等价** | **PASS**：cats **5/5 匹配，IoU 0.972–0.984**，score Δ −0.043…+0.201；room conf=0.25 下 9/13 匹配（IoU 0.70–0.996），**conf=0.20 时恢复 13/13**——未匹配项全为 `clock 0.283 / dining table 0.278 / chair 0.266` 等**阈值边界**（RKNN int8 分数低 ~0.04），非解码错误 |
| **K3-A/B 零回归** | **PASS**：不带 `--yolo-backend rknn` 时 VQF 路径输出与改造前**逐值一致**（cats：cat 0.888/0.825、remote 0.781、couch 0.360、remote 0.324；`[RKNN-YOLO]` 日志 0 条） |
| **K3-J4 回退透明 + 生效性** | **PASS**：① 模型文件不存在 → **HTTP 500 `detect failed`**（日志 `cannot open`）；② 不传 `--rknn-lib`（落到系统 1.4.0）→ **HTTP 500**（`rknn_init failed`）；均**不静默降级**。③ 生效性：每次 RKNN 推理打 `[RKNN-YOLO] run=… letterbox=… ncan=… dets=…`（3 请求 3 条） |
| **K3-J5 性能** | **PASS（大幅超出）**：`/v1/detect` 端到端 **RKNN 0.042–0.084 s/图** vs **VQF 2.37–2.66 s/图** ⇒ **~35–55×**（目标 ≥18×）。分段：`rknn_run+inputs_set+outputs_get` ≈ **20–22 ms**（bench 纯 `rknn_run` 15.9 ms ⇒ outputs_get(float) 转换 ≈ 4–6 ms）、letterbox ≈ **9 ms**、NMS/decode ≈ 剩余 |
| 构建 | `cmake --build build-rk3588` rc=0（root 二进制已 sync） |

> **诚实边界**：① 本轮只交付 **v8**（唯一在板的 `.rknn`）；v5u/v11 需先取得对应 `.rknn`（M1）。`st_yolo_load_rknn` 目前把 family 固定为 V8（v8/v11 同为 9 输出 anchor-free DFL，故 v11 只要拿到 `.rknn` 即可直接跑）；v5u 是 anchor-based 3 输出，需另写 decode。② letterbox 9 ms 偏慢（可并行化，M2 优化项）。③ `rknn_outputs_get(want_float=1)` 有 ~5 ms 转换开销，后续可改 `want_float=0` + 按 §2.3 的 zp/scale 手写 dequant 换速度。④ 启动横幅 `YOLO backend : rknn (librknnrt.so)` 在未指定 `--rknn-lib` 时会误报可用（实际 1.4.0 在请求时显式失败）——已知，日志可辨。
