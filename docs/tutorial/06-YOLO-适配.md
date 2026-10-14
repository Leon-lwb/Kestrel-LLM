# 06 · YOLO · 适配篇：从官方 `.pt` 到自有 VQF 检测格式

> 系列教程 · 第 6 篇 ｜ [返回总览](01-总览.md) ｜ 下一篇：[07 YOLO·优化](07-YOLO-优化.md)
> 真源：[K-YOLO-001](../立项-K-YOLO-001-G0-NPU卷积检测适配.md)

前四篇是「编码器/解码器」类模型。这一篇换到**卷积检测网络**，而且面对一条最硬的红线：**部署格式必须是引擎自有 VQF，不引入 ONNX / `.rknn`**。这意味着适配 = **自己写 `.pt` 解析器 + 自己实现整条卷积前向**。

---

## 1. 原理：为什么卷积检测是「另一种工程」

| 维度 | LLM（Qwen/Laya） | YOLO 检测 |
|---|---|---|
| 主体算子 | 矩阵乘（GEMM） | **卷积**（kernel>1 / stride / pad） |
| 计算形态 | 权重 × 少数激活（memory-light） | 权重 × 大特征图（含 im2col 放大） |
| 后处理 | 采样/池化 | **DFL 解码 + anchor 解码 + NMS** |
| 版本差异 | 结构基本一致 | v5 anchor-based / v8 anchor-free+DFL / v11 有 **C2PSA 注意力** |

**关键事实**：引擎既有的 NPU 通路（DIRECT 驱动）**只标定过 GEMM（1×1 卷积 / FC 模式）**，`kernel>1` 的 CNA `conv_mode≠0` 字段**完全未标定**。所以「上 NPU」不是调库，而是攻坚（第七篇的主角）。

---

## 2. 三版 YOLO 的算子共性

三版骨干是同一族算子，**卷积占绝对主导**：

| 算子 | 说明 | 落点 |
|---|---|---|
| **Conv(k,s,p) + BN + SiLU** | 3×3 / 1×1 为主 | **核心攻坚** |
| **C2f / C3 / C3k2** | 多 bottleneck + concat | 由多个 Conv + Concat 组成 |
| **SPPF** | 3×maxpool 级联 | DPU/CPU |
| **Upsample / Resize** | nearest | CPU |
| **Concat** | 通道拼接 | CPU（拷贝） |
| **Detect head** | 1×1 conv + DFL + anchor(±) 解码 | 1×1 conv 可复用 GEMM；DFL/解码 CPU |

> **原则**：结构以**官方权重经解析得到的 graph 为准**，不凭记忆硬编码。差异点（v5/v8/v11）必须按版分别适配。

---

## 3. 适配的四个步骤（M0）

### 3.1 自有格式 VQF v4（检测扩展）

**红线**：部署/运行时载体必须是引擎自有格式；`.pt` 官方原始权重是唯一数据源，由**离线工具直接解析**为自有格式；`ultralytics`/`torch` **仅作离线解析与 golden 参考**，不进部署链。

`tools/yolo/yolo_to_vqf.py` 用 `torch`/`ultralytics` **读取 `.pt` 的模块图与权重**，按 VQF v4 规格写出：

```
graph 张量     : 算子序列 + 超参（kernel/stride/pad/通道…）
权重张量目录   : 按 qtype（F32 / Q8_0 / F16）编成张量
```

**不经过 ONNX / `.rknn`**——这是本项与「主流做法」最大的不同，也是全部工作量的来源。

### 3.2 C 引擎前向

新增 `src/model/vllm_yolo.c`，实现：

```
Conv / BN / SiLU
C3 / C2f / C3k2（多分支 + Concat）
SPPF（maxpool 级联）
Concat / Upsample
Detect: 1×1 conv + DFL + anchor(±) decode + NMS
```

`vqf.c` 扩展 v4 分支读取检测模型。

### 3.3 serve 端点

`vllm_server.c` 新增 `POST /v1/detect`（图片 base64 → 检测框 JSON）。图片解码复用既有的 stb_image。

### 3.4 判据（先冻结）

| 判据 | 内容 | 门限 |
|---|---|---|
| **J1 算子数值** | 单算子（conv/maxpool/silu/concat/resize/DFL） vs CPU 参考 | F32 位级或紧容差 |
| **J2 检测等价** | 同图同权重，输出框（cls/conf/xyxy） vs CPU golden | IoU 匹配 + conf `|Δ| ≤ tol`，无 NaN/Inf |
| **J3 覆盖** | v5 / v8 / v11 各 ≥5 张真图 | 全部过 J2 |
| **J4 回退透明** | NPU 不可用/失败 | 显式回退 CPU，日志可辨，**绝不静默** |
| **J5 性能** | 单图端到端（板端 serve 常驻，t4@A76） | **待基线采数后冻结** |

---

## 4. int8 权重量化（M1）：省了 75% 体积，CPU 反而变慢

### 4.1 做法

转换器 `--q8`：**per-output-channel 对称 int8**（`collapse_priority_rule`：跨通道值域大时必须细粒度 scale，否则失真）：

```
权重存 c%04d_w (QT_Q8) + c%04d_s (F32 scale[oc])；bias 恒 F32
```

引擎侧 `yconv_q8`：int8 权重 × **F32 激活**，逐 oc 一次 scale。C2PSA 注意力的 qkv/proj/pe 经 `conv_w_f32` 反量化（注意力占比小，不值得量化）。

**产物体积**：yolov8n **12.04 → 3.07 MB（−75%）**；yolo11n 10.00 → 2.58 MB（−74%）。

### 4.2 J2 质量门（int8 vs ultralytics `predict`，同阈值）

| 模型 | 匹配 | 检测集 | maxIoU | score 差 | 框差 |
|---|---|---|---|---|---|
| yolov8n_q8 | **18/18** | 与 F32 **完全相同**（5/13） | ≥0.959 | ≤0.073 | ≤2.2 px |
| yolo11n_q8 | **17/17** | 与 F32 **完全相同**（4/13） | ≥0.937 | ≤0.033 | ≤2.5 px |

⇒ **量化坍缩在 YOLO 检测上是安全的**（不改变检测集）。

### 4.3 但是：CPU 上 int8 反而慢 8–11%

| 模型 | F32 cats/room | Q8 cats/room | 变化 | RSS |
|---|---|---|---|---|
| yolov8n | 2352.7 / 2140.6 ms | 2537.9 / 2306.5 ms | **+7.9% / +7.8%** | 538 → 529 MB |
| yolo11n | 2020.9 / 1816.0 ms | 2253.7 / 2003.4 ms | **+11.5% / +10.3%** | 299.5 → 291.4 MB |

**诚实判读**（`nonmonotonic_benefit_no_cherry_pick`）：

- CPU 路径是 **compute-bound（非带宽）**；标量 int8 多出 int8→float 转换开销，且**当时没有 dotprod int8 内核** ⇒ **int8 在 CPU 上无收益**（只省内存 −2%）。
- ⇒ **int8 的价值在 NPU**（NPU 的 int8 GEMM 才是加速面）；CPU int8 内核（dotprod）当时判定**不投入**。

> **教学点**：这个「int8 CPU 反而慢」的结论，正是第七篇 CPU dotprod 优化要**反转**的东西——它说明「无 dotprod 内核的 int8」只是伪 int8。有 dotprod（`vdotq_s32`）后，int8 才真正变快（14×）。

---

## 5. 本篇要点回收

| 教学点 | 内容 |
|---|---|
| 格式红线改变工作性质 | 不做 ONNX/`.rknn` ⇒ 要自己写 `.pt` 解析器 + 整条卷积前向 |
| 结构以解析为准 | v5/v8/v11 差异（anchor / DFL / C2PSA）必须按版分别适配 |
| per-channel int8 | 跨通道值域大 ⇒ 必须细粒度 scale（公理红线） |
| **量化不改变检测集** | J2 18/18、17/17 ⇒ 检测对 int8 鲁棒 |
| **伪 int8 vs 真 int8** | 无 dotprod 内核时 int8 反而慢 8–11%；有 `vdotq_s32` 后才变快 |
| 判据先冻结 | J1–J4 立即冻、J5 待基线（无分母不下结论） |

**残余（交给第七篇）**：功能闭环后性能仍不可用（CPU 基线 **1.82–2.36 s/图**）；NPU 真卷积通路未标定；`POST /v1/detect` 已可用。
