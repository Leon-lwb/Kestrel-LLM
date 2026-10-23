# Kestrel（红隼）— ARM(aarch64) 边缘多模型推理引擎

[English](README.en.md) | **简体中文**

**纯 C11、零第三方运行时依赖的 ARM（aarch64）CPU 多模型推理引擎（LLM / MoE / 向量嵌入 / 决策 / 检测）。**

> **版本：v1.0（测试版）** | 平台基准：RK3588 开发板

---

## 核心优势

一块 ARM 开发板（RK3588 / Orange Pi 5 Plus）+ 一个约 **1.4 MB** 的单文件可执行程序 = 原生多模型推理服务。

| 核心优势 | 说明 |
|---|---|
| **五类模型同进程承载** | Qwen3-VL 2B/8B（文本+图/视频）、Qwen3-30B-A3B（MoE）、Qwen3-Embedding-0.6B（向量）、Laya（决策）、YOLO v5/v8/v11（检测）|
| **统一联合入口 `/v1/ask`** | 带 `image`→scene、带 `kb_dir`→rag、带 `questions`→decide、带 `query`→chat（可选 `VLLM_ROUTE_POLICY` 语义路由）|
| **逐层推理** | 权重常驻与模型体积解耦：8B 权重常驻 4.19 GB → 0.48 GB（8.7×），serve 峰值仅 0.82 GB |
| **长文本与多轮** | 组合①（L3 驱逐 × 前缀复用）追问轮 prefill 减少 80%~96% |
| **可验证推理** | SM2 供应链签名护权重 + 逐请求 attestation 凭证（schema 3，含请求原文绑定）|

---

## 快速开始

### 1. 构建（RK3588 / aarch64 Linux）

```bash
# 板上原生构建
./build_rk3588.sh                    # 构建 + 复制到 ./vllm_kestrel
./build_rk3588.sh --run-tests        # 构建 + 运行自检

# x86_64 交叉编译（需要 gcc-aarch64-linux-gnu）
./build_rk3588.sh --cross
```

### 2. 权重准备

引擎**只加载 VQF v2 单文件**（需用 `vqf_convert/` 转换）：

```bash
# 转换 safetensors → VQF
cd vqf_convert && ./build.sh
./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
```

模型目录结构：
```
model-dir/
├── config.json
├── model.vqf          # 单文件 VQF v2
└── vocab.bin          # 可选（缺失时使用内置词表）
```

### 3. 启动服务

```bash
# 基础启动（默认端口 8080）
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4

# 逐层驻留省内存模式（只支持明文 VQF）
VLLM_VQF_STREAM=1 ./vllm_kestrel --serve --model <model-dir> --auto-load --wmode q4
```

### 4. 验证

```bash
# 健康检查
curl http://<board>:8080/health

# 对话
curl http://<board>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

---

## 支持的模型

| 类型 | 模型 | 说明 |
|---|---|---|
| **LLM** | Qwen3-VL 2B/8B | 文本 + 图片/视频多模态 |
| **MoE** | Qwen3-30B-A3B | 混合专家模型 |
| **Embedding** | Qwen3-Embedding-0.6B | 句子向量嵌入 |
| **决策** | Laya | ModernBERT-large + 决策头 |
| **检测** | YOLO v5/v8/v11 | 目标检测 |

**端点对应**：
- `/v1/chat/completions` - 对话
- `/v1/embeddings` - 向量嵌入
- `/v1/systemone` - Laya 决策
- `/v1/detect` - YOLO 检测
- `/v1/ask` - 统一联合入口

---

## 性能（2026-09-12 测点）

### 逐层推理内存优势

| 模型 | 全层：权重常驻 | 逐层：权重常驻 | 省内存 |
|---|---|---|---|
| 2B Qwen3-VL | 1.81 GB | **0.39 GB** | 4.6× |
| 8B Qwen3-VL | 4.18 GB | **0.47 GB** | 8.9× |
| 30B-A3B MoE | 11.73 GB | **0.51 GB** | 23.2× |

### 推荐档（8B 逐层）

热态 TTFT 中位 **1.734 s**，tpot 中位 **330.2 ms**，峰值内存 **0.82 GB**。

---

## 安全特性

| 特性 | 说明 |
|---|---|
| **VQF-Enc** | SM4-CTR + HMAC-SM3 存储态加密（`VLLM_VQF_KEY`） |
| **SM2 签名** | 供应链签名，绑定权重来源（`VLLM_VQF_SIGN_PRIV`）|
| **Attestation** | 推理出证（schema 3，含请求原文绑定 SM3 摘要）|

---

## 体量与依赖

| 项目 | 数值 |
|---|---|
| 可执行文件 | **~1.4 MB**（Release, `-O2 -s`）|
| 源码 | 28 个 C 文件（C11，单工程）|
| 运行时依赖 | **零第三方运行时**（仅需 gcc + libm）|
| 部署 | 单文件 + 可选 vocab.bin，拷贝即运行 |

---

## 目录结构

```
├── CMakeLists.txt             # 构建配置
├── build_rk3588.sh            # RK3588 构建入口
├── include/                   # 头文件
│   ├── core/                  # 推理内核（GEMM/Attention/FHE/国密）
│   ├── model/                 # 权重加载、视觉、分词
│   ├── serve/                 # HTTP/管理页/批处理
│   └── npu/                   # NPU 直驱
├── src/                       # C 源码
│   ├── core/                  # 核心推理逻辑
│   ├── model/                 # 模型实现
│   ├── serve/                 # 服务层
│   └── npu/                   # NPU 驱动
├── tools/                     # 工具
│   ├── bench/                 # 基准测试
│   ├── build/                 # 构建工具
│   └── security/              # 安全工具
├── vqf_convert/               # 权重转换工具
└── docs/                      # 技术文档
```

---

## 文档

- [docs/技术文档.md](docs/技术文档.md) - 架构与实现细节
- [docs/优化配置与边界说明.md](docs/优化配置与边界说明.md) - 优化开关与配置
- [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md) - 安全方案

---

## 许可

**Apache License 2.0** - 代码全部公开，可自由使用、修改、分发、商用（含闭源集成），无需另行授权。

历史版本为 AGPL-3.0-or-later，详见 [LICENSE](LICENSE) 与 [LICENSING.md](LICENSING.md)。

---

## 联系

- Email: 398152090@qq.com
- Issues: https://gitee.com/pei-xiaoguang/kestrel-llm/issues