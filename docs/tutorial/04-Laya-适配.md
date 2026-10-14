# 04 · Laya · 适配篇：非自回归决策模型的「逐字钉死」

> 系列教程 · 第 4 篇 ｜ [返回总览](01-总览.md) ｜ 下一篇：[05 Laya·优化](05-Laya-优化.md)
> 真源：[K-LAYA-001](../K-LAYA-001-非自回归决策模型-Laya-ModernBERT适配.md)

前两篇的嵌入模型是「因果解码器 + 池化」。这一篇换一种形态：**非自回归**的编码器模型，一次前向直接输出**类型化决策**。

---

## 1. 原理：什么是「非自回归决策模型」

`convaiinnovations/laya`（英文根检查点）是一个 **System-1 决策模型**：

```
输入：一个 state（文本/邮件/JSON）+ 若干带类型的问题
输出：每个问题的**类型化答案 + 校准概率**（不生成文本，单次前向）
```

问题是三种类型：

| 类型 | 语义 | 输出 |
|---|---|---|
| `choice` | 多选一 | argmax 选项 |
| `score` | 打分 | 期望值 |
| `noul` | 是非 | p[1] |

**与生成式模型的关键差别**：没有 decode 循环、没有 KV cache、没有 token 生成。整条链路就是「拼序列 → 一次编码 → 在每个选项的 marker 位打一个分 → 按温度校准」。这让它天然适合板端（无自回归延迟），但**计算量集中在一整段编码上**——这正是下一篇优化的战场。

---

## 2. 模型解构：把结构「逐字钉死」

适配编码器模型的第一原则：**结构以官方权重导出的 graph 为准，不凭记忆硬编码**。骨干是 **ModernBERT-large**：

| 项 | 值 |
|---|---|
| 层数 / d / heads / head_dim | 28 / 1024 / 16 / 64 |
| inter（MLP 中间维） | 2624 |
| vocab / max_pos | 50368 / 8192 |
| 归一化 | **LayerNorm**（非 RMSNorm），`norm_eps=1e-5`，**norm_bias=false** |
| `is_causal` | **false**（双向） |

四个「与 Qwen 不同」的点，每一个都可能让结果错得面目全非：

### 2.1 交替注意力（full / sliding）

`layer_types` 里**每 3 层一个 `full_attention`**（层 0/3/…/27），其余是 `sliding_attention`：

```
滑窗 = local_attention/2 = 64，掩码 |q-kv| ≤ 64（含边界）
```

### 2.2 双 RoPE theta

```
full 层  : theta = 160000
sliding 层: theta = 10000
inv_freq[j] = 1 / θ^(2j/hd),   emb = cat(freq, freq)  覆盖 hd=64
```

### 2.3 GLU-MLP（GELU 精确 erf 版）

```
input, gate = Wi(x).chunk(2)      →  Wi = [5248, 1024]
out = Wo( gelu_exact(input) * gate )  →  Wo = [1024, 2624]
```

注意是 **`erf` 精确 GELU**，不是 tanh 近似——用近似会在 logits 上引入系统性偏差。

### 2.4 层 0 的 `attn_norm = Identity`

由 `embeddings.norm` 承担首层归一化 ⇒ 逐层权重指针解析时要特别处理第 0 层。

### 2.5 决策头（从零训练）

```
h += type_emb[qtype]                                  # type_emb[3]，加到每个位置
2 × nn.TransformerEncoderLayer(norm_first=True,
      activation=relu, ffn=4096)                       # 标准 MHA，带 bias，无 RoPE
scorer = LN → Linear(1024,1024) → GELU → Linear(1024,1)   # 在每个 [MASK] marker 位打分
```

### 2.6 输入拼接与校准

```
[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
markers = 各 [MASK] 的位置
```

输出侧：每选项 logits → 按 `temperature_by_options[bucket]`（choice:3-5=1.760 / score:3-5=1.251 / noul:2=1.983 …）缩放 → softmax → `choice` 取 argmax、`score` 取期望、`noul` 取 p[1]。

> **教学点**：温度是 **per-(qtype, #opts)** 的查表，不是全局常量。漏掉它，概率就永远对不上官方。

---

## 3. 工程实现：三段式

### 3.1 先造 golden（Python 参考）

`tools/laya_ref.py`：本地加载 tokenizer + `DecisionModel`，复现官方 `system_one`，导出 golden（ids / markers / encoder hidden / marker hidden / logits / probs）+ `*.cases.txt`（短/长两场景）。

> **原则**：**先有 golden，再写 C**。否则「引擎对不对」没有裁判。

### 3.2 打包权重

`tools/laya_pack.py`：206 张量 → `laya_f16.bin`（842.6 MB F16）+ `laya_manifest.txt`。

### 3.3 引擎侧：新增独立决策路径

| 文件 | 作用 |
|---|---|
| `include/model/vllm_laya.h` / `src/model/vllm_laya.c` | 权重加载 + ModernBERT 编码器前向 + 决策头 + scorer |
| `src/main.c` | `--laya-pack` / `--laya-cases` / `--laya-tok` / `--laya-decide` |
| `src/serve/vllm_server.c` | `POST /v1/systemone` + 懒加载 `LayaRunner`（专用锁串行） |

### 3.4 分词器复用（重要的架构巧合）

ModernBERT 用 **ByteLevel BPE**，与 Qwen 同属 GPT-2 家族 ⇒ **复用引擎里已实现的 BPE 引擎**，只换预分词规则：

- 新增 `pretoken_span_gpt2` + `qwen_tokenizer_encode_gpt2`；
- `tools/gen_modernbert_bin.py` 生成引擎域 `vocab.bin`(50368) + `merges.bin`(50009 merges / 116 added)。

> **教学点**：把「BPE 引擎」和「预分词规则」解耦，第二个同族模型几乎零成本接入。这是第 02 篇分词器改造的**复用红利**。

### 3.5 一条性能注记（预示下一篇）

初版把 F16 权重**逐元素**在标量内层转 F32 ⇒ 480-token 用例 **~14 min**。改为「**载入即整表转 F32 arena**」后 **<2 min**——因为 matmul 内层不再有逐元素转换，可被编译器向量化。

---

## 4. 实测结果（质量门）

### 4.1 J1 加载 / J2 batch 不变性

- **J1**：206 张量 strict 加载 `missing=[] unexpected=[]`。
- **J2**：官方 batch（含 padding）vs 逐条 batch=1，`max|Δlogits| = 9.06e-6` ⇒ **引擎可用 batch=1 免 padding**（少一个等价性变量）。

### 4.2 J3/J4 数值 + 覆盖门 6/6

`tools/laya_gate.py` → **6/6 PASS**（短 + 长两场景 × choice/score/noul）：

| 用例 | 类型 | 序号 | max&#124;Δlogits&#124; | max rel | argmax |
|---|---|---|---|---|---|
| s_department | choice | 59 | 7.2e-6 | 1.0e-6 | 0 = 0 |
| s_churn_risk | noul | 56 | 2.5e-5 | 6.7e-6 | 1 = 1 |
| L_department | choice | 480 | 6.7e-6 | 1.5e-6 | 0 = 0 |
| L_churn_risk | noul | 477 | 4.0e-6 | 5.1e-7 | 0 = 0 |

长例 **470–480 token > 129** ⇒ **必须触发滑窗层**且过门 ⇒ `sliding window=64` 常量与掩码语义被实证。

### 4.3 分词器位级门 26/26 + 端到端 d=0

- **分词**：`qwen_tokenizer_encode_gpt2` vs HF `tokenizers` **逐 id 全等 26/26**（含标点/缩进/大小写/长空格 run/`|||EMAIL_ADDRESS|||` 等 added token）。
- **端到端**：`--laya-decide` vs 官方 `RLAgent.system_one` **6/6 PASS，全部 `d=0.0e+00`**（逐位一致）。例：department→`billing`(0.9865)、urgency→`1.7722`、churn_risk→`0.8790`。
- **HTTP**：`POST /v1/systemone` → 200，body 与 CLI 逐字节一致；`LayaRunner` 懒加载一次后复用；非法 JSON → 400；未带 `--laya-pack` → 404。

### 4.4 板端部署（RK3588 aarch64）

`./build_rk3588.sh` rc=0；板端分词/decide/HTTP 三项门全过，输出与 x64/golden **逐字节相同**；回归：Qwen 分词器（K-EMB-001 的 BPE 路径）**20/20 PASS**（未回归）。

---

## 5. 本篇要点回收

| 教学点 | 内容 |
|---|---|
| 形态先判 | 非自回归 ⇒ 无 decode/KV；一次编码出决策 ⇒ 优化面是「整段编码」 |
| 结构以 graph 为准 | 交替注意力 / 双 RoPE theta / LayerNorm / erf-GELU / 层 0 Identity，任一错都致命 |
| 温度是查表 | per-(qtype, #opts)，漏掉就永远对不上官方 |
| **先 golden 再写 C** | 没有裁判的适配 = 盲写 |
| 解耦复用 | BPE 引擎与预分词规则解耦 ⇒ 第二个同族模型零成本接入 |
| 性能注记 | 「载入即整表转 F32 arena」14min→2min，说明**内层类型转换**是大坑 |

**残余限制（诚实披露）**：`act_head` 未接入（README 自述信号不可用，AUROC 0.30）；`multilingual`/`typed-decisions` 未适配；NFC 归一化未做；本阶段性能仍为**标量 F32 单线程**（板端单问题 ~19 s）——这直接引出下一篇。
