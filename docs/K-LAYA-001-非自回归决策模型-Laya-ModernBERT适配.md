# K-LAYA-001 非自回归决策模型 Laya（ModernBERT-large + 决策头）适配

> 状态：**L6 已收口（2026-10-10）**——x64 与板端（RK3588 aarch64）全链路打通：分词器位级 26/26、端到端 decide 与官方 `RLAgent` 逐位一致、HTTP `/v1/systemone` 板端冒烟通过。
> 口径：单一真源本文件；配套真源 [落地台账.md](落地台账.md)。

## 1. 目标与判据（先冻结）

**模型**：`convaiinnovations/laya`（英文根检查点）——**非自回归 System-1 决策模型**。输入一个 state（文本/邮件/JSON）+ 若干带类型的问题（`choice`/`score`/`noul`），**单次前向**返回带校准概率的类型化答案。**不生成文本**。

**判据（先冻结，再采数）**
- **J1 加载**：`model.safetensors` 206 张量 strict 加载无损（`missing=[] unexpected=[]`）。
- **J2 batch 不变性**：官方 batch（含 padding）与逐条 batch=1 的 scorer logits 一致（≤1e-5）⇒ 引擎可用 batch=1 免 padding。
- **J3 数值门**（引擎 vs Python golden）：逐选项 logits `max|Δ| ≤ 5e-3` 且 `max rel ≤ 1e-3`；经官方 per-(qtype,#opts) 温度 + softmax 的 **argmax 完全一致**；无 NaN/Inf。
- **J4 覆盖**：短 state（序号 < 滑窗）与长 state（>129 token，触发滑窗层）均须过。

## 2. 模型解构（逐字钉死）

**骨干 ModernBERT-large（双向编码器）**：28 层、`d=1024`、`H=16`、`hd=64`、`inter=2624`、`vocab=50368`、`max_pos=8192`。
- **交替注意力**：`layer_types` 每 3 层一个 `full_attention`（层 0/3/…/27），其余 `sliding_attention`；滑窗 = `local_attention/2 = 64`，掩码为 `|q-kv| ≤ 64`（含边界）。
- **双 RoPE theta**：full → 160000，sliding → 10000；`inv_freq[j]=1/θ^(2j/hd)`，`emb=cat(freq,freq)` 覆盖 `hd=64`。
- **LayerNorm**（非 RMSNorm，`norm_eps=1e-5`，**norm_bias=false**）；层 0 的 `attn_norm = Identity`（由 `embeddings.norm` 承担）。
- **GLU-MLP（GELU 精确 erf 版）**：`input,gate = Wi(x).chunk(2)` → `Wo(gelu(input) * gate)`；`Wi=[5248,1024]`，`Wo=[1024,2624]`。
- **注意力**：`Wqkv=[3072,1024]` 融合（q|k|v 各 1024），`Wo=[1024,1024]`，无 bias，`is_causal=false`。

**决策头（从零训练）**：`type_emb[3]`（加到每个位置）+ 2× `nn.TransformerEncoderLayer(norm_first=True, activation=relu, ffn=4096)`（标准 MHA，带 bias，无 RoPE）+ `scorer = LN → Linear(1024,1024) → GELU → Linear(1024,1)` 在**每个选项的 [MASK] marker 位**打分。

**输入拼接**（`rl_common.build_sequence`）：`[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`；`markers` = 各 [MASK] 的位置。
**输出**：每选项 logits → 按 `temperature_by_options[bucket]`（choice:3-5=1.760 / score:3-5=1.251 / noul:2=1.983 …）缩放 → softmax；`choice`→argmax、`score`→期望、`noul`→p[1]。

## 3. 工程改动

| 文件 | 作用 |
|---|---|
| [tools/laya_ref.py](../../tools/laya_ref.py) | Python golden 参考：本地加载 tokenizer+`DecisionModel`，复现官方 `system_one`，导出 golden（ids/markers/encoder hidden/marker hidden/logits/probs）+ `*.cases.txt`；两场景（短/长） |
| [tools/laya_pack.py](../../tools/laya_pack.py) | 206 张量打包为 `laya_f16.bin` + `laya_manifest.txt`（842.6 MB F16） |
| [include/model/vllm_laya.h](../../include/model/vllm_laya.h) / [src/model/vllm_laya.c](../../src/model/vllm_laya.c) | **新增独立决策路径**：权重加载（载入即转 F32 arena）+ ModernBERT 编码器前向 + 决策头 + scorer；`laya_run_cases()` 测试入口 |
| [src/main.c](../../src/main.c) | `--laya-pack <dir>` + `--laya-cases <file>`（离线跑用例） |
| [tools/laya_gate.py](../../tools/laya_gate.py) | 质量门：跑引擎 + 与 golden 逐项比对（logits 容差 + argmax + NaN） |
| [tools/gen_modernbert_bin.py](../../tools/gen_modernbert_bin.py) | 由 `tokenizer.json` 生成引擎域 `vocab.bin`(50368) + `merges.bin`(50009 merges/116 added) |
| [tools/laya_tok_gate.py](../../tools/laya_tok_gate.py) | 分词器位级门：引擎 BPE vs HF `tokenizers` 逐 id 比对 |
| [tools/laya_decide_gate.py](../../tools/laya_decide_gate.py) | 端到端门：CLI `--laya-decide` vs 官方 `RLAgent.system_one`（choice/score/noul，容差 5e-3） |
| [tools/build/build_x64.ps1](../../tools/build/build_x64.ps1) | 纳入 `src/model/vllm_laya.c` |

**L5 新增引擎面**：
- [src/model/vllm_tokenizer_qwen.c](../../src/model/vllm_tokenizer_qwen.c)：新增 GPT-2/ByteLevel 预分词 `pretoken_span_gpt2` + `qwen_tokenizer_encode_gpt2`（ModernBERT 复用同一 BPE 引擎，仅预分词规则不同）。
- [src/model/vllm_laya.c](../../src/model/vllm_laya.c)：`laya_tok_dump()`（text→ids 打印）、`laya_decide()`（CLI 端到端）、`LayaRunner`（模型+分词器+温度配置**常驻复用**，供 HTTP 长驻）、`laya_runner_decide()`（请求文本→answers JSON 缓冲）。
- [src/main.c](../../src/main.c)：`--laya-tok <dir>`（分词 dump）+ `--laya-decide`（端到端）；serve 模式把 `--laya-pack` 透传到 `ctx.laya_pack`。
- [src/serve/vllm_server.c](../../src/serve/vllm_server.c)：`POST /v1/systemone` 路由 + 懒加载 `LayaRunner`（专用锁串行；未带 `--laya-pack` 时 404）。

**当前边界（重要）**：引擎侧接受**文本**（`--laya-decide` / `/v1/systemone` 内部完成 text→ids）；`--laya-cases` 仍保留为「已分词的 ids」离线数值测试入口。

## 4. 质量门结果

- **J1**：strict 加载 `missing=[] unexpected=[]`。
- **J2**：batch（含 padding）vs batch=1，`max|Δlogits| = 9.06e-6` ⇒ 引擎 batch=1 免 padding 成立。
- **J3/J4**：`tools/laya_gate.py` → **6/6 PASS**。

| 用例 | 类型 | 选项 | 序号 | max&#124;Δlogits&#124; | max rel | argmax |
|---|---|---|---|---|---|---|
| s_department | choice | 3 | 59 | 7.2e-6 | 1.0e-6 | 0 = 0 |
| s_urgency | score | 3 | 49 | 3.7e-6 | 2.1e-6 | 2 = 2 |
| s_churn_risk | noul | 2 | 56 | 2.5e-5 | 6.7e-6 | 1 = 1 |
| L_department | choice | 3 | 480 | 6.7e-6 | 1.5e-6 | 0 = 0 |
| L_urgency | score | 3 | 470 | 1.1e-5 | 4.0e-6 | 2 = 2 |
| L_churn_risk | noul | 2 | 477 | 4.0e-6 | 5.1e-7 | 0 = 0 |

长例（470–480 token > 129）**必须触发滑窗层**且过门 ⇒ `sliding window = 64` 常量与掩码语义已被实证。
参考端到端（Python）：`department→billing 0.9907`、`urgency→blocking`、`churn_risk noul 0.879`（与 README 一致）。

**实现注记**：初版逐元素 F16→F32 在标量内层 → 480-token 例 ~14 min；改为**载入即整表转 F32 arena** 后 <2 min（matmul 可被编译器向量化）。

### L5 门（分词器 / 端到端 / HTTP，2026-10-10）

- **分词器位级门** `tools/laya_tok_gate.py` → **26/26 PASS**：引擎 `qwen_tokenizer_encode_gpt2` 与 HF `tokenizers`（ModernBERT ByteLevel BPE）逐 id 全等（含标点/缩进/大小写/长空格 run/`|||EMAIL_ADDRESS|||` 等 added token）。
- **端到端门** `tools/laya_decide_gate.py` → **6/6 PASS**（短 + 长两场景 × choice/score/noul，容差 5e-3）：全部 `d=0.0e+00`（引擎与官方 `RLAgent.system_one` 逐位一致）。样例：

| 场景 | 问题 | 结果（引擎 / golden） |
|---|---|---|
| 短 | department (choice) | billing（p: billing 0.9865 / technical 0.0080 / other 0.0055）|
| 短 | urgency (score) | 1.7722 |
| 短 | churn_risk (noul) | 0.8790 |
| 长(≈480tok) | department (choice) | billing |
| 长 | urgency (score) | 1.3027 |
| 长 | churn_risk (noul) | 0.0295 |

- **HTTP 冒烟**（x64 本地 `--serve --port 8099 --laya-pack build_laya`）：`POST /v1/systemone` → `200`，body 与 CLI 输出逐字节一致；`LayaRunner` 懒加载一次后复用（日志 `[LAYA] systemone runner ready` 仅一次）；非法 JSON → `400`；未带 `--laya-pack` 时路由 `404`。

**性能注记（实测）**：单次 decide（3 问，短 state）≈ 35 s（标量 F32、无 SIMD/多线程），`LayaRunner` 复用不重复加载，瓶颈在标量 matmul（见 §6-5）。

### L6 板端部署与验证（RK3588 aarch64，2026-10-10）

**部署**：源码树 `/root/_vfy/gitee3`（git 5fadadd + K-EMB 补丁）；本次覆盖 5 文件（`vllm_tokenizer_qwen.c/.h`、`main.c`、`vllm_server.c/.h`）+ 新增 2 文件（`vllm_laya.c/.h`）+ 改 `CMakeLists.txt`（加 `src/model/vllm_laya.c`）；旧文件备份 `/root/_vfy/_laya_bak/`。模型包落 `/mnt/VQF/laya/`（`laya_f16.bin` 842.6MB + `laya_manifest.txt` + `vocab.bin` + `merges.bin` + `rl_agent_config.json`）。`./build_rk3588.sh` **rc=0**，产物 aarch64 ELF 1.105 MB。

**板端判据（全部 PASS）**：
- **分词器一致**：板端 `--laya-tok` 与 x64 逐行 **26/26 完全相同**（x64 侧已与 HF `tokenizers` 逐 id 全等 ⇒ 板端亦等价 golden）。
- **端到端 decide**：板端 `--laya-decide --laya-pack /mnt/VQF/laya --laya-cases req.json` 输出与 x64/golden **逐字节相同**（department billing 0.9865 / urgency 1.7722 / churn_risk 0.8790）；板端 **63 s**（x64 ~35 s，≈1.8×，标量 F32 无 SIMD）。
- **HTTP `/v1/systemone`**：板端 `--serve --port 8099 --laya-pack /mnt/VQF/laya` → 两次请求均 **200 且 body 逐字节相同**、`systemone runner ready` 仅 1 次（复用）、非法 JSON → **400**、`/health` → **200**。
- **回归**：Qwen 分词器（K-EMB-001 的 BPE 路径）板端门 **20/20 PASS**（覆盖 tokenizer 覆盖式替换未回归）。


## 5. 复现（本地 x64）

```bash
# 1) 打包权重（一次）
python tools/laya_pack.py                       # -> build_laya/{laya_f16.bin,laya_manifest.txt}
# 2) golden（一次）
python tools/laya_ref.py --dump build_laya_golden.npz   # -> 同时写 *.cases.txt
# 3) 构建
powershell -ExecutionPolicy Bypass -File tools/build/build_x64.ps1 \
  -Gcc <mingw>\gcc.exe -OutDir <out>
# 4) 引擎单跑 / 质量门
<out>/vllm_kestrel_x64.exe --laya-pack build_laya --laya-cases build_laya_golden.npz.cases.txt
python tools/laya_gate.py <exe> build_laya build_laya_golden.npz
# 5) L5 分词器门（一次生成词表 + 逐 id 比对）
python tools/gen_modernbert_bin.py                # -> build_laya/{vocab.bin,merges.bin}
python tools/laya_tok_gate.py
# 6) L5 端到端门（CLI --laya-decide vs 官方 RLAgent）
python tools/laya_decide_gate.py
# 7) L5 HTTP（需先起服务；未带 --laya-pack 时 /v1/systemone 返回 404）
<out>/vllm_kestrel_x64.exe --serve --port 8099 --laya-pack build_laya
curl -s -X POST localhost:8099/v1/systemone -H "Content-Type: application/json" \
  --data-binary @build_laya_req.json
```

**板端复现（RK3588 aarch64）**：把 `vllm_laya.c/.h`（新增）与 `vllm_tokenizer_qwen.c/.h`、`main.c`、`vllm_server.c/.h`（覆盖）同步到板端源码树，`CMakeLists.txt` 的 `VLLM_SOURCES` 加 `src/model/vllm_laya.c`（`VLLM_HEADERS` 加 `include/model/vllm_laya.h`），模型包放 `<pack>`，然后：

```bash
cd <board_src> && ./build_rk3588.sh           # rc=0
./vllm_kestrel --laya-tok <pack> --laya-cases cases.txt          # == x64/HF
./vllm_kestrel --laya-decide --laya-pack <pack> --laya-cases req.json
./vllm_kestrel --serve --port 8099 --laya-pack <pack>            # POST /v1/systemone
```

## 6. 未覆盖 / 下一步（登记备查）

1. ~~**ModernBERT 分词器移植**~~（**已完成 2026-10-10**）：`tools/gen_modernbert_bin.py` 生成引擎域 `vocab.bin`(50368)/`merges.bin`(50009/116)，`qwen_tokenizer_encode_gpt2` 复用引擎 BPE（ByteLevel 预分词），位级门 26/26。
2. ~~**CLI `--decide`**~~（**已完成**）：`--laya-decide` 端到端，门 6/6（d=0）。
3. ~~**HTTP `/v1/systemone`**~~（**已完成**）：`POST /v1/systemone`，`LayaRunner` 常驻复用，冒烟 200/400/404 全覆盖。
4. **`act_head` 未接入**（`act_probability`，README 自述其信号不可用，AUROC 0.30）。
5. ~~**板端 aarch64 部署（L6）**~~（**已完成 2026-10-10**）：`/root/_vfy/gitee3` 构建 rc=0、分词/decide/HTTP 三项板端门全过（见 §L6）。**性能仍为标量 F32**——未做 SIMD/多线程/量化（F16 权重 842MB → F32 驻留 1.69GB）；板端单次 decide ≈ 63 s（x64 ~35 s）。
6. **其它检查点**：`multilingual`(mmBERT-base) / `typed-decisions` 未适配（用户本轮只要求英文根）。
7. **NFC 归一化**（对 ASCII/Latin 恒等，未做）、`head_max_len` 自适应缩放的边界用例、选项数 >20 的高基数场景未门。
8. **文档/台账登记 L5/L6**（本轮已补）。

## 7. 板端性能（RK3588，2026-10-10）

**口径**：`--serve --port 8099 --laya-pack /mnt/VQF/laya`；governor=**performance**（基准后已复原 ondemand）；`POST /v1/systemone`，每档 warmup 1 次（触发 `LayaRunner` 懒加载）+ 2 次取 median；`req.json`（短 state 97 字符）/ `req_long.json`（长 state 1222 字符 ≈480 tok，触发滑窗层），各 3 个 typed 问题。

| 场景 | 序号≈ | 单请求 median | 单问题 median | 吞吐 |
|---|---|---|---|---|
| 短 state（97 字符） | ~55 tok | **58.0 s** | 19.3 s | 0.017 req/s（0.052 q/s）|
| 长 state（1222 字符） | ~512 tok（截断上限） | **294.9 s** | 98.3 s | 0.0034 req/s（0.010 q/s）|

- **线程扩展**：OMP=1 与 OMP=8 **逐档完全相同**（短 state 58043.5 ms vs 57943.7 ms，差 0.17%）⇒ **确认 Laya 路径为纯标量单线程**（`vllm_laya.c` 无 OpenMP/SIMD），`OMP_NUM_THREADS` 对 Laya 无影响。
- **上下文扩展**：长/短延迟比 **5.08×**（294.9/58.0），小于 token 数比（~9×）——因 2/3 层为滑窗（`|q-kv|≤64`）注意力，未随上下文平方增长。
- **RSS**：稳态/峰值 **~1.66 GB**（`VmHWM 1748796 kB`）——与 F32 arena 1.685 GB 吻合；模型包 F16 文件 842.6 MB 载入即转 F32 常驻。
- **HTTP vs CLI**：CLI `--laya-decide`（单进程，含整模加载）短 state 3 问 **63.0 s**，与 HTTP 稳态 58.0 s 同量级（差 ~8%，含加载/启动）⇒ 常驻复用无额外开销。

**瓶颈判读**：单问题 ~19 s 全在标量 F32（`lay_linear` 逐元素），无并行/向量化；这是当前 Laya 端到端延迟的主因（未做 SIMD/多线程/量化）。
**优化进展**（详见 [K-LAYA-002](立项-K-LAYA-002-G0-性能优化.md) §8，本表数字为其 M0 基线）：M1 SIMD → M2a F16 存储（RSS −50%）→ M4 位置级多线程（4×A76）→ M3 分块 MatMul → M5 int8 dotprod → M5b int8 分块 → M5c 向量化量化/Wo 分块 → M5d 量化块 32→128 → M7 滑窗区间裁剪，板端短 state 3 问 **58.0 s → 0.963 s（60.2×）**、长 state **294.9 s → ~7.13 s（41×）**，RSS ~0.47 GB。**短 state 单问 321 ms（≈ ≤0.3 s 目标）**；M3/M7 输出与基线逐字节一致、M5/M5b/M5c/M5d 决策等价。**收官验证（M8）**：int8 QB=128 vs F16 基线 **32 请求 / 96 决策 100% 一致**；MatMul 双累加链/行块缓存两个进一步实验实测退化已回退 ⇒ MatMul 已达 A76 寄存器/发射平衡点。**K-LAYA-002 已收口（见其 §9）**；仅剩可选 attention Q-block（长 state ~1.1×）。
**未覆盖**：`/v1/systemone` 并发请求（队列）表现、act_head。

---
**修复历史**：初版 C 有两处笔误（宏名断字），已修；`laya_gate.py` 初版按字母序对齐 key 导致误判，改为**按 ids 匹配**用例↔golden（引擎本身自始正确）。
