# K-EMB-001 句子向量嵌入：Qwen3-Embedding-0.6B 适配 + byte-level BPE 分词器

- **日期**：2026-10-10
- **对象**：`d:\项目\NewVLLM\modl\Qwen--Qwen3-Embedding-0.6B`（`Qwen/Qwen3-Embedding-0.6B`）
- **载体**：kestrel x64（`tools/build/build_x64.ps1` 副产物，仅作可编译+自检回归）+ 板端 RK3588（`/root/_vfy/gitee3`，`build_rk3588.sh`）
- **相关真源**：`docs/落地台账.md`（本项 K-EMB-001 行）

---

## 1. 目标与判据（先冻结）

1. 让引擎支持 **Qwen3-Embedding-0.6B** 的句子向量（embedding）推理：CLI + HTTP 都要。
2. 向量口径必须与官方一致：**last-token pooling + L2 normalize**（`1_Pooling/config.json: pooling_mode_lasttoken=true`、`modules.json` 的 `2_Normalize`）。
3. 质量门：
   - **分词位级门**：引擎 `qwen_tokenizer_encode` 的 id 序列 == `tokenizers`（HF `tokenizer.json`）`encode(text, add_special_tokens=False)`。
   - **端到端保真门**：引擎向量 vs HF `AutoModel`（last-token + L2）的**余弦**。

> ⚠️ 关键事实：**旧引擎分词器不是 BPE**，是「逐位置取最长词表前缀」的贪心匹配，与 HF 的 BPE 合并序不一致（`Explain` → 贪心 `Expla`+`in`，HF 为 `Ex`+`plain`）。因此本项包含一次**引擎级分词器改造**（byte-level BPE）。

---

## 2. 模型兼容性（几何 100% 同构）

Qwen3-Embedding-0.6B 的 `config.json` 为 `Qwen3ForCausalLM`，文本骨干与 Qwen3-0.6B 同构：

| 项 | 值 |
|---|---|
| dim / layers / heads / kv_heads / head_dim | 1024 / 28 / 16 / 8 / 128 |
| ffn_dim / vocab_size / max_pos | 3072 / **151669** / 32768 |
| rope_theta / rms_eps | 1e6 / 1e-6 |
| tied embeddings | **true**（**无 `lm_head` 张量**） |

- 张量命名对比：剥离 `model.` / `model.language_model.` 前缀后，与 Qwen3-0.6B **310 个张量名+形状完全一致**，差异仅 ①embed 行数=151669（0.6B base 151936）②无 `lm_head`。
- 采样口径：`tokenizer_config.json` 的 post_processor 为 `[ByteLevel, TemplateProcessing(single=[Seq, <|endoftext|>])]` ⇒ **encode 恒在尾部追加 EOS=151643**；last-token pooling 取到的就是**该 EOS 位置**。官方 README 的 Transformers 用法据此，实测可**逐位复现**其公布分数（见 §5.3）。

---

## 3. 工程改动

### 3.1 转换器 `vqf_convert`

- `src/model_cfgio.c`：`find_tensor` 增加 `strip_wrap` 归一化回退（把两侧的 `model.` / `model.language_model.` 包裹前缀统一剥掉）⇒ 覆盖**裸名导出**（`embed_tokens.weight` / `layers.N.*` / `norm.weight`）。`st_load_tensor` / `st_load_tensor_slice` 共用该函数，一处改即覆盖全部命名调用。
- `lm_head` 缺失：既有 `st_load_tensor_slice("lm_head.weight", …)` 失败后**回退 `memcpy(token_embed+off)`（tied）** 已覆盖，无需再改。
- `src/conv_main.c`：**修复 Q4-4x4 repack 全局门**。原逻辑「任一矩阵 vc/q_out/k_out/dim/ff 几何不满足则**全局**关闭 q4 repack」在本模型（`vocab_size=151669`，**奇数**）下触发 ⇒ 写出 `flags` 缺 `VQF_FLAG_Q4_4X4` 的文件 ⇒ 被引擎 `vqf_load` 的 layout 校验拒绝（`file=0x21 engine=0x3`）。**引擎侧本身不做全局几何门**（`g_st_q4_repack` 恒 1，只在各内核调用处按 `(rows&3)==0` 逐矩阵门控）⇒ 转换器改为与引擎一致（不做全局门），逐矩阵 repack 自带 `rows%4/cols%32` 门控，几何不合矩阵自动跳过。修正后 `flags=0x23`。
- 落盘：`--wmode dual`（Q8+Q4，与在线主模型同格式）→ `model.vqf` **1219.7 MB**；输出目录须手工放 `config.json` + `vocab.bin` + `merges.bin`。

### 3.2 引擎取值钩子（`src/model/vllm_safetensors.c` / `include/model/vllm_safetensors.h`）

- 新增 `g_embed_out` / `g_embed_dim` + `void st_embed_request(float *out, int dim)`。
- 在 `st_qwen_model_prefill_batch` **收尾**（`normed_b = dyn_rms_norm(末位 final-norm 隐状态)` 之后、`VLLM_DUMP_LOGITS` 之前）插入：把 `normed_b` 做 **L2 归一化**后拷出 `out[0..dim)` 并**一次性清空钩子**。未登记时对生成路径**零行为改变**（lm_head 照常计算）。

### 3.3 CLI（`src/main.c`）

- `--embed <text>` / `--embed-file <p>`（每行一个）/ `--embed-out <p>`（fp32 空格分隔）/ **`--embed-stdin`**（stdin 整段为一个输入，**保 UTF-8**，规避 Windows CRT 把 argv 里的 UTF-8 当 ANSI(GBK) 转码；Linux argv 本无此问题）。
- `run_embed()`：裸文本分词（**不套 chat 模板**）→ 每条输入全量复位 KV → prefill → 取向量。**复刻官方 recipe：分词后追加 `eos_id`**（否则末位落在最后一个实词，向量与 HF 不一致）。
- `VLLM_EMB_IDS=1`：打印 `[EMB-IDS]` 供分词位级对拍（诊断用，env 门控）。

### 3.4 HTTP（`src/serve/vllm_server.c`）

- 新增 `handle_embeddings` + 路由 `POST /v1/embeddings`（OpenAI 兼容 `{"object":"list","data":[{"object":"embedding","index":i,"embedding":[…]}],"model":…,"usage":{prompt_tokens,total_tokens}}`）。`input` 支持 string 或 string 数组；裸文本编码；复用 `ensure_model_ready` / `infer_gate` / `ist_reset` 语义（此处用全量复位）。加入 `is_infer_path`（卸载门控）。`--batch-max` 开启时显式 400（批调度器持独立 engine 锁且无 pooling 分支）。

### 3.5 引擎分词器：byte-level BPE（`src/model/vllm_tokenizer_qwen.c` / `.h`）

- `QwenTokenizer` 增加 BPE 表字段（`bpe_ready` / `bpe_str` / `bm_*` merges 哈希 / `at_*` added_tokens）。
- `tok_bpe_load_bin`：载入 `merges.bin`（开放寻址哈希查 `(left,right)` → rank）。
- `tok_bpe_encode`：
  1. **added_tokens 整串最长匹配**（`<|im_*|>` / `<think>` 等 26 个）→ 直接出 id；
  2. **Qwen 预分词正则近似**（GPT2 家族，g1 收缩 / g2 可选前缀+字母 / g3 单 `\p{N}` / g4 其它段+`[\r\n]*` / g5 `\s*[\r\n]+` / g6 `\s+(?!\S)` 回溯到 R−1 / g7 `\s+`）切 piece；
  3. raw 字节 → **引擎词表域**（见下）；
  4. 按 **merge rank 最小合并**（左优先）；
  5. 逐符号查词表（repr 域）出 id。
  6. **added 硬边界**：piece 不得跨过后面出现的特殊 token 起始（否则 `。`+`<|im_end|>` 被 g4 并成一段）。
- `merges.bin` 缺失 → **回退旧贪心**（行为与改造前一致）；`VLLM_TOK_LEGACY=1` 可强制旧路径（位级回归对照）。
- **引擎词表域（关键发现）**：`vocab.bin` 并非标准 GPT2 repr 域，而是「**原始字节**，仅把 `<=0x20` 的字节做 bytes_to_unicode 编码」（空格→`Ġ`(C4 A0)、`\n`→`Ċ`(C4 8A)；CJK 等 ≥0x80 字节**原样**）。实测该变换与 `vocab.bin` **151643/151643 逐字节一致**。故 `tools/gen_merges_bin.py` 把 `tokenizer.json` 的 repr merges **转到引擎域**再落盘，`added_tokens` 内容按**原始文本**存。

### 3.6 工具（`tools/`）

- `gen_merges_bin.py`：由 `tokenizer.json` 生成引擎 `merges.bin`（`QBM1` 二进制：u32 头 + n_merges × (u16 llen,rlen,bytes) + n_added × (u32 id,u16 len,bytes)）。
- `tok_gate.py`：引擎 ids vs `tokenizers` golden 的位级门（经 `--embed-stdin`，自动去掉末尾 EOS）。

---

## 4. 板端同步与构建

- 板端 `gitee3` 与本地树同一基线（锚点逐处一致）。`vllm_tokenizer_qwen.c/.h`、`vllm_server.c`、`include/model/vllm_safetensors.h` 相对 git **干净** ⇒ 直接覆盖；`main.c`、`vllm_safetensors.c` 有**板端本地改动** ⇒ 以**插入**方式打补丁（`/root/_vfy/apply_emb_patch.py`，幂等、带标记）。原文件已备份至 `/root/_vfy/_emb_bak/`。
- `build_rk3588.sh` **rc=0**（首轮仅 1 处编译错：`vllm_tokenizer_qwen.h` 用 `size_t` 未含 `<stddef.h>`，已修）。
- 板端模型目录 `/root/emb_model/`：`model.vqf`(1219.7 MB) + `config.json` + `vocab.bin` + `merges.bin`。

---

## 5. 质量门结果

### 5.1 分词位级门（20/20）

`tools/tok_gate.py`（x64）与 `board_tokcheck.py`（板端，仅 stdlib，跑同一批 golden）**均 20/20 PASS**。覆盖：golden 句（`A train travels … 2.5 hours …`）、列号/冒号指令串（`Instruct:` / `Query:`）、收缩词（`can't` / `it's`）、中文（`你好，世界！12×8=96。`）、全角标点（`“”（）；，`）、**ChatML 模板全串**、代码缩进（`def add(a, b):\n    return a + b`）、URL/邮箱、数字/小数/负数、多空格/换行段。

> 改造前（贪心）同批仅 12/20，且端到端向量在 `Instruct:/Query:` 类文本上余弦低至 **0.83**。

### 5.2 端到端保真门（vs HF `AutoModel` + last-token + L2，本地 x64）

| 文本 | BPE 前 E2E 余弦 | BPE 后 E2E 余弦 | 仅数值（同 ids 喂 HF） |
|---|---|---|---|
| `hello world` | 0.9973 | **0.9973** | 0.9973 |
| `The capital of China is Beijing.` | 0.9965 | **0.9965** | 0.9965 |
| `Explain gravity` | 0.979 | **0.9965** | 0.9913 |
| `Query:Explain gravity` | 0.904 | **0.9967** | 0.9667 |
| `Instruct: test` | 0.849 | **0.9956**（同批） | 0.982 |
| 完整 `Instruct: …\nQuery:…` | 0.832 | **0.9947** | 0.988 |

⇒ BPE 后 E2E 余弦 **0.994–0.997**（≈仅数值上限）。残余 ≈0.3–0.5% = **Q8_0 权重 × int8 激活**的量化误差（引擎 prefill 用 Q8 整数点积），非分词/几何问题。

### 5.3 README 分数复现 + 板端

- **oracle 自证**：用官方 Transformers 代码（`tokenizer(padding=True)` 默认追加 EOS + `last_token_pool` + L2）复现 model card 公布值 —— `[[0.76456,0.14143],[0.13550,0.59995]]`，**逐位吻合**。
- 引擎 2x2：
  - x64：`[[0.7569,0.1571],[0.1359,0.6083]]`
  - 板端：`[[0.7598,0.1648],[0.1427,0.6317]]`
  ⇒ 与 golden 同量级（差 ≤~0.03，量化+架构浮点差异）。

### 5.4 板端冒烟

`./vllm_kestrel --model /root/emb_model --embed 'hello world'`：`vqf_load OK (dim=1024 …vocab=151669)`、`[TOK] BPE merges loaded: 151387 merges, 26 added tokens`、`tok=3 prefill=0.325s |v|=1.000000`（有限、单位范数）。

### 5.5 HTTP `/v1/embeddings` 实弹压测（板端，serve 8098）

启动：`./vllm_kestrel --serve --auto-load --model /root/emb_model --model-id qwen3-embedding-0.6b --port 8098`（⚠️ 不传 `--model-id` 时响应 `model` 会回落到 serve 硬编码默认 `qwen3-vl-8b`，属既有 serve 行为，部署 embedding 服务务必显式指定）。

| 用例 | 结果 |
|---|---|
| 单串 `{"input":"hello world"}` | **200**；`object=list`、`model=qwen3-embedding-0.6b`、`data[0].index=0`、**dim=1024**、`|v|=1.000000`、`usage={prompt_tokens:3,total_tokens:3}`（3 = BPE 2 + EOS） |
| 数组 `{"input":[…4 段…]}` | **200**；`n_data=4`、dim 全 1024、`usage.prompt_tokens=42` |
| **HTTP vs CLI 同文本** | 4/4 **余弦 = 1.000000**（逐位一致） |
| 缺 `input` | **400** `'input' must be a string or an array of strings` |
| serve 横幅 | 新增 `POST /v1/embeddings  (K-EMB-001; last-token pooling + L2)`（可发现性） |

### 5.6 语义可用性 sanity（检索排序）

用官方推荐格式（query 侧加 `Instruct: …\nQuery:…` 前缀）对 4 篇候选文档排序，**3/3 把正确文档排到第一，且裕度清晰**：

| query | 正确 doc 分 | 其它最高 |
|---|---|---|
| `What is the capital of China?` | **0.760**（doc0） | 0.356 |
| `Explain gravity` | **0.583**（doc1） | 0.195 |
| `Which city is the Eiffel Tower in?` | **0.694**（doc3） | 0.394 |

⇒ 引擎可作**真正可用的句向量/检索嵌入器**（非仅数值对齐）。

---

## 6. 残余限制（诚实披露）

1. **量化残差**：Q8 int8-激活量化 ⇒ E2E 余弦 ≈0.994–0.997（非 1.0）。若需更高保真需 F16/F32 计算路径（当前 VQF 无该档）。
2. **NFC 归一化未实现**：HF `tokenizer.json` 有 `normalizer: NFC`；引擎无。ASCII/CJK 常见文本 NFC 为恒等，含组合字符（如 `é` 的两码点写法）时 ids 可能与 HF 不一致（未纳入本门用例）。
3. **Unicode 分类近似**：`\p{L}`/`\p{N}` 用工程近似（非 ASCII 默认字母，显式排除常见标点/符号/emoji 块）。门集覆盖英文/中文/全角；罕见族（其它文字/花式符号）未穷举。
4. **`merges.bin` 现为「按模型目录」的可选件**：缺失回退旧贪心。⚠️ **切勿把某个 `merges.bin` 放到可执行文件同目录**（引擎有 exe-dir 回退）——否则**所有**无自身 `merges.bin` 的模型（含在线 chat）会切到 BPE，属未经验证的全局行为变更。
5. **q8-only VQF 观察（未闭合）**：纯 `--wmode q8`（`flags=0x21`）的 embedding 文件在 **x64** 上 `vqf_load` 后 prefill **AV(0xC0000005)**；`--wmode dual`（`flags=0x23`）正常。本项交付用 **dual**。该 AV 是否 x64 专属 / 引擎 q8-only 路径缺陷**未定位**，登记待查。
6. **多模态/EOS 之外的 recipe 未做**：MRL 自定义维度、`prompt_name="query"` 的指令自动前置、32K 长文本、批量 `input` 数组的 batching 优化均未实现（当前逐条串行）。
7. **HTTP `/v1/embeddings` 已实弹压测**（板端 serve 8098，见 §5.5）—— 单串/数组/错误处理/与 CLI 逐位一致均通过。⚠️ 部署需显式 `--model-id`（否则响应 `model` 回落到 serve 默认 `qwen3-vl-8b`）。

---

## 7. 未覆盖（登记备查）

- `merges.bin` 在**在线 chat** 路径上的 BPE 切换与质量回归：**已完成（见 §9）**。
- 与 HF/llama.cpp/vLLM 的**端到端检索任务**（MTEB 子集）对齐——本项只做向量余弦门。
- 长文本（>1K）分块/池化、16K/32K 上下文。
- q8-only AV 的根因定位。

---

## 8. 部署与调用（quickstart）

```bash
# 1) 转换（本地 x64 或板端；输出目录须手工放 config.json + vocab.bin + merges.bin）
vqf_conv --model <hf_dir> --convert-vqf <out_dir>/model.vqf --wmode dual
python tools/gen_merges_bin.py <hf_dir>/tokenizer.json <out_dir>/merges.bin
cp <hf_dir>/config.json <out_dir>/  &&  cp <任一 Qwen 目录>/vocab.bin <out_dir>/

# 2) CLI（裸文本，不套 chat 模板）
./vllm_kestrel --model <out_dir> --embed "hello world"
echo -n "任意中文/多行文本" | ./vllm_kestrel --model <out_dir> --embed-stdin   # UTF-8 安全

# 3) HTTP（OpenAI 兼容）
./vllm_kestrel --serve --auto-load --model <out_dir> \
     --model-id qwen3-embedding-0.6b --port 8098
curl http://127.0.0.1:8098/v1/embeddings -H 'Content-Type: application/json' \
     -d '{"input":["What is 12*8?","Gravity is a force."]}'
```

**调用约定**：
- 输入为**裸文本**；官方检索用法建议 query 侧自行拼 `f'Instruct: {task}\nQuery:{q}'`（引擎不自动前置）。
- 输出向量**已 L2 归一化**（`|v|=1`），余弦相似度 = 内积。
- **`merges.bin` 是每模型目录一份的可选件**：缺失回退旧贪心；**切勿放到可执行文件同目录**（引擎有 exe-dir 回退，会让所有模型切到 BPE）。

---

## 9. 在线 chat 路径启用 byte-level BPE + 对话回归（2026-10-10）

### 9.1 现状与家族事实
- 各 **chat 模型目录原有 `merges.bin` 是 09-06 旧格式**（magic `0x00024f5b`，非当前 `QBM1=0x314D4251`）⇒ 被 `tok_bpe_load_bin` 的 magic 校验**拒绝**，`[TOK] ... greedy longest-prefix (legacy)`。即：**在线 chat 一直走旧贪心分词器**（非 BPE）。
- **家族 tokenizer 逐字节同源**：Qwen3-0.6B/1.7B/4B、Qwen3-VL-2B、Qwen3-Embedding-0.6B 的 `model.merges`（151387 条）与 `added_tokens`（26 条，id 151643–151668）**完全一致** ⇒ 一份 `merges.bin` 通用（`md5=57D71D88ABF1F41D2AAF75A19ADD4D81`，1646277 B）。

### 9.2 改动
- 用当前 [gen_merges_bin.py](../../tools/gen_merges_bin.py) 对各 chat 模型目录**重生成** `merges.bin`（本地 `modl/Qwen3-0.6B`、`1.7B`、`4B`、`Qwen3-VL-2B-Instruct`；板端 `/mnt/VQF/2b`、`/mnt/VQF/8b/serve`、`/mnt/VQF/qwen3-30B-A3B-q4`）。
- 开关：**默认 BPE**；`VLLM_TOK_LEGACY=1` 强制旧贪心（对照用）。仍**按模型目录**生效，未放 exe 同目录。

### 9.3 判据（先冻结）
- **分词位级门**：引擎 BPE ids == HF `tokenizers` golden（含 ChatML 模板全串）。
- **对话回归**：同一模型 + 同一二进制，仅 `VLLM_TOK_LEGACY` 不同（A=LEGACY / B=BPE）。硬判据 = 无 `U+FFFD`/控制字符乱码、无特殊 token 泄漏、回复非空；质量 = 算术题正确数 **B ≥ A**。

### 9.4 结果
- **分词门（Qwen3-0.6B 目录）**：BPE **20/20 逐位一致**（含 ChatML 模板）；对照旧贪心在同句缺 `220`（`at 60`/`2.5` 前空格 token 丢失）⇒ 语法上 BPE 才是对的。
- **对话回归（板端 VL-2B，10 例〔8 算术 + 2 开放〕，max_tokens=256，temp=0）**：**A=8/8、B=8/8**；**乱码/泄漏 A=0 / B=0**；10 题中 **5 题仅措辞不同**（如 #7 A `24 students × 3 pencils each…` / B `24 students × 3 pencils each = 72 pencils`），正确性一致。
- **多模态 smoke（VL-2B，3 张真图）**：`zebra`/`apple`/`stopsign` 两臂**答案相同且正确**（`zebra`/`apple`/`stop sign`）⇒ 视觉链路的文本分词切换无回归。
- **VL-8B text smoke**：`The capital of France is **Paris**…` / `Hello!` / `4`（连贯）。
- **MoE-30B text smoke**：连贯 `<think>` 推理（无乱码）。
- **⚠️ x64 重要披露**：本地 Windows x64 serve 的 chat decode 对本批 VQF 输出退化 `!!!!`（argmax 恒为 id 0 = 数值退化）——**LEGACY 对照同样如此** ⇒ **属 x64 既有问题（与 BPE 无关）**；因此对话 A/B 平台选**板端（生产平台 aarch64）**。x64 仅用于分词门（`--embed-stdin`，不 decode）。

### 9.5 未覆盖（登记备查）
- 板端 **VL-8B / MoE 仅 text smoke**（未跑全量对话回归）；MoE 首请求**冷页入极慢**（17.6 GB 权重，13-token prefill **204 s**）。
- 其它语言、超长上下文、多轮对话的 BPE 回归。
- x64 chat decode 退化 `!` 的根因（与本项无关，另行登记）。

## 10. 板端性能（RK3588，2026-10-10）

**口径**：`--serve --port 8099 --model /root/emb_model --load-on-use`（`model.vqf` 1.279 GB；dim1024/28L/vocab151669）；governor=**performance**（基准后已复原 ondemand）；`POST /v1/embeddings`，每档 warmup 1 次 + 3 次取 median；`usage.total_tokens` 计算吞吐。用 `tools` 侧临时脚本 `bench_new.py`（板端 stdlib）。

**单请求延迟 + 吞吐**（单条目，`input=[text]`）：

| 输入 | tokens | OMP=1 | OMP=4 | OMP=8 |
|---|---|---|---|---|
| `Hello world` | 3 | 247.8 ms (12.1 tok/s) | 215.8 ms | 222.9 ms |
| 中 (~150 tok) | 150 | 2712 ms (**55.3 tok/s**) | **950 ms (157.9 tok/s)** | 1184 ms (126.7 tok/s) |
| 长 (~594 tok) | 594 | 12528 ms (47.4 tok/s) | **3922 ms (151.4 tok/s)** | 4973 ms (119.5 tok/s) |
| 批 8×150 | 1200 | 21518 ms (55.8 tok/s) | **6549 ms (183.2 tok/s)** | 9225 ms (130.1 tok/s) |

**线程扩展**：prefill 吞吐 1→4 线程 **≈2.85×**（55→158 tok/s），**4→8 线程反降 0.8×**（158→127）——RK3588 为 4×A76 + 4×A55 异构，8 线程会把 A55 拉进 GEMM 拖慢。**最优 OMP=4**。批 8 相对单条串行的微弱增益（单条 0.95 s × 8 = 7.6 s vs 批 6.55 s，**≈1.16×**）来自一次权重读取摊销。
**吞吐上界**：单请求 **~1.05 req/s**（150 tok 档 @4 线程）；prefill **~183 tok/s**（8×150 @4 线程）。
**RSS**：稳态/峰值 **880 MB**（`VmHWM 880956 kB`）——远小于文件 1.279 GB（VQF 分块/量化驻留）。
**冷启动**：CLI `--embed` 每次含整模加载（进程级），不适合作为稳态延迟口径；生产走 HTTP 常驻。

**未覆盖**：并发多请求（队列）吞吐、fp16 vs Q8 档对比、`/v1/embeddings` 数组输入的更大 batch、A76/A55 绑核（taskset）细分。
