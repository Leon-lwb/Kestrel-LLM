# 立项 K-COLLAB-001 · G0 多模型协作平台（感知 / 记忆 / 决策 / 深度思考）

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-COLLAB-001** |
| **标题** | 多模型协作平台：YOLO（感知）+ Embedding（记忆）+ Laya（决策）+ Qwen LLM（深度思考）的进程内编排 |
| **项目定位** | **边缘云协调平台**（不再对外表述为「单独推理引擎」）：边缘节点 = 可被云端纳管的编排体，内核为自研 CPU LLM 推理引擎；云端为**可选** uplink 回退（默认 `never`） |
| **目标** | 在已有四个独立 serve 端点之上，加一层**进程内编排器**，把「单模型服务」升级为「多模型协作 / 边缘云协调平台」 |
| **阶段** | **G0 ✅ → P0 向量库 ✅ → P1 检索质量 ✅（5/5）→ P2 多模型注册表 ✅ → P1 RAG 链 ✅（J2）→ P3 感知接入 ✅（场景理解链）→ P4 任务路由 ✅（`/v1/ask` 四链）** |
| **日期** | 2026-10-11 |
| **载体** | 板端 `192.168.1.92`，`/root/_vfy/gitee3`；四模型已落地：YOLO（v5u/v8/v11）`POST /v1/detect`、Laya `POST /v1/systemone`、Embedding `POST /v1/embeddings`、Qwen LLM `POST /v1/chat/completions` |
| **前置** | [K-EMB-001/002](K-EMB-001-句子向量嵌入-Qwen3-Embedding-0.6B适配与byte-level-BPE分词器.md)、[K-LAYA-001/002](K-LAYA-001-非自回归决策模型-Laya-ModernBERT适配.md)、[K-YOLO-001/002](立项-K-YOLO-001-G0-NPU卷积检测适配.md) |
| **判定** | **G0 ✅；P0 向量库 J1 位级 ✅；P1 检索质量 5/5 ✅；P2 多模型注册表 J3/J3'/J4 ✅；P1 RAG 链 J2 端到端 ✅（Qwen3-4B-Instruct）；P3 场景理解链 J2/J4 ✅（图→YOLO→叙事→LLM）；P4 `/v1/ask` 四链路由 J2/J4/J5 ✅（路由开销≈0%）** —— **P0–P4 全链路贯通** |

---

## 1. 目标与判据（先冻结）

### 1.1 四角色映射（用户给定的心智模型 → 工程落点）

| 角色 | 模型 | 语义 | 已落地接口 | 板端成本 |
|---|---|---|---|---|
| **眼睛（感知）** | YOLO | 图 → 检测框（cls/conf/box） | `/v1/detect` | ~142 ms/图（CPU dotprod）；q8 **3 MB** |
| **记忆（知识库）** | Qwen3-Embedding-0.6B | 文本 → 1024 维向量 | `/v1/embeddings` | prefill 位级对齐；检索 sanity 3/3 |
| **决策（System-1）** | Laya | state+typed questions → 类型化答案 | `/v1/systemone` | 0.321 s/问；int8 RSS **0.47 GB** |
| **深度思考（System-2）** | Qwen LLM | 生成/多模态/推理 | `/v1/chat/completions` | decode 44–580 ms/tok |

### 1.2 架构现状（**读代码确认，非臆造**）

- `VLLMServerCtx`（[vllm_server.h](../include/serve/vllm_server.h#L80-L140)）是**单模型**设计：`w/cfg/tok/ist` 单一实例。
- Laya（`laya_pack` 路径 + 独立 `LayaRunner`）与 YOLO（`yolo_model` 路径 + `g_yolo`）是**旁路独立加载**，不共享主 `ctx->ist` 的池化。
- **Embedding 复用主 `ctx->ist`**（`handle_embeddings` 走 `ctx->ist` 的 prefill + `g_embed_out` 钩子）⇒ **embedding 与 chat 共用同一 ist**，当前二者必须是同一模型。
- 已有**单模型** load-on-use / auto-unload / 驻留分级（L0–L4），但**无多模型注册表**、**无向量库存储/检索**、**无编排/路由**。

### 1.3 冻结判据

| 判据 | 内容 | 门限 |
|---|---|---|
| **J1 检索正确性** | 建库向量 + query 向量，top-k 余弦排序 vs 暴力参考 | **位级一致**（同实现即同，无浮点近似） |
| **J2 RAG 端到端** | 给定知识库 + query，检索 top-k 正确，LLM 回答**引用正确事实** | 检索 top-1 = 正确文档；回答含关键事实 |
| **J3 数值等价** | 编排器**不改变**各模型单请求数值（embedding/LLM decode 逐位不变） | 位级（编排只是 prompt 拼接 + 调用序） |
| **J4 回退透明** | 向量库缺失/检索失败/模型未加载 → 显式回退**纯 LLM**（不带上下文）并打日志 | 绝不静默降级 |
| **J5 延迟可测** | RAG 链各段（embed / 检索 / LLM）可分解计时 | `VLLM_COLLAB_TIMING=1`（env 门控，默认关） |

---

## 2. 协作架构设计

```
                    ┌──────────────────────────────┐
     用户输入 ────▶ │  任务路由器（P4；先 Laya/规则） │
  (文本/图片)       └──────────────┬───────────────┘
                    ┌─────────────┼──────────────┐
              ┌─────▼─────┐ ┌─────▼─────┐ ┌──────▼──────┐
              │ 感知 YOLO │ │ 记忆 Embed │ │  决策 Laya   │
              │ 图→场景叙事│ │ 向量库检索 │ │ 类型化路由/动作│
              └─────┬─────┘ └─────┬─────┘ └──────┬──────┘
                    │   结构化上下文（文本）      │
                    └─────────────┬──────────────┘
                          ┌───────▼────────┐
                          │ 深度思考 Qwen LLM│ → 综合回答/动作
                          └────────────────┘
```

### 2.1 要补的四个组件（把「裸端点」变成「协作能力」）

1. **向量库存储 + 检索**（`vllm_kvstore`）：建库（文档→向量→持久化）+ 检索（query→top-k 余弦）。
   - **零依赖**：1024 维、万级 doc 暴力余弦在 4×A76 上可行（万级 × 1024 × 4B ≈ **40 MB**，余弦 1024×10000 ≈ 1e7 MAC，单核 <10 ms）。
   - 持久化二进制：`u32 dim | u32 n | [n × (u32 meta_len + meta + dim×f32)]`。
2. **感知→场景叙事转换器**（YOLO 检测框 → 结构化文本）：`a cat at [340,26,640,370] conf 0.87; …`，供决策/推理消费。
3. **中间态文本协议**：四层之间只传**文本/结构化 JSON**，不传隐状态（见 §4 硬约束）。
4. **编排器 + 任务路由**（`vllm_orchestrator`）：把四层按需串联，支持短路与专家接力。

### 2.2 两种编排模式

- **A. Pipeline 串联**：感知→记忆→决策→推理，逐层产出结构化中间态。
- **B. Router 分流 + 专家接力**（贴合项目「系统/任务/路由/专家」偏好）：路由先判复杂度——低复杂度直接由 Laya 回，高复杂度才上 LLM（并注入前几层上下文）。

---

## 3. 公理映射

| 公理 | 工程映射 |
|---|---|
| `tensor_iterated_addition`（跨 question 结构复用） | **多模型编排**：同一请求的多角色接力，公共上下文（检索到的 doc / 场景叙事）只产一次，复用给决策与推理 |
| `collapse_to_classic` + `collapse_priority_rule` | 检索用 L2 归一向量（余弦 = 内积，经典坍缩）；结构化中间态坍缩为文本 |
| `conservation_law_under_collapse`（弱约束 ⇒ 实测） | RAG 注入不破坏 LLM 输出正确性（J2 实测） |
| `operator_composition` | 中间态文本协议 = 把多模型编成「算子链」 |
| `isomorphic_superposition` | 视觉→场景叙事 = 卷积检测坍缩为文本场景图 |
| `silent_fallback_false_pass` | 检索失败显式回退纯 LLM（J4） |
| `measurement_fairness_equivalence` | 各模型 A/B 同口径（同 governor/绑核/中位） |
| `feasibility_conclusion_timeliness` | 每 Phase 设上界；P0 判别不可行即收口 |

---

## 4. 三条硬约束（诚实边界，先冻）

| 约束 | 事实 | 设计含义 |
|---|---|---|
| **只能「文本级」协作** | 跨模型 KV 复用已证否（异模型对齐漂移大，K-KV 系列） | 每层产出文本/结构化，下一层**重新编码**；不做跨模型隐状态传递 |
| **RAM 池化** | 30B-A3B 17.67 GB **超 15.6 GB RAM**；小模型可同进程常驻（emb 1.27 + laya 0.47 + yolo 0.003 + 4B-Q8 2.9 ≈ 4.6 GB） | 「眼/记忆/决策」常驻；「深度思考」按需加载/卸载 |
| **单 `ctx->ist` 现状** | embedding 与 chat 共用 `ist` | P0/P1 先复用（见 §5），P2 才引入多模型注册表 |
| **延迟叠加** | pipeline 各层延迟相加 | 路由短路；深度思考流式；感知/记忆可并行预取 |

---

## 5. 分级方案

> 用户选定：进程内编排器 + 先打通「知识库→深度思考」（RAG）。顺序按依赖：**P0 向量库 → P1 RAG 链 → P2 多模型注册表 → P3 感知/决策接入 → P4 任务路由**。

### P0 — 向量库存储 + 检索（零依赖，独立可用）
- 落地 `vllm_kvstore.c`：建库（文本→`/v1/embeddings` 向量→持久化）+ 检索（query 向量 → top-k 余弦）。
- CLI：`--kb-add <doc>` / `--kb-file <p>` / `--kb-build <dir>` / `--kb-search "<q>" -k N`。
- **判据**：J1（位级）。**上界**：纯内存暴力检索，万级 doc 单核 <10 ms（微基准先行）。

### P1 — RAG 链打通（知识库 → 深度思考）
- 编排器串起：`query → embed(query) → 检索 top-k → 拼 prompt → LLM 回答`。
- **关键决策（先判别）**：embedding 复用主 LLM（`g_embed_out` 钩子，**零新架构**，但检索质量未校准）vs 专用 0.6B（**需 P2 多模型注册表**）。**P1 先用前者验证端到端正确性**。
- 端点：`POST /v1/rag`（`{query, kb_dir, top_k}` → `{retrieved:[…], answer}`）。
- **判据**：J2 / J3 / J4 / J5。

### P2 — 多模型注册表 + 池化（架构底座）
- 把 `VLLMServerCtx` 的「单 ist」扩展为「模型注册表 + 每模型独立 ist/w/cfg/tok + 统一 load-on-use/驻留分级」。
- 使 embedding(0.6B) / LLM(4B) / Laya / YOLO 可**同进程按需共存**（眼/记忆/决策常驻，深度思考池化）。
- **判据**：J3（各模型数值不变）+ 多模型并发/切换正确。

### P3 — 感知 + 决策接入
- 场景理解链：`图 → YOLO 检测 → 场景叙事 → LLM/VL 综合回答`。
- 低延迟分流：`query → Laya 路由/决策 → 复杂才上 LLM`。
- **判据**：J2（各链端到端正确）+ J4（各层失败显式回退）。

### P4 — 任务路由器（收口）
- 统一入口 `POST /v1/ask`：路由到 {检索问答 / 场景理解 / 决策动作 / 深度推理} 四链。
- **判据**：J2 + J4 + J5（路由开销 < 5% 端到端）。

---

## 6. 待决策（进 G1 前须定）

> **已决策（2026-10-11，AskUserQuestion）**：① embedding = **专用 0.6B**（⇒ 先落地 P2 多模型注册表，检索质量已 5/5 验证）；② 任务路由器（P4）= **轻量规则**（按显式请求字段分流，零重训）；③ 编排 API = **独立 `/v1/rag` 端点**。

1. ~~**P1 的 embedding 复用 vs 专用**~~ → ✅ **专用 0.6B**（触发 P2；复用主 LLM 因 4B 非 embedding 训练、检索质量未校准而弃）。
2. ~~**任务路由器（P4）的实现**~~ → ✅ **轻量规则**（`/v1/ask` 按 `image/kb_dir/questions` 显式字段分流四链；Laya 当路由器需重训决策头而弃，见 §12）。
3. ~~**编排 API 形态**~~ → ✅ **独立 `/v1/rag`**（`{query, kb_dir, top_k}` → `{retrieved, answer}`；不改动现有 chat 端点行为）。

---

## 7. 风险与回退

| 风险 | 应对 |
|---|---|
| P1 复用主 LLM 出向量 ⇒ 检索质量不达 J2 | 回退专用 0.6B（触发 P2）；或先冻结检索质量阈值 |
| 多模型注册表（P2）blast radius 大 | 单模型路径**零行为改变**；新注册表 env 门控（默认关），逐模型迁移 |
| RAM 超限（30B + 其它） | 池化调度：深度思考加载时驱逐小模型；或拒绝并发并显式报错 |
| 文本级协作导致「语义损失」（场景→文本丢空间信息） | 诚实登记为已知边界；保留结构化 JSON（框坐标）供下游，不强制扁平化 |
| 延迟叠加不可接受 | 路由短路 + 流式 + 感知/记忆并行预取 |

---

## 8. 执行记录

| 轮次 | 内容 | 结论 |
|---|---|---|
| G0（2026-10-11） | 只读勘察：确认四端点现状 + 单 `ctx->ist` 架构事实 + 无向量库/无编排 | 冻结判据 J1–J5 与四层设计；**不动代码** |
| **P0**（2026-10-11） | **向量库核心落地**（`vllm_kvstore.c/.h` + `--kb-test` 自测） | **全 PASS**（下） |

### 8.1 P0 微基准（板端 `/root/kb.c`，1024 维 × 10000 doc，A76 pinned）

| 形态 | 单核 | 4 核 |
|---|---|---|
| 标量余弦全扫 | **8.98 ms** | — |
| NEON 4 累加器全扫 | 2.49 ms | 1.60 ms |

- **检索远非瓶颈**（万级 doc 标量 9 ms，远小于 LLM decode 数百 ms/tok）。
- **NEON 与标量非位级**（`max|Δ|=1.37e-6`，浮点累加重排），但 **top-1 排序一致**。
- ⇒ **P0 采用标量余弦**（确定性、够快），NEON 留作后续可选（需放宽 J1 为「排序一致」）。

### 8.2 P0 实现（`include/core/vllm_kvstore.h` + `src/core/vllm_kvstore.c`）

- 结构 `VLLMKVStore { dim/n/cap/meta/vec }`；`vllm_kv_add / vllm_kv_search / vllm_kv_save / vllm_kv_load`。
- 检索 = 标量余弦（`for i` 固定累加序 ⇒ 确定性）+ 稳定插入 top-K（并列按下标升序）。
- 持久化二进制：`magic 'VKB1' | u32 dim | u32 n | n×(u32 meta_len | meta | dim×f32)`。
- CLI `--kb-test`（`main.c` run_kb_test）：合成 100×64 向量验证。
- 构建注册：板端 `CMakeLists.txt` 的 `VLLM_SOURCES` + `VLLM_HEADERS` 各加一行。

### 8.3 P0 判据结果（板端，合成向量 100×64，固定 seed）

| 判据 | 结果 |
|---|---|
| **J1 检索排序 == 暴力参考** | **PASS**（top-5 下标全一致） |
| **J1 分数逐位一致** | **PASS**（`memcmp` 全等） |
| **持久化 round-trip**（n/dim/vec/meta） | **PASS** |
| 板端构建 | `cmake --build` rc=0 |

> **方法学沉淀**：P0 的「纯向量库」与「embedding 前向」解耦——用合成向量验证 J1，不混入 embedding 数值；「文本→向量」的接线留到 P1。

### 8.4 P1 文件向量检索质量验证（2026-10-11，板端真实 embedding）

- **形态**：`--embed-file` 用 Qwen3-Embedding-0.6B（`/root/emb_model/model.vqf`，dim=1024，28L）对 8 条真实中文文档 + 5 条 query 各产 1024 维向量（`--embed-out`），`--kb-build` 落库，`--kb-search` 逐 query 检索 top-3。
- **结果：5/5 top-1 全命中，且分数余量清晰**（正确项 ~0.74–0.89 vs 干扰项 ~0.24–0.41）：

| query | top-1（正确文档） | score | 余量 |
|---|---|---|---|
| iPhone 16 用的是什么芯片？ | iPhone 16 / A18 | 0.785 | ≥0.48 |
| 火箭要飞多快才能离开地球？ | 逃逸速度 11.2 km/s | 0.805 | ≥0.50 |
| 怎么做出好吃的红烧肉？ | 红烧肉家常做法 | 0.796 | ≥0.51 |
| 目标检测用什么神经网络结构？ | 卷积神经网络/目标检测 | 0.736 | ≥0.33 |
| 用紫砂壶泡茶要注意什么？ | 紫砂壶泡茶温壶 | 0.889 | ≥0.59 |

- **判读**：真实文本→向量→建库→检索端到端闭环成立，检索质量达标（J2 的「检索 top-1 = 正确文档」子项通过）。P1 剩余 = 编排器把检索结果拼进 LLM prompt（已定：embedding 用专用 0.6B → 先落地 P2 多模型注册表，见 §9）。
- **修复记录**：`run_kb_build` 首行定 dim 后未回填 `kb.dim`（`vllm_kv_init(&kb,0)` 使 dim=0，`vllm_kv_add` 判 `dim<=0` 直接拒报「add OOM」）→ 首行解析后 `kb.dim = dim` 一行修复，板端重建后 8×1024 落库成功。

---

## 9. P2 实现方案（冻结，2026-10-11）

> 决策：embedding = 专用 0.6B（§6 决策 1），故 P1 前先落地 P2。**最小 2 槽位**：主 chat + 专用 embedding 两个 causal-LM 推理状态共存；Laya/YOLO 已用旁路全局变量模式（`g_laya_runner`/`g_yolo`），无需注册表。

### 9.1 架构事实（读代码确认）

- `STQwenInferenceState` **内嵌** `weights` + `cfg`（[vllm_safetensors.h](../include/model/vllm_safetensors.h#L229-L231)），`st_qwen_model_prefill_batch(st,…)` 只收 `st` 不从全局取权重 ⇒ **推理状态本身自包含，多实例可共存**（内存够即可）。
- 当前 serve 单模型：`ctx->w/cfg/tok/ist` 单实例；`handle_embeddings` 复用 `ctx->ist/tok/cfg`。
- 全局单例（需串行保护）：`g_embed_out` 钩子、量化开关（`g_st_prefill_q8` 等）、NPU strip 缓存。⇒ **embedding 与 chat 必须共用同一 `inf_lock` 串行**，不能像 Laya/YOLO 用独立锁。

### 9.2 数据结构（最小 2 槽位）

`VLLMServerCtx` 新增专用 embedding 槽位（`vllm_server.h`）：

```c
const char *embed_model_dir;      /* --embed-model <dir>：专用 embedding 模型；NULL=复用主模型 */
STModelWeights       *embed_w;
STModelConfig        *embed_cfg;
QwenTokenizer        *embed_tok;
STQwenInferenceState *embed_ist;
pthread_mutex_t       embed_lock;   /* 保护 embed_* 加载/释放 */
int                   embed_loaded; /* 0=未加载 1=就绪 -1=失败 */
char                  embed_error[256];
```

### 9.3 迁移路径（零回归）

- `handle_embeddings` 分支：`embed_model_dir==NULL` → 走**现有主模型路径**（零改动）；非空 → 懒加载专用 0.6B 到独立槽位，用 `embed_tok/ist/cfg/w` 出向量（复用同一段 last-token pooling 逻辑，仅换来源指针）。
- 加载在 `infer_gate` 之后、`embed_lock` 内执行（保证与 chat 串行 + 加载互斥）。
- 新增 `--embed-model <dir>` CLI（缺省 NULL ⇒ 现有行为不变）。

### 9.4 判据与回退门

| 判据 | 门限 |
|---|---|
| **J3 数值等价** | 主 chat 路径（`embed_model_dir==NULL` 及 `/v1/chat/completions`）**逐位不变**（不改主路径代码逻辑） |
| **J3' embedding 专用正确** | `--embed-model /root/emb_model` 时 `/v1/embeddings` 向量 == CLI `--embed` 同输入向量（同模型同路径，位级） |
| **J4 回退透明** | 专用模型加载失败 → 显式 503 + `embed_error` 日志，**绝不**静默回退主模型出错误向量 |
| 切换正确 | chat 与 embedding 交替请求，各自数值不变、无全局状态串扰 |

### 9.5 边界（诚实登记）

- 内存：embedding 0.6B（~1.27GB）与 chat 4B（~2.9GB Q8）同进程常驻 ≈ 4.2GB，仍在 15.6GB RAM 内；不做「用后卸载」先，P3 再接驻留分级。
- `serve_load_model` 的 `st_npu_gw_clear_all()` 换模型副作用：embedding 专用模型加载会清 NPU strip 缓存（仅性能，无正确性影响；NPU 默认关）。
- tokenizer：Qwen3 全族 vocab 哈希全等（151669），专用模型独立加载（生命周期解耦），内容与主模型一致。

### 9.6 P2 判据结果（2026-10-11，板端已实施）

| 判据 | 结果 |
|---|---|
| **J3' embedding 专用正确** | **PASS**：`--embed-model /root/emb_model` → `/v1/embeddings` 与 CLI `--embed-stdin` **cos=1.000000（4/4 位级一致）**；日志现 `[EMB-MODEL] dedicated embedding model ready (dim=1024 layers=28)` |
| **J3 主路径回归** | **PASS**：不带 `--embed-model` → 复用主模型（else 分支逐位等价），cos=1.000000，`grep -c EMB-MODEL`=0 |
| **J4 回退透明** | **PASS**：`--embed-model /nonexistent` → **HTTP 503** `{"error":{"message":"no model.vqf in /nonexistent"}}`，不静默回退 |
| 构建 | `cmake --build build-rk3588` rc=0 |

> **诚实边界**：板端当前无 chat VQF（>4B 已删、1.7B/4B 不在板），本验证的主模型=embedding 模型（同模型双角色）。「4B chat + 0.6B embedding **两个不同模型共存**」的端到端验证留待 P1 RAG（需先恢复一个 chat VQF 到板端）。P2 本轮已验证的是**独立槽位懒加载 + 向量位级一致 + 回退透明 + 主路径零回归**四个机制。

---

## 10. P1 RAG 链实现（`POST /v1/rag`）

### 10.1 实现

- 新增 `handle_rag` + `rag_embed_text` 辅助（[vllm_server.c](../src/serve/vllm_server.c)），路由 `/v1/rag` 加入 `server_handler` 与 `is_infer_path`（unload gate 保护）。
- 请求 `{query, kb_dir, top_k, max_tokens}` → 响应 `{retrieved:[{index,meta,score}], answer}`。
- 流程：`embed(query)`（专用 `--embed-model` 槽位或复用主模型，复用 last-token pooling）→ `vllm_kv_load` + `vllm_kv_search` top-k → 拼 `参考资料：…\n问题：…\n回答：` → `build_completion_prompt` + `run_completion`（贪心 0.0 / 主 chat 模型）→ 返回。
- 依赖 P0（`vllm_kvstore`）+ P2（embedding 独立槽位）；`#include "vllm_kvstore.h"`。

### 10.2 判据结果（板端，J2 端到端，2026-10-11）

主模型 = **Qwen3-4B-Instruct-2507**（`/root/chat4bi`，Q8 VQF 4.7GB，新下载+转换），专用 embedding = Qwen3-Embedding-0.6B。

| 判据 | 结果 |
|---|---|
| **J2 检索 top-1 正确** | **PASS**：iPhone query → `retrieved[0]`=iPhone/A18 文档（score 0.785）；红烧肉 query → `retrieved[0]`=红烧肉文档（0.796） |
| **J2 answer 引用正确事实** | **PASS**：iPhone → `answer="iPhone 16 用的是 A18 芯片。"`；红烧肉 → `answer="要做出好吃的红烧肉，可以按照以下步骤操作：1. 准备材料…2. 焯水…"`（引用 retrieved 文档事实，连贯中文） |
| **J3 数值等价** | PASS（embedding 专用槽位向量位级不变，见 §9.6） |
| 构建 | `cmake --build` rc=0 |

### 10.3 关键教训（vocab 钳制证否回退）

- **误判修复**：最初以为「4B/1.7B/VL 的 `model.vqf` vocab=151936 但 tokenizer=151669」是 bug，在 serve 加载时把 `cfg->vocab_size` 钳到 151669。**实测证否**：钳制使 lm_head 计算错位，Instruct 模型 decode 生成 `</think>` 循环。
- **正确结论**：`lm_head` 151936 行（含 267 个无定义 reserved token）与 tokenizer 151669 不一致是**正常的**——Instruct 模型的 argmax 自然落在有效 token（0..151668），无需钳制。reserved token 只在 **base 模型（未微调）** 下命中（生成 `!`/`151935`），属模型能力边界，非引擎 bug。
- **回退**：已移除钳制，`serve_load_model` 保留注释说明（[main.c](../src/main.c)）；板端复测 J2 全 PASS。
- **方法学**：之前 base 模型「生成 reserved token / `!`」的根因是**模型未指令微调**（非引擎/量化 bug），误判为 vocab mismatch 并做了有害修复——「先证伪假设再上修复」再次成立。

---

## 11. P3 感知接入：场景理解链（图 → YOLO → 场景叙事 → LLM，2026-10-11）

> P3 分两子项：**① 场景理解链**（感知接入，本轮完成）与 **② 低延迟分流**（决策接入，依赖 §6 决策 2，见 §11.3）。本轮先落地 ①（无分叉、自包含）。

### 11.1 方案冻结

- **端点**：`POST /v1/scene`。Body `{"image":"<base64 或 data URL>","query":"...","max_tokens":M}`（非 JSON 体按原始图像字节，query 缺省"请描述图中场景"）。Reply `{"model","detections":[{cls,name,score,box}],"scene":"...","answer":"..."}`。
- **场景叙事格式**（§2.1 组件 2）：`<class> at [x1,y1,x2,y2] conf <s>; ...`，无检测 → `（图中未检测到物体）`。文本级协作（§4 硬约束：只传文本，不传隐状态）。
- **链**：`image → media_load_image_memory → st_yolo_detect(g_yolo) → scene_narrative → build_completion_prompt + run_completion(主 chat 模型)`。YOLO 走 `g_yolo_lock`（与 `/v1/detect` 共享 `g_yolo` 单例）；LLM 走 `infer_gate/inf_lock`；两锁不同持。
- **判据**：J2（detections top-1 正确 + answer 引用场景事实）；J4（yolo 未启用 404 / 检测失败 500 / 图解码失败 400 / chat 未加载 503，绝不静默降级）。
- **依赖**：`--yolo-model`（感知）+ 主 chat 模型（深度思考）。`/v1/scene` 已加入 `is_infer_path`（unload gate 保护）。

### 11.2 判据结果（板端，Qwen3-4B-Instruct-2507 + yolov8n.vqf）

| 判据 | 结果 |
|---|---|
| **J2 检测正确** | **PASS**：cats.jpg 5 检测（cat×2 0.89/0.83、remote×2、couch）；room.jpg 13 检测（tv/chair×5/person/potted plant/vase/refrigerator/clock/dining table×2） |
| **J2 叙事正确** | **PASS**：`scene` 字段与 detections 逐条对应，格式 `cat at [341,26,640,370] conf 0.89; …` |
| **J2 answer 引用场景事实** | **PASS**：`"图中有什么？"` → 列举 cat/remote/couch 及置信度；`"图里有猫吗？"` → `"是的，图里有猫。检测结果中提到了两个猫…conf 0.89 和…conf 0.83"`（引用场景叙事，非臆造） |
| **J4 回退透明** | 代码级显式：yolo 未启用 404 / 检测失败 500 / 图解码失败 400 / chat 未加载 503（`ensure_model_ready`）；无静默降级路径 |
| 构建 | `cmake --build build-rk3588` rc=0（root 二进制已 sync） |

### 11.3 ② 低延迟分流（决策接入）→ 并入 P4 统一路由

- 原计划「Laya 判复杂度 → 简单回 Laya、复杂上 LLM」在实现上与 **P4 统一入口 `/v1/ask`** 同构（都是「按请求特征选链」），故**合并为 P4**（见 §12）。
- 决策已定（§6 决策 2 = 轻量规则）：路由只依据**显式请求字段**（image / kb_dir / questions），不做语义猜测（避免 silent mis-route）。
- `POST /v1/systemone`（Laya，0.321s/问）作为 `decide` 链被 `/v1/ask` 直接复用。

---

## 12. P4 任务路由器：统一入口 `/v1/ask`（2026-10-11）

### 12.1 方案冻结

- **端点**：`POST /v1/ask`。Body `{query, image?, kb_dir?, top_k?, max_tokens?, questions?, state?}`（与 §11 scene、§10 rag 同构的**统一 schema**）。
- **路由规则（轻量规则，§6 决策 2）**，只认显式字段，优先级从高到低：

| 条件 | route | 链 | 复用 |
|---|---|---|---|
| `image` 非空 | `scene` | 场景理解（YOLO → 叙事 → LLM） | `handle_scene` |
| `kb_dir` 非空 | `rag` | 检索增强（embedding → 向量库 → LLM） | `handle_rag` |
| `questions` 为非空对象 | `decide` | 决策（Laya 类型化问答） | `handle_systemone` |
| 否则（仅有 `query`） | `chat` | 深度推理（纯 LLM，贪心） | 内置 `run_completion` |

- **响应**：子链响应 + `"route"` 字段（`ask_tag_route` 在子链 JSON 首键插入，自描述「本次由哪条链作答」）。
- **回退透明（J4）**：子链错误码**原样透传**（如 `kb_dir` 不存在 → 404 "vector store load failed"；坏图 → 400 "no image payload"；缺字段 → 400），无静默降级。
- **计时（J5）**：`VLLM_COLLAB_TIMING=1` 时打 `[COLLAB-T] ask route=<r> total=…ms`；RAG 链额外分段 `embed/search/llm`。每请求另打 `[ASK] route=<r>`。

### 12.2 判据结果（板端，Qwen3-4B-Instruct + Embedding-0.6B + yolov8n + Laya）

| 判据 | 结果 |
|---|---|
| **J2 四链路由正确** | **PASS**：`{query}` → `route=chat`（`"2 + 3 等于 5。"`）；`{query,kb_dir}` → `route=rag`（`retrieved[0]` iPhone/A18 0.785，`answer="iPhone 16 用的是 A18 芯片。"`）；`{image,query}` → `route=scene`（`answer="图里有猫、遥控器和沙发。"`）；Laya 请求 → `route=decide`（`department=billing 0.9857 / urgency 1.7613 / churn_risk…`，与 K-LAYA-001 一致） |
| **J4 回退透明** | **PASS**：坏 `kb_dir` → **404**；空 body → **400**；坏图 → **400**（均为子链原始错误，显式透传） |
| **J5 延迟可分解 + 路由开销** | **PASS**：RAG 链 `embed=712.9ms search=0.1ms llm=11679.0ms total=12391.9ms`；`/v1/ask` 总 `12392.1ms` ⇒ **路由开销 ≈0.2ms（≈0%）< 5%** |
| 构建 | `cmake --build build-rk3588` rc=0（root 二进制已 sync） |

> **实现说明**：`/v1/ask` 复用三个既有 handler（不重写链逻辑），故三条非 chat 链的数值/行为与直连各自端点**完全一致**（J3 数值等价天然成立，无重复实现漂移）。
> **诚实边界**：`/v1/ask` 不支持流式（用于非流式编排验证）；chat 兜底沿用 `resolve_thinking`（默认思考，与 `/v1/chat/completions` 一致），需精确控制请用原端点。

---

## 13. 公理库审视：优化空间评估（2026-10-11）

> 依据 `axiom_registry.json`（通用公理库）+ `kestrel_prefill_axioms.json`（prefill 性能公理库），对 K-COLLAB-001 已落地的 P0–P4 逐条审视「是否还有优化空间」。结论：**协作平台层的性能优化空间已闭合；唯一真实空间是一条「回退透明」缺陷**（silent_fallback_false_pass），非性能问题。

### 13.1 逐条审视（候选优化点 × 公理 × 实测/结论）

| # | 候选优化点 | 公理依据 | 实测/依据 | 结论 |
|---|---|---|---|---|
| 1 | 编排/路由性能 | `bottleneck_scaling_attribution`（唯一随规模增长的桶才是目标） | J5 实测：路由 ≈0.2ms（≈0%）；search 0.1ms；embed 712.9ms；**LLM decode 11679ms = 94%** | ❌ **无空间**：瓶颈在引擎层 LLM decode，非协作平台层；且已有 K-ACTQ/K-EMB 专项覆盖 |
| 2 | 检索加速（NEON/索引） | `measured_benefit_region`（收益随区间反转，须实测界定） | P0 微基准：标量余弦 8.98ms/万级 doc；实际 8 doc 更小 | ❌ **已收口**（P0 定论：标量够快，NEON 非位级留可选） |
| 3 | RAM 池化/驻留分级联动 | `todo_requires_measured_share`（未量占比 + 可达性前，不构成立项） | 板端 16GB；全模型常驻 ≈8GB **不超限**；30B 已删（模型库收敛） | ❌ **无实测瓶颈依据**（§4 硬约束② 的池化是为 30B 级准备，当前小模型常驻不构成瓶颈） |
| 4 | 专家接力 / verifier | 记忆（K-CAP 系列）已证否 | 8B critic 续写 / 1.7B NLL 重打分 / ALIGNED verifier **均净负** | ❌ **已证否**，不重复立项 |
| 5 | 并发锁串行（embedding/chat 共用 `inf_lock`） | `todo_requires_measured_share` | P2 决策：`g_embed_out`/量化开关/NPU strip 是全局单例 ⇒ 必须串行；无「并发排队」实测占比 | ❌ **无实测依据**（单机边缘无高并发场景；解耦全局单例 blast radius 大） |
| 6 | **RAG embedding 缺省静默回退** | `silent_fallback_false_pass` + 判据 J4（绝不静默降级） | `handle_rag` 缺省 `--embed-model` 时**静默复用主模型（4B Instruct）出向量** → 检索质量崩但无告警 | ✅ **真实空间**（回退透明修复，见 §13.2） |

### 13.2 唯一真实空间：RAG 回退透明修复（silent_fallback_false_pass）

- **缺陷**：`handle_rag` 在 `--embed-model` 缺省时走 `else` 分支 `st=ctx->ist; tok=ctx->tok`（复用主模型 4B Instruct 出向量）。4B Instruct **非 embedding 训练**、last-token pooling 对生成模型非良好句子向量 ⇒ 检索质量崩，但**无任何日志/标记**，用户拿到「看似正常」的 RAG 结果——正是 `silent_fallback_false_pass` 的「凡有 fallback 的路径，判据必须附生效性凭证」反向场景。
- **修复方案（二选一，倾向 A）**：
  - **A（符合 J4 字面）**：缺省 `--embed-model` → 显式**回退纯 LLM**（不拼检索上下文），打日志 `[RAG] no --embed-model, plain-LLM fallback (no retrieval)`，响应 `retrieved:[]` + `"fallback":"plain_llm"` 标记。
  - B（更强硬）：缺省 → **503**「RAG requires --embed-model」。
- **判据（先冻结）**：
  | 判据 | 门限 |
  |---|---|
  | **R1 回退显式** | 不带 `--embed-model` 的 `/v1/rag` → 200 纯 LLM 回答 + `retrieved:[]` + `fallback:"plain_llm"`；日志含 `[RAG] … fallback` 标记 |
  | **R2 无静默降级** | 同上请求，**不得**出现「复用主模型出向量 + 检索」路径（响应无 `retrieved` 非空项、无静默 `score`） |
  | **R3 生效性凭证** | `silent_fallback_false_pass` 反向要求：日志**必须**打印回退诊断（与「优化路径必须打印生效诊断」同构） |
  | **R4 回归** | 带 `--embed-model` 的 `/v1/rag` 检索正常（J2 不退化：iPhone→A18 0.785 仍 PASS） |
  | **R5 主路径零回归** | `/v1/embeddings`（不带 `--embed-model` 复用主模型）行为**不变**（该端点语义即「用主模型出向量」，非 RAG，不在此修复范围） |

- **范围边界**：只改 `handle_rag` 的 embedding 缺省分支；`handle_embeddings`（`/v1/embeddings`）保持复用主模型不变（那是合法语义，非 RAG）。

### 13.3 修复判据结果（2026-10-11，板端已实施，方案 A）

| 判据 | 结果 |
|---|---|
| **R1 回退显式** | **PASS**：不带 `--embed-model` 的 `/v1/rag` → 200，`retrieved:[]` + `fallback:"plain_llm"` + 纯 LLM 回答（`"截至2024年…尚未发布…猜测"`——证明**未检索到 A18 文档**，是纯 LLM 通用知识） |
| **R2 无静默降级** | **PASS**：同上请求 `retrieved:[]`（无 `index/meta/score` 假结果），无「复用主模型出向量 + 检索」路径 |
| **R3 生效性凭证** | **PASS**：日志打印 `[RAG] no --embed-model, plain-LLM fallback (no retrieval)`（每请求一条） |
| **R4 回归** | **PASS**：带 `--embed-model` 的 `/v1/rag` → `retrieved[0]` iPhone/A18 0.785 + `answer="iPhone 16 用的是 A18 芯片。"`，`fallback` 字段**不存在**（检索正常） |
| **R5 主路径零回归** | **PASS**：`handle_embeddings`（`/v1/embeddings`）未改动，复用主模型语义不变（代码层面确认，无 diff） |
| 构建 | `cmake --build build-rk3588` rc=0（root 二进制已 sync） |

> **对比实证（修复前 vs 修复后）**：带 `--embed-model` 时 `answer="iPhone 16 用的是 A18 芯片。"`（检索增强）；不带时 `answer="截至2024年…猜测…"`（纯 LLM，无 A18）——修复前两者**无法区分**（静默复用 4B 出向量，检索质量崩但无告警），修复后**显式可辨**（`fallback` 标记 + `retrieved:[]` + 日志）。
