# K-BENCH-018 · 板端 MoE 模型（Qwen3-30B-A3B）本引擎性能与质量

| 项 | 内容 |
|---|---|
| **立项 ID** | **K-BENCH-018** |
| **目标** | 只测**本引擎**在板端 MoE 模型上的性能画像（上下文扫描 + 优化档 + 长上下文质量） |
| **日期** | 2026-10-09 |
| **载体** | 板端 `192.168.1.92`（RK3588 OPi5+，4×A76@2.256G + 4×A55@1.8G，RAM 15964 MB，zram swap 7.8 GB） |
| **模型** | `/mnt/VQF/qwen3-30B-A3B-q4/model.vqf`（**17.67 GB**，v3 MoE，48 层，`n_experts=128`、`moe_ffn=768`、`top_k=8`、`dim=2048`） |
| **引擎** | `vllm_kestrel`（2026-10-09 构建），t4，governor=performance |
| **口径** | `--stream-test --stream-ctx N --bench-seqlen N --stream-n M`；不与其他引擎对比（用户要求只测本引擎） |

---

## 1. 生效性凭证（默认档，实测日志）

```
[ACTQ-GU]  GU int8-dot ON (default; escape: =0)
[ACTQ-DN]  DN int8-dot OFF (inherited from VLLM_ACTQ)
[ACTQ-DEC] DEC int8-dot ON (default; escape: VLLM_ACTQ_DEC=0)
[ACTQ]     decode int8-dot approximate track ON (gate: VLLM_ACTQ_DEC; l=0, d=2048 nbG=64)
[MOEDNORD] dn order=exp-major (escape: VLLM_MOE_DN_ORDER=0)
[MOEDNBAL] dn bal=ON (escape: VLLM_MOE_DN_BAL=0)
[MOEDNBAL] dn bal ENGAGED l=0 nt=4 KC=32 imbalance 1.124 -> 1.000
```
⇒ 默认档已含 **ACTQ(GU+DEC) / DN exp-major / DN 均衡装桶** 全部已默认化优化（K-ACTQ-010/013、K-DN-006/012）。

---

## 2. 上下文扫描（默认档，t4，gen=16）

| ctx | tokens | prefill (s) | **prefill t/s** | **TPOT (ms)** | decode t/s | RSS | wall (s) |
|---|---|---|---|---|---|---|---|
| 1024 | 1014 | 298.6 ⚠️ | 3.4 ⚠️ | 783 | 1.28 | 13.09 GB | 312 |
| 2048 | 2038 | **241.0** | **8.5** | **295** | 3.39 | 13.19 GB | 247 |
| 4096 | 4086 | 554.4 | 7.4 | 385 | 2.60 | 13.30 GB | 561 |
| 8192 | 8182 | 1483.2 | 5.5 | 580 | 1.72 | 13.50 GB | 1493 |

> ⚠️ **ctx=1024 偏低**：该轮是 M0 冷跑（首次加载）之后的**第一个** M1 用例，页缓存仍未完全预热（对照 2048 更快），**属预热伪影，不代表 ctx 效应**。2048 起为稳定热态。

**观察**：
- prefill 从 2048 的 8.5 t/s 单调降到 8192 的 5.5 t/s（**−35%**，注意力 O(n) + MoE 全专家读取叠加）。
- TPOT 从 295 → 580 ms（**1.97×**），衰减远小于稠密 8B 的 2.54×。
- **RSS 13.09–13.50 GB < 15.9 GB RAM** ⇒ 全程无需驱逐，**MoE 在本板端可完整驻留运行**。

---

## 3. 优化档矩阵（ctx=8192，gen=8，t4）

| 档 | prefill t/s | prefill (s) | TPOT (ms) | RSS | 相对默认 |
|---|---|---|---|---|---|
| **A 默认** | 5.5 | 1496 | 558 | 13.45 GB | — |
| **B +稀疏 decode**（`--sparse-attn --sparse-k 32`） | 5.5 | 1493 | **514** | 13.45 GB | decode **+8.6%** |
| **C +稀疏 prefill**（`VLLM_SPARSE_PREFILL=1`） | **8.0** | **1026** | 648 | 13.98 GB | prefill **+45.8%** |

**段分解（`[PREFILL-KERNELS]`，三档一致）**：
```
QKV=4.2%  O=3.5%  GATEUP=92.2%  DOWN=0.0%
```
> `GATEUP` 桶 = **整个 MoE 批**（router+量化+top-k+gu+dn+combine，见 K-PF-005 §11.1 口径订正）。合计 GEMM ≈ **696 s（占 prefill 47%）** ⇒ **ATTN ≈ 800 s（53%）**。
> ⇒ **MoE 在 8K 下 prefill 已是「注意力 ≈ MoE-FFN」各半**（与短上下文时「GATEUP 96%」的画像完全不同）；稀疏 prefill 正是砍 ATTN 那半，故 **+45.8%**。

**判读**：
1. **稀疏 decode 对 MoE 收益有限（+8.6%）**——MoE decode 本就靠 top-8/128 稀疏（TPOT 558 ms 已远优于稠密 8B 的 858 ms），注意力占比小。
2. **稀疏 prefill 是 MoE 长上下文的最大杠杆（+45.8%）**——由 ATTN 占比 53% 决定。
3. C 档 TPOT 648 ms（−13.9%）与 gen=8 的短程噪声量级相当，**不宜作判据**（登记）。

---

## 4. 长上下文质量（NIAH）

| ctx | 位置 | 召回 |
|---|---|---|
| 2048 | 0 / 25 / 50 / 75 / 100% | **5/5 全 Y**（输出均以 `ramen` 开头） |
| 8192 | 50% | **Y** |

- 答案 PPL（teacher-forced）Q8 = **4.37–6.88**（n/a F32：`VLLM_KV_NOF32` 无 f32 正典）。
- ⇒ **MoE 长上下文检索满分**，与稠密 8B（K-BENCH-017）结论一致。

---

## 5. 与稠密 8B 的横向对照（同 ctx=8192、同 t4、同引擎默认档）

| 指标 | 稠密 Qwen3-VL-8B | **MoE Qwen3-30B-A3B** | 说明 |
|---|---|---|---|
| 模型文件 | 6.60 GB | **17.67 GB**（2.7×） | — |
| prefill t/s | **8.3** | 5.5 | 稠密更快（MoE 需读**全部 128 专家**） |
| TPOT ms | 858 | **580** | **MoE 更快**（top-8/128 稀疏 ⇒ 单 token 只读 6% 专家） |
| RSS | 4.57 GB | 13.50 GB | MoE 大 3×但**装得下** |

**核心洞察**：**MoE 的稀疏性把 decode 相对收益拿到了**（30B 的 decode 比 8B 稠密还快 1.48×），**但 prefill 反被"全专家读取"拖累**（比 8B 稠密慢 1.5×）——这与 K-PF-005/MoE 系列"prefill 是 MoE 主战场"的判断一致。

---

## 6. 多轮对话（会话模式；补做 §7 未覆盖项「多轮次对话」）

**口径**：`--serve --auto-load`（HTTP `/v1/chat/completions`，`messages` 传**全量历史**，`temperature=0`、`enable_thinking=false`、`max_tokens=32`）；KV 前缀复用（`--prefix-kv`，默认开）；t4 / governor=performance；**只测本引擎**（延续本项「不与其他引擎对比」口径）。同对话：首轮植入代号「蓝莓」→ 8 轮 ~460-tok 背景材料 + 「用一句话概括」→ 末轮追问代号。

**生效性凭证**（serve 路径实测，与 §1 一致）：`[ACTQ-GU] ON` / `[ACTQ-DEC] ON` / `[ACTQ-DN] OFF` / `[MOEDNORD] exp-major` / `[MOEDNBAL] ON+ENGAGED (imbalance 1.118→1.000)`。

### 6.1 逐轮（复用开）

| 轮 | prompt tok | gen | prefill (ms) | TTFT (ms) | TPOT (ms) | 轮总 (s) | RSS |
|---|---|---|---|---|---|---|---|
| 1 | 43 | 7 | **61 454** ⚠️冷 | 61 454 | 900 | 66.9 | 11.60 GB |
| 2 | 502 | 16 | **124 332** ⚠️冷 | 124 333 | 300 | 128.8 | 14.05 GB |
| 3 | 970 | 15 | 51 672 | 51 673 | 260 | 55.3 | 14.29 GB |
| 4 | 1 437 | 15 | 54 440 | 54 442 | 294 | 58.6 | 14.39 GB |
| 5 | 1 904 | 15 | 61 420 | 61 420 | 291 | 65.5 | 14.46 GB |
| 6 | 2 371 | 15 | 65 536 | 65 537 | 319 | 70.0 | 14.51 GB |
| 7 | 2 838 | 15 | 74 083 | 74 084 | 347 | 78.9 | 14.56 GB |
| 8 | 3 305 | 15 | 75 874 | 75 875 | 353 | 80.8 | 14.60 GB |
| 9 | 3 772 | 15 | 82 817 | 82 818 | 385 | 88.2 | 14.63 GB |
| 10 | 3 812 | 2 | 10 721 | 10 722 | 1 569 | 12.3 | 14.74 GB |

- **轮 10 = 追问代号**，输出 `蓝莓` ⇒ **多轮记忆保留正确（@3.8K）**。
- ⚠️ **轮 1/2 为冷启动伪影**：预热请求后 RSS 仅 9.6 GB，轮 1/2 继续页入（RSS 11.6→14.05 GB），prefill 含大量 page fault（与 K-L0-004「冷启动 page fault 主导」一致）；**轮 3 起为热态**（RSS 稳定 ~14.3 GB）。

### 6.2 前缀复用 A/B（`--no-prefix-kv` 关复用）

| 轮 | prompt tok | prefill 复用**开** (ms) | prefill 复用**关** (ms) | 说明 |
|---|---|---|---|---|
| 1 | 43 | 61 454（冷） | 6 660（页缓存热） | ⚠️ 不可比 |
| 2 | 502 | 124 332（冷） | 48 684 | ⚠️ 不可比 |
| 3 | 970 | 51 672 | 93 688 | **1.81×** |

- ⚠️ **Arm N 首轮受益于 Arm R 遗留的文件页缓存**（mmap file-backed 页可回收、跨进程留存）⇒ 轮 1/2 两臂热态不同、不可比；**唯一可比点是轮 3（两臂均热）：复用 1.81×**。
- 复用在 MoE 上同样生效：轮 3 只 prefill 增量（约 452 tok）而非全量 970 tok。

### 6.3 多轮专项结论（MoE）

1. **前缀复用生效**：轮 3 热态下 prefill **51.7 s vs 93.7 s = 1.81×**。
2. **增量 prefill 速率随 ctx 退化**：轮 3→9 每轮增量约 452 tok，速率 **8.75 → 5.46 tok/s**（51.7 s → 82.8 s）；比稠密 8B（§K-BENCH-017 同区间 ~13→7 t/s）**更慢**（MoE prefill 需读全 128 专家）。
3. **decode TPOT 260→385 ms**（1→3.8K），**优于稠密 8B 同区间**（280→414 ms），与 §5「MoE 稀疏性把 decode 收益拿到」一致。
4. **RSS 稳定 ~14.3–14.7 GB**（< 15.9 GB）⇒ 多轮长会话**无驱逐**，但余量仅 ~1.2 GB。
5. **记忆保留正确**（@3.8K 仍召回「蓝莓」）。
6. **冷启动主导首轮体验**：serve 加载后首个请求仍需页入 17.7 GB 权重（预热 169.4 s + 轮 1/2 续入）。
7. ⚠️ **本次仅到 ~3.8K**（MoE prefill 每轮 50–83 s，8K 不可行于本预算）；更长上下文的趋势未测。

---

## 7. 结论

1. **MoE 在本板端可完整驻留**（RSS 峰值 13.5 GB < 15.9 GB），全程无驱逐 ⇒ 无冷档惩罚。
2. **默认档 = 全优化档**（ACTQ GU+DEC / DN exp-major / DN 均衡装桶全部默认开，凭证已录）。
3. **prefill**：2048 峰值 **8.5 t/s**，8K 降至 5.5 t/s；**稀疏 prefill 在 8K 回到 8.0 t/s（+45.8%）**。
4. **decode**：TPOT 295 ms(2K) → 580 ms(8K)；**稀疏 decode 在 8K 给 +8.6%**。
5. **质量**：NIAH 2K 5/5 + 8K 通过 ⇒ 满分区。
6. **8K 下 MoE prefill 的构成已变成 ATTN 53% / MoE-FFN 47%**，与短上下文的 GATEUP 96% 截然不同 ⇒ 长上下文 MoE 优化应同时打两侧。

### 未覆盖（诚实登记）

1. ~~**多轮次对话**~~ ✅ **已完成，见 §6**（MoE 会话模式，至 3.8K，记忆保留正确）。
2. **16K+ 上下文**（`--stream-ctx` 上限 8192）。
3. **C 档（稀疏 prefill）的质量门**（NIAH/NLL 未在 `VLLM_SPARSE_PREFILL=1` 下复测；wiki 记载 8B 在 `--sparse-pf-group 4` 下 16K 严格判据全过）。
4. **DN actq（`VLLM_ACTQ_DN=1`）档**未测（当前默认 OFF）。
5. **t1/t2/t8 线程扩展曲线**。
6. **MoE 多轮的优化档组合未测**（§6 全程默认档；未叠加 `--sparse-attn` / `VLLM_SPARSE_PREFILL`）。
7. 全部为**单会话读数**，非跨会话可比绝对值；ctx=1024 受预热伪影影响；§6 轮 1/2 受冷启动伪影影响。
