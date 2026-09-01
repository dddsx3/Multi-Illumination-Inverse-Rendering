# R5-B′ · A10 (24 GB / 20 vCPU / 116 GB) 算力校准 · 24 机时免费额度选型

> **作者**: ZCode agent · 2026-09-01
> **算力**: 1× NVIDIA A10 24GB / 20 vCPU / 116 GB RAM / 抵扣 3.3× 机时
> **免费额度**: 24 机时 = 实际 ~7.27 GPU-小时等效
> **依据**: `r5/R5_RESOURCE_PRECISION_MATRIX.md` + R4§ 历史 solver timing

---

## 1. A10 与既有算力的差异校准

| 实例 | GPU | vCPU | RAM | per-call @ P=2000 | per-solver-run (restarts=1) |
|---|---|---|---|---:|---:|
| RTX 5070 Ti Laptop（本机）| 12 GB | 24 | 32 GB | 1.60 s (单线程实测) | N/A (WinError 1455) |
| T4 (Cloud Studio) | 16 GB | 8 | 32 GB | 1.60 s × 1/4.0 = **0.40 s** | 6–9 s (R4§ 估测) |
| **A10** | **24 GB** | **20** | **116 GB** | **~0.25 s** | **~2–3 s** |
| A100 (Lambda) | 40 GB | 30 | 200 GB | ~0.25 s | ~2.5 s |
| H100 (Lambda) | 80 GB | 26 | 252 GB | ~0.25 s | ~2.5 s |

**校准要点**:
- A10 FP32 = 31.2 TFLOPS，比 T4 (8.1) 强 ~3.8× ⇒ solver per-run 估 **2–3 s**（vs R4§ T4 6–9 s）
- A10 20 vCPU 是 T4 8 vCPU 的 2.5×，OpenBLAS scaling 从 4× → 5.3× ⇒ GSIQ per-call **0.25 s**
- RAM 116 GB 对 P=2000 GSIQ（peak RSS ~76 MB）+ solver (~2 GB) 绰绰有余

---

## 2. 各阶段 A10 wall-clock 估算

### 2.1 P1-A full (6 dev scenes, N{3,5}, P=2000, +solver arm +Task G)

| 子阶段 | calls/runs | per-unit | wall-clock |
|---|---:|---:|---:|
| GSIQ (O + A) | 83,520 calls | 0.25 s | **5.80 h** |
| solver arm (360 runs) | 360 | 2.5 s | 0.25 h |
| Task G (240 runs) | 240 | 2.5 s | 0.17 h |
| env + 数据下载 | — | — | 0.20 h |
| **合计** | | | **~6.4 h wall-clock** |

**抵扣机时 = 6.4 × 3.3 = 21.1 机时**

> ✅ **24 机时免费额度** 够跑 P1-A full + Task G，剩余 **2.9 机时**做冗余 (debug / re-run / 二次抽样)。

### 2.2 P1-A full + P1-B (无 P2)

| 子阶段 | calls | wall-clock |
|---|---:|---:|
| P1-A full | 83,520 | 5.80 h |
| P1-B (10 scenes, 5 score variants 新增 3 个: N + L + P) | ~125,000 | 8.70 h |
| solver (~600 runs) | — | 0.42 h |
| env | | 0.2 h |
| **合计** | | **~15.1 h wall-clock** |

**抵扣机时 = 15.1 × 3.3 = 49.8 机时**

> ❌ 24 机时**不够**。

### 2.3 P1-A full + P1-B + P2 held-out（完整论文）

| 子阶段 | wall-clock |
|---|---:|
| P1-A full | 5.80 h |
| P1-B | 8.70 h |
| P2 held-out (12 scenes × 3 N × 8960 × 2 score variants × P=2000) | 14.9 h |
| solver (~2,000 runs) | 1.4 h |
| env + 调试 | 1.0 h |
| **合计** | **~32 h wall-clock** |

**抵扣机时 = 32 × 3.3 = ~105 机时**

> ❌ 24 机时**远不够**（仅 23% 覆盖）。

---

## 3. 选型决策

### 选项 A：P1-A full + Task G（24 机时内**完整覆盖**）✅

- **wall-clock**: ~6.4 h
- **抵扣机时**: 21.1 / 24
- **产出**: `r5/r5_p1_*.csv` (6 scenes × P=2000 全量) + `r4pp/07_local_vs_global_init.csv` (Task G)
- **回答**: Q1（oracle↔albedo）+ Q2（local-global）
- **推荐**: ✅ 这是当前 24 机时免费额度能完整覆盖的唯一选项
- **剩余 2.9 机时**: 留作冗余（遇到 OOM 重启 / 部分 cell 重跑 / 调试）

### 选项 B: P1-A full only (无 Task G)（最保守）

- **wall-clock**: ~6.3 h
- **抵扣机时**: ~20.8 / 24
- **产出**: 仅 P1-A full
- **回答**: Q1 only
- **推荐**: ⚠️ 不建议，Q2 是 reviewer 必问的，"global optimizer difficulty proxy" 必须答。

### 选项 C: P1-A + P1-B（超额）

- **wall-clock**: ~15.1 h
- **抵扣机时**: ~49.8
- **不足**: 缺 ~26 机时；要么自己付费 ($26 spot ~$0.3/h × 15h = $5)，要么放弃 P1-B 一半 scene 数。

### 选项 D: P1-A full + Task G + 部分 P1-B（最贪心）

- 跑 P1-A full (21 机时) → 剩余 3 机时
- 用剩余 3 机时跑部分 P1-B（2 scenes × 2000 subsets × 3 score = 12,000 calls × 0.25 s = 0.83 h wall-clock = 2.7 机时）
- **产出**: P1-A full + Task G + 2 scenes P1-B 验证

> 这是性价比最高的方式，但**剩 0.3 机时无冗余**。如果中途 OOM 就完蛋了。

---

## 4. 我的建议

**做选项 A（21 机时）**。理由：

1. **24 机时免费额度足够**，剩余 2.9 机时做冗余
2. **Q1 + Q2 都是 reviewer 必问**，是论文成立的**必要不充分**条件
3. **Q3 (selection) 依赖 P1-B** — 但 P1-B 需要 proxy 估计（不是纯 GSIQ），CPU 时间 8.7 h ≈ 30 机时，性价比不高
4. **如果你已经知道 Q1 答案**（P1-A smoke 已 PASS-A），那么选项 A 的边际收益最大
5. **如果你要 P1-B**，考虑单独再申请一次 A10 或换算力（Vast.ai spot 更便宜）

---

## 5. 选项 A 详细执行预算（A10 21.1 机时）

### 5.1 时间切片（建议 6.5 h 内完成，留 0.1 h 调试）

| t (h) | 阶段 | 备注 |
|---|---|---|
| 0.0–0.2 | 数据 LFS pull + env setup (torch cu128 + numpy + scipy) | |
| 0.2–0.4 | 数据 sanity（nvidia-smi / df / 验证 19 scene 完整） | |
| 0.4–3.4 | **P1-A GSIQ N=3**（6 scenes × 4960 subsets × 2 score = 59,520 calls × 0.25 s ≈ 4.1 h wall-clock） | CPU-bound；GPU 空闲可塞 Task G |
| 1.0–1.3 | **Task G**（240 runs × 2.5 s ≈ 10 min，**和 P1-A 并行**）| GPU 跑 Task G，CPU 跑 GSIQ |
| 3.4–5.8 | **P1-A GSIQ N=5**（6 scenes × 2000 × 2 = 24,000 calls × 0.25 s ≈ 1.7 h） | |
| 5.8–6.1 | **P1-A solver arm**（360 runs × 2.5 s ≈ 0.3 h） | GPU-only |
| 6.1–6.3 | boundary outlier 分析 + gate verdict + 报告 | 轻量 |
| 6.3–6.5 | git commit + push | |

> 关键洞察：**P1-A GSIQ 是 CPU-bound，Task G 与 solver arm 是 GPU-bound**。在 A10 上两者可完全 overlap，wall-clock ≈ max(CPU, GPU) = **5.8 h CPU + 0.6 h GPU = 5.8 h**（不是 6.4 h 线性相加）。

### 5.2 抵扣机时精算

```
P1-A full wall-clock = 6.5 h
抵扣因子 = 3.3
抵扣机时 = 6.5 × 3.3 = 21.45 机时
预算 = 24 机时
剩余 = 2.55 机时（10.6% 冗余）
```

---

## 6. 关键约束与 mitigations

### 6.1 OpenBLAS 线程

A10 有 20 vCPU，但 **eigh 多核 scaling 上限 ~5.3×**。保守设置：

```bash
export OMP_NUM_THREADS=10      # eigh 用 10 线程
export OPENBLAS_NUM_THREADS=10
# 留 10 vCPU 给 OS / Task G / solver
```

### 6.2 GPU 共享

A10 24GB 同时跑 solver (batched, ~2 GB) + Task G (batched, ~2 GB) 共 4 GB，**留 20 GB buffer**。但 GPU compute 不能真并行（CUDA stream 调度），所以 Task G 与 solver 串行跑：

| step | CPU (10 thread) | GPU |
|---|---|---|
| 0.4–5.8 h | GSIQ P1-A | idle / 跑 Task G（10 min）|
| 5.8–6.1 h | idle | solver arm |

### 6.3 数据下载慢

LFS data 约 300 MB。在 A10 实例上首次 `git lfs pull` 可能 5-10 min（取决于出网带宽）。

### 6.4 时间窗风险

如果实例在 24 机时内被强制停机（比如 spot 抢占 / 平台限额），**R5 脚本都支持 incremental resume**：
- `r5_p1_albedo_ablation.py`：每次运行覆盖写入 CSV（无 resume，但可以增量跑剩余 scene/N）
- `r4pp_local_vs_global.py`：自带 incremental（按 `(scene, N, subset, init_mode)` skip done 行）

**强烈建议**：拆成 6 个 scene × 2 N = 12 cell 跑，每次跑完一个 cell 立即 `git commit + push`，避免丢失进度。

---

## 7. 与"完整论文"的差距

24 机时免费额度只够 **P1-A full + Task G（21 机时）**。要补上 P1-B + P2：

| 缺口 | 所需机时 | 等效 wall-clock | 备注 |
|---|---:|---:|---|
| P1-B (10 scenes × 5 score variants @ P=2000) | ~30 | 9 h | proxy 估计 1 h + GSIQ 8 h |
| P2 (12 scenes × 3 N × 8960 × 2 score @ P=2000) | ~50 | 15 h | held-out benchmark |
| 剩余 P1-B 一半 + P2 完整 | ~80 | 24 h | 完整论文 |
| **合计（再申请一次 80 机时 = $30 spot）** | **80** | **24 h** | 仍不到完整论文 |

完整论文总需求 ~105 机时 ≈ **$400 spot / $1300 on-demand**。

---

## 8. 申请清单（按选项 A）

```
申请类型: 已批 A10 实例 + 24 机时免费额度
用途: R5-B′ P1-A full + Task G（21 机时）+ 冗余（3 机时）
依据: 
  - r5/R5_RESOURCE_PRECISION_MATRIX.md §6 (实例选型对照)
  - r5/R5_H100_RESOURCE_INVENTORY.md（原 H100 清点）  
  - r5/R5_A10_FIT_CHECK.md (本文件)

算力规格:
  - GPU: 1× NVIDIA A10 24 GB
  - vCPU: 20
  - RAM: 116 GB
  - 抵扣: 3.3× 机时
  - 预算: 24 机时（21 + 3 冗余）

执行计划（选项 A）:
  - 0.0–0.4h: env setup + data LFS pull
  - 0.4–5.8h: P1-A GSIQ (CPU-bound, 与 Task G 并行)
  - 1.0–1.3h: Task G (并行于 GSIQ, 占用 GPU 10 min)
  - 5.8–6.1h: P1-A solver arm
  - 6.1–6.5h: 报告 + commit
  - 总 wall-clock: ~6.5 h
  - 抵扣机时: ~21.5 / 24 (留 2.5 冗余)

数据:
  - git clone dddsx3/Multi-Illumination-Inverse-Rendering @ 9796884 (含 LFS)
  - p1/calibration_set/data_sun_confirmatory/ (19 scenes, ~300 MB)

代码入口:
  - p1/source/information_audit/r5_p1_albedo_ablation.py (P1-A)
  - p1/source/information_audit/r4pp_local_vs_global.py (Task G)

预期产出:
  - r5/r5_p1_albedo_ablation.csv (6 scenes × 13920 rows @ P=2000)
  - r5/r5_p1_albedo_ablation_ranking.csv
  - r5/r5_p1_albedo_ablation_outliers.csv
  - r5/r5_p1_albedo_ablation_gate.md
  - r5/r5_p1_albedo_ablation_selection.csv (solver arm)
  - r4pp/07_local_vs_global_init.csv (Task G)

成功标准:
  - P1-A: median(ρ) ≥ 0.95 + median(top10 overlap) ≥ 0.80 → PASS-A
  - Task G: β_global<0 AND β_oracle_local<0 → Case 1 (intrinsic identifiability)

不做的（24 机时内不现实）:
  - P1-B (proxy 估计 + 全量 GSIQ, 30 机时)
  - P2 held-out (50 机时)
```

---

## 9. 总结

| 阶段 | 所需 wall-clock | 抵扣机时 | 24 机时内? |
|---|---:|---:|---|
| P1-A full + Task G（推荐） | 6.5 h | 21.5 | ✅ |
| P1-A only（最保守）| 6.3 h | 20.8 | ✅ |
| P1-A + P1-B | 15.1 h | 49.8 | ❌ |
| 全量（P1-A + P1-B + P2）| 32 h | 105 | ❌ |

**我的最终建议：选项 A（21 机时）**。Q1 + Q2 拿到，剩余预算留给后续 P1-B / P2（再申请一次 A10 或换算力）。如果 P1-A smoke 已经 PASS-A，Q1 大概率能过；Q2（Task G）是 reviewer 必问，必跑。

---

*作者: ZCode agent · 2026-09-01*
*A10 单调用 per-call 估算: 0.25 s = 1.60 s / (5.3 vCPU speedup × 1.2 主频优势)*
*Solver per-run 估算: 2.5 s = R4§ T4 6-9 s / 3 (A10 FP32 优势)*
*抵扣机时: wall-clock × 3.3*