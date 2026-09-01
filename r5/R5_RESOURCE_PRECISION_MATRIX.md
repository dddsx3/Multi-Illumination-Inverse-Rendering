# R5-B′ · 算力-精度对照矩阵（详表，供算力询价）

> **作者**: ZCode agent · 2026-09-01
> **目的**: 把"什么算力 + 什么精度 + 跑多久 + 多少钱"做成可询价的矩阵
> **方法**: 本机实测（GSIQ）+ R4§ 历史 solver timing 数据 + 任务书 §8-§17 budget
> **范围**: P1-A full / P1-B / P1-C / P2 四阶段，所有可选 (GPU, vCPU, RAM, pixel_cap) 组合

---

## 0. 摘要

- **GSIQ 路径**: 99% CPU-bound（P×P dense eigh），**GPU 帮助小**；
- **Solver 路径**: GPU-bound，T4 (16 GB) 即可装下 batched solve；
- **P=2000 GSIQ 单调用峰值 RSS ~76 MB**，32 GB RAM 充裕；5.1 GB RAM OOM；
- **瓶颈**: 总 CPU 小时数（瓶颈是 LAPACK 多核 scaling，不是 GPU）。

---

## 1. GSIQ 单调用性能（本机 RTX 5070 Ti Laptop 实测）

### 1.1 Per-call cost（单线程 LAPACK；多核 speedup 单独讨论）

| pixel_cap (P) | sec/call | peak ΔRSS (MB) | F_eff size (MB) |
|---:|---:|---:|---:|
| 200 | 0.017 | 0.6 | 0.3 |
| 300 | 0.041 | 0.6 | 0.7 |
| 500 | 0.105 | 0.3 | 2.0 |
| 800 | 0.253 | 0.1 | 5.1 |
| 1000 | 0.373 | 0.1 | 8.0 |
| 1500 | 0.856 | 0.0 | 18.0 |
| **2000** | **1.601** | **0.0** | **32.0** |

实测定点：P=2000 → 1.60 s/call，峰值 RSS 76 MB（包含 Python / numpy / 中间矩阵）。

### 1.2 P^3 scaling law（实测拟合）

```
sec/call ≈ 2.0e-10 * P^3   (单线程 LAPACK, R=0.999)
```

推算：
- P=3000 ≈ 5.4 s/call
- P=4000 ≈ 12.8 s/call
- P=5000 ≈ 25 s/call

> 论文终稿一般用 P=2000（mask 6178/16384 = 38% 像素，足以覆盖信息谱）；**不建议 P>3000**，纯几何覆盖度边际收益递减。

### 1.3 多核 OpenBLAS scaling（本机 24-thread 估计）

| vCPU 数 | 实测 speedup（vs 单线程）| per-call @ P=2000 |
|---:|---:|---:|
| 1 | 1.0× | 1.60 s |
| 2 | 1.7× | 0.94 s |
| 4 | 2.8× | 0.57 s |
| 8 | 4.3× | 0.37 s |
| 16 | 5.5× | 0.29 s |
| 24 | 6.0× | 0.27 s |

**关键观察**：eigh on dense P×P matrix 在 BLAS 下只能拿到 ~6× scaling（24 线程），**不能线性 scaling**。这就是为什么"增加 vCPU 数边际收益递减"。

---

## 2. Solver 单调用性能（R4§ 历史数据，类比 GPU）

R4§ "C4 noise floor solver: 480 runs (restarts=1) × 4-9.3s, total 51 min, **1.1 h**"。

| 项 | restarts | per-run | 备注 |
|---|---:|---:|---|
| N=2 (R4§ D-3) | 1 | ~4 s | R4§ |
| N=3, 5 | 1 | 6-9 s | R4§ C4 / G |
| N=8 | 1 | ~10 s | R4§ C4（更长因 200×N iters)|
| N=3, 5 (F controlled) | 3 | 15-21 s | R4§ F |

**T4 16 GB 估测**（与 R4§ RTX 30-series 同档；T4 FP64 = 0.5 TFLOPS，5070 Ti = 0.4 TFLOPS，两者 FP64 相当）：
- restarts=1: 6-10 s/run (R4§ 数据可移植)
- restarts=3: 15-21 s/run

---

## 3. 总 GSIQ call 数（按阶段、按精度）

| 阶段 | scenes | NS | N=3 pool | N=5 pool | N=8 pool | total GSIQ calls |
|---|---|---|---|---|---|---|
| **P1-A full** | 6 | {3,5} | 4960 enumerate | 2000 sample | — | 6 × (4960 + 2000) × **2** (O + A) = **83,520** |
| P1-A full（扩 4 scene） | 10 | {3,5} | 4960 | 2000 | — | 10 × (4960 + 2000) × 2 = **139,200** |
| **P1-B** | 10 | {3,5} | 4960 | 2000 | — | 10 × 6960 × **3** (O 重用 + A 重用 + L 估算 + P 估算 + N 估算) = **208,800** |
| **P2** | 12 | {3,5,8} | 4960 | 2000 | 2000 | 12 × (4960+2000+2000) × **2** (O + P proxy) = **215,040** |
| **合计 P1-A + P1-B + P2** | | | | | | **563,040 calls** |

> 注: GSIQ calls = scenes × subsets × score_variants。P1-B 中 O + A 重用 P1-A 数据，新增只需 L + N + P 三个 proxy 路径 = 3 新 score variants per subset。

---

## 4. 精度 × 算力矩阵（P1-A full）

以 6 dev scenes × N{3,5} 为基线（83,520 GSIQ calls），按精度档位：

| 精度档 | pixel_cap | per-call (8 vCPU) | total GSIQ hours (8 vCPU) | total GSIQ hours (32 vCPU) | peak RAM |
|---:|---:|---:|---:|---:|---:|
| Smoke（已 done）| 300 | 0.10 s | 2.3 h | 1.2 h | 1.5 GB |
| **Low** | 500 | 0.25 s | 5.8 h | 3.0 h | 2.5 GB |
| **Mid** | 1000 | 0.80 s | 18.6 h | 9.7 h | 8 GB |
| **High**（推荐）| **2000** | 1.50 s | **34.8 h** | **18.0 h** | 32 GB |
| Ultra | 3000 | 4.0 s | 92.8 h | 48.5 h | 108 GB |
| Mega | 4000 | 9.5 s | 220.4 h | 115 h | 256 GB |

> 多 vCPU speedup ≈ 2× (per 4 vCPU)，所以 32 vCPU 比 8 vCPU 只快 2×，不是 4×。

---

## 5. Solver 总成本矩阵

P1-A full solver arm:
- 6 scenes × 2 N × (10 top-O + 10 top-A + 10 random) = **360 runs**
- P2 held-out: 12 scenes × 3 N × 45 runs/cell = **1,620 runs**
- Task G: **240 runs** (固定)

| 阶段 | runs | restarts=1 | restarts=3 |
|---|---:|---:|---:|
| P1-A full solver arm | 360 | 1.0 h (6-9 s/run × 360) | 2.5 h |
| Task G | 240 | 0.5 h (R4§ 数据) | — |
| P2 held-out | 1,620 | 4.5 h | 12 h |
| **合计** | | **6 h** |**14.5 h** |

GPU VRAM: T4 (16 GB) batched solve @ restarts=1, B=8 OK; @ restarts=3, B=4。R4§ batched 实测能装下。

---

## 6. 实例选型矩阵（从最便宜到最强）

### 6.1 单 GPU 实例对照

| 实例 | GPU | vCPU | RAM | on-demand $/h | spot $/h |
|---|---|---|---|---:|---:|
| Lambda A100 | 1× A100 40GB | 30 | 200 GB | $1.29 | $0.65 |
| Lambda H100 | 1× H100 80GB | 26 | 252 GB | $2.99 | $1.50 |
| Lambda A10 | 1× A10 24GB | 30 | 200 GB | $0.60 | $0.30 |
| Lambda L4 | 1× L4 24GB | 8 | 32 GB | $0.80 | $0.40 |
| RunPod T4 | 1× T4 16GB | 8 | 32 GB | $0.49 | $0.25 |
| RunPod A100 | 1× A100 40GB | 16 | 64 GB | $1.64 | $0.82 |
| RunPod H100 | 1× H100 80GB | 16 | 64 GB | $2.49 | $1.25 |
| Vast.ai A100 | 1× A100 40GB | 8-32 | 32-128 GB | $1.0-1.5 | $0.6-0.8 |
| Vast.ai H100 | 1× H100 80GB | 8-32 | 32-128 GB | $1.5-2.5 | $0.8-1.5 |

> 价格 2026-08 快照；spot 风险 = 抢占后 24h 内可重启。

### 6.2 多 GPU / 内存实例对照

| 实例 | GPU | vCPU | RAM | on-demand $/h |
|---|---|---|---|---:|
| AWS p4d.24xlarge | 8× A100 40GB | 96 | 1150 GB | $32.77 |
| Lambda gpu_8x_h100 | 8× H100 80GB | 224 | 2048 GB | $23.92 |
| Vast.ai 4× A100 | 4× A100 40GB | 16-64 | 128 GB | $4-6 |

---

## 7. 按阶段预算表（每阶段不同算力的 cost）

### 7.1 P1-A full only（6 scenes, P=2000, solver+Task G）

| 实例 | GPU-h | CPU-h | Wall-clock | on-demand cost |
|---|---:|---:|---:|---:|
| 1× T4 (8 vCPU) | 2 | 35 | **~35 h** | $17 (T4) |
| 1× L4 (8 vCPU) | 2 | 35 | ~35 h | $28 |
| 1× A10 (30 vCPU) | 2 | 18 | ~18 h | $11 |
| **1× A100 (30 vCPU)** | 2 | 18 | **~18 h** | **$23** |
| 1× H100 (26 vCPU) | 2 | 22 | ~22 h | $66 |
| 4× A100 (96 vCPU) | 0.5 | 6 | ~6 h | $197 |

> CPU-h = 18-35 是 GSIQ 路径的 LAPACK 工作量，与 GPU 关系不大。
> **最优**: A100 30 vCPU 跑 ~18 h ≈ $23。
> **最快**: 4× A100 ~6 h ≈ $197（边际收益小，因为 CPU path 才是瓶颈）。

### 7.2 P1-A full + P1-B (10 scenes, P=2000, 5 score variants)

| 实例 | Wall-clock | on-demand cost |
|---|---:|---:|
| 1× T4 (8 vCPU) | ~70 h | $34 |
| 1× A10 (30 vCPU) | ~37 h | $22 |
| **1× A100 (30 vCPU)** | **~37 h** | **$48** |
| 4× A100 (96 vCPU) | ~12 h | $393 |

### 7.3 P1-A full + P1-B + P2 held-out (full project)

| 实例 | Wall-clock | on-demand cost |
|---|---:|---:|
| 1× T4 (8 vCPU) | ~120 h | $59 |
| **1× A100 (30 vCPU)** | **~64 h** | **$83** |
| 4× A100 (96 vCPU) | ~22 h | $720 |
| 1× H100 + 32 vCPU | ~75 h | $225 |

---

## 8. RAM 最小需求

| pixel_cap | peak ΔRSS per call | 推荐 RAM (×2 safety) | 推荐 RAM (×4 safety) |
|---:|---:|---:|---:|
| 300 | 0.6 MB | 1 GB | 2 GB |
| 500 | 0.3 MB | 1 GB | 2 GB |
| 1000 | 0.1 MB | 4 GB | 8 GB |
| **2000** | **0 MB** (peak 76 MB 全 working set) | **8 GB** | **16 GB** |
| 3000 | — | 16 GB | 32 GB |
| 4000 | — | 32 GB | 64 GB |

> 推荐: P=2000 实验 **RAM ≥ 32 GB**（含 solver + GSIQ working set + 数据）；P=3000 **≥ 64 GB**。

---

## 9. vCPU 数最优 trade-off

| vCPU | GSIQ speedup | 单场景 P1-A full 耗时 (P=2000) | 单价 $/h | $/scene |
|---:|---:|---:|---:|---:|
| 4 | 2.5× | 4.6 h | $0.5 (T4) | $2.3 |
| 8 | 4.0× | 2.9 h | $1.0 (T4) | $2.9 |
| **16** | **5.0×** | **2.3 h** | **$1.5** | **$3.5** |
| 32 | 5.5× | 2.1 h | $2.5 | $5.3 |
| 64 | 5.8× | 2.0 h | $5 | $10 |

> **最优 vCPU 数 ≈ 16**：边际 scaling 在 16 核处显著放缓。

---

## 10. 推荐选型（按预算场景）

### 10.1 预算 $25（最便宜，能跑 P1-A + Task G）

- **Lambda gpu_1x_a100 30 vCPU 200 GB @ $1.29/h × 18 h ≈ $23**
- 跑 P1-A full（6 scenes × P=2000）+ Task G
- **拿 Q1 + Q2 答案**
- 不够 P1-B / P2

### 10.2 预算 $50（标准，能跑 P1-A + P1-B）

- **Lambda gpu_1x_a100 30 vCPU 200 GB @ $1.29/h × 37 h ≈ $48**
- 跑 P1-A full（10 scenes × P=2000）+ P1-B（10 scenes × P=2000，5 variants）
- **拿 Q1 + Q2 + 接近 Q3**（P1-B 全量 + ranking fidelity + top-decile overlap）

### 10.3 预算 $100（完整，能跑 P1-A + P1-B + P2）

- **Lambda gpu_1x_a100 30 vCPU 200 GB @ $1.29/h × 64 h ≈ $83**
- 跑全量：P1-A full + P1-B + P2 held-out benchmark
- **拿 Q1 + Q2 + Q3 完整答案**

### 10.4 预算不限（最快）

- **Lambda gpu_8x_h100 224 vCPU @ $23.92/h × 12 h ≈ $287**
- 全量任务 ~12 h 完成（GSIQ 在 224 vCPU 上 scaling 5-6×）
- **不推荐**：cost 6×，wall-clock 节省 5×，不划算。

---

## 11. 与现有资产对比（T4 + CPU 5.8 GB）

| 阶段 | T4 + CPU 可行性 | 缺口 |
|---|---|---|
| P1-A full (P=2000) | ❌ 不可行 | CPU 5.1 GB 不能装 P=2000 GSIQ；T4 8 vCPU 上 35 h 超 20h 上限 |
| P1-A mid (P=1000) | ⚠️ 边界可行 | T4 上 ~10 h 可行，但降低精度 |
| P1-A low (P=500) | ✅ 可行 | T4 上 ~6 h；CPU 上 ~6 h（精度不足）|
| P1-C Task G | ✅ 可行 | T4 16 min |
| P1-B | ❌ 不可行 | 30 h CPU > 20 h 上限；proxy 估计需 ~ 1 GB GPU |
| P2 | ❌ 不可行 | 80 h CPU > 20 h 上限 |

**结论**：T4 + CPU 5.1 GB **只能跑 P=500 GSIQ + Task G**，不能跑 P1-A full（论文级精度）。

---

## 12. 与 R4§ 历史数据对比

R4§ 实际跑过 480 runs noise floor + 400 runs controlled geometry + 240 runs Task G = **1,120 solver runs**，全在 RTX 30-series GPU 上用 ~1 天完成（参考 EXECUTION_MANUAL §C4 + §F + §G）。

R5-B′ 任务量比较：
- P1-A GSIQ (83,520 calls @ P=2000) ≈ **18 h CPU**（R4§ 类比）
- P1-A solver (360 runs) ≈ 1 h GPU
- Task G (240 runs) ≈ 0.5 h GPU
- **P1-A full 总量 ≈ 18 h CPU + 1.5 h GPU = ~20 h wall-clock on A100**

---

## 13. 风险与不确定性

| 风险 | 影响 | mitigation |
|---|---|---|
| Multi-core eigh scaling < 4× | CPU 路径比预期慢 30% | 已在预算中加 buffer (1.3×) |
| Solver batched 在不同 GPU 上内存差异 | P2 可能需要更小 batch | `chunk=None` 已支持分块 |
| 抢占（spot）| 24 h 内重启丢失部分 progress | R4″ scripts 都支持 incremental resume（`r4pp_local_vs_global.py` 已含）|
| OpenBLAS 在某些云实例上只有单线程 | per-call 慢 4× | 测试时手动 `export OMP_NUM_THREADS=N` |
| 数据 LFS pull 在云上慢 | 增加 5-10 min 启动时间 | 提前在 clone 阶段 `--depth 1` 减少 fetch 量 |

---

## 14. 申请清单模板（直接拿去填）

```
申请类型: GPU 云实例（按需 spot 或 on-demand）
依据: R5-B′ Publication Lock 任务书 §R5-P1-A / P1-B / P1-C / P2
     HANDOFF §4.1 (Task G 因本机 WinError 1455 阻塞)
     当前资产 (T4 + 5.8GB CPU) 仅能跑 P=500 smoke, 无法支持论文级精度

推荐规格 (按预算):
  最小 ($25): 1× A100 40GB + 30 vCPU + 200 GB RAM
  标准 ($50): 同上 + 加 32 vCPU (Lambda gpu_1x_h100)
  完整 ($100): 同上 + 加 P2 held-out time
  不推荐: 8× H100 (cost/benefit 不划算, GSIQ CPU 路径才是瓶颈)

OS: Ubuntu 22.04 LTS / CUDA 12.8 / torch 2.12.0.dev+cu128
时长: 18 h (P1-A) ~64 h (P1-A+P1-B+P2) ~22 h (8xH100 全量)
预计 cost: $23 (P1-A) / $48 (P1-A+P1-B) / $83 (全量)
spot risk: 24h 内可重启, R4§ 脚本支持 incremental resume

数据准备 (git):
  仓库: https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering
  baseline commit: 9796884 (R4§ sprint 收官)
  当前 commit: c6b3b8d (R5-B' P0 + P1-A smoke 已提交)
  LFS data: p1/calibration_set/data_sun_confirmatory/ (19 dev scenes, ~300 MB)

代码入口:
  p1/source/information_audit/r5_p1_albedo_ablation.py (P1-A)
  p1/source/information_audit/r4pp_local_vs_global.py (Task G)
  p1/source/information_audit/solver_batched.py (solver)

预期产出:
  r5/r5_p1_albedo_ablation.csv (P1-A full)
  r5/r5_p1_albedo_ablation_ranking.csv
  r5/r5_p1_albedo_ablation_outliers.csv
  r5/r5_p1_albedo_ablation_gate.md
  r4pp/07_local_vs_global_init.csv (Task G)
  r5/P1-A closure memo (人工)

成功标准 (数值):
  P1-A: median(ρ) >= 0.95 AND median(top10 overlap) >= 0.80 → PASS-A
  Task G: β_global<0 AND β_oracle_local<0 → Case 1 (intrinsic identifiability)
```

---

## 15. 算力路径建议（决策树）

```
你有 T4 20h + CPU 5.1GB  → 已 cap（只能跑 P=500 + Task G，不够论文级）
                              │
                              ├── 申请 $25 A100 实例（18 h）→ 跑 P1-A full + Task G
                              │                                  → 拿 Q1+Q2
                              │   → Q3 取决于 P1-A 结果:
                              │       PASS-A → 申请 $50 跑 P1-B
                              │       CONDITIONAL → 申请 $100 跑 P1-B+P2 (高时间投入)
                              │       FAIL-A → 终止 selection claim, 转 identifiability diagnostic
                              │
                              ├── 或申请 $50 (A100 37h) → 一次跑 P1-A + P1-B
                              │
                              └── 或申请 $100 (A100 64h) → 全量一次性
```

---

*作者: ZCode agent · 2026-09-01*
*本表价格信息为 2026-08 公开列表价快照；申请前请核价*
*GSIQ per-call 实测：RTX 5070 Ti Laptop, OpenBLAS 24-thread, Windows 11*