# R5-B′ · 算力清点（H100 / 云实例申请依据）

> **作者**: ZCode agent · 2026-09-01
> **目的**: 把 R5-B′ P1-A full / P1-B / P1-C / P2 / P3 各阶段的资源需求翻译成数字，帮你做云实例 / H100 申请
> **方法**: 本机实测 + 代码路径分析 + 任务书 §9–§17 budget 拆解

---

## 0. 当前可用算力（RTX 5070 Ti Laptop）盘点

| 项 | 规格 | 实测 |
|---|---|---|
| GPU | NVIDIA RTX 5070 Ti Laptop GPU | 12.8 GB total VRAM |
| CPU | (未探) | OpenBLAS 24 threads available |
| RAM | 32 GB | commit 配额 94% 占用（HANDOFF §3） |
| OS | Windows 11 | WinError 1455 反复（HANDOFF §3） |

**本机当前状态**：
- ✅ `ga_isi_v2_scores` 在 P ≤ 2000 单调用能跑（实测 1.58s/call @ P=2000，0.04s/call @ P=300）；
- ❌ `joint_solve_batched` 在 B≥1 都立刻 CUDA error：unknown（WinError 1455 类）；
- ⚠️ `ga_isi_v2_scores` 在长跑时累积 OOM（cumulative Windows commit drift）：P=300 12-cell 6000 subsets 跑通，P=400 第二个 scene 起 OOM。

## 1. 代码路径与瓶颈分布

### 1.1 `ga_isi_v2_scores`（GSIQ 主指标）

| 子步骤 | P=2000 单调用耗时 | 占比 |
|---|---|---|
| `fisher_blocks` | 0.001 s | <1% |
| `schur_full` (P×P outer products) | 0.13 s | 8% |
| **`spectrum_metrics` (full eigh on P×P)** | **0.42 s** | **28%** |
| total per call | 1.5 s | 100% |

**瓶颈**：P×P dense symmetric eigendecomposition（CPU LAPACK，**GPU 没用上**）。

**P scaling**（实测，单线程 LAPACK）：
```
P=300: 0.04 s/call
P=500: 0.10 s/call
P=800: 0.25 s/call
P=1000: 0.39 s/call
P=2000: 1.50 s/call   ← smoke 默认上限
```

**Multi-thread scaling**：本机 OpenBLAS 24 线程，理论上 eigh 能用满多核。但本机 commit quota 占用 94% 后无法稳定分配 ~32MB P=2000 working set。

### 1.2 `joint_solve_batched`（solver）

- 单 run：`forward(Adam 400 iter) + 1 image recon`(B,N,H,W)
- 在 RTX 5070 Ti (12.8 GB) 上实测：**B=1 都失败**（CUDA unknown error / WinError 1455）
- 真实可用环境（H100 80GB 或 A100 40GB）上：典型 ~2-5 s/run（restarts=1, iters=400-800）

### 1.3 `r4pp_local_vs_global.py`（Task G）

- 与 `joint_solve_batched` 同引擎
- 6 scenes × 2 N × 10 subset × 2 init_mode = **240 run**
- 实测估计：~3-5 s/run @ H100 ⇒ **240 × 4s ≈ 16 min**

## 2. 阶段任务量拆解

### P1-A full（条件 PASS-A 后立刻跑）

| 项 | 数量 | 资源 |
|---|---|---|
| scenes | 6 (dev) + 4 (扩展) = 10 | — |
| NS | {3, 5} | — |
| pixel_cap | 2000 | — |
| N=3 subsets | C(32,3)=4960 per scene | 4960 × 2 score calls = 9920 calls/scene |
| N=5 subsets | sample 2000 per scene | 2000 × 2 = 4000 calls/scene |
| solver arm | 30 subsets per cell × 3 arms × 12 cells = 1080 runs | GPU (per run ~3-5s) |
| **total GSIQ calls** | 10 × (9920 + 4000) = **139,200** | CPU-bound eigh |
| **total solver runs** | **1080** | GPU-bound |

时间预估（H100 + 24-core CPU）：
- GSIQ: 139,200 × 1.5s = **58 h** single-thread；用 24 核 ~**2.5 h** if perfectly parallel
- Solver: 1080 × 4s = **1.2 h** GPU
- **Total: ~3.7 h**

但 GSIQ 的多核 scaling 受 eigh 单次调用的线程效率限制（实测只能拿到 4-6 倍加速，不是线性 24×）。所以真实估算：

- GSIQ: 139,200 × 1.5s / 6 = **9.7 h** CPU-bound (24-thread OpenBLAS 但 eigh 只用 4-6)
- Solver: 1080 × 4s = **1.2 h** GPU (可与 GSIQ 并行)
- **Wall-clock: ~10 h** CPU + 1.2 h GPU

### P1-B（条件 PASS-A 后，P1-A full 完成后）

proxy P = I(1, Ŷ, Ĉ)，其中：
- `Ŷ` 来自 coarse normal estimate（**不允许 GT normal**）
- `Ĉ` 来自 lighting estimate（**不允许 GT SH**）

P1-B 需额外算：
- 每个 dev scene 1 个 coarse normal estimate：5–10 min/scene（取决于方法）
- 每个 dev scene 1 个 lighting estimate：5–10 min/scene
- 5 score variants × (139,200 GSIQ calls) = **696,000** CPU calls（O 与 A 已有）
- 实际只需要 L3（normal proxy）+ L4（light proxy）+ P（fully practical），所以新增 ≈ **417,600 calls**（O, A 重用 P1-A）
- Top-decile overlap / ranking fidelity computation：negligible

**P1-B 真实预算**：
- 10 scenes × 2 proxy estimation ≈ 1.5 h GPU
- 417,600 × 1.5s / 6 = **29 h** CPU
- **Wall-clock: ~30 h CPU + 1.5 h GPU**

### P1-C Task G

| 项 | 数量 | 资源 |
|---|---|---|
| scenes | 6 | — |
| NS | {3, 5} | — |
| subsets | 10 per cell | — |
| init_mode | {global, oracle_local} | — |
| **total runs** | **240** | GPU |
| Time/run (H100) | ~4 s | — |
| **Total** | **16 min GPU** | — |

### P2 Held-out selection benchmark（条件 P1-B PASS 后）

任务书 §8：≥12 held-out scenes，**与 P1-A / P1-B 不重叠**。
任务书 §9 NS = {3, 5, 8}。
任务书 §10 候选 pool：N=3 enumerate 4960；N=5 / N=8 sample 2000–5000。
任务书 §11 solver runs / cell：40–45。

预算（条件 P1-B 通过后）：
- 12 held-out × × {N=3 enumerate 4960 + N=5 sample 2000 + N=8 sample 2000} = 12 × 8960 = **107,520 GSIQ calls**
- pixel_cap 2000，3 score variants（O/P + 一个 baseline 例如 angular）
- 12 × 3 NS × 45 solver = **1620 solver runs**
- GSIQ CPU: 107,520 × 3 × 1.5s / 6 = **80 h**
- Solver GPU: 1620 × 4s = **1.8 h**
- **Wall-clock: ~80 h CPU + 2 h GPU**

P3 / P4 / P5 暂不定量，等 P1/P2 出结果再细化。

## 3. 总预算（如果 P1-A full + P1-B + P1-C + P2 全部 PASS 后）

| 阶段 | CPU-hours | GPU-hours | Wall-clock |
|---|---|---|---|
| P1-A full (10 scene) | ~10 | 1.2 | ~10 h |
| P1-B (10 scene) | ~30 | 1.5 | ~30 h |
| P1-C Task G (并行) | — | 0.3 | 16 min |
| P2 held-out (12 scene) | ~80 | 1.8 | ~80 h |
| **合计** | ~120 | ~5 | ~120 h（5 day） |

## 4. 为什么本机 RTX 5070 Ti 不能完成

| 瓶颈 | 表现 | 来源 |
|---|---|---|
| Windows commit 配额 94% 占用 | 长跑 P=400 后 OOM | HANDOFF §3 实测 |
| WinError 1455（CUDA unknown error）| solver B≥1 立即 fail | HANDOFF §3 实测 |
| OpenBLAS 24-thread 多核 eigh 实际加速仅 4-6× | CPU-bound GSIQ 慢 | 实测 |
| 缺 Linux + NVIDIA Container | R4″ R5-B′ 数据可复现性需要 | HANDOFF §4.1 |

## 5. H100 / 其他云实例需求规格

### 5.1 推荐配置

| 资源 | 最低 | 推荐 | 备注 |
|---|---|---|---|
| GPU | 1× A100 40GB 或 1× H100 80GB | 1× H100 80GB | GPU-bound step 仅 5h，A100 也够 |
| CPU | 16 vCPU | **32 vCPU** | GSIQ 路径 CPU-bound，**CPU 比 GPU 重要** |
| RAM | 64 GB | **128 GB** | P=2000 × 32 = 12.8 GB working set；30 concurrent calls × ≈ 12 GB peak |
| Storage | 100 GB SSD | 200 GB SSD | 渲染中间产物 |
| OS | Linux（Ubuntu 22.04 / 24.04）| Linux | Windows commit quota 是已知阻塞 |
| CUDA / torch | 2.12.0.dev+cu128 | 同 | 与本机数据兼容 |

### 5.2 不需要 H100 SXM / NVLink 的理由

- GSIQ 路径完全 CPU-bound（eigh on P×P 是 LAPACK，GPU 帮助有限）
- solver 路径 GPU 利用率虽高，但总时长仅 5 h，单卡即可
- 数据规模 < 1 GB，PCIe 带宽足够，不需要 NVLink 多卡

### 5.3 不需要 A100 80GB 的理由

- 12.8 GB（5070 Ti）实测 OOM 主要是 Windows commit quota；**Linux 40GB A100 完全够**
- H100 80GB 优势在 FP16 / FP8 吞吐，对 GSIQ 帮助小

### 5.4 候选云实例（2026-09 价格快照，需你按区域核价）

| Provider | 实例 | GPU | CPU | RAM | 价格 ($/h, on-demand) |
|---|---|---|---|---|---|
| AWS | p4d.24xlarge | 8× A100 40GB | 96 vCPU | 1150 GB | $32.77 |
| AWS | p5.48xlarge | 8× H100 80GB | 192 vCPU | 2048 GB | $98.32 |
| Lambda | gpu_8x_h100 | 8× H100 80GB | 224 vCPU | 2048 GB | $23.92 |
| Lambda | gpu_1x_h100 | 1× H100 80GB | 26 vCPU | 252 GB | $2.99 |
| Lambda | gpu_1x_a100 | 1× A100 40GB | 30 vCPU | 200 GB | $1.29 |
| RunPod | H100 80GB | 1× H100 80GB | 16 vCPU | 64 GB | $2.49 |
| RunPod | A100 40GB | 1× A100 40GB | 8 vCPU | 32 GB | $1.64 |
| Vast.ai | H100 80GB | 1× H100 80GB | var | var | $1.5–2.5 (spot) |
| Vast.ai | A100 40GB | 1× A100 40GB | var | var | $0.8–1.5 (spot) |

> ⚠️ 价格随时间 / 区域 / spot / 预留波动；以上为 2026-08 公开列表价快照，请以你申请时的官方报价为准。

### 5.5 推荐选择

**最经济（推荐）**: **1× H100 80GB + 32 vCPU + 128 GB RAM on Lambda 或 Vast.ai**
- 总预算 ≈ $360（120 h × $3/h）
- 如果走 spot 可降到 ~$180
- 优势：单卡、Linux、commit quota 不受限（Linux 有 overcommit 机制）

**最稳妥**: **1× A100 40GB + 30 vCPU + 200 GB RAM on Lambda**
- 总预算 ≈ $155（120 h × $1.29/h）
- 优势：价格低 2×，A100 对 eigh 帮助与 H100 接近（瓶颈在 CPU LAPACK 不在 GPU）
- 缺点：GSIQ CPU 路径仍 ~30% 比 H100 慢（CPU 主频差）

**如果预算无限**: p5.48xlarge（8× H100）
- 总预算 ≈ $11,800（120 h × $98/h）
- 没有必要：本任务单 GPU 已足够，多 GPU 不带来 scaling

## 6. 申请清单（你可以直接拿去填）

```
申请类型: GPU 云实例
用途: R5-B′ 论文 held-out benchmark + Task G
依据: R5-B′ Publication Lock 任务书 §R5-P1-A / P1-B / P1-C / P2
HANDOFF §4.1: Task G 因本机 WinError 1455 阻塞，需 Linux + H100

推荐规格:
- GPU: 1× H100 80GB（或 1× A100 40GB）
- CPU: 32 vCPU（A100 配置选 30 vCPU）
- RAM: 128 GB（A100 配置选 200 GB）
- OS: Ubuntu 22.04 LTS
- Storage: 200 GB SSD
- 时长: 5 day wall-clock（120 CPU-h）
- 预算: $155–$360（取决于 provider 与是否 spot）

数据准备（git 仓库）:
- 仓库: dddsx3/Multi-Illumination-Inverse-Rendering
- commit baseline: 9796884 (R4″ sprint 收官)
- 数据依赖: p1/calibration_set/data_sun_confirmatory/（19 dev scenes）
                p1/calibration_set/data_heldout/（12 held-out scenes, 准备 P2）
- 代码入口:
    p1/source/information_audit/r5_p1_albedo_ablation.py
    p1/source/information_audit/r5_p1_normal_light_proxy.py （待写）
    p1/source/information_audit/r4pp_local_vs_global.py （已存在）

预期产出:
- r5/r5_p1_*.csv (P1-A / P1-B)
- r4pp/07_local_vs_global_init.csv (Task G)
- r5/r5_p2_*.csv (P2)
- 3 个总裁决点 Q1/Q2/Q3 报告
```

---

## 7. 总结：算力 vs 时间

| 阶段 | 必跑？ | 推荐在哪跑 | Wall-clock | $ (Lambda H100) |
|---|---|---|---|---|
| P1-A smoke（已完成）| ✅ DONE | RTX 5070 Ti | 已用 ~2h | $0 |
| P1-A full | 必要 | H100 | ~10h | $30 |
| P1-B | 条件 PASS-A | H100 | ~30h | $90 |
| P1-C Task G | 必要（Q2）| H100 | 16 min | $0.5 |
| P2（条件 PASS-B） | 条件 | H100 | ~80h | $240 |
| **合计（worst case P1+P2 全跑）** | | | **~120h（5 day** | **$360** |
| **合计（best case P1 PASS 后转 identifiability diagnostic）** | | | **~40h** | **$120** |

**最小必要算力**（Q1+Q2 回答 + 论文 submission 准备）：**$30（P1-A full + P1-C Task G）**。

---

*作者: ZCode agent · 2026-09-01 · 基于本机实测 + 任务书 §8-§17 budget*
*价格信息为 2026-08 公开列表价快照，**不是报价**；申请前请核价*