# R4″ 执行手册 · 逐小时表 / 实验矩阵 / 算力预算 / 脚本 I/O 契约

> **来源**：`R4″ 项目任务书 V1.0`（Scene–Illumination Observability 全新裁决版）下钻。
> **目的**：任何 agent 接手后可直接开跑，无需重新推导。
> **基线**：commit `006cdbf` · 单机 RTX 5070 Ti 12GB · 32GB RAM（commit 配额 ~94%）· Windows git-bash
> **纪律**：本文件不新增任何科学决策；一切阈值以任务书 §23–§27 为准。

---

## 0. 开工前必须拍板的三件事（阻塞 Day 1）

| # | 决策 | 我的推荐 | 理由 |
|---|---|---|---|
| **D-1** | 是否终止仍在跑的旧 R4′ solve（已 989/1620，剩 ~3.5h GPU） | **终止** | 任务书 §0 已把旧 R4′ 降级为 instrument-development dataset；§29 明确旧 18 scene 不进新 confirmatory。989 trial × 11 scene 对"找 bug / 比 metric / power planning"已充分。3.5h GPU 应转入 Task C（关键路径）。终止可逆（断点续跑）。 |
| **D-2** | Controlled geometry 用几个 family | **2 family × 5 level = 10 scene** | 任务书 §32 禁止"low-G 全 cube、high-G 全 sphere"。单 family 5 level 无法排除 family 混淆，2 family 才能交叉验证 gating 趋势。代价仅 +8min 渲染 +1.2h solver。 |
| **D-3** | N=2 是否纳入 | **纳入** | 任务书 §C2/§18 均标"算力允许则加"。N=2 单次 solve 仅 ~4s，全 Task C 只 +8min。而 §24 Gate 2 要看 N=2–5，缺 N=2 会削弱 low-N 结论。 |

---

## 1. 关于任务书三处前提的事实修正（必须先读，否则会白做工）

### 1.1 §5 Task A "先恢复所有被旧 convergence filter 排除的 trials" —— **已满足，无需恢复**

旧 filter 是**分析时**施加的（`r4p_confirmatory_gate._load_trials()` 里算的组内 P75），
**不是采集时**。`r4p_confirmatory_trials.csv` 本身就是全量未筛选记录，
`diagnostics/r4p_raw_trials_joined.csv` 已是 all-trials 视图。
⇒ Task A 退化为"格式化 + 补字段"，不是"数据恢复"。**省 Day 1 约 2h。**

### 1.2 §6 Task B 的 "objective stability" 判据 —— **旧数据无法回溯计算**

`joint_solve` 只返回 `final_loss` / `grad_norm`，**不落盘 loss 迭代轨迹**。
所以 `objective_rel_change`（连续 k 步相对变化）对旧 989 trial **不可计算**。
⇒ 绝对收敛判据只能在**新跑的 trial** 上生效；旧数据仅能用 finite + grad_norm 两条。
必须先改 solver 落盘 loss trace（见 §5 代码改动 C-1），否则 Task B 无法冻结。

### 1.3 §10 的 M1/M4/M5 —— **已经算好了，不用重写**

`diagnostics/r4p_trial_eigenspectrum.csv`（1620×51）已含：

| 任务书候选 | 现成列 |
|---|---|
| M1 normalized log pdet | `logdet_pos_mean_log`（= `(1/d)Σlog λ_i`） |
| M4 lower-spectrum quantile | `eig_norm_q{0.1,0.5,1,2,5,10,25}` |
| M5 effective rank | `eff_rank_entropy`、`participation_ratio` |
| M3 A-optimal | scores 表 `full_a_opt_pos_norm` |
| — | `cond_p1_p99`（抗离群条件数，可作 M4 变体） |

只需新增 **M2 regularized usable information** `(1/d)Σlog(1+λ_i/τ)`。
⇒ Task D 的 metric 计算部分省 ~1h，重心全部转到 §11 的六个 stability test。

---

## 2. 实验矩阵

### 2.1 Task C · Noise floor（Day 1 最高优先级）

**Scene 选择**（按 §C1，覆盖 G 三档 + 两类形状族，**不看旧 ρ 正负**）：

| scene | sh_gram_rank | normal_eff_rank | G 档 | 形状族 |
|---|---|---|---|---|
| `conf_cube_axis` | 4 | 1.00 | low | sparse-normal |
| `conf_prism8` | 5 | 1.18 | low | sparse-normal |
| `conf_cylinder_r03_d12` | 6 | 1.20 | medium-low | ruled |
| `conf_cone_r04_d12` | 9 | 1.24 | medium | ruled |
| `conf_egg` | 9 | 2.16 | high | smooth |
| `conf_icosphere_sub3` | 9 | 2.27 | high | smooth |

**矩阵**：6 scene × N{2,3,5,8} × 4 fixed subset × 5 solver seed = **480 solver run**（restarts=1）
**+ render repeat**：6 scene × 2 额外 realization → 6×4N×4subset×3realization×1seed = **288 run**

方差分解：
```
σ²_solver  = Var over 5 seeds  | fixed (scene,N,subset,render)
σ²_render  = Var over 3 renders| fixed (scene,N,subset,seed)
σ²_subset  = Var over 4 subsets| fixed (scene,N)，取 seed/render 均值后
σ²_repeat  = σ²_solver + σ²_render
R_signal   = σ_subset / σ_repeat
```

### 2.2 Task D · Metric bake-off（CPU，可与 GPU 交错但**不并发**）

评估单元：Task C 的 6 scene × N{3,5} × 30 subset = **360 unit**（子集来自旧 scores 序列，保证可复现）

| Test | 变体 | 单元数 | 通过标准（§11） |
|---|---|---|---|
| M-A cutoff stability | cutoff ∈ {1e-9, 1e-8, 1e-7} | 360×3 | ρ_rank > 0.95 |
| M-B pixel-cap stability | P ∈ {500, 1000, 2000} | 360×3 | ρ_rank > 0.90 |
| M-C pixel bootstrap | 同 cap=1000，5 个重采种子 | 360×5 | score 变异小、排序不漂 |
| M-D duplicate-light | 每 subset 加 1 盏与已有光夹角 <5° 的复制光 | 360 | 指标不得大幅上升 |
| M-E complementary-light | 加 1 盏与 subset span 正交度最高的光 | 360 | 指标应合理上升 |
| M-F extreme-mode robustness | 删最低 1 / 3 / 5 个 eigenmode 后重算 | 360×3 | 排序不得彻底改变 |

对每个候选 metric（M1–M5）都跑全部六项 → **metric × test 矩阵 5×6**。

### 2.3 Task E · Geometry observability（CPU，便宜）

25 scene（18 旧 + ~7 新 controlled）× 5 候选 G（rank / normalized logdet / eff rank / cond / λ_min(9×9)）
× 4 稳定性扰动（pixel resample ×5、mesh 分辨率 ×2、旋转 sanity ×3、scale normalization）

### 2.4 Task F · Controlled geometry pilot（Day 2 核心）

**两个 family，各 5 level，只有 normal coverage 变化**（相机/材质/尺度/renderer/光池全固定）：

| family | level 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **A 棱柱→柱** | prism4 | prism8 | prism16 | prism32 | cylinder(64) |
| **B 立方→球** | cube | bevel 0.05 | bevel 0.15 | bevel 0.30 | rounded cube |

**矩阵**：10 scene × N{3,5} × 5 information strata × 4 subset = **400 solver subset**（restarts=3）

分层抽样（§19）：
1. 每 (geometry, N) 生成 **1000** candidate subset
2. 只算 **cheap tier** information（frozen primary metric，P=300）
3. 按 quintile 分 5 层，每层等量抽 4 个 → 20 subset/cell
4. 被抽中的 subset 再用 **full tier**（P=1000）重算 information 入表

### 2.5 Task G · Local vs global initialization（§21，防 reviewer 攻击）

6 scene（Task C 同批）× N{3,5} × 10 subset × 2 init mode × 1 seed = **240 run**

| mode | 初始化 | 目的 |
|---|---|---|
| `global` | 现行（a=softplus⁻¹(0.3) 常数，c=0.01 噪声 + DC 0.3） | practical recoverability |
| `oracle_local` | θ₀ = θ_GT + δ，δ 固定相对幅度（albedo 5% RMS、c 5% norm） | intrinsic/local conditioning |

---

## 3. 算力预算表

**实测基准**（来自 R4′ 989 trial 与诊断跑）：

| 操作 | 实测单价 |
|---|---|
| `joint_solve` restarts=3 · 128² | N=2 ~12s / N=3 15s / N=5 21s / N=8 28s |
| `joint_solve` restarts=1（÷3） | N=2 4s / N=3 5s / N=5 7s / N=8 9.3s |
| Fisher `schur_full`+`eigvalsh` P=1000 | 0.45 s |
| 同上 P=500 / P=2000 | 0.06 s / 3.6 s |
| 同上 P=300（cheap tier） | 0.012 s |
| 渲染 1 scene × 32 SUN · 128²/32spp | 46 s |
| scene Gram spectrum（全掩码） | 1.7 s/scene |

### 3.1 GPU 预算

| 任务 | job 数 | 单价 | 小计 | 含 1.3× 余量 |
|---|---|---|---|---|
| C4 noise floor solver | 480 (restarts=1) | 4–9.3s | 51 min | **1.1 h** |
| C5 render repeat（渲染） | 12 render | 46 s | 9 min | **0.2 h** |
| C5 render repeat（solver） | 288 (restarts=1) | 6.3s 均 | 30 min | **0.7 h** |
| F controlled 渲染 | 10 scene | 46 s | 8 min | **0.2 h** |
| F controlled solver | 400 (restarts=3) | 15/21s | 2.0 h | **2.5 h** |
| G local-vs-global | 240 (restarts=1) | 6s 均 | 24 min | **0.5 h** |
| **GPU 合计** | **1408 solver run + 22 render** | | **4.0 h** | **5.2 h** |

### 3.2 CPU 预算

| 任务 | 计算量 | 小计 |
|---|---|---|
| A master table（格式化，无重算） | 989 行 join | 10 min |
| B convergence audit（旧数据 finite+grad 两条） | 989 行 | 5 min |
| D-A cutoff sweep | 360×3 @0.45s | 8 min |
| D-B pixel-cap sweep | 360×(0.06+0.45+3.6) | 25 min |
| D-C pixel bootstrap | 360×5 @0.45s | 14 min |
| D-D/E/F sanity + mode-drop | 360×5 @0.45s | 14 min |
| E geometry metrics + 稳定性 | 25×(1.7s×11) | 8 min |
| F candidate screening（cheap tier） | 10×2×1000 @0.012s | 4 min |
| F full-tier rescoring | 400 @0.45s | 3 min |
| 统计 + dashboard | — | 30 min |
| **CPU 合计** | | **~2.1 h** |

### 3.3 关键约束（违反会重演已发生的事故）

1. **GPU 与重 CPU Fisher 不得并发**：已实测 CUDA host queue OOM（commit 配额 94%）。P=2000 的 D-B 必须在 GPU 空闲窗口跑。
2. **P=2000 有 OOM 风险**：R4′ 阶段 P=2000 dense 反复分配失败。D-B 的 cap2000 分支必须带 MemoryError 退避重试（现成模式见 `r4p_diagnostics.py`）。
3. **每场景独立 blenderproc 进程**（踩坑 #2，`enable_*` 进程级一次性）。
4. **渲染必须走 INC-001 修复后的 `render_multilight.py`**（帧级完整性校验）。
5. 机器总时间 **~7.3 h**，48h sprint 有 6× 余量 → 瓶颈是**人/agent 的编码时间，不是算力**。

---

## 4. 逐小时执行表

> H = 从 sprint 开始计的小时。"编码"指 agent 写脚本，"跑"指机器执行。
> 所有 GPU 任务串行；CPU 任务标注可否与 GPU 交错。

### Day 0（H0–H1，1h）· 冻结

| H | 动作 | 产出 | 阻塞 |
|---|---|---|---|
| H0.0 | **拍板 D-1/D-2/D-3**（§0） | — | 需用户 |
| H0.0 | 若 D-1=终止：kill 后台 solve，记录终止点 | `archive/R4prime_frozen/TERMINATION.md` | — |
| H0.1 | `r4pp_freeze_archive.py` 建只读归档 | `archive/R4prime_frozen/**` | — |
| H0.5 | 写 failure memo（P0-1…P3-7 七行表，内容已在 `R4P_DIAGNOSTIC_BUNDLE.md`） | `R4prime_failure_audit.md` | — |
| H1.0 | commit + push「R4′ 冻结」 | tag `r4prime-frozen` | — |

### Day 1 上半（H1–H5，4h）· Master table + 绝对收敛 + Noise floor 启动

| H | 动作 | 类型 | 产出 |
|---|---|---|---|
| H1.0 | **代码改动 C-1**：`joint_solve` 加 `seed` / `theta0` / `return_trace` 参数（§5） | 编码 30min | patch |
| H1.5 | 单测：同 seed 两次结果逐位一致；`theta0` 生效 | 跑 5min | `tests/test_solver_repro.py` |
| H1.7 | `r4pp_master_table.py` | 编码 40min | — |
| H2.4 | 跑 master table | CPU 10min | `01_master_trial_table.parquet` |
| H2.6 | `r4pp_convergence_audit.py`（绝对判据 + 旧 filter 偏差） | 编码 40min | — |
| H3.3 | 跑 audit | CPU 5min | `05_old_filter_bias_report.pdf/md` |
| H3.4 | `r4pp_noise_floor.py` | 编码 50min | — |
| H4.2 | **启动 C4 noise floor（GPU 1.1h）** | GPU 后台 | 部分 `02_noise_floor.csv` |
| H4.2 | 并行（CPU 轻）：`r4pp_geometry_metrics.py` | 编码 40min | — |
| H5.0 | — | — | — |

### Day 1 下半（H5–H9，4h）· Noise floor 收尾 + Metric bake-off

| H | 动作 | 类型 | 产出 |
|---|---|---|---|
| H5.3 | C4 完成 → 启动 C5 render repeat（GPU 0.9h） | GPU 后台 | — |
| H5.3 | 跑 geometry metrics（CPU 轻，可交错） | CPU 8min | `04_geometry_spectrum.csv` |
| H5.5 | `r4pp_metric_bakeoff.py`（M1–M5 × M-A…M-F） | 编码 1.5h | — |
| H7.0 | C5 完成 → 计算 σ 分解 + R_signal | CPU 5min | `02_noise_floor.csv` 定稿 |
| H7.1 | **Gate 2 初判**（low-N R_signal > 1？） | 判读 | dashboard 行 2 |
| H7.2 | 启动 metric bake-off（CPU 1.1h，GPU 此时空闲，cap2000 安全） | CPU 后台 | `03_metric_stability.csv` |
| H7.2 | `make_controlled_geometry_meshes.py`（2 family × 5 level） | 编码 50min | — |
| H8.1 | blenderproc 生成 10 个 mesh | GPU 5min | `meshes_controlled/**` |
| H8.3 | 渲染 10 scene（每场景独立进程） | GPU 8min | `data_sun_controlled/**` |
| H8.5 | 数据 Gate + Oracle（复用 `r4p_data_gates.py`） | CPU 3min | Gate 报告 |
| H8.8 | **Gate 1 初判** + 冻结 primary illumination metric（§12 优先级，**最后才看 error 关联**） | 判读 | dashboard 行 1 |
| H9.0 | commit + push「Day 1 交付」 | — | 5 份 Day 1 产物 |

### Day 2 上半（H9–H13，4h）· Controlled geometry pilot

| H | 动作 | 类型 | 产出 |
|---|---|---|---|
| H9.0 | `r4pp_controlled_pilot.py`（3 阶段分层抽样 + solver） | 编码 1h | — |
| H10.0 | Stage 1–2：1000 candidate/cell cheap 打分 + quintile 分层 | CPU 4min | `candidates_*.csv` |
| H10.1 | **Stage 3：400 subset solver（GPU 2.5h）** | GPU 后台 | `06_controlled_geometry_results.csv` |
| H10.1 | 并行编码：`r4pp_local_vs_global.py` | 编码 50min | — |
| H11.0 | 并行编码：`r4pp_dashboard.py`（6 行 Gate 表 + 5 张 figure 骨架） | 编码 1.5h | — |
| H12.6 | pilot 仍在跑 → 写 `09_R4pp_decision.md` 模板（三分支预填，结论留空） | 编码 20min | — |
| H13.0 | — | — | — |

### Day 2 下半（H13–H16，3h）· 解耦实验 + 裁决

| H | 动作 | 类型 | 产出 |
|---|---|---|---|
| H12.6 | pilot 完成 → 估 per-geometry β_G，画 Figure 3 | CPU 10min | `fig3_geometry_mechanism.png` |
| H12.8 | **Gate 4 初判**（G↑ ⇒ |β_G|↑ 连续趋势？） | 判读 | dashboard 行 4 |
| H13.0 | 启动 G local-vs-global（GPU 0.5h） | GPU | `07_local_vs_global_init.csv` |
| H13.5 | **Gate 6 初判**（oracle-local 下关系是否存活） | 判读 | dashboard 行 6 |
| H13.7 | Gate 3（info→error 方向）+ Gate 5（N=8 saturation）计算 | CPU 15min | dashboard 行 3/5 |
| H14.0 | 生成 `08_go_no_go_dashboard.pdf`（**只含 6 行，禁止加第 7 项**） | CPU 5min | dashboard |
| H14.1 | Figure 1/2/4 生成 | CPU 20min | figures |
| H14.5 | **GO / PIVOT / KILL 会议**（只看 dashboard） | 判读 | `09_R4pp_decision.md` |
| H15.0 | 按裁决更新 `CLAIM_REGISTRY.md`（→ v0.3）+ `P1_R0_STOP_LINE.md` 附录 | 编码 30min | — |
| H15.5 | commit + push + 打包 zip 给顾问 | — | `R4PP_SPRINT_*.zip` |
| H16.0 | 结束 | — | — |

**机器占用**：GPU 5.2h / CPU 2.1h，分布在 16h 窗口内 → 有 ~9h 空闲缓冲吸收编码超时与重跑。

---

## 5. 必须先做的代码改动（阻塞多项任务）

| ID | 文件 | 改动 | 阻塞 |
|---|---|---|---|
| **C-1** | `information_audit_v2.py::joint_solve` | ① 加 `seed=None`（替代写死的 `torch.manual_seed(20260830+rs)`）② 加 `theta0=None`（(a_raw, c) 初值，供 oracle-local）③ 加 `return_trace=False` → 返回 `loss_trace` 数组 ④ 返回值补 `tail_rel_change`、`proj_grad_norm`（scale-normalized） | Task B/C/G 全部 |
| **C-2** | 新 `r4pp_metrics.py` | 统一 Fisher 预处理（§D1：nuisance elim → gauge projection → 对称化 → eigh → dimension normalization → structural nullity 声明），所有 metric 共用同一份谱 | Task D 全部 |
| **C-3** | 新 `r4pp_metrics.py::M2` | `I_τ = (1/d)Σlog(1+λ_i/τ)`，τ **只允许**由 Task C 的 σ_repeat 标定，禁止调参 | Task D |
| **C-4** | `render_multilight.py` | 加 `--cycles_seed`（固定 Cycles 采样种子）以便区分 render 随机性 vs 确定性重渲 | Task C5 |

**C-1 的关键细节**：现行 `torch.manual_seed(20260830 + rs)` 在每个 restart 前重置，
使同一 `rs` 的 c 初始化在所有 (scene, subset) 上相同。改为显式 `seed` 后
**必须保留这一语义**（否则与旧 989 trial 不可比）：`seed=None` 时行为不变。

---

## 6. 脚本 I/O 契约

### 6.1 `r4pp_freeze_archive.py`

```
IN : p1/information_audit/{r4p_confirmatory_*.csv, r4p_conv_thresholds.json,
                            diagnostics/*, R4P_*.md}
     p1/protocol/{R4P_PREREGISTRATION.md, CLAIM_REGISTRY.md}
     git rev-parse HEAD, pip freeze, nvidia-smi -q
OUT: archive/R4prime_frozen/
       ├─ data/**              (逐文件复制，chmod 只读)
       ├─ MANIFEST.csv          path, sha256, bytes, mtime
       ├─ ENVIRONMENT.txt       commit / python / torch / numpy / scipy / GPU / OS
       └─ TERMINATION.md        若 D-1=终止：终止时的 trial 数与最后 scene
EXIT: 非 0 当且仅当 sha256 校验失败
```

### 6.2 `r4pp_master_table.py`

```
IN : archive/R4prime_frozen/data/r4p_confirmatory_trials.csv     (全量，未筛选)
     archive/R4prime_frozen/data/r4p_confirmatory_scores.csv
     archive/R4prime_frozen/data/diagnostics/r4p_trial_eigenspectrum.csv
     archive/R4prime_frozen/data/diagnostics/r4p_scene_gram_spectrum.csv
OUT: r4pp/01_master_trial_table.parquet
列（≥ 任务书 §5 最低字段）:
  identity      : scene_id, geometry_family, dataset_tag(=R4prime_exploratory)
  budget        : N
  subset        : illumination_ids (逗号串), n_lights
  randomness    : solver_seed(旧数据=20260830+rs 语义), pixel_seed
  optimization  : solver_status, iteration_count, final_objective, grad_norm,
                  objective_rel_change (旧数据 = NaN，见 §1.2)
  error         : reconstruction_error(=si_mae_A), ho_psnr
  old filtering : old_converged_flag(solver 当场), old_p75_success_flag(分析时重算)
  Fisher        : eig_norm_q*(14), n_above_*(9), logdet_pos_mean_log,
                  eff_rank_entropy, participation_ratio, cond_p1_p99, trace,
                  n_negative, neg_mass_over_trace
  Fisher rank   : rank_Fk_min/max/mean, Fk_eigmax_min, Fk_minpos_min
  geometry      : sh_gram_rank, sh_gram_logdet, sh_gram_eff_rank,
                  normal_cov_eff_rank, normal_cov_anisotropy
断言: 行数 == trials.csv 数据行数（不得因 join 丢行）；old_p75_success_flag 的
      均值应 ≈ 0.5625（复现 §1 的 endogenous filtering 证据）
```

### 6.3 `r4pp_convergence_audit.py`

```
IN : r4pp/01_master_trial_table.parquet
     (新数据) r4pp/traces/*.npy      loss_trace per trial
OUT: r4pp/05_old_filter_bias_report.md  +  .csv
内容:
  1. 绝对判据定义（冻结）:
       finite      : all(isfinite(objective, grad, params))
       stationarity: proj_grad_norm / max(|f|,eps) < eps_g
       stability   : max_k |f_t - f_{t-k}| / max(|f_t|,eps) < eps_f,  k=50
     (eps_g, eps_f) 由 Day 1 pilot 的数值尺度定一次后**永久冻结**，写入
     r4pp/CONV_CRITERIA_FROZEN.json —— 禁止事后调整（任务书 §6/§43.2）
  2. 旧 P75 filter 的选择偏差:
       - E[error | old_p75_success=1] vs E[error | =0]
       - P(old_p75_success=1) 对 information / N / G 的 logistic 回归系数
         （若显著 ⇒ endogenous selection 证据）
  3. 新旧判据的一致性矩阵（2×2 混淆表）
```

### 6.4 `r4pp_noise_floor.py`

```
IN : p1/calibration_set/data_sun_confirmatory/{6 个选定 scene}/
     r4pp/config/noise_floor_matrix.json    scene/N/subset/seed/render 列表（预生成、冻结）
OUT: r4pp/02_noise_floor.csv               每 run 一行
     r4pp/02_noise_floor_summary.csv       每 (scene,N) 一行
per-run 列 : scene, N, subset, solver_seed, render_realization,
             reconstruction_error, final_objective, proj_grad_norm,
             tail_rel_change, conv_finite, conv_stationary, conv_stable, iters
summary 列 : scene, N, sigma_solver, sigma_render, sigma_repeat, sigma_subset,
             R_signal, n_runs, n_converged, G_bucket, sh_gram_rank
断言       : 每 (scene,N,subset) 至少 5 个不同 solver_seed 且结果非逐位相同
             （若相同 ⇒ seed 未生效，C-1 改动失败）
```

### 6.5 `r4pp_metric_bakeoff.py`

```
IN : p1/calibration_set/data_sun_confirmatory/{6 scene}/
     r4pp/config/bakeoff_units.json     360 个 (scene,N,subset) 冻结列表
OUT: r4pp/03_metric_stability.csv
列 : metric_id(M1..M5), test_id(M-A..M-F), variant(cutoff/cap/boot_seed/...),
     unit_count, rho_rank_vs_reference, score_cv, ordering_drift,
     dup_light_delta, comp_light_delta, mode_drop_rho, PASS/FAIL
     + r4pp/03_metric_raw_scores.csv   每 (unit × metric × variant) 的原始分数
参考基准: 每个 metric 以 (cutoff=1e-8, cap=1000, boot_seed=0) 为 reference
判读     : 严格按 §11 阈值（M-A ρ>0.95, M-B ρ>0.90），**不看 error 关联**
禁止     : 任何依据 error correlation 的排序（§43.1）
```

### 6.6 `r4pp_geometry_metrics.py`

```
IN : {18 旧 scene} + {10 controlled scene} 的 normal_mesh.npy / mask.npy / albedo.npy
OUT: r4pp/04_geometry_spectrum.csv
列 : scene, G1_rank, G2_norm_logdet(=(1/9)logdet(G_s+eps I)), G3_eff_rank,
     G4_cond, G5_min_nonzero_eig,
     + 稳定性: 上述 5 个在 {pixel resample ×5, mesh 分辨率 ×2, 旋转 ×3} 下的 CV
     + W 权重方案标注（W=I 与 W=diag(a²) 两版，见任务书 §13 的 G_s = Y^T W Y）
冻结  : 通过全部稳定性检查且 CV 最小者 → r4pp/GEOMETRY_METRIC_FROZEN.json
```

### 6.7 `make_controlled_geometry_meshes.py`（blenderproc）

```
IN : --out_dir, --families A,B --levels 5
OUT: p1/calibration_set/meshes_controlled/{famA_L1..L5, famB_L1..L5}.obj
     + controlled_list.txt
约束: 所有 level 的 bounding-box 最大边长归一到同值（渲染端还会归一到 1.6，
      但此处先归一以保证 bevel 比例语义一致）
首行必须 `import blenderproc`（踩坑 #1）
```

### 6.8 `r4pp_controlled_pilot.py`

```
IN : p1/calibration_set/data_sun_controlled/{10 scene}/
     r4pp/GEOMETRY_METRIC_FROZEN.json
     r4pp/METRIC_FROZEN.json                 (primary illumination metric)
OUT: r4pp/controlled_candidates_{scene}_{N}.csv   1000 行 cheap 打分
     r4pp/06_controlled_geometry_results.csv      400 行 solver 结果
     r4pp/06_beta_per_geometry.csv                10 行（每 geometry level 一行）
stage : 1) candidate 生成 + cheap 打分(P=300)
        2) quintile 分层 → 每层抽 4
        3) full-tier 重算 information(P=1000) + solver(restarts=3)
        4) 每 geometry level 拟合 log Error ~ z(I) → beta_G, se, bootstrap CI
断言  : 抽样后 information 的 5 个 strata 均值单调递增（否则分层失败）
```

### 6.9 `r4pp_local_vs_global.py`

```
IN : {6 scene} + r4pp/config/local_global_units.json  (6×2×10 冻结列表)
OUT: r4pp/07_local_vs_global_init.csv
列 : scene, N, subset, init_mode(global|oracle_local), perturb_rel,
     reconstruction_error, converged, proj_grad_norm, information, G
判读: 分别在两 mode 下估 β；若 oracle_local 的 β 仍显著为负
      ⇒ effect 非纯 optimization artifact（Gate 6 PASS）
```

### 6.10 `r4pp_dashboard.py`

```
IN : r4pp/{01..07}*
OUT: r4pp/08_go_no_go_dashboard.pdf   (+ .md)
     r4pp/09_R4pp_decision.md
     r4pp/figures/fig{1..5}_*.png
dashboard 严格 6 行（任务书 §41），不得增删:
  | Gate        | 指标                    | 判据                       |
  | Instrument  | metric stability       | M-A ρ>0.95 且 M-B ρ>0.90 且 M-F 稳 |
  | Signal      | low-N signal/noise     | 多数 high-G scene 在 N≤5 R_signal>1（>2 更佳） |
  | Direction   | info→error             | 方向一致、非单场景驱动、bootstrap 不覆盖任意方向 |
  | Interaction | geometry gating        | G↑ ⇒ |β_G|↑ 连续趋势        |
  | Saturation  | N=8 noise-floor        | N=8 R_signal≈1 且 N=3/5 明显>1 |
  | Externality | local-init replication | oracle_local 下 β 仍为负     |
裁决映射（§44）:
  Instrument+Signal+Direction+Interaction 全 PASS      → GO  (A2)
  Instrument+Signal+Direction PASS, Interaction FAIL   → PIVOT (B′)
  Instrument FAIL 或 Signal FAIL                        → KILL H-COND
```

---

## 7. 复用资产映射（不要重写）

| 需求 | 现成资产 | 需改 |
|---|---|---|
| Fisher 块 / full-Schur / gauge 投影 | `gauge_fisher_v2.py`（28/28 单测 PASS） | 无 |
| 全谱分位 / bulk metric | `r4p_diagnostics.py::stage_eigenspectrum` | 加 M2(τ) |
| scene Gram / normal 协方差谱 | `r4p_diagnostics.py::stage_scene_gram` | 加旋转 / 分辨率扰动 |
| all-trials 视图 | `r4p_diagnostics.py::stage_raw_join` | 输出改 parquet |
| 渲染（INC-001 已修） | `render_multilight.py` | 加 `--cycles_seed` |
| mesh 参数化家族 | `make_confirmatory_meshes.py` | 复制改 bevel/prism 序列 |
| 数据 Gate + Oracle | `r4p_data_gates.py` | 改 data_root |
| solver | `information_audit_v2.py::joint_solve` | **C-1（阻塞）** |
| scene-bootstrap / LOO-CV 统计骨架 | `r4p_confirmatory_gate.py::{g2_stats,e3_stats}` | 换 hierarchical，弃 binning |

---

## 8. 每小时自查清单（防止重演 R4′ 的失败）

每完成一个 task 前逐条核对（任务书 §43）：

- [ ] 没有因为 correlation 不够而更换 metric
- [ ] 没有调 rank cutoff 直到结果显著
- [ ] 报告了 all trials，不只是 converged
- [ ] N=8 有 noise 测量，不是直接删掉
- [ ] 没有把 `rank_Fk` 当 nuisance covariate 塞进去就宣布解决
- [ ] 没有在旧 18 scene 上重新挑假设后称 confirmatory
- [ ] controlled mechanism 实验没有被"更多随机场景"替代
- [ ] 没有用 p-value 决定 48h pilot 是否存在 signal
- [ ] 没有因为 sunk cost 降低 KILL 门槛
- [ ] 所有冻结项（conv 判据 / primary metric / geometry metric）都已落 JSON 且带时间戳

---

## 9. 交付物清单（Day 1 / Day 2）

| Day | 文件 | 来源脚本 |
|---|---|---|
| 0 | `archive/R4prime_frozen/**` + `R4prime_failure_audit.md` | `r4pp_freeze_archive.py` |
| 1 | `r4pp/01_master_trial_table.parquet` | `r4pp_master_table.py` |
| 1 | `r4pp/02_noise_floor.csv` + `_summary.csv` | `r4pp_noise_floor.py` |
| 1 | `r4pp/03_metric_stability.csv` + `03_metric_raw_scores.csv` | `r4pp_metric_bakeoff.py` |
| 1 | `r4pp/04_geometry_spectrum.csv` | `r4pp_geometry_metrics.py` |
| 1 | `r4pp/05_old_filter_bias_report.md` | `r4pp_convergence_audit.py` |
| 1 | `r4pp/{CONV_CRITERIA,METRIC,GEOMETRY_METRIC}_FROZEN.json` | 各自脚本 |
| 2 | `r4pp/06_controlled_geometry_results.csv` + `06_beta_per_geometry.csv` | `r4pp_controlled_pilot.py` |
| 2 | `r4pp/07_local_vs_global_init.csv` | `r4pp_local_vs_global.py` |
| 2 | `r4pp/08_go_no_go_dashboard.pdf` | `r4pp_dashboard.py` |
| 2 | `r4pp/09_R4pp_decision.md` | 会议产出 |
| 2 | `r4pp/figures/fig{1..5}_*.png` | `r4pp_dashboard.py` |

---

*编制：R4″ sprint 执行层 · 2026-08-31 · 基线 commit `006cdbf`*
*本文件不含任何科学裁决；阈值与裁决规则一律以 `R4″ 项目任务书 V1.0` 为准。*
