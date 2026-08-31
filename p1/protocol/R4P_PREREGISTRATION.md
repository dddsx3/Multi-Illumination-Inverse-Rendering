# R4′ Confirmatory Gate · 预注册（冻结版 v1.0）

> **冻结时点**：本文件 commit 先于任何 R4′-C 确认数据的统计分析。
> **依据**：`P1_NEXT_STAGE_EXECUTION_TASKBOOK v1.0` §4（T4′.0–T4′.6）、§8 纪律 #8。
> 冻结后**禁止**更换 primary、阈值、N、subsets 数、回归式（纪律 #9 不救故事）。

## 0. 防双重使用（T4′.0）

- Discovery Set = `p1/calibration_set/data_sun`（cube/cylinder/hemisphere/sphere，
  seed 20260830）。仅用于：R4′-D 实现稳定性检查、solver 收敛判据标定、
  批量 solver 与串行实现的一致性验证。**其任何统计量不进入 confirmatory 证据。**
- Confirmatory Set = `p1/calibration_set/data_sun_confirmatory`（25 scene ×32 SUN，
  seed 20260901，`make_confirmatory_meshes.py` 参数化家族）。与未来 P1-13
  test split 隔离（注册于 `split_manifest_r4p_confirmatory.json`）。

## 1. 冻结指标（T4′.2）

- **Primary**：`full_lam_min_pos_norm` = scene-normalized λ_min⁺
  （`gauge_fisher_v2.spectrum_metrics`：F_eff/trace 正谱最小值）。
  数值策略冻结：cutoff(F_k 伪逆) = 1e-8；**pixel_cap = 1000**
  （冻结理由：本机 Windows commit 配额 ~32GB 已用 ≥94%，P=2000 dense 的
  ~500MB 瞬时峰值反复分配失败——同一评估重复 3 次死于 30.5MB 分配；
  P=1000 峰值 ~150MB 稳健。cap 对全部 scene/subset 统一适用，指标为
  trace 归一无量纲；cap1000 vs cap500 的 primary 排名稳定性由 R4′-D
  变体实证，见 R4P_DISCOVERY_RERUN_REPORT.md）；spec_cutoff = 1e-8；
  gauge 处理 = 投影等效的正谱指标（T4b 已证不变）。
- **Secondary**：`full_logdet_pos_norm`、`full_a_opt_pos_norm`、
  `full_d_pos`、angular diversity（32 SUN 方向子集的成对夹角均值，
  从 metadata 单位方向计算）。
- 看完 confirmatory 结果后禁止更换 primary。

## 2. E2 · same-N different-conditioning（T4′.3）

- N ∈ {3, 5, 8, 12}；每 scene 每 N = **50** random subsets（无放回，
  `np.random.default_rng(20260902)` 统一采样，seed 落盘于 CSV；100 预
  注册时算力预算为 18×4×100×3×12s ≈ 72h 超单机/单日，降至 50 控预算在
  36h 量级、仍满足 per-scene Spearman n≥30 与 scene-bootstrap 10000 的
  统计门槛）。
- Solver：`joint_solve_batched`（批量 = 串行实现的逐元素等价形式；
  冻结验证门槛：4 个 (scene,N) 验证用例上 SI-MAE 相对差 ≤ 1e-3）。
  - restarts = 3（固定种子 20260830+rs）；max_iters = 800 + 200·N（与 R4 相同）；
  - 收敛判据（先在 Discovery 上标定后冻结）：solver 落盘 raw 诊断
    tail_range（末 50 iter loss 极差）与 grad_norm；pilot（Discovery 4 scene ×
    N=5 × 6 subsets = 24 trials，串行 solver）取 rel 标度后冻结
    **tail_range < 【pilot P75】且 grad_norm < 【pilot P75】**，写
    `p1/information_audit/r4p_conv_thresholds.json` 后不得更改；
    最终 success rate 必须报告；**只纳入收敛 trials**，失败率单独列报，
    不得静默删除；另报全 trials（不筛）的 E2 相关系数作稳健性对照（不改裁决）。
  - solver = `joint_solve`（串行；batched 版 Bp=2 vs 串行验证有 ~1e-3 相对
    漂移，根因为 `c[b]=noise` 赋值与 Adam buffer reshape kernel 累加序差；
    6.5× speedup 不值 1e-3 风险，batched v3 重做）；
  - 记录：iters / grad-norm / final loss / objective gap（loss − 饱和下界代理）/ success。
- 误差：`si_mae_A`（albedo SI-MAE，scale gauge，mesh normal GT 口径）+
  `ho_psnr`（oracle-query-light held-out relighting，q = 子集外最小索引）。
- **主统计（控制 scene effect）**：
  1. 每 (scene, N) 内 Spearman ρ(primary, si_mae_A)（n=100）；
  2. scene 级汇总：per-scene ρ 的中位数 + **scene-level bootstrap**（重采样
     scene，B=10000）95% CI；
  3. 方向冻结：primary 越大 ⇒ 信息越大 ⇒ 误差越低 ⇒ **ρ < 0** 为正确符号。
- **E2 PASS**（预注册阈值）：per-scene ρ 中位数 ≤ −0.30，且 ≥ 80% scene 符号为负，
  且 scene-bootstrap 95% CI 上界 < 0。三条同时满足。

## 3. G2 · beyond-N explanatory power（T4′.4）

- baseline：`Error ~ 1 + logN + scene 固定效应`（场景哑变量）；
- full：`Error ~ 1 + logN + scene 固定效应 + primary`；
- 估计：pooled OLS（scene 内 z-score 不再需要——固定效应已吸收场景尺度）；
- **out-of-sample**：leave-one-scene-out CV（固定效应按训练 scene 拟合，
  测试 scene 用自身均值截距），报 ΔR²_oos = R²_full,oos − R²_base,oos；
- 不确定性：scene-level bootstrap 95% CI（B=10000）；
- **G2 PASS**（任务书冻结阈值）：ΔR²_oos ≥ 0.05 且 95% CI 下界 > 0，
  且 primary 系数符号为负（正确方向）。三条同时满足。
- 附加（不改变裁决）：以 secondary 指标重复同一回归，仅作报告。

## 4. E3 · matched-conditioning different-N（T4′.5）

- 匹配：每 scene 内，将 N∈{3,5,8,12} 的 100 subsets 按 primary 分十分位；
- 量：对每 scene 拟合 `Error ~ logN`（未匹配全部数据 → slope_unmatched）与
  同十分位中心对齐后的 `Error ~ logN`（matched；即每个 N 取各分位均值误差）；
- 统计：ratio = slope_matched / slope_unmatched，per-scene → 中位数 +
  scene-bootstrap 95% CI；
- **E3 PASS**（预注册阈值）：ratio 中位数 ≤ 0.25 且 CI 上界 ≤ 0.5
  （= conditioning 匹配后 N 的残解释 ≤ 1/4，"N 曲线是 conditioning 曲线
  的投影"强命题成立）；
- ratio ∈ (0.25, 0.5] = PARTIAL（按任务书分支规则记 FAIL）；> 0.5 = FAIL。

## 5. 三分支裁决（T4′.6，任务书原文，不得改写）

| 分支 | 条件 | 裁决 |
|---|---|---|
| A | E2 + G2 + E3 全 PASS | 锁定 H-COND；允许升级题目/主贡献；进入 P1-13 Go/No-Go |
| B | E2 PASS；G2/E3 任一 FAIL | 只保留 "fixed-N subset quality predicts difficulty"；是否进 P1-13 由 R5 fallback novelty 决定 |
| C | 新确认集 E2 FAIL | 杀掉 H-COND；停止以 conditioning 为理论核心；不准换 proxy/挑 N 救故事 |

Claim 状态更新写入 `CLAIM_REGISTRY.md`（版本号递增），随后更新
`P1_R0_STOP_LINE.md` 附录与 `AGENT_HANDOFF.md`。

## 6. 算力与 canary（纪律 #15）

- 批量 solver 估算：20 scene × 4N × 100 subsets = 8000 runs × 3 restarts，
  批 B=16 → 约 500 个 GPU 调用 ≈ 4–7h（先 1 个调用计时校准）；
- >2 GPU-hour：canary = 首个 scene 的 4N×100 全量跑通 + 全部诊断落盘后
  再继续其余 scene；
- GPU 任务与 CPU Fisher 任务**严格串行**（本机实证：并发导致 CUDA host OOM）。

## 7. 冻结清单（commit 时逐项打钩）

- [x] primary 定义与数值策略（cutoff/spec_cutoff）
- [x] pixel_cap = 1000（commit 配额约束，理由与稳定性对照见 §1）
- [ ] solver 收敛判据数值（待 Discovery pilot 标定 → r4p_conv_thresholds.json）
- [x] solver 选择：串行 joint_solve（batched 1e-3 风险不可接受；标尺 6.5× 速度不值）
- [x] subsets_per_N = 50（算力预算 18×4×50×3×12s ≈ 36h）
- [x] N 集合、subsets 数、seed
- [x] E2/G2/E3 回归式与阈值
- [x] 三分支裁决
