# R4PP_HANDOFF · R4″ sprint 继任者 handoff

> **写给下一个 agent**（论文发表方向 R4prime/R4″ 阶段的接手者）
> **目的**：任何后续 agent 打开仓库后能在 **15 分钟内**建立完整上下文，**不重蹈本会话 10 个 commit 链里任何已识别的事故**
> **基线 commit**：`9796884`（R4″ sprint 收官点；本文件同一 commit 落盘）
> **对应任务书**：R4″ 项目任务书 V1.0（Scene–Illumination Observability 全新裁决版）§0–§46
> **阅读顺序**：§0–§1（30 秒定位）→ §2（裁决）→ §3（环境与节奏）→ §4（重启清单）

---

## 0. 30 秒现状

**R3′ 数学封口通过**（commit `236f895`）+ **R4″ 6 行 Gate dashboard 4/6 PASS**（commit `9796884`）→ 终裁 **PIVOT (B′)**。

- 原 H-COND（"光照子集质量决定误差"）**主效应在弱化版上成立**（Direction PASS，β median −0.348, 81% 负号）
- **Geometry × Information interaction 未观察到稳定趋势**（Interaction FAIL，family A N=3/5 ρ 反向）
- **M1 log pdet 是稳健信息度量**（Instrument PASS，5/5 stability）
- **N=8 是真 saturation 不是噪声**（Saturation PASS，σ_subset/err 3.2%, R_signal 22.6）

**PIVOT 后的新方向**（任务书 §28 B′ 主线）：
> *Beyond Cardinality: Effective Information at Fixed Illumination Budget*

---

## 1. 关键裁决（CLAIM_REGISTRY v0.3 已冻结）

`p1/protocol/CLAIM_REGISTRY.md` v0.3（6 行 Gate 表述 + 裁决映射）

| Gate | 状态 | 证据文件 |
|---|---|---|
| Instrument | ✅ PASS | `r4pp/03_metric_stability.csv`（M1 5/5 PASS）|
| Signal | ✅ PASS | `r4pp/02_noise_floor_summary.csv`（R_signal 全 24 cell > 2）|
| Direction | ✅ PASS | `r4pp/06_beta_per_geometry.csv`（β med=-0.348, 81% 负）|
| **Interaction** | ❌ **FAIL** | family A N=3 ρ=+0.29, N=5 ρ=+0.72（**反向**）|
| Saturation | ✅ PASS | `r4pp/02_noise_floor_summary.csv`（N=8 R_signal 22.6）|
| Externality | ⚪ PENDING | Task G 因本机 OOM 未执行（详见 §3）|

裁决：**PIVOT (B′)** —— 论文方向收缩到"fixed-budget effective information"，**放弃"illumination set 决定一切"的大统一理论**。

---

## 2. 已识别的 8 个 R4prime 时代事故（**绝对不可重演**）

> 完整证据：`archive/R4prime_frozen/R4prime_failure_audit.md`（P0-1 ~ P3-7 七行表）

| # | 事故 | 后果 | 修复 |
|---|---|---|---|
| P0-1 | `λ_min⁺` 退化（90 维极值统计量在 P=1000 下贴数值地板） | primary metric invalid | R3′ v2 改 M1 log pdet（5/5 stability）|
| P0-2 | N=8 动态范围坍塌但未测噪声 | saturation 误判为 correlation 弱 | Task C 480 run 实测 R_signal |
| P0-3 | 收敛判据 P75×P75 内生 | 0.75×0.75=0.5625 vs 实测 0.584 | `CONV_CRITERIA_FROZEN.json`（abs_step200）|
| P1-4 | G2 primary 跨 4.9 个数量级 | 线性回归 ΔR²=-54 | G2 必须对 primary 取对数 |
| P1-5 | E3 十分位 × 8 subset 恒 INSUFFICIENT | 端点从未评估 | 永久弃 binning，模型比较 |
| P2-6 | 5 项功效连续削减 | 统计功效不足 | 现已扩 N={3,5,8}×30 = 1620 trial |
| P3-7 | geometry × information 调制 | 新科学发现 | 写 R4prime_failure_audit 永久标记 |
| **P0-3 新** | ρ(grad_norm, err) = -0.471 (91% cell 为负) | 旧判据**反向**选择 | pgn 仅诊断，禁作筛选 |

**H-COND 状态**：保留 hypothesis（弱化版 PASS）；未杀。

---

## 3. 环境与硬约束（接 R4PP_EXECUTION_MANUAL §5 关键约束）

| 资源 | 规格 | 风险点 |
|---|---|---|
| GPU | RTX 5070 Ti 12GB（driver 591.86）| 显存充足；commit 配额常满 |
| RAM | 32GB（Windows commit 配额 ~94% 经常占用）| 8MB 以上分配经常失败 |
| Python | 3.14.2（torch 2.12.0.dev+cu128）/ 3.10（仅 blenderproc 2.8.0）| torch 加载时 `WinError 1455` 反复出现 |
| BlenderProc | Blender 4.2.1 LTS（任务书红线：每场景独立进程）| INC-001 帧级校验已嵌入 |
| GitHub | `dddsx3/Multi-Illumination-Inverse-Rendering` | 网络不稳，commit 后 push 常需多次重试 |

**两个必须先做判断的硬约束**：
1. **本机 WinError 1455**：Task G（local-vs-global）跑不了（torch 加载失败）。换算力环境（≥32GB commit + ≥4GB 页面文件）即可补，~6h GPU。
2. **GPU 与重 CPU Fisher 不得并发**：P=2000 dense 频繁 OOM。Task D 任务的 M-B cap2000 必须 GPU 空闲窗口。

**Git tag**：`r4prime-frozen`（旧数据冻结点，不可改写）。`9796884`（R4″ sprint 收官点）。

---

## 4. 重启清单（按优先级，**前 3 项不算后不能动**）

### 4.1 必做的前 3 项
1. **重跑 Task G**（`r4pp_local_vs_global.py`）以确认 Externality verdict。在 H100+32GB commit 环境下，~6h GPU。脚本已就绪，输出 `07_local_vs_global_init.csv`。
2. **写 PIVOT B′ 方向论文初稿**（CVPR/ICCV/NeurIPS，~8 页正文）。Figure 1+5（noise+标定）+ 2+3+4（数据）。骨架与 Figure 排序见 `EXPERT_PUBLICATION_BRIEF_V2.md`。
3. **R5 文献封口**（任务书 §5 强制）：逐篇原文级核实 IDArb / LINO / GeoUniPS / ReLeaPS 2025–2026 新工作，更新 `p1/literature/RELATED_WORK_MATRIX_v3`。

### 4.2 可选做（按价值）
4. 补 3 个 B 族场景（bevel30 / rounded）— Task F 已被本机 OOM 阻止；换算力可补 ~1.5h 渲染 + 0.5h solver
5. §3 noise-floor 完整：渲染 3 realization 估计 σ_render（已留 RENDER_REALIZATION=0 入口）
6. 论文 Figure 5（定性重建 Case A/B/C）—— 数据已生成

### 4.3 **明确禁止**（任务书 §43）
- ❌ 换 metric 救 correlation
- ❌ 调 rank cutoff 直到显著
- ❌ 只报告 converged trials
- ❌ 删 N=8 不测 noise
- ❌ 把 rank_Fk 当 nuisance covariate 塞进去
- ❌ 旧 18 scene 重挑假设后称 confirmatory
- ❌ 用更多随机场景替代 controlled mechanism
- ❌ 用 p-value 决定 pilot 是否存在 signal
- ❌ 因 sunk cost 降低 KILL 门槛
- ❌ **任何对 `archive/R4prime_frozen/` 的改写**（冻结）

---

## 5. 关键文件路径索引（按接手后使用频率）

### 5.1 必读（先看 5 个）
1. `p1/protocol/CLAIM_REGISTRY.md` — 三句话契约 + v0.3 裁决
2. `p1/information_audit/09_R4pp_decision.md` — PIVOT 终裁详细
3. `p1/information_audit/08_go_no_go_dashboard.md` — 6 行 Gate 表
4. `archive/R4prime_frozen/R4prime_failure_audit.md` — 旧数据失效清单
5. `p1/protocol/R4PP_EXECUTION_MANUAL.md` — 48h sprint 逐小时表

### 5.2 论文级证据（GitHub 是真相来源）
| 文件 | 内容 |
|---|---|
| `p1/source/information_audit/gauge_fisher_v2.py` | M1 实现（full-Schur）|
| `p1/protocol/IDENTIFIABILITY_v2.md` | 数学公式 |
| `r4pp/03_metric_stability.csv` | 5 metric × 6 stability test |
| `r4pp/02_noise_floor_summary.csv` | 24 cell × R_signal |
| `r4pp/06_beta_per_geometry.csv` | β_G vs G（Gate 4 数据）|
| `r4pp/09_R4pp_decision.md` | 终裁报告 |
| `r4pp/04_geometry_spectrum.csv` | 28 scene normal/SH Gram |
| `r4pp/06_controlled_geometry_results.csv` | 400 run solver raw |
| `r4pp/traces/*.npy` | 480 run loss trace（每 trial 一个）|
| `r4pp/01_master_trial_table.parquet` | 988 trial × 91 列 master |

### 5.3 给外部顾问（仓库外）
- `D:\MIR_Archive_20260829\EXPERT_PUBLICATION_BRIEF_V2.md` — 论文方向 V2 简报
- `D:\MIR_Archive_20260829\EXPERT_PAPER_POTENTIAL_BRIEF.md` — V0 简报（**已过期**）
- `D:\MIR_Archive_20260829\EXPERT_BRIEFING.md` — 学术准确性层简报
- `D:\MIR_Archive_20260829\R4PP_PAPER_MIN_006_FILES_9796884.zip` — 6 文件最小交付包（21 KB）

### 5.4 冻结归档（不可改）
- `archive/R4prime_frozen/` — 30 文件只读 + sha256 校验（`MANIFEST.csv`）
- 重建需显式 `--force`：`python p1/source/information_audit/r4pp_freeze_archive.py --force`
- git tag `r4prime-frozen` 标记冻结点

### 5.5 不在仓库的
- `D:\MIR_Archive_20260829\` 根目录下的 EXPERT_*.md 与 R4PP_*.zip — 跨项目咨询与交付物，单独留存

---

## 6. 实施细节备忘（避免重蹈本会话 10 个 commit 的微工程坑）

### 6.1 代码修改
- **`joint_solve` 新签名**（C-1，commit `73b3e53`）：`seed=None, theta0=None, return_trace=False, tail_k=50`，所有默认保持旧行为（逐位复现冻结 988 trial）
- **三个新绝对收敛原始量**：`proj_grad_norm`（gauge 投影 + RMS 归一）、`tail_rel_change`、`conv_finite`；pgn 仅诊断不作筛选（实测 ρ(pgn, err)=-0.748，仍反向）
- **INC-001 渲染器修复**（commit `1aa45f3`）：ratio 下界 0.15→0.02 + 真丢光判据（fg≈0∧bg≈0）
- **noise_floor lazy import**：`config/summarize` 纯 numpy，`run` 才加载 torch（绕开本机 WinError 1455）

### 6.2 数据陷阱
- **trials.csv 旧 filter 是分析时施加**——**采集未丢数据**，Master table 不需要"恢复"
- **plan 必须晚于渲染完成**——Task F 教训：build_plan 冻结 6 scene 早于补渲的 A_prism4/B_bevel15，重建 plan + 断点续跑才能覆盖完整集合
- **geometry metric 在 controlled 上用 G3_effrank 不要 G1_rank**——rank 只有 5/6/9 三档，effrank 连续

### 6.3 统计精度提示
- E2 per-cell Spearman 中位数 0.25（v0.2 旧数据）—— 主效应成立但功效差
- bootstrap B=10000 在 24 cell 上 ~10s 可完成，**不要用更少**
- scene-bootstrap 25（v0.2 旧场景数）做 10000 重采样统计功效不足

---

## 7. 6 文件最小交付包

按论文投稿所需，6 个最关键文件打包在：
`D:\MIR_Archive_20260829\R4PP_PAPER_MIN_006_FILES_9796884.zip`（21 KB）

含 sha256 MANIFEST + 上述 6 个核心文件。**论文投稿时把这一个 zip 附进 supplementary 即可**。完整数据在 GitHub，顾问想看 480 run 的 loss trace 等可以在仓库里取。

---

## 8. 签发

本 handoff 已冻结，与 commit `9796884` 同 commit 落盘。下次 sprint 开始时：

1. 读本文件 → §0-§4（15 分钟定位）
2. 跑 §4.1 三项（重 Task G、写 PIVOT 论文、R5 文献）
3. 任何对 R4prime 时代的引用 → 查 `archive/R4prime_frozen/R4prime_failure_audit.md` 确认该数字是否仍在

下一个 agent 接手后的 30 分钟内应能完成：Task G 启动 + 论文 §3-§4 草稿（用本 handoff 索引的数据）。

---

*写作者：ZCode agent · 2026-08-31 · 基线 `9796884` · 配套 `EXPERT_PUBLICATION_BRIEF_V2.md`*
*本文件不含任何科学裁决的修改——所有裁决以 `p1/protocol/CLAIM_REGISTRY.md` v0.3 为准*
