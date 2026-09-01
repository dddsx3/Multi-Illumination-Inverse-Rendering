# R5-P0 · 收官注记（2026-09-01）

> **状态**：P0 Gate 五项全部完成 ✅
> **下一步**：R5-P1 Oracle→Proxy availability audit（条件 PASS 后才能进 P2）

## 本次 P0 落盘清单

| 项 | 文件 | 状态 |
|---|---|---|
| T0.1 数学文档 v3 | `p1/protocol/IDENTIFIABILITY_v3.md` | ✅ |
| T0.2 structural-null gate | `p1/source/information_audit/gauge_fisher_v2.py::structural_null_gate` + `ga_isi_v2_scores` 集成 | ✅ |
| T0.3 M1 primary 冻结理由 | `IDENTIFIABILITY_v3.md` §6.4 + `CLAIM_REGISTRY.md` §"Primary metric 冻结理由" | ✅ |
| T0.4 术语扫描修订 | `r4pp/08_go_no_go_dashboard.md` / `r4pp/09_R4pp_decision.md` / `r4pp/02_noise_floor_report.md` | ✅ |
| CLAIM_REGISTRY v0.4 | `p1/protocol/CLAIM_REGISTRY.md` | ✅ |

## structural-null gate smoke test（已通过）

toy: P=50, N=3, random normals + random albedo + random SH →

```
P=50, n_dead=1, d_expected=48, d_pos=46, d_extra_null=2, structural_status=deficient
full_logdet_pos_norm = -5.17  (M1 仍可算，但 d_extra_null>0 → 必须与 structural-null 同时报告)
```

random 数据上 d_extra_null=2 是预期的（结构 rank 不到 full 不代表数学错）；
真实 development scene 上 d_extra_null 应主要在边缘 / 低 N / 重复光场景触发。

## 字面禁词（CLAIM_REGISTRY §"字面禁词清单"）

R5-B′ 正文一律改用：

- GSIQ / Gauge-Schur Information Quality（不再称 M1 log pdet；M1 仍可作内部符号）
- selection-leverage compression / subset-sensitivity saturation（不再称 noise-floor saturation）
- solver-repeat noise / repeatability floor（不再称 render noise floor）
- M1 chosen for 5 frozen reasons（不再称 M1 uniquely stable）

## 下一步 R5-P1 启动准备（不立即执行，仅清单）

P1 需要新建脚本：

1. `p1/source/information_audit/r5_oracle_proxy_audit.py`
   - 加载 R4″ development scenes（已存在 r4pp/06_controlled_geometry_results.csv 等可复用）
   - 计算 5 种 score：
     - O = `ga_isi_v2_scores(a_GT, Y_GT, C_GT)`
     - A = `ga_isi_v2_scores(1, Y_GT, C_GT)`
     - N = `ga_isi_v2_scores(1, Ŷ, C_GT)`，Ŷ 来自 coarse normal estimate
     - L = `ga_isi_v2_scores(1, Y_GT, Ĉ)`，Ĉ 来自 lighting estimate
     - P = `ga_isi_v2_scores(â, Ŷ, Ĉ)` 或 `ga_isi_v2_scores(1, Ŷ, Ĉ)`（取决于 A 实验）
   - 输出：(scene, N, subset) × (O, A, N, L, P) 全表 + structural-null gate 全列
2. `p1/source/information_audit/r5_proxy_metrics.py`
   - 计算 within scene + within N 的 `ρ(I_proxy, I_oracle)`
   - Top-decile overlap (oracle top 10% ∩ proxy top 10%)
   - Selection regret：(oracle-selected, proxy-selected, random-selected) 各取 mean error
3. P1 Gate 判定脚本（median ρ ≥ 0.70 + ≥75% scene×N cell proxy>random）

### P1 启动前需要确认的 2 件事（任务书 §4 已有 hedge）

- **proxy P 的具体来源**：A 实验若证 albedo 不重要 → P 用 (1, Ŷ, Ĉ)；否则需加 â估计路径
- **算力环境**：Task G 与 P2/P3 都需要 Linux H100+32GB commit（避免 WinError 1455）；
  HANDOFF §3 已记。本机 RTX 5070 Ti 可跑 N≤3、pixel_cap=2000 的小 P 烟雾测试。

## 与 HANDOFF §4.3 的衔接

T0.4 严格执行了"❌ 换 metric 救 correlation"等禁令：本次**未**改 M1 primary 地位、
**未**改 R_signal 数字、**未**改写 `archive/R4prime_frozen/`、**未**重排 6 行 Gate 数字。

P0 的所有改动都集中在 wording 与报告口径（claim 收紧），不触碰裁决数字。