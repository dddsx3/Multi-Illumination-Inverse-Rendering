# R5-P1-A · Albedo Ablation Smoke · Status: PASS-A (smoke, RTX 5070 Ti)

> **Date**: 2026-09-01
> **Stage**: R5-P1-A (smoke budget on RTX 5070 Ti; full pool = Linux H100, awaiting算力)
> **依据**: R5-P1-A task book (本会话批准拍板)
> **脚本**: `p1/source/information_audit/r5_p1_albedo_ablation.py`
> **输出**:
> - `r5/r5_p1_albedo_ablation.csv` — per-(scene, N, subset) raw O + A + structural-null
> - `r5/r5_p1_albedo_ablation_ranking.csv` — per-(scene, N) ranking diagnostics
> - `r5/r5_p1_albedo_ablation_gate.md` — pre-registered gate verdict

## Smoke budget (RTX 5070 Ti)

| 项 | smoke budget | full P1-A budget (Linux H100) |
|---|---|---|
| scenes | 6 (stratified low/med/high from `data_sun_confirmatory/`) | same 6 (or扩展到 10–12 dev scenes) |
| N | {3, 5} | {3, 5} |
| pixel_cap | 400 | 2000 |
| N=3 pool | enumerate first 500 | enumerate all C(32,3)=4960 |
| N=5 pool | sample 500 from C(32,5) | sample 2000 from C(32,5) |
| solver arm | off (--solver not passed) | on, ~360 solver runs |

## Sanity check (preliminary, 2 scenes × 2 N)

```
abs(I_O - I_A) range: [4.14e-11, 8.53e-04]
relative (I_O - I_A)/|I_O|: [4.25e-12, 1.09e-04]
max|I_O - I_A|/std(I_O): [0.000, 0.003]
```

Differences are 4e-11 to 8.5e-4 — orders of magnitude below the score's natural
spread (std ≈ 0.37). Subset ranking is identical to 3 decimal places.

**This is not a numerical artifact**: GSIQ = `mean log(λ̃⁺)` after trace
normalization. F_eff scales as `a_p²` inside both F_ss_diag and the Schur
subtraction, so the **ratio λ/trace(F_eff) is invariant to per-pixel albedo
scaling**. This is a structural property of the metric, not an accident.

## 为什么这个 PASS 是"干净"的

任务书 §4 P1-A Gate 裁决：

| Verdict | criterion | action |
|---|---|---|
| **PASS-A** | median(ρ) ≥ 0.95 AND median(top10) ≥ 0.80 | freeze a=1; 进入 P1-B normal/light proxy |
| CONDITIONAL | 0.80 < median(ρ) < 0.95 | albedo secondary spectral weighting; **不**进入 P1-B with â |
| FAIL-A | median(ρ) ≤ 0.80 | halt practical selector; 进入 albedo-proxy 分支 |

当前 smoke 上 6 scene × 2 N = 12 cells 全部 ρ=1.0 / top10=1.0 ⇒ **PASS-A**。

## 后续 (等你 / 算力到位)

1. **full pool run (Linux H100)**：pixel_cap=2000, N=3 全枚举, N=5 采样 2000。
   主要时间花在 P×P dense eigh（pixel_cap=2000 ⇒ 8e9 flops/subset）。
   估计 ~6 scene × (4960+2000) subsets × 2 score calls × ~0.5s ≈ 1.5 h GPU。
2. **solver arm (Linux H100)**：本机 smoke 阶段未跑 solver（避免与 Task G 撞 commit）。
   full P1-A 上加 360 solver runs (~30 min GPU)。
3. **Task G (Linux H100, 不等 P1-A full)**：240 run local-vs-global。
   准备脚本：`r4pp/r4pp_local_vs_global.py`（已存在）；数据打包：`r4pp/06_controlled_geometry_results.csv`。
4. **正式 PASS-A 判定** = full pool 的 median(ρ) + top10 + 任一 solver arm 反向 sanity。
   当前 smoke PASS-A 是必要但不充分；正式裁决在 H100 全量之后。

## 不变量（per CLAIM_REGISTRY v0.4 字面禁词清单）

- ❌ 仍不写"joint recoverability"
- ❌ 仍不写"noise-floor saturation"
- ❌ 仍不写"M1 uniquely stable"
- ❌ 仍不写"render noise floor"
- PASS-A 来自 *trace normalization 下 GSIQ 对 albedo 的结构无关性*，**不是**
  "albedo 不重要" 的弱 claim。这是 metric 设计层面的数学事实，不允许渲染成更弱 /
  更强的 wording。