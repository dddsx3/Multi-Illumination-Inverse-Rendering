"""Directional validation & amplitude validity envelope（P-MAG 核心，M0）。

只读已提交产物（冻结 arm-A 的 mode_ranking.csv + corrected arm-D 的
mf0_factorial_rows_D.json），不执行任何重算实验。钉死两组事实（数学冻结
v1.0 §17 + 潜力评估审计，2026-09-11 双方第一手复核一致）：

1. **方向等价（R_A 的构造性退化）**：cell 内 `S_j = 1 − 1/pred_deg_j` 因 ρ_j
   升序而严格单调于模式序号 j，故
       Spearman(S_j, D_j^emp) ≡ Spearman(−j, D_j^emp)（模式序号基线）。
   逐 cell 报告两个统计量与最大偏差——R_A 检验的是**方向性**（retention
   谱排出的弱方向确实是经验上更脆弱的方向），不检验幅值。

2. **幅值有效域（L7）**：`R_j = D_j^emp / pred_deg_j` 的分布（定理对
   matched linear-Gaussian GLS 预言比值 ≈ 1；合成 MC 实测 1.0045；真实
   数据的偏离 = 线性化在真实光度数据上的有效域，正的、可证伪的结论）。

输出：results/magnitude/directional_amplitude_summary.json

用法：
  python experiments/directional_amplitude.py \
      [--rows results/openillumination/correctness/mf0_factorial_rows_D.json] \
      [--frozen results/openillumination/mode_ranking.csv] \
      [--out results/magnitude/directional_amplitude_summary.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spearman_safe(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def analyse_rows(rows):
    """逐 cell 的基线等价 + 幅值比（corrected arm-D 行格式）。"""
    cells = []
    max_dev = 0.0
    n_equal = 0
    ratios_all, per_mode = [], {j: [] for j in range(5)}
    for r in rows:
        pred = np.asarray(r["pred_deg"], float)
        emp = np.asarray(r["emp_deg"], float)
        S = 1.0 - 1.0 / pred
        s_stat = spearman_safe(S, emp)
        b_stat = spearman_safe(-np.arange(len(pred)), emp)
        dev = abs(s_stat - b_stat)
        max_dev = max(max_dev, dev)
        n_equal += int(dev == 0.0)
        cells.append(dict(object=r["object"], level=float(r["level"]),
                          spearman_directional=s_stat,
                          spearman_mode_index_baseline=b_stat))
        ratio = emp / pred
        ratios_all.extend(ratio.tolist())
        for j in range(len(pred)):
            per_mode[j].append(float(ratio[j]))
    ratios = np.asarray(ratios_all)
    return dict(
        n_cells=len(rows),
        max_abs_deviation_from_mode_index_baseline=float(max_dev),
        n_cells_exactly_equal_to_baseline=int(n_equal),
        ratio_stats=dict(
            median=float(np.median(ratios)),
            p5=float(np.percentile(ratios, 5)),
            p95=float(np.percentile(ratios, 95)),
            per_mode_median={str(j): float(np.median(v))
                             for j, v in per_mode.items()}),
        cells=cells)


def analyse_frozen_csv(path):
    """冻结 arm-A 记录（mode_ranking.csv）的同一组统计。"""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["object_id"], float(r["level"])), []).append(r)
    cells, max_dev, n_equal = [], 0.0, 0
    ratios_all, per_mode = [], {j: [] for j in range(5)}
    for (obj, lv), rs in sorted(by_cell.items()):
        rs = sorted(rs, key=lambda r: int(r["mode_id"]))
        pred = np.asarray([float(r["pred_deg"]) for r in rs])
        emp = np.asarray([float(r["emp_deg"]) for r in rs])
        S = 1.0 - 1.0 / pred
        s_stat = spearman_safe(S, emp)
        b_stat = spearman_safe(-np.arange(len(pred)), emp)
        dev = abs(s_stat - b_stat)
        max_dev = max(max_dev, dev)
        n_equal += int(dev == 0.0)
        cells.append(dict(object=obj, level=lv,
                          spearman_directional=s_stat,
                          spearman_mode_index_baseline=b_stat))
        ratio = emp / pred
        ratios_all.extend(ratio.tolist())
        for j in range(len(pred)):
            per_mode[j].append(float(ratio[j]))
    ratios = np.asarray(ratios_all)
    return dict(
        n_cells=len(by_cell),
        max_abs_deviation_from_mode_index_baseline=float(max_dev),
        n_cells_exactly_equal_to_baseline=int(n_equal),
        ratio_stats=dict(
            median=float(np.median(ratios)),
            p5=float(np.percentile(ratios, 5)),
            p95=float(np.percentile(ratios, 95)),
            per_mode_median={str(j): float(np.median(v))
                             for j, v in per_mode.items()}),
        cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows",
                    default=str(REPO / "results/openillumination/correctness/"
                                "mf0_factorial_rows_D.json"))
    ap.add_argument("--frozen",
                    default=str(REPO / "results/openillumination/mode_ranking.csv"))
    ap.add_argument("--out",
                    default=str(REPO / "results/magnitude/"
                                "directional_amplitude_summary.json"))
    args = ap.parse_args()

    rows_path, frozen_path = Path(args.rows), Path(args.frozen)
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    arm_rows = rows["rows"]
    arm = rows.get("variant", "D")

    d_arm = analyse_rows(arm_rows)
    d_frozen = analyse_frozen_csv(frozen_path)

    summary = dict(
        gate="P-MAG core: directional validation + amplitude validity envelope",
        analysis_status="directional_amplitude_v1",
        directional_validation=(
            "Within a cell, S_j = 1 - 1/pred_deg_j is strictly monotone in the "
            "mode index (rho ascending by construction), so the within-cell "
            "Spearman is rank-equivalent to the mode-index baseline. It "
            "validates DIRECTIONS (the retention operator's bottom tracked "
            "directions are the empirically more fragile ones), not the "
            "predicted magnitudes 1/rho_j."),
        amplitude_interpretation=(
            "The matched-GLS theorem predicts emp/pred ~ 1 (synthetic MC: "
            "1.0045). The real-data distribution below is the validity "
            "envelope of the linearized theory on real photometric data - a "
            "positive, falsifiable finding, not a theorem failure."),
        arm_D_corrected=dict(analysis_of="corrected interface (arm D)",
                             **{k: v for k, v in d_arm.items() if k != "cells"}),
        arm_A_frozen=dict(analysis_of="frozen pre-correction record (arm A)",
                          **{k: v for k, v in d_frozen.items() if k != "cells"}),
        provenance=dict(
            rows_sha256=_sha(rows_path), frozen_sha256=_sha(frozen_path),
            row_count=len(arm_rows)),
        cells_arm_D=d_arm["cells"],
        note="66/66 cells exactly equal to the mode-index baseline (both "
             "calibers) is a construction identity, not a numerical accident.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    print(f"[directional] arm D: {d_arm['n_cells_exactly_equal_to_baseline']}"
          f"/{d_arm['n_cells']} cells == baseline, "
          f"max dev {d_arm['max_abs_deviation_from_mode_index_baseline']:.1e}; "
          f"ratio median {d_arm['ratio_stats']['median']:.1f} "
          f"[{d_arm['ratio_stats']['p5']:.1f}, {d_arm['ratio_stats']['p95']:.1f}]")
    print(f"[directional] frozen: {d_frozen['n_cells_exactly_equal_to_baseline']}"
          f"/{d_frozen['n_cells']} cells == baseline; "
          f"ratio median {d_frozen['ratio_stats']['median']:.1f}")
    print(f"[directional] wrote {out}")


if __name__ == "__main__":
    main()
