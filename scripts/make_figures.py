#!/usr/bin/env python3
"""make_figures · 论文图一键重建（卡 C04 空壳，图 1–9 接口冻结）。

宪法 §11：Figure/Table 只从 artifacts/frozen/ 的机器可读摘要重建；
每张图的唯一 recipe = `python scripts/make_figures.py --figure N`。
C20（图表冻结卡）前逐图实现；当前实现：Fig.1 占位（CI01 双路线误差热图 draft）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts" / "frozen"
FIGS = ROOT / "paper" / "figures"

FIGURES = {f"Fig.{i}" for i in range(1, 10)}


def _make_fig6(out_dir):
    """Fig.6 · 线性化 validity envelope（卡 C13）：mask-flip rate × k 热图，
    颜色 = 理论误差（弱模式 median |emp/pred − 1|）；10% 边界线。"""
    import numpy as np
    src = FROZEN / "ci03nl_nl_formal_summary.json"
    if not src.exists():
        raise SystemExit(f"[make_figures] 缺 {src}——先跑 run_ci.py --experiment ci03nl")
    data = json.loads(src.read_text(encoding="utf-8"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("[make_figures] 需要 matplotlib")
    geos = sorted({r["geometry"] for r in data["checks"]})
    ks = sorted({r["k"] for r in data["checks"]})
    fig, axes = plt.subplots(1, len(geos), figsize=(4.2 * len(geos), 3.4), sharey=True)
    for ax, geo in zip(np.atleast_1d(axes), geos):
        rows = [r for r in data["checks"] if r["geometry"] == geo]
        # 同一 k 可能多场景：取场景中位
        grid_err = np.full((len(ks),), np.nan)
        for i, k in enumerate(ks):
            errs = [r["pred_err_median"] for r in rows if r["k"] == k]
            if errs:
                grid_err[i] = np.median(errs)
        ax.plot(range(len(ks)), grid_err, "o-", color="#1f77b4")
        ax.axhline(data["gate_valid_pred_err"], color="r", ls="--", lw=1,
                   label="10% validity gate")
        env = data["envelope"][geo]
        ax.set_title(f"{geo} | valid flip<={env['max_valid_flip_rate']:.3f} "
                     f"(k<={env['max_valid_k']:g})", fontsize=9)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k:g}" for k in ks], fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("corruption strength k (×Σ_c0)")
    np.atleast_1d(axes)[0].set_ylabel("median |emp/pred − 1| (weak 5 modes)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Fig.6 (draft) — Linearization validity envelope (B-arm analytic SH+ReLU)", fontsize=10)
    out = out_dir / "fig6_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.6 -> {out}")
    return out


def make_figure(n, out_dir=FIGS):
    out_dir.mkdir(parents=True, exist_ok=True)
    if n == 6:
        return _make_fig6(out_dir)
    src = FROZEN / f"ci01_pilot_summary.json"
    if not src.exists():
        raise SystemExit(f"[make_figures] 缺 {src}——先跑 run_ci.py（图表只读 frozen）")
    data = json.loads(src.read_text(encoding="utf-8"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("[make_figures] 需要 matplotlib（pip install matplotlib）")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    cases = [r["case"] for r in data["checks"]]
    errs = [r["rel_err"] for r in data["checks"]]
    ax.bar(range(len(errs)), [max(e, 1e-18) for e in errs], log=True)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=7)
    ax.axhline(1e-10, color="r", ls="--", lw=1, label="1e-10 gate")
    ax.set_ylabel("dual-route rel err (log)")
    ax.set_title(f"Fig.{n} (draft) — CI01 dual-route checks, run {data.get('run_name','pilot')}")
    ax.legend()
    out = out_dir / f"fig{n}_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.{n} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=int, required=True)
    args = ap.parse_args()
    if args.figure not in range(1, 10):
        raise SystemExit(f"--figure 须在 1–9（{FIGURES}）")
    make_figure(args.figure)


if __name__ == "__main__":
    main()
