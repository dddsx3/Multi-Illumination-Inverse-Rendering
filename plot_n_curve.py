"""N 曲线绘图：双轨（合成 v3 + DiLiGenT）

读取：
- eval_output/n_curve_synth_v3/n_curve_agg.json
- eval_diligent/n_curve/diligent_n_curve.json

输出：
- report_assets/n_curve_synth.png
- report_assets/n_curve_diligent.png
- report_assets/n_curve_combined.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 优先使用支持中文的字体（避免 CJK 字符 fallback 警告）
for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "PingFang SC", "Source Han Sans CN", "Arial Unicode MS"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False


def load_synth():
    p = "eval_output/n_curve_synth_v3/n_curve_agg.json"
    if not os.path.exists(p):
        print(f"[WARN] {p} not found, skip")
        return None
    with open(p) as f:
        return json.load(f)


def load_diligent():
    p = "eval_diligent/n_curve/diligent_n_curve.json"
    if not os.path.exists(p):
        print(f"[WARN] {p} not found, skip")
        return None
    with open(p) as f:
        return json.load(f)


def plot_synth(agg, out):
    if agg is None:
        return
    ns = sorted([int(k) for k in agg.keys()])
    keys = [("image_psnr", "PSNR (dB)", True),
            ("albedo_si_mae", "albedo si-MAE", False),
            ("normal_mae_deg", "normal MAE (°)", False),
            ("depth_rmse", "depth RMSE", False)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (k, label, higher_better) in zip(axes, keys):
        means = [agg[str(N)][k]["mean"] for N in ns]
        stds  = [agg[str(N)][k]["std"] for N in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4,
                    color="C0" if higher_better else "C1",
                    label="synth v3 test (124 scenes × 3 subsets)")
        ax.set_xlabel("# lights N")
        ax.set_ylabel(label)
        ax.set_title(f"N-curve synth: {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("N sensitivity curve — synthetic v3 test set (FusionUNet v2 best)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"OK -> {out}")


def plot_diligent(diligent, out):
    if diligent is None:
        return
    nc = diligent.get("n_curve", {})
    ns = sorted([int(k) for k in nc.keys()])
    keys = [("mae_mean", "MAE (°)", False),
            ("acc_11_25_mean", "acc@11.25°", True)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (k, label, higher_better) in zip(axes, keys):
        means = [nc[str(N)][k] for N in ns]
        stds  = [nc[str(N)].get("mae_std", 0) for N in ns] if k == "mae_mean" else [0] * len(ns)
        ax.errorbar(ns, means, yerr=stds, marker="s", capsize=4,
                    color="C2" if higher_better else "C3",
                    label="DiLiGenT (10 objects × 3 subsets)")
        ax.set_xlabel("# lights N")
        ax.set_ylabel(label)
        ax.set_title(f"N-curve DiLiGenT: {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("N sensitivity curve — DiLiGenT (zero-shot)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"OK -> {out}")


def plot_combined(agg_synth, agg_diligent, out):
    if agg_synth is None and agg_diligent is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左图：合成 v3 normal_mae
    if agg_synth is not None:
        ns = sorted([int(k) for k in agg_synth.keys()])
        means = [agg_synth[str(N)]["normal_mae_deg"]["mean"] for N in ns]
        stds  = [agg_synth[str(N)]["normal_mae_deg"]["std"] for N in ns]
        axes[0].errorbar(ns, means, yerr=stds, marker="o", capsize=4,
                         color="C0", label="synth v3 test (124 scenes × 3 subsets)")
    # 右图：DiLiGenT MAE
    if agg_diligent is not None:
        nc = agg_diligent.get("n_curve", {})
        ns_d = sorted([int(k) for k in nc.keys()])
        means_d = [nc[str(N)]["mae_mean"] for N in ns_d]
        stds_d  = [nc[str(N)].get("mae_std", 0) for N in ns_d]
        axes[1].errorbar(ns_d, means_d, yerr=stds_d, marker="s", capsize=4,
                         color="C2", label="DiLiGenT zero-shot (10 objects × 3 subsets)")
    axes[0].set_xlabel("# lights N")
    axes[0].set_ylabel("normal MAE (°)")
    axes[0].set_title("N-curve synth v3 (FusionUNet v2 best)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("# lights N")
    axes[1].set_ylabel("normal MAE (°)")
    axes[1].set_title("N-curve DiLiGenT (zero-shot)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle("N sensitivity curves — Phase 2 T2.5 dual-track protocol (post INC-0012)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"OK -> {out}")


def main():
    os.makedirs("report_assets", exist_ok=True)
    agg_synth = load_synth()
    agg_diligent = load_diligent()
    if agg_synth is None and agg_diligent is None:
        print("[FAIL] no input found")
        sys.exit(1)
    plot_synth(agg_synth, "report_assets/n_curve_synth.png")
    plot_diligent(agg_diligent, "report_assets/n_curve_diligent.png")
    plot_combined(agg_synth, agg_diligent, "report_assets/n_curve_combined.png")


if __name__ == "__main__":
    main()
