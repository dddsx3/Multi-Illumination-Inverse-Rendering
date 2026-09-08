"""figure_gens · Fig.3–9 生成器（卡 C20；输入一律 artifacts/frozen，禁 notebook 手工导出）。"""

import json
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "artifacts" / "frozen"
FIGS = Path(__file__).resolve().parents[1] / "paper" / "figures"


def _load(name):
    p = FROZEN / name
    if not p.exists():
        raise SystemExit(f"[make_figures] missing {p}; run corresponding run_ci first")
    return json.loads(p.read_text(encoding="utf-8"))


def fig3(out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci02_formal_summary.json")["fig3"]
    lam = np.array(d["lam_grid"])
    closed = np.array(d["closed"])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(lam, closed, "-", lw=2, label="Prop.2 closed form")
    ax.plot(lam, np.array(d["direct"]), "o", ms=4, mfc="none", label="direct Rayleigh")
    ax.axhline(d["floor"], color="gray", ls=":", label="mu_floor")
    ax.axvline(d["lam_star"], color="r", ls="--", lw=1,
               label="lambda*=%.1e" % d["lam_star"])
    ax.axvline(d["lam_star_pred"], color="g", ls="-.", lw=1,
               label="lambda*_lin=%.1e" % d["lam_star_pred"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("gauge absolute info")
    ax.set_title("Fig.3 - gauge lifting (%s)" % d["scene_id"], fontsize=10)
    ax.legend(fontsize=7)
    out = out_dir / "fig3_gauge_lifting.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.3 ->", out)
    return out


def fig4(out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci02_formal_summary.json")["fig4"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (tag, ttl) in zip(axes[:2], (("n1", "N=1"), ("n3", "N=3"))):
        f = d[tag]
        rho = np.array(f["rho_tracked"])
        im = ax.imshow(rho.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("%s: tracked rho (angle %.1f deg)"
                     % (ttl, f["bottom_subspace_angle_deg"]), fontsize=9)
        ax.set_xlabel("lambda idx (1e-2..1e2 s_med^2)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    f = d["n3"]
    axes[2].plot(f["grid"], f["trace_ratio"], "o-", ms=3)
    axes[2].set_xscale("log")
    axes[2].set_title("trace ratio (near-constant counterexample)", fontsize=9)
    axes[2].set_xlabel("lambda")
    axes[0].set_ylabel("tracked mode id")
    fig.suptitle("Fig.4 - mode-resolved continuum (continuity tracking, no index sort)",
                 fontsize=10)
    out = out_dir / "fig4_modes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.4 ->", out)
    return out


def fig5(out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci03_formal_summary.json")
    rows = d["checks"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax2 = ax.twinx()
    ax.bar(x - 0.2, [r["var_ratio_median"] for r in rows], 0.4,
           label="var ratio (weak 5)")
    ax2.bar(x + 0.2, [r["cov68"] for r in rows], 0.4, color="orange",
            label="coverage 68%")
    ax2.axhline(0.68, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["case"] for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("variance ratio")
    ax2.set_ylabel("coverage 68%")
    ax.set_title("Fig.5 (draft) - finite-sample calibration (CI03 linear MC)",
                fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    out = out_dir / "fig5_gateB.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.5 ->", out)
    return out


def fig7(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci04_formal_summary.json")
    per = {}
    for r in d["rows"]:
        for p, e in zip(r["pred_deg"], r["emp_deg"]):
            per.setdefault(r["object"], []).append((p, e))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for o, pts in per.items():
        p, e = zip(*pts)
        ax.scatter(p, e, s=12, alpha=0.7, label=o)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted degradation")
    ax.set_ylabel("empirical degradation (vs control arm)")
    ax.set_title("Fig.7 - OpenIllumination, Spearman=%.3f [%.2f,%.2f]"
                 % (d["spearman"], d["spearman_ci95"][0], d["spearman_ci95"][1]),
                 fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    out = out_dir / "fig7_ci04.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.7 ->", out)
    return out


def fig8(out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci05_formal_summary.json")
    rows = sorted(d["rows"], key=lambda r: r["lambert_residual"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - 0.2, [r["lambert_residual"] for r in rows], 0.4,
           label="Lambert residual")
    ax.bar(x + 0.2, [100 * r["floor_over_med"] for r in rows], 0.4,
           label="weak-mode floor/med x100")
    ax.set_xticks(x)
    ax.set_xticklabels([r["object"] for r in rows], rotation=30, ha="right",
                       fontsize=7)
    ax.set_title("Fig.8 (draft) - DiLiGenT sanity & failure taxonomy", fontsize=10)
    ax.legend(fontsize=8)
    out = out_dir / "fig8_sanity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.8 ->", out)
    return out


def fig9(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci05abl_ablation_summary.json")["summary"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    s = d["whitening_vs_raw"]
    axes[0].bar(["whiten vs raw"], [s["median"]],
                yerr=[[max(s["median"] - s["iqr"][0], 0)],
                      [max(s["iqr"][1] - s["median"], 0)]])
    axes[0].set_title("whitening vs raw lambdaI", fontsize=9)
    s = d["lambda_misspec"]
    axes[1].bar(["Lambda 4x"], [s["median_pct"]],
                yerr=[[max(s["median_pct"] - s["iqr_pct"][0], 1e-9)],
                      [max(s["iqr_pct"][1] - s["median_pct"], 1e-9)]])
    axes[1].set_title("Lambda 4x misspec (weak-mode trace %)", fontsize=9)
    lc = d["light_count"]
    ks = sorted(lc, key=int)
    axes[2].bar(ks, [lc[k]["median_log10"] for k in ks])
    axes[2].set_title("#lights -> weakest mode (log10)", fontsize=9)
    s = d["noise_model"]
    axes[3].bar(["het vs hom"], [s["median"]],
                yerr=[[max(s["median"] - s["iqr"][0], 0)],
                      [max(s["iqr"][1] - s["median"], 0)]])
    axes[3].set_title("noise model (het vs hom)", fontsize=9)
    fig.suptitle("Fig.9 - ablation / robustness (within-scene effect sizes)",
                 fontsize=10)
    out = out_dir / "fig9_ablation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.9 ->", out)
    return out


def fig2(out_dir):
    """Fig.2 (draft) — theory anatomy：Schur 消元 / ΔF 绝对谱 / R∈[0,1] 三 panel
    （数据：CI01 pilot summary 的双路线 checks + CI02 固定网格 retention）。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load("ci02_formal_summary.json")["fig3"]
    lam = np.array(d["lam_grid"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    # 左: 块信息结构（示意）
    axes[0].imshow([[1, 0.6, 0], [0.6, 1, 0.3], [0, 0.3, 1]], cmap="Blues", vmin=0, vmax=1.4)
    axes[0].set_xticks([0, 1, 2]); axes[0].set_xticklabels(["x", "c", "x-cov"])
    axes[0].set_yticks([0, 1, 2]); axes[0].set_yticklabels(["x", "c", "x-cov"])
    axes[0].set_title("joint info H; Schur over c -> DeltaF", fontsize=9)
    # 中: 绝对信息谱（ΔF 谱随 λ）
    axes[1].plot(lam, np.array(d["closed"]), label="gauge dir (Prop.2)")
    axes[1].axhline(d["floor"], color="gray", ls=":", label="mu_floor")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_title("absolute information spectrum", fontsize=9)
    axes[1].legend(fontsize=7)
    # 右: retention 谱（固定网格，弱 3 模式）
    grid = np.array(d["fixed_log_grid"])
    rho = np.array(d["rho_fixed_grid"])[:, :3]
    for j in range(3):
        axes[2].plot(10.0 ** grid, rho[:, j], label="mode %d" % j)
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("retention spectrum R in [0,1]", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("Fig.2 (draft) - theory anatomy: two readouts, strictly separated",
                 fontsize=10)
    out = out_dir / "fig2_anatomy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[make_figures] Fig.2 ->", out)
    return out
