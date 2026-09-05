#!/usr/bin/env python3
"""关键实验组 · 图表生成（数值 → PNG，供视觉检查与论文/汇报引用）

按 WORKSTREAM_PROTOCOL 的接口约定, 每张图的数据来自 critical_experiments/*.json。
输出: critical_experiments/figures/*.png (150 dpi)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def fig_exp4_spectral():
    """exp4 · 谱层次 log-log: x=log10(σ²/λ_k), y=log10 Var(u_kᵀe), 斜率参考线。"""
    d = json.loads((HERE / "exp4_crb_validity.json").read_text(encoding="utf-8"))
    scenes = list(d["scenes"].keys())
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.2 * len(scenes), 3.0),
                             sharey=False)
    for ax, scene in zip(np.atleast_1d(axes), scenes):
        r = d["scenes"][scene]["results"]["0.01"]   # 中间噪声水平为代表
        x = np.array(r["loglog_x"]); y = np.array(r["loglog_y"])
        sel = np.array(r["loglog_sel"], dtype=bool)
        ax.scatter(x[sel], y[sel], s=8, alpha=0.6, color="#1f77b4",
                   label="可估带 (λ>1e-2λmax)")
        bad = ~sel
        if bad.any():
            ax.scatter(x[bad], y[bad], s=10, alpha=0.6, color="#d62728",
                       marker="x", label="收缩带")
        # 斜率=1 参考线（过可估带中位点）
        if sel.any():
            x0 = np.median(x[sel]); y0 = np.median(y[sel])
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, y0 + (xs - x0), "--", color="gray", lw=1,
                    label="斜率=1 参考")
        ax.set_title(f"{scene}  (斜率={r['spectral_slope']:.2f}, "
                     f"中位比={r['param_ratio_median']:.2f})")
        ax.set_xlabel(r"$\log_{10}(\sigma^2/\lambda_k)$ = CRB 对角")
        ax.set_ylabel(r"$\log_{10}\,\mathrm{Var}(u_k^\top e)$")
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(r"实验4 · CRB 谱层次有效性: $\mathrm{Var}(u_k^{T} e)$ vs $\sigma^2/\lambda_k$ (log-log, 噪声=1%峰值, 200 trials/格)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "exp4_spectral_crb.png")
    plt.close(fig)


def fig_exp7_crb_vs_n():
    """exp7 · 双轴: 左网络 N-curve(MAE), 右 log 理论 E_min 相对倍率。"""
    d = json.loads((HERE / "exp7_crb_vs_n.json").read_text(encoding="utf-8"))
    Ns = np.arange(1, 6)
    nc = d["network_ncurve"]
    mae = [nc[str(N)]["normal_mae_deg"]["mean"] for N in Ns]
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(Ns, mae, "o-", color="#1f77b4", lw=2, label="网络实测 normal MAE (EX-01)")
    ax1.set_xlabel("光照数 N")
    ax1.set_ylabel("网络 normal MAE (°)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(14.85, 14.92)
    ax2 = ax1.twinx()
    colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for r, c in zip(d["crb_curve"], colors):
        if "E_min_rel_to_N2" not in r:
            continue
        rel = [r["E_min_rel_to_N2"][str(N)] for N in Ns]
        ax2.plot(Ns, rel, "s--", color=c, alpha=0.75, lw=1.4,
                 label=f"理论 E_min 相对 ({r['scene']})")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"理论 $E_{\min}(N)/E_{\min}(2)$ (log)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(Ns)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper right", framealpha=0.9)
    ax1.set_title("实验7 · 理论 CRB 带(N1→N5 降 52–78%) vs 网络 N-curve(平坦 0.017°)\n"
                  "——信息在数据里, 不在网络用法里")
    fig.tight_layout()
    fig.savefig(FIG / "exp7_crb_vs_n.png")
    plt.close(fig)


def fig_exp5_dispersion():
    """exp5+5b · 散点: 全局 λ_min⁺ 与局部 λ_loc vs σ_min²。"""
    d5 = json.loads((HERE / "exp5_dispersion_corrected.json").read_text(encoding="utf-8"))
    d5b = json.loads((HERE / "exp5b_local_normal_fisher.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ok = [r for r in d5["runs"] if "error" not in r]
    scenes = sorted({r["scene"] for r in ok})
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(scenes):
        xs = [r["sigma_min_sq"] for r in ok if r["scene"] == s]
        ys = [r["lam_min_pos"] for r in ok if r["scene"] == s]
        axes[0].scatter(xs, ys, s=18, alpha=0.8, color=cmap(i), label=s)
    axes[0].set_xlabel(r"$\sigma_{\min}(C)^2$"); axes[0].set_ylabel(r"$\lambda_{\min}^+$ (几何块)")
    axes[0].set_title(f"全局口径: Spearman={d5['spearman_all']['rho']:.2f} (p="
                      f"{d5['spearman_all']['p']:.2f})\n低秩扰动保护(秩≤45)")
    axes[0].legend(fontsize=6)
    okb = d5b["runs"]
    for i, s in enumerate(scenes):
        xs = [r["sigma_min_sq"] for r in okb if r["scene"] == s]
        ys = [r["lam_min_mean"] for r in okb if r["scene"] == s]
        axes[1].scatter(xs, ys, s=18, alpha=0.8, color=cmap(i), label=s)
    axes[1].set_xlabel(r"$\sigma_{\min}(C)^2$")
    axes[1].set_ylabel(r"$\overline{\lambda_{\min}}(F_p)$ 局部法线")
    axes[1].set_title(f"逐像素口径: Spearman={d5b['spearman_all']['rho']:.2f} (p="
                      f"{d5b['spearman_all']['p']:.2f})\n三口径不一致 → σ_min 预测子命题不成立")
    fig.suptitle("实验5/5b · 修正散布度检验: 经典预期(σ_min→几何信息)在本 SH-2 模型不成立",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "exp5_dispersion.png")
    plt.close(fig)


def fig_exp3_decomposition():
    """exp3 · 近零维数分解堆叠柱(64 级)。"""
    d = json.loads((HERE / "exp3_direction_separation.json").read_text(encoding="utf-8"))
    scenes = ["sphere", "cube", "cylinder", "hemisphere"]
    gauge, sh_edge, gbr, measured = [], [], [], []
    for scene in scenes:
        r = next(x for x in d["runs"] if x["scene"] == scene and x["res"] == 64)
        m = r["measured_near0"]; N = r["N"]
        e = r.get("n_gy_edge_band_1e8_1e5", 0)
        g = 1
        sh_n = N * e
        gb = m - g - sh_n
        gauge.append(g); sh_edge.append(sh_n); gbr.append(max(gb, 0)); measured.append(m)
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x, gauge, 0.55, label="全局尺度 gauge(1)", color="#1f77b4")
    ax.bar(x, sh_edge, 0.55, bottom=gauge, label="SH 边缘病态 ×N", color="#ff7f0e")
    ax.bar(x, gbr, 0.55, bottom=np.array(gauge) + np.array(sh_edge),
           label="GBR(可见维)", color="#2ca02c")
    ax.scatter(x, measured, marker="_", s=300, color="black", lw=2,
               label="实测近零数")
    for i, m in enumerate(measured):
        ax.text(i, m + 0.4, str(m), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(scenes)
    ax.set_ylabel("近零特征值个数 (阈值 1e-6 相对)")
    ax.set_title("实验3 · S 谱近零维数三类解析分解\n(cube/cylinder 1+10+3 完美闭合, 其余部分维被截断)")

    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "exp3_decomposition.png")
    plt.close(fig)


def fig_exp6_forward():
    """exp6 · 复制/打乱漂移分布 + 敏感度份额。"""
    d = json.loads((HERE / "exp6_network_forward_tests.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    rep = np.array([r["output_drift_deg"] for r in d["replicate_test"]["per_scene"]])
    shf = np.array([r["output_drift_deg"] for r in d["shuffle_test"]["per_scene"]])
    axes[0].hist(rep, bins=30, color="#1f77b4", alpha=0.8)
    axes[0].set_title(f"复制测试输出漂移 (median {np.median(rep):.3f}°)\n——对光照多样性近零响应")
    axes[0].set_xlabel("漂移 (°)"); axes[0].set_ylabel("场景数")
    axes[1].hist(shf, bins=30, color="#ff7f0e", alpha=0.8)
    axes[1].set_title(f"打乱测试输出漂移 (median {np.median(shf):.2f}°)\n——对外来内容敏感(分布偏移响应)")
    axes[1].set_xlabel("漂移 (°)")
    sens = d["sensitivity_test"]["per_scene"]
    pl = np.array([r["grad_norm_per_light"] for r in sens])          # (S,5) 范数
    shares = pl**2 / np.maximum((pl**2).sum(1, keepdims=True), 1e-30)  # 能量份额
    bp = axes[2].boxplot([shares[:, k] for k in range(5)], tick_labels=[f"光{k+1}" for k in range(5)],
                         showfliers=False)
    axes[2].axhline(0.2, ls="--", color="gray", lw=1, label="均匀基准 0.20")
    axes[2].set_ylim(0.19, 0.21)
    axes[2].set_title("敏感度: 单图梯度能量份额\n——场景内完全均匀(无偏图)")
    axes[2].set_ylabel("份额"); axes[2].legend(fontsize=7)
    fig.suptitle("实验6 · 网络三前向测试 (A3-0, 124 场景, 零训练)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "exp6_forward_tests.png")
    plt.close(fig)


def fig_exp2_spectrum():
    """exp2 · S 谱(相对 λmax, log10) 4 校准场景 64 级。"""
    d = json.loads((HERE / "exp2_joint_fisher_spectrum.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    scenes = ["sphere", "cube", "cylinder", "hemisphere"]
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(scenes):
        r = next(x for x in d["runs"] if x.get("scene", "").startswith(s)
                 and x.get("res") == 64)
        e = np.array(r["eig_rel"])          # 升序全 45 个
        ax.plot(np.arange(1, len(e) + 1), np.log10(np.maximum(e, 1e-16)),
                "o-", ms=3, lw=1.2, color=cmap(i), label=s)
    ax.axhline(-6, ls=":", color="gray", lw=1)
    ax.text(38, -5.8, "near0 阈值 1e-6", fontsize=7, color="gray")
    ax.set_xlabel("特征值序号 (升序, 9N=45)")
    ax.set_ylabel(r"$\log_{10}(\lambda/\lambda_{\max})$")
    ax.set_title("实验2 · 光照 Schur 补 S 谱 (64×64, N=5)\n"
                 "底部: 全局 gauge(≈1e-9~1e-11) + SH 边缘病态带 + GBR")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "exp2_spectrum.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_exp4_spectral()
    fig_exp7_crb_vs_n()
    fig_exp5_dispersion()
    fig_exp3_decomposition()
    fig_exp6_forward()
    fig_exp2_spectrum()
    print("figures ->", FIG)
    for f in sorted(FIG.glob("*.png")):
        print(" ", f.name)
