"""C19 · 稳健性/ablation 附录（宪法 §4.5 robustness appendix → Fig.9 / Table VI）。

四项 ablation（红队 N5 式 4× Λ misspec 当前只支持单场景二阶效应，禁泛化）：
  1 whitening_vs_raw：白化（异方差 a+bI 权重）vs raw λI（V4 陷阱的正式版）——
    ΔF 读数差与红队 V4 实测 10% 对照；
  2 lambda_misspec：Λ 4× 错设（N5 口径）——弱模式方差预测敏感性（单场景二阶）；
  3 light_count：灯数 N ∈ {6,12,24,48} 对弱模式谱/λ⋆ 的影响（within-scene）；
  4 noise_model：同方差 vs 对角异方差噪声口径下的 retention 谱稳定性（V5 纪律）。

全部 within-scene 读出，只报告效应量与敏感性带（median/IQR），不设胜负判据。
"""

from __future__ import annotations

import numpy as np

from calibinfo.datasets.synthetic import make_scene
from calibinfo.information.gauge import gauge_response
from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import delta_f


def _weak_spectra(sc, lam, use_whitening=True, Sigma_y=None):
    """逐灯 Schur → ΔF 与弱 5 模式 retention 谱（within-scene）。"""
    P = sc["P"]
    q = sc["B_blk"].shape[1]
    per_l = sc["meta"]["n_lights"]
    ss = np.asarray(sc["ss"])                       # (L,P) 工厂返回 list → 数组化
    DF = np.zeros((P, P))
    for k in range(per_l):
        sl = slice(k * P, (k + 1) * P)
        Bk = sc["B_blk"][sl, k * 9:(k + 1) * 9]
        Ak = np.diag(ss[k])
        if use_whitening and Sigma_y is not None:
            wk = 1.0 / np.sqrt(Sigma_y[sl])          # 白化行权
            Ak, Bk = wk[:, None] * Ak, wk[:, None] * Bk
        DFk, _, _ = delta_f(Ak, Bk, lam * np.eye(9))
        DF += DFk
    if Sigma_y is None:
        Finf = np.diag((ss ** 2).sum(0))
    else:
        Finf = np.diag(((1.0 / Sigma_y.reshape(per_l, P)) * ss ** 2).sum(0))
    out = retention_spectrum(DF, Finf)
    return DF, out


def run(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    rows = {k: [] for k in ("whitening_vs_raw", "lambda_misspec",
                            "light_count", "noise_model")}
    for si in range(config.get("n_scenes", 4)):
        sc = make_scene(np.random.default_rng(rng.integers(0, 2**63 - 1)),
                        config.get("P", 300), 3, geometry=config.get("geometry", "bumpy"))
        P = sc["P"]
        m_obs = sc["A_st"].shape[0]
        # 公共噪声口径：异方差 a+bI
        sig = config.get("sigma", 0.02)
        Sigma_y = np.full(m_obs, sig ** 2)
        Sigma_y[:m_obs // 2] *= 4.0                    # 对角异方差（两档）

        # 1 whitening vs raw λI：同一 λ，白化 vs 不白化，弱模式 retention 差
        s_med2 = float(np.median(np.linalg.eigvalsh(
            sc["B_blk"].T @ sc["B_blk"])))
        lam0 = s_med2 / 100
        _, w_out = _weak_spectra(sc, lam0, True, Sigma_y)
        _, r_out = _weak_spectra(sc, lam0, False)
        rows["whitening_vs_raw"].append(dict(
            scene=si, lam=lam0,
            rho_whiten=w_out["rho"][:5].tolist(),
            rho_raw=r_out["rho"][:5].tolist(),
            max_abs_diff=float(np.abs(w_out["rho"][:5] - r_out["rho"][:5]).max())))

        # 2 Λ 4× misspec（N5 单场景二阶口径）
        def tr_weak(lam):
            DF, _ = _weak_spectra(sc, lam, True, Sigma_y)
            ev = np.linalg.eigvalsh(DF)
            pos = ev[ev > 1e-10 * max(ev.max(), 1e-300)]
            return float((1.0 / pos[pos <= np.percentile(pos, 20)]).sum())
        t_true, t_miss = tr_weak(lam0), tr_weak(4 * lam0)
        rows["lambda_misspec"].append(dict(
            scene=si, lam=lam0,
            weak_tr_pinv_true=t_true, weak_tr_pinv_4x=t_miss,
            rel_change_pct=float(100 * (t_miss - t_true) / t_true)))

        # 3 灯数扫描（within-scene：用每灯独立块，取前 N 灯）
        for N in config.get("light_counts", [1, 2, 3]):
            sc_n = make_scene(np.random.default_rng(rng.integers(0, 2**63 - 1)),
                              config.get("P", 300), N, geometry=config.get("geometry", "bumpy"))
            e = np.linalg.eigvalsh(sc_n["B_blk"].T @ sc_n["B_blk"])
            s2 = float(np.median(e[e > 1e-3 * e.max()])) if (e > 1e-3 * e.max()).any() else 1.0
            lam_n = s2 / 100
            DFn, o = _weak_spectra(sc_n, lam_n, True, np.full(sc_n["A_st"].shape[0], sig ** 2))
            resp = gauge_response(sc_n["B_blk"][:sc_n["P"], :9],
                                  sc_n["gauge_cbar"][:9], lam_n)
            evn = np.linalg.eigvalsh(DFn)
            pos = evn[evn > 1e-10 * max(evn.max(), 1e-300)]
            rows["light_count"].append(dict(
                scene=si, n_lights=N, rho5=o["rho"][:5].tolist(),
                weakest_mode=float(pos[0]) if pos.size else float("nan")))

        # 4 噪声模型口径（同方差 vs 异方差白化）——retention 谱稳定性
        _, het = _weak_spectra(sc, lam0, True, Sigma_y)
        _, hom = _weak_spectra(sc, lam0, True, np.full(m_obs, sig ** 2))
        rows["noise_model"].append(dict(
            scene=si, rho_het=het["rho"][:5].tolist(), rho_hom=hom["rho"][:5].tolist(),
            max_abs_diff=float(np.abs(het["rho"][:5] - hom["rho"][:5]).max())))

    # 汇总（效应量 median/IQR，不设胜负判据）
    summary = {}
    wr = rows["whitening_vs_raw"]
    d = [r["max_abs_diff"] for r in wr]
    summary["whitening_vs_raw"] = dict(
        median=float(np.median(d)), iqr=[float(np.percentile(d, 25)),
                                         float(np.percentile(d, 75))], n=len(d))
    lm = rows["lambda_misspec"]
    d = [r["rel_change_pct"] for r in lm]
    summary["lambda_misspec"] = dict(
        median_pct=float(np.median(d)), iqr_pct=[float(np.percentile(d, 25)),
                                                 float(np.percentile(d, 75))], n=len(d))
    lc = {}
    for r in rows["light_count"]:
        lc.setdefault(r["n_lights"], []).append(np.log10(max(r["weakest_mode"], 1e-300)))
    summary["light_count"] = {str(k): dict(median_log10=float(np.median(v)),
                                           n=len(v)) for k, v in lc.items()}
    nm = rows["noise_model"]
    d = [r["max_abs_diff"] for r in nm]
    summary["noise_model"] = dict(
        median=float(np.median(d)), iqr=[float(np.percentile(d, 25)),
                                         float(np.percentile(d, 75))], n=len(d))
    return dict(run_name=config.get("run_name", "ablation"), seed=config["seed"],
                summary=summary, rows=rows,
                note="ablation 全部 within-scene；N5 式 4× misspec 单场景二阶，禁泛化（红队报告二攻击五）")
