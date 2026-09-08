"""CI03 · Monte-Carlo 紧性与线性化有效域（宪法 §4.3，卡 C12 线性 oracle 部分）。

两臂（R4 修订 + 宪法 §0 禁令）：
  A. joint ensemble（δc 与 ε 联合抽样）→ Cov(x̂) = σ²ΔF⁻¹（Prop 1 / Gate B 主判据）
     + 最弱 5 模式（预测协方差特征基，mode-resolved）的 68%/95% coverage；
  B. fixed-δc 对照诊断 → bias=ΔF⁻¹AᵀMBδc 与条件方差 σ²ΔF⁻¹AᵀM²AΔF⁻¹
     （不用于 tightness 声明；方向性结论：条件方差 < marginal）。

可测性纪律（跑前写死，gate_bias_snr_min）：bias 检查只在
SNR = ‖Fh·bias‖ / sqrt(tr(C_pred_w)/trials) > 10 时判定——Λ 小的随机块场景
bias 低于 MC 均值噪声地板，比率是噪声（首轮 4.5×/69× 假象的根因），
此时如实记 not-measurable，不做 gate 判定。

实现红线（本卡首轮事故，如实入 memo）：系综循环必须用 trials_c（per-case 扩样），
误用全局 trials 会把未初始化的 np.empty 行混进经验协方差（0.40 紧 IQR 假象）。
"""

from __future__ import annotations

import numpy as np

from calibinfo.datasets.synthetic import make_scene
from calibinfo.information.schur import delta_f


def run(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    gate_median = config["gate_variance_ratio_median"]
    gate_iqr = config["gate_variance_ratio_iqr"]
    gate_cov = config["gate_coverage_abs_tol"]
    gate_bias = config["gate_bias_ratio"]
    gate_condvar = config["gate_condvar_ratio_median"]
    gate_bias_snr = config["gate_bias_snr_min"]
    trials = config["trials"]

    rows = []
    for ci, case in enumerate(config["cases"]):
        for si in range(config.get("n_scenes_per_case", 3)):
            s_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + ci * 100 + si)
            P = case["P"]
            if case.get("structure", "random") == "photometric":
                sc = make_scene(s_rng, P, case["n_lights"],
                                geometry=case.get("geometry", "sphere"),
                                albedo=case.get("albedo", "medium"))
                A_st, B_blk = sc["A_st"], sc["B_blk"]
                n, q = A_st.shape[1], B_blk.shape[1]
                a_true = sc["gauge_a"]
            else:
                n, q = case["n"], case["q"]
                A_st = s_rng.normal(size=(P, n))
                B_blk = s_rng.normal(size=(P, q))
                a_true = s_rng.normal(size=n)
            sigma = case["sigma"]
            sig_c = case["sig_c"]
            m_obs = A_st.shape[0]                     # photometric: N·P'(kept)，random: P
            trials_c = int(case.get("trials", trials))

            Lam = (sigma / sig_c) ** 2 * np.eye(q)
            DF, M, diag = delta_f(A_st, B_blk, Lam)
            DFinv = np.linalg.inv(DF)                 # 联合先验下可逆（gauge 已 lift）
            G = B_blk.T @ B_blk + Lam
            Ginv_Bt = np.linalg.lstsq(G, B_blk.T, rcond=None)[0]

            # 分析基：F∞^{-1/2} 白化（range 限制）+ 预测协方差特征基（mode-resolved）
            Finf = A_st.T @ A_st
            wv, Vv = np.linalg.eigh(Finf)
            kp = wv > 1e-12 * max(wv.max(), 1e-300)
            Fh = Vv[:, kp] @ np.diag(1 / np.sqrt(wv[kp])) @ Vv[:, kp].T
            C_pred_w = Fh @ (sigma ** 2 * DFinv) @ Fh          # 白化空间预测协方差
            ev_c, V_c = np.linalg.eigh(C_pred_w)
            weak = np.argsort(ev_c)[-5:]                       # 最弱 5 模式（最大方差）
            Vw = V_c[:, weak]                                  # (n, 5) 模式向量（确定性）
            pred_var = ev_c[weak]
            scores = (Fh.T @ Vw)                                # x̂ → 白化模式得分算子 (n,5)

            # ---- A: joint ensemble（trials_c：per-case 扩样，禁用全局 trials）----
            E = np.empty((trials_c, n))
            for t in range(trials_c):
                dc = s_rng.normal(0, sig_c, size=q)
                eps = s_rng.normal(0, sigma, size=m_obs)
                y = A_st @ a_true + B_blk @ dc + eps
                My = y - B_blk @ (Ginv_Bt @ y)
                E[t] = DFinv @ (A_st.T @ My) - a_true
            Sc = E @ scores                                     # (trials, 5) 模式得分
            emp_var = Sc.var(axis=0, ddof=1)
            ratio = emp_var / pred_var
            cov = {}
            for lvl, zz in {68: 1.0, 95: 1.96}.items():
                cov[lvl] = float(np.median(
                    (np.abs(Sc) <= zz * np.sqrt(pred_var)).mean(axis=0)))

            # ---- B: fixed-δc 诊断（trials_c）----
            dc_fix = s_rng.normal(0, sig_c, size=q)
            bias = DFinv @ A_st.T @ M @ (B_blk @ dc_fix)
            bias_w = Fh @ bias
            cond_pred_w = Fh @ (sigma ** 2 * DFinv @ (A_st.T @ (M @ M) @ A_st) @ DFinv) @ Fh
            mean_noise_floor = float(np.sqrt(np.trace(cond_pred_w) / trials_c))
            bias_snr = float(np.linalg.norm(bias_w) / mean_noise_floor)
            Eb = np.empty((trials_c, n))
            for t in range(trials_c):
                y = A_st @ a_true + B_blk @ dc_fix \
                    + s_rng.normal(0, sigma, size=m_obs)
                My = y - B_blk @ (Ginv_Bt @ y)
                Eb[t] = DFinv @ (A_st.T @ My) - a_true
            Scb = Eb @ scores
            # 条件方差预测 = 同一模式向量上的二次型（R4 闭式 σ²ΔF⁻¹AᵀM²AΔF⁻¹）
            Vw5 = V_c[:, weak]
            cond_pred_var = np.diag(Vw5.T @ cond_pred_w @ Vw5)
            cond_ratio = Scb.var(axis=0, ddof=1) / cond_pred_var

            bias_measurable = bool(bias_snr > gate_bias_snr)
            bias_ratio = float(np.linalg.norm(Eb.mean(0)) / np.linalg.norm(bias))
            marg_res = float(np.linalg.norm(E.mean(0)) / np.linalg.norm(bias))
            # 方向性：条件方差 < marginal（M 收缩映射）——可测性无关，恒成立
            cond_lt_marg = bool(np.median(cond_ratio) < np.median(ratio))

            # cond_lt_marginal 为方向性诊断（M 收缩映射由数学保证，MC 噪声内不 gate）
            gate_pass = bool(
                gate_median[0] < np.median(ratio) < gate_median[1]
                and np.percentile(ratio, 25) > gate_iqr[0]
                and np.percentile(ratio, 75) < gate_iqr[1]
                and abs(cov[68] - 0.68) < gate_cov
                and abs(cov[95] - 0.95) < gate_cov
                and gate_condvar[0] < np.median(cond_ratio) < gate_condvar[1]
                and (not bias_measurable
                     or (gate_bias[0] < bias_ratio < gate_bias[1]
                         and marg_res < 0.2)))
            rows.append(dict(
                case=case.get("tag", f"c{ci}"), scene=si, P=P, m_obs=m_obs, n=n, q=q,
                cond_DF=float(diag["g_max"] / diag["g_min"]), trials=trials_c,
                var_ratio_median=float(np.median(ratio)),
                var_ratio_iqr=[float(np.percentile(ratio, 25)),
                               float(np.percentile(ratio, 75))],
                cov68=cov[68], cov95=cov[95],
                bias_ratio=bias_ratio, marginal_mean_residual=marg_res,
                bias_snr=bias_snr, bias_measurable=bias_measurable,
                condvar_ratio_median=float(np.median(cond_ratio)),
                cond_lt_marginal=cond_lt_marg,
                gate_pass=gate_pass))

    all_pass = all(r["gate_pass"] for r in rows)
    return dict(run_name=config.get("run_name", "run"), seed=config["seed"],
                n_checks=len(rows), all_pass=bool(all_pass),
                gates=dict(gate_variance_ratio_median=gate_median,
                           gate_variance_ratio_iqr=gate_iqr,
                           gate_coverage_abs_tol=gate_cov,
                           gate_bias_ratio=gate_bias,
                           gate_condvar_ratio_median=gate_condvar,
                           gate_bias_snr_min=gate_bias_snr),
                checks=rows,
                note="joint ensemble = Gate B 主判据（Prop 1）；fixed-δc 仅诊断"
                     "（R4：条件方差 σ²ΔF⁻¹AᵀM²AΔF⁻¹，不用于 tightness 声明；"
                     "bias 检查仅 SNR>gate_bias_snr_min 时判定）")
