"""C13 · CI03 非线性 validity envelope（B 臂：解析非线性 SH+ReLU 管线）。

宪法 §4.3 / R-B 风险：线性理论在真实（非线性）渲染下的有效域显式化。

协议（每 cell = 场景 × 扰动强度 k）：
  1. 场景工厂出 nominal：法线 n、反照率 a、灯方向 d₀ → 名义 s₀=relu(Yc₀)、H₀；
  2. 线性理论（nominal 线性化）：A=D(s₀)、B=D(a·H₀)Y、先验 Σ_c=k²σ_{c0}²I →
     Λ、ΔF、M、弱 5 模式（预测协方差特征基）；
  3. MC（每 draw）：δc~N(0,k²Σ₀) → **真渲染** y = a·relu(Y(c₀+δc)) + ε；
     估计器 = 线性理论的 x̂=ΔF⁻¹AᵀMy（theory-matched，隔离模型失配效应）；
  4. 记录三轴：mask-flip 率（|H(δc)≠H₀|/P，含方向拆分）、‖δc‖、SNR；
     理论误差 = |emp_var/pred_var − 1| 的弱 5 模式 median。

验收（预注册）：mask flip 必须被记录（缺失=本卡不合格）；
envelope 边界 = median 误差 <10% 的最大 flip 率（按几何分层）；
若不存在 <10% 有效域 → Gate C 失败动作（只留线性理论），如实执行。
增强臂 A（BlenderProc 真渲染）环境问题时记 BLOCKER 不阻塞主线。
"""

from __future__ import annotations

import numpy as np

from calibinfo.datasets.synthetic import make_scene
from calibinfo.information.schur import delta_f

SIG_C0 = 0.05
SIGMA = 0.02
TRIALS = 2000


def _one_cell(sc, k, trials, seed):
    P = sc["P"]
    Y = sc["Y"]
    a_true = sc["gauge_a"]
    c0 = sc["cs"][0]                                   # N=1 gauge 灯
    s0 = sc["ss"][0]
    H0 = sc["Hs"][0]
    A = np.diag(s0)
    B = (a_true * H0)[:, None] * Y
    sigma = SIGMA
    Sig_c = (k * SIG_C0) ** 2 * np.eye(9)
    Lam = sigma ** 2 * np.linalg.inv(Sig_c)
    DF, M, _ = delta_f(A, B, Lam)
    DFinv = np.linalg.inv(DF)
    Ginv_Bt = np.linalg.lstsq(B.T @ B + Lam, B.T, rcond=None)[0]

    Finf = A.T @ A
    w, V = np.linalg.eigh(Finf)
    kp = w > 1e-12 * max(w.max(), 1e-300)
    Fh = V[:, kp] @ np.diag(1 / np.sqrt(w[kp])) @ V[:, kp].T
    C_pred_w = Fh @ (sigma ** 2 * DFinv) @ Fh
    ev_c, V_c = np.linalg.eigh(C_pred_w)
    weak = np.argsort(ev_c)[-5:]
    pred_var = ev_c[weak]
    scores = Fh.T @ V_c[:, weak]

    rng = np.random.default_rng(seed)
    E = np.empty((trials, P))
    flips = []
    for t in range(trials):
        dc = rng.normal(0, k * SIG_C0, size=9)
        c = c0 + dc
        s_true = np.maximum(Y @ c, 0.0)
        flips.append(int(np.sum((s_true > 0) != (H0 > 0))))
        y = a_true * s_true + rng.normal(0, sigma, size=P)
        My = y - B @ (Ginv_Bt @ y)
        E[t] = DFinv @ (A.T @ My) - a_true
    Sc = E @ scores
    emp_var = Sc.var(axis=0, ddof=1)
    var_ratio = emp_var / pred_var
    dc_norm = float(np.linalg.norm(rng.normal(0, k * SIG_C0, size=9)))  # 典型 ‖δc‖
    snr = float(np.linalg.norm(a_true * s0) / (sigma * np.sqrt(P)))
    flip_rate = float(np.mean(flips) / P)
    return dict(k=k, sig_c=k * SIG_C0, dc_norm_typical=dc_norm, snr=snr,
                mask_flip_rate=flip_rate,
                var_ratio_median=float(np.median(var_ratio)),
                var_ratio_iqr=[float(np.percentile(var_ratio, 25)),
                               float(np.percentile(var_ratio, 75))],
                pred_err_median=float(abs(np.median(var_ratio) - 1.0)))


def run(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    gate_valid = config["gate_valid_pred_err"]
    rows = []
    for geo in config["geometries"]:
        for si in range(config.get("n_scenes_per_geo", 2)):
            s_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + si)
            sc = make_scene(s_rng, config["P"], 1, geometry=geo,
                            albedo=config.get("albedo", "medium"))
            for k in config["k_grid"]:
                r = _one_cell(sc, float(k), config.get("trials", TRIALS),
                              seed=rng.integers(0, 2**63 - 1))
                r.update(scene=f"{geo}_s{si}", geometry=geo, scene_i=si)
                rows.append(r)

    # envelope 边界（按几何）：median 误差 <10% 的最大 mask-flip 率
    envelope = {}
    for geo in config["geometries"]:
        gr = [r for r in rows if r["geometry"] == geo]
        ok = [r for r in gr if r["pred_err_median"] < gate_valid]
        envelope[geo] = dict(
            max_valid_flip_rate=float(max((r["mask_flip_rate"] for r in ok), default=0.0)),
            max_valid_k=float(max((r["k"] for r in ok), default=0.0)),
            n_valid_cells=len(ok), n_cells=len(gr))
    all_have_domain = all(e["n_valid_cells"] > 0 for e in envelope.values())
    return dict(run_name=config.get("run_name", "run"), seed=config["seed"],
                gate_valid_pred_err=gate_valid, n_cells=len(rows),
                envelope=envelope,
                valid_domain_exists=bool(all_have_domain),
                gate_c_action=("有效域存在 → 真实预测主张限定在低 flip 区"
                               if all_have_domain else
                               "Gate C 失败动作：论文只留线性理论，不做真实预测主张（如实执行）"),
                checks=rows,
                note="B 臂 = 解析非线性 SH+ReLU（同 GT 重渲染）；估计器 = 线性理论"
                     "估计器（隔离模型失配）；A 臂 BlenderProc 见卡 C13 卡住时条款")
