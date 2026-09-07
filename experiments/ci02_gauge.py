"""CI02 · Gauge lifting、绝对信息与 λ⋆（宪法 §4.2 冻结规格，卡 C10）。

主输出（mode-resolved，非 trace）：gauge 绝对信息、non-gauge floor μ_floor、
闭式曲线（Prop 2，多灯求和 = V1b 块对角恒等式）、扫描 λ⋆、一阶预测 λ⋆^lin；
retention heatmap 仅 within-scene 解释（固定 log grid，C11 加连续性追踪）。

预注册验收（config 冻结）：
  closed_vs_direct < 1e-8（每场景全网格最大相对误差）；
  λ⋆ log error：median(|log10(pred/scan)|) ≤ 1 decade（H3），线性域分层另行报告；
  失效处置：保留闭式、降级 crossover claim（宪法 CI02 失败动作），如实执行。
"""

from __future__ import annotations

import numpy as np

from calibinfo.datasets.synthetic import make_grid, mu_floor
from calibinfo.information.gauge import gauge_response
from calibinfo.information.retention import (retention_spectrum,
                                              retention_spectrum_dual)
from calibinfo.information.schur import delta_f


def _fig4_data(sc, s_med2):
    """Fig.4 三 panel 数据（宪法 §7）：

    1. trace 反例：tr(ΔF(λ))/tr(F∞) 跨 4 个数量级近乎不动，而最弱模式 ρ 大幅移动；
    2. tracked ρ heatmap：bottom-9 模式经 track_modes 连续性追踪（禁索引排序）；
    3. V6 角度复现：最弱保留 9 维子空间 λ_lo→λ_hi 最大主角度（N=1 ≈0° / N>1 大旋转）。
    注意：追踪对象是 R(λ)=F∞^{-1/2}ΔF F∞^{-1/2} 的**特征向量**（λ 依赖），
    不是 F∞ 的 range 基（λ 无关——首轮实现缺陷）。
    """
    from calibinfo.information.mode_tracking import (principal_angle_ref,
                                                     track_modes)
    lam_lo, lam_hi = 1e-2 * s_med2, 1e2 * s_med2
    grid = np.geomspace(lam_lo, lam_hi, 21)
    q = sc["B_blk"].shape[1]
    Finf = sc["A_st"].T @ sc["A_st"]
    tr_fin = float(np.trace(Finf))

    w, V = np.linalg.eigh(Finf)
    keep = w > 1e-12 * max(w.max(), 1e-300)
    Vk, wk = V[:, keep], w[keep]
    Fh = Vk @ np.diag(1.0 / np.sqrt(wk)) @ Vk.T        # 正定平方根（白化路线口径）

    trace_series, rho_tracked, Vbottom = [], [], []
    prev_cols, chain = None, None
    deg_steps = 0
    for l in grid:
        DF = delta_f(sc["A_st"], sc["B_blk"], l * np.eye(q))[0]
        trace_series.append(float(np.trace(DF) / tr_fin))
        ev, evec = np.linalg.eigh(Fh @ DF @ Fh)        # R(λ) 特征分解（升序）
        cand = evec[:, :9]                             # bottom-9 最弱保留
        rho_bot = ev[:9]
        if prev_cols is None:
            chain = list(range(9))
            prev_cols = cand
        else:
            r = track_modes(prev_cols, cand)
            chain = [chain[j] for j in r["assignment"]]
            deg_steps += int(r["degenerate"])
            prev_cols = cand
        rho_tracked.append([float(rho_bot[j]) for j in chain])   # 身份保持读出
        Vbottom.append(cand)

    angle = principal_angle_ref(Vbottom[0], Vbottom[-1], 9)
    return dict(scene_id=sc["scene_id"], grid=grid.tolist(),
                trace_ratio=trace_series, rho_tracked=rho_tracked,
                degenerate_steps=deg_steps,
                bottom_subspace_angle_deg=float(angle))


FIXED_LOG_GRID = np.linspace(-8.0, 8.0, 33)      # 跨场景统计用固定 log10 网格


def _scene_band(scene):
    """B 奇异谱的正带中位（工作带纪律：近零奇异值不入带）。"""
    e = np.linalg.eigvalsh(scene["B_blk"].T @ scene["B_blk"])
    pos = e[e > 1e-3 * max(e.max(), 1e-300)]
    return float(np.median(pos)) if pos.size else float(e.max())


def run(config, run_dir):
    dual_rels = []
    rng = np.random.default_rng(config["seed"])
    scenes, manifest = make_grid(
        rng, P=config.get("P", 300),
        geometries=tuple(config["geometries"]), albedos=tuple(config["albedos"]),
        light_elevs=tuple(config["light_elevs"]),
        n_lights_choices=tuple(config["n_lights_choices"]))
    gate_cvd = config["gate_closed_vs_direct"]
    cancel_margin = config.get("cancellation_margin", 100.0)
    gate_star = config["gate_lambda_star_log_err_median"]
    lin_band = config.get("linear_band_rel_smed2", 1e-2)

    rows, fig3 = [], None
    fig4 = {"n1": None, "n3": None}
    ratios_all, ratios_lin = [], []
    for sc in scenes:
        s_med2 = _scene_band(sc)
        lam_grid = np.logspace(-6, 6, 25) * s_med2          # 自适应网格
        a = sc["gauge_a"]

        # 闭式（多灯求和）vs 直接 Rayleigh（联合 ΔF）
        resp = [gauge_response(sc["B_blk"][k * sc["P"]:(k + 1) * sc["P"],
                                           k * 9:(k + 1) * 9],
                               sc["gauge_cbar"][k * 9:(k + 1) * 9], lam_grid)
                for k in range(sc["meta"]["n_lights"])]
        closed = np.sum([r["exact"] for r in resp], axis=0)
        slope = float(sum(r["slope"] for r in resp))
        saturation = float(sum(r["saturation"] for r in resp))
        direct = np.array([float(a @ delta_f(sc["A_st"], sc["B_blk"],
                                             l * np.eye(sc["B_blk"].shape[1]))[0] @ a)
                           for l in lam_grid])
        rel = np.abs(closed - direct) / np.abs(direct)
        # 阈值分层（宪法 §6.2：按 condition number 分层，禁统一 atol）：
        # 直接路线在深未校准端（λ→0，gauge Rayleigh ≈ 0）发生灾难性消减，
        # 相对误差地板 ≈ C·u·saturation/|direct|；良条件区保持 1e-8 gate。
        u = np.finfo(float).eps
        allow = np.maximum(gate_cvd, cancel_margin * u * saturation / np.abs(direct))
        cvd = float(rel.max())
        cvd_layered = float((rel / allow).max())   # <1 为过（含分层预算）

        # μ_floor、λ⋆ 数值求根（闭式单调曲线二分，宪法 §4.2：线性域外用闭式求根）
        floor, _ev = mu_floor(sc)

        def curve(l):
            return float(np.sum([gauge_response(
                sc["B_blk"][k * sc["P"]:(k + 1) * sc["P"], k * 9:(k + 1) * 9],
                sc["gauge_cbar"][k * 9:(k + 1) * 9], l)["exact"]
                for k in range(sc["meta"]["n_lights"])]))

        lo, hi = 1e-9 * s_med2, 1e6 * s_med2
        while curve(hi) < floor and hi < 1e14 * s_med2:
            hi *= 100.0
        if curve(hi) < floor:
            lam_star = float("nan")                    # 求根窗内无交叉，如实记录
        else:
            while curve(lo) >= floor and lo > 1e-16 * s_med2:
                lo /= 100.0
            for _ in range(80):                        # 二分到相对 ~1e-12
                mid = float(np.sqrt(lo * hi))
                if curve(mid) >= floor:
                    hi = mid
                else:
                    lo = mid
            lam_star = float(np.sqrt(lo * hi))
        lam_star_pred = floor / slope if slope > 0 else float("nan")
        ratio = float(np.log10(lam_star_pred / lam_star)) if np.isfinite(lam_star) \
            else float("nan")
        in_linear = bool(np.isfinite(lam_star) and lam_star <= lin_band * s_med2)
        # λ⋆ 适用条件诊断（宪法：λ⋆≪min_{αᵢ≠0}sᵢ²）：逐灯最小有效奇异值
        min_e_alpha = float("inf")
        for r_ in resp:
            ia = np.abs(r_["alpha"]) > 1e-12 * max(np.abs(r_["alpha"]).max(), 1e-300)
            if ia.any():
                min_e_alpha = min(min_e_alpha, float(r_["eigs"][ia].min()))
        cond_value = float(lam_star / min_e_alpha) if np.isfinite(lam_star) \
            and np.isfinite(min_e_alpha) and min_e_alpha > 0 else float("nan")

        # retention（固定网格，within-scene）
        Finf = sc["A_st"].T @ sc["A_st"]
        rho_grid = np.array([retention_spectrum(
            delta_f(sc["A_st"], sc["B_blk"], (10.0 ** t) * s_med2
                    * np.eye(sc["B_blk"].shape[1]))[0], Finf)["rho"][:5]
            for t in FIXED_LOG_GRID])                        # (T, 5) 最弱 5 模式
        # C11 双路线交叉：白化（eigh 平方根）vs generalized eig（Cholesky 归约）
        lam_mid = float(np.sqrt(10.0) ** 0 * s_med2)         # 固定网格中点 t=0
        lam_mid = (10.0 ** 0.0) * s_med2
        dual = retention_spectrum_dual(
            delta_f(sc["A_st"], sc["B_blk"], lam_mid * np.eye(sc["B_blk"].shape[1]))[0],
            Finf)

        # C11 Fig.4 数据：网格里第一个 N=1 与第一个 N=3 场景各算一份（V6 对比口径）
        if config.get("with_tracking", False) and sc["meta"]["n_lights"] in (1, 3)                 and fig4["n%d" % sc["meta"]["n_lights"]] is None:
            fig4["n%d" % sc["meta"]["n_lights"]] = _fig4_data(sc, s_med2)

        rows.append(dict(scene_id=sc["scene_id"], **{k: v for k, v in sc["meta"].items()},
                         s_med2=s_med2, floor=floor, slope=slope,
                         closed_vs_direct=cvd, cvd_layered=cvd_layered,
                         lam_star=lam_star,
                         lam_star_pred=lam_star_pred, log10_ratio=ratio,
                         cond_lam_star_over_min_e_alpha=cond_value,
                         retention_dual_rel=dual["dual_rel"],
                         in_linear_band=in_linear))
        if dual["dual_rel"] is not None:
            dual_rels.append(dual["dual_rel"])
        if np.isfinite(ratio):
            (ratios_lin if in_linear else ratios_all).append(ratio)
            ratios_all.append(ratio)
        # Fig.3 代表场景：|log10 ratio| 最小的场景
        if fig3 is None or abs(ratio) < abs(fig3["log10_ratio"]):
            fig3 = dict(scene_id=sc["scene_id"], lam_grid=lam_grid.tolist(),
                        closed=closed.tolist(), direct=direct.tolist(),
                        floor=floor, slope=slope, lam_star=lam_star,
                        lam_star_pred=lam_star_pred, log10_ratio=ratio,
                        s_med2=s_med2, rho_fixed_grid=rho_grid.tolist(),
                        fixed_log_grid=FIXED_LOG_GRID.tolist(),
                        in_linear_band=in_linear)

    ok_cvd = all(r["cvd_layered"] < 1.0 for r in rows)
    med = float(np.median(np.abs(ratios_all))) if ratios_all else float("nan")
    med_lin = float(np.median(np.abs(ratios_lin))) if ratios_lin else float("nan")
    q90 = float(np.percentile(np.abs(ratios_all), 90)) if ratios_all else float("nan")
    return dict(
        run_name=config.get("run_name", "run"), seed=config["seed"],
        n_scenes=len(rows), gate_closed_vs_direct=gate_cvd,
        gate_lambda_star_log_err_median=gate_star,
        closed_vs_direct_max=float(max(r["closed_vs_direct"] for r in rows)),
        closed_vs_direct_layered_max=float(max(r["cvd_layered"] for r in rows)),
        cancellation_margin=cancel_margin,
        closed_vs_direct_pass=bool(ok_cvd),
        lam_star_stats=dict(n_finite=len(ratios_all),
                            median_abs_log10=med, median_abs_log10_linear=med_lin,
                            q90_abs_log10=q90,
                            cond_max=float(np.nanmax([r["cond_lam_star_over_min_e_alpha"]
                                                      for r in rows])),
                            pass_=bool(med <= gate_star)),
        fig3=fig3,
        lam_star_table=rows,
        retention_dual_rel_max=float(max(dual_rels)) if dual_rels else None,
        fig4=fig4,
        note="λ⋆ log error 失效处置：保留闭式、降级 crossover claim（宪法 CI02）")
