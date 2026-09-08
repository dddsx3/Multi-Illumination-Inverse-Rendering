"""CI05 · DiLiGenT 外部 sanity + failure taxonomy（宪法 §4.5，卡 C17/C18）。

职责边界（宪法原文）：确认真实非 Lambertian、shadow/specularity 条件下
弱模式现象仍有解释力——**不承担**理论精确吻合、**不用于** calibration-only 因果。
失败案例显式归因：模型失配（R-D）vs 校准不确定度（R-A），二者不许混写。

读出（within-scene，10 对象）：
  - 逐对象朗伯残差 + 离群占比（继承 exp8R 系口径：全 96 光标定）；
  - 弱模式存在性：ΔF(Λ→0) 的最小正特征值 << λ_med（非 gauge 结构弱模式）；
  - gauge 交叉可观测性：Prop 2 闭式曲线 vs μ_floor（λ⋆ 存在性 per object）；
  - 失败分类：mask 内像素的模型失配方差占比 vs 校准传播方差占比的量级对比
    （只做 taxonomy，不做因果 claim）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibinfo.datasets.diligent import load_object
from calibinfo.datasets.synthetic import sh_basis
from calibinfo.information.gauge import gauge_response
from calibinfo.information.schur import delta_f

DATA_ROOT = "D:/data/DiLiGenT/pmsData"
N_LIGHTS_SUB = 48
N_PIXEL_SUB = 1200


def run(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    data_root = config.get("data_root", DATA_ROOT)
    objects = config.get("objects") or sorted(
        d.name for d in Path(data_root).iterdir() if d.is_dir())
    rows = []
    for name in objects:
        obj = load_object(Path(data_root) / name)
        dirs = obj["dirs"]
        I = obj["I_norm"]                                   # (96, P_full)
        n_gt = obj["normals_gt"]
        sel = np.sort(rng.choice(96, N_LIGHTS_SUB, replace=False))
        P_full = I.shape[1]
        pidx = np.sort(rng.choice(P_full, min(P_full, N_PIXEL_SUB), replace=False))
        d_sub = dirs[sel]
        I_s = I[sel][:, pidx]
        n_s = n_gt[pidx]
        # 朗伯残差（exp8R calibrate 同口径：全 96 光）
        nl_full = np.clip(n_gt @ dirs.T, 0, None)
        rho_full = (I.T * nl_full).sum(1) / np.maximum((nl_full * nl_full).sum(1), 1e-12)
        I_hat = (rho_full[:, None] * nl_full).T
        resid = I - I_hat
        lambert_resid = float(np.linalg.norm(resid) / np.linalg.norm(I))
        r3 = np.abs(resid)
        outlier_frac = float((r3 > 3 * r3.std()).mean())
        # 弱模式结构（within-scene）：A=D(ŝ), B=ρ̂h·Y（geometry-known——DiLiGenT 有法线 GT，
        # 这是与 exp8R 的合法差别；GT 用在标定侧不作估计初始化）
        s_hat = np.clip(n_s @ d_sub.T, 0, None).T          # (L,P)
        A_st = np.vstack([np.diag(s_k) for s_k in s_hat])
        Y = sh_basis(n_s)
        h = (s_hat > 0).astype(float)
        B_blk = np.zeros((len(sel) * len(pidx), len(sel) * 9))
        rho_s = rho_full[pidx]
        for k in range(len(sel)):
            B_blk[k * len(pidx):(k + 1) * len(pidx), k * 9:(k + 1) * 9] = \
                (rho_s * h[k])[:, None] * Y
        # 弱模式谱（逐灯 Schur，避免 N·P 全矩阵——OOM 教训：M=eye(N·P) 24.7GB）
        # Λ→0 极限下 ΔF 的谱 = Σ_k D(ŝ_k) 投影掉 col(B_k) 后的谱（V1b 块对角恒等式）；
        # 全体非零特征值由各灯块非零谱并集近似其分布（floor 取全体像素白化谱的分位）：
        ev_all = []
        for k in range(len(sel)):
            Ak = s_hat[k]
            Bk = B_blk[k * len(pidx):(k + 1) * len(pidx), k * 9:(k + 1) * 9].copy()
            DFk, _, _ = delta_f(np.diag(Ak), Bk, np.full((9, 9), 1e-12))
            evk = np.linalg.eigvalsh(DFk)
            ev_all.append(evk[evk > 1e-10 * max(evk.max(), 1e-300)])
        ev_all = np.concatenate(ev_all)
        pos = ev_all[ev_all > 1e-10 * max(ev_all.max(), 1e-300)]
        floor = float(np.percentile(pos, 1)) if pos.size else float("nan")
        med = float(np.median(pos)) if pos.size else float("nan")
        # gauge 交叉（首灯，与 CI02 同口径）
        c0 = sh_basis(d_sub[0][None, :])[0]
        resp = gauge_response(B_blk[:, :9], -c0, 1.0)
        lam_star_pred = floor / resp["slope"] if resp["slope"] > 0 else float("nan")
        # 失败分类（量级对比，仅 taxonomy）
        # 模型失配方差：朗伯残差方差（截断 3σ 内的稳健口径）
        miss_var = float(np.median(r3[r3 <= 3 * r3.std()] ** 2))
        # 校准传播方差：弱模式在 λ⋆ 预设不确定度下的方差 σ²/λ_j —— 用 floor 的对偶
        calib_var = float(1.0 / floor) if floor > 0 else float("nan")
        taxonomy = ("model_mismatch_dominant" if miss_var > calib_var
                    else "calibration_propagation_comparable")
        rows.append(dict(object=name, P=int(len(pidx)), L=int(len(sel)),
                         lambert_residual=lambert_resid, outlier_frac=outlier_frac,
                         weak_mode_floor=floor, weak_mode_median=med,
                         floor_over_med=float(floor / med) if med else float("nan"),
                         gauge_slope=resp["slope"], saturation=resp["saturation"],
                         lam_star_pred=lam_star_pred,
                         miss_var=miss_var, calib_var=calib_var,
                         taxonomy=taxonomy))
    return dict(run_name=config.get("run_name", "sanity"), seed=config["seed"],
                n_objects=len(rows), rows=rows,
                note="DiLiGenT 只承担 real sanity（宪法 §4.5）：弱模式存在性 + gauge "
                     "交叉 + 失败分类（模型失配 vs 校准传播，量级对比，不混写因果）")
