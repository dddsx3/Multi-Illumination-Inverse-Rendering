#!/usr/bin/env python3
"""关键实验 12(卡 D)· 合成跨单元判别力 + 粗几何鲁棒性

设计(任务书 v3.0 卡 D, 预注册):
  ① 判别力: 复用卡 C 场景内设计(每场景 30-50 配置, 这里 20 × 4 场景);
     每配置跑无先验 ALS(固定噪声=峰值 1%, T=1000 次迭代), 经验法线角误差
     (与诊断同量纲); 场景内 Spearman(诊断 tr(S_θ⁻¹) 的倒数——迹越小信息越大,
     误差应越大 → 用 1/迹 或直接 Spearman(迹, 误差) 预期【正】);
  ② 粗几何: 深度代理三档——高斯平滑(σ=2px)/加噪(5%)/降分辨率(32² 上采);
     每档重算诊断并排序, 与真值几何排序做场景内 Spearman;
  验收(预注册): ① 逐场景多数显著为正 → 支柱③合成版成立;
                ② 排序一致性 ≥0.8 → "训练前可用"; <0.8 → 降格"后验诊断"。

诊断量(R-D1 口径): 法线角迹 = tr(S_θ⁻¹)(exp11v2 的解析 Woodbury 迹,
含 (ρ,C) 消去 + gauge 吸收)。经验误差 = ALS 估计的法线角 MAE(度)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import DATA, load_scene_compat, sh2, sh2_d, sobel_sparse  # noqa: E402
from exp11v3_kappa_expansion import trace_theta_inv_safe  # noqa: E402

OUT = HERE / "exp12_discrimination_robustness.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
N_LIGHTS = 5
N_CONFIGS = 20
N_TRIALS = 50          # 每配置 ALS 重复次数(任务书 T≥50)
ALS_ITERS = 1000       # T=1000(exp10 已证斜率迭代无关, 1000 足够)
NOISE = 0.01
SEED = 20260906


def scene_pixels(scene, res=RES):
    """几何已知口径: GT 法线 + ρ + 32 灯池。"""
    import os
    sc = load_scene_compat(str(DATA / scene))
    n_mesh = np.load(os.path.join(str(DATA / scene), "normal_mesh.npy")).transpose(1, 2, 0)
    H0, W0 = n_mesh.shape[:2]
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    nm = n_mesh[i0:i0+H, j0:j0+W].reshape(-1, 3)
    nm = nm / np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    a = sc["albedo"][i0:i0+H, j0:j0+W].ravel()
    mk = sc["mask"][i0:i0+H, j0:j0+W].ravel() > 0
    nm, a = nm[mk], a[mk]
    pool = sc["sh"][:32].astype(float)
    # 深度(粗几何用): GT depth 裁剪
    z = sc["depth"][i0:i0+H, j0:j0+W].ravel().astype(float)
    z = np.where(mk & (z < 1e8), z, np.nan)
    return nm, a, pool, z, int(mk.sum())


def als_normal_error(I_n, Y, rho, C_init, n_true, iters=ALS_ITERS):
    """无先验 ALS(固定 ρ 初值=真值邻域, h 冻结真值口径同 exp4)→ 法线角 MAE(°)。
    注意: 几何已知口径下 ALS 只估 (ρ, C); 法线角误差来自 ρ/C 误差经渲染的非线性
    传播——但几何已知时法线不变! 修正: 经验误差 = 渲染图像相对误差(信息受限的
    直接量测, 与法线角同受 (ρ,C) 信息量控制)。任务书"经验法线角误差"在几何已知
    口径下退化为【光照角误差】: 由 C 反解方向光, 与真值方向比角度。"""
    rho, C = rho.copy(), C_init.copy()
    for _ in range(iters):
        S = np.maximum(Y @ C.T, 0.0)
        rho = (S * I_n).sum(1) / np.maximum((S * S).sum(1), 1e-12)
        for k in range(C.shape[0]):
            zk = Y @ C[k]
            h = (zk > 0).astype(float)
            w = rho * h
            A9 = (Y * w[:, None]).T @ Y
            b9 = (Y * w[:, None]).T @ I_n[:, k]
            C[k] = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
    # 光照角误差: 每光反解方向(l=1 块)与真值方向比
    A_kernel = np.array([np.pi, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3] + [np.pi/4]*5)
    errs = []
    for k in range(C.shape[0]):
        l1 = C[k, 1:4] / (A_kernel[1] * 0.488603)
        n1 = np.linalg.norm(l1)
        if n1 < 1e-9:
            errs.append(180.0)
            continue
        l1 /= n1
        # 真值方向: 同款反解 C_true
        errs.append(l1)
    return errs, C


def light_angle_errors(C_est, C_true):
    """由 SH 系数反解方向光(l=1 块/标准核), 每光与真值方向比夹角(°), 均值。"""
    A1 = 2 * np.pi / 3
    C1c = 0.488603
    angs = []
    for k in range(C_est.shape[0]):
        l_e = C_est[k, 1:4] / (A1 * C1c)
        l_t = C_true[k, 1:4] / (A1 * C1c)
        ne, nt = np.linalg.norm(l_e), np.linalg.norm(l_t)
        if ne < 1e-9 or nt < 1e-9:
            angs.append(180.0)
            continue
        c = np.clip(l_e @ l_t / (ne * nt), -1, 1)
        angs.append(np.degrees(np.arccos(c)))
    return float(np.mean(angs))


def main():
    rng = np.random.default_rng(SEED)
    out = {"scenes": {}, "meta": dict(n_configs=N_CONFIGS, n_trials=N_TRIALS,
                                      als_iters=ALS_ITERS, noise=NOISE,
                                      coarse_proxies=["smooth_σ2", "noise_5%", "downres_32"])}
    for scene in SCENES:
        nrm_t, rho_t, pool, z_t, n_px = scene_pixels(scene)
        Y = sh2(nrm_t)
        I_true = rho_t[:, None] * np.maximum(Y @ pool[:N_LIGHTS].astype(float).T, 0)
        sigma = NOISE * np.abs(I_true).max()
        rows = []
        for ci in range(N_CONFIGS):
            sel = np.sort(rng.choice(32, N_LIGHTS, replace=False))
            C_cfg = pool[sel].astype(float)
            S_cfg = np.maximum(Y @ C_cfg.T, 0)
            # 经验误差: T 次ALS, 法线角误差均值(光照角)
            ang_trials = []
            for t in range(N_TRIALS):
                I_n = I_true + rng.normal(0, sigma, I_true.shape)
                _, C_e = als_light(I_n, Y, rho_t, C_cfg)
                ang_trials.append(light_angle_errors(C_e, C_cfg))
            emp_err = float(np.mean(ang_trials))
            # 诊断: 真值几何 tr(S_θ⁻¹)(schur 块)
            try:
                tr_true = trace_theta_inv_safe(nrm_t, rho_t, C_cfg)[1]   # Schur 块迹
            except Exception:
                tr_true = float('nan')
            rows.append(dict(config=ci, emp_err=emp_err, tr_schur=tr_true))
        # 场景内 Spearman(tr, 误差)——预期【正】(信息少→误差大)
        trs = np.array([r["tr_schur"] for r in rows])
        errs = np.array([r["emp_err"] for r in rows])
        fin = np.isfinite(trs) & np.isfinite(errs)
        rho_s, p_s = spearmanr(trs[fin], errs[fin])
        out["scenes"][scene] = dict(rows=rows,
                                    spearman_trace_err=dict(rho=float(rho_s), p=float(p_s),
                                                            n=int(fin.sum())))
        print(f"{scene:10s} 判别: Spearman(tr, err) = {rho_s:.3f} (p={p_s:.4e}, n={fin.sum()})")
    # 判据①
    sig_pos = sum(1 for s in SCENES
                  if out["scenes"][s]["spearman_trace_err"]["p"] < 0.05
                  and out["scenes"][s]["spearman_trace_err"]["rho"] > 0)
    out["verdict_discrimination"] = dict(
        n_significant_positive=sig_pos,
        acceptance="逐场景多数显著为正 → 支柱③合成版成立",
        result=("成立" if sig_pos >= 3 else "不成立(负结果入文)"))
    print(f"[exp12-①] 判别力: {sig_pos}/4 场景显著正 → {out['verdict_discrimination']['result']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp12] 阶段落盘 -> {OUT}(粗几何部分见 exp12b)")


def als_light(I_n, Y, rho, C_init):
    """只估 C 的 ALS(ρ 固定真值——判别口径: 光照已知强度? 不, ρ 也估)。"""
    # 任务书: 无先验 ALS 估 ρ 与光强。几何已知下光"强度"进 C 的幅度。
    # 实现: ρ 与 C 交替(与 exp10.als 同构), 但法线角误差只看方向 → ρ 误差不影响。
    rho_, C = rho.copy(), C_init.copy()
    for _ in range(1000):
        S = np.maximum(Y @ C.T, 0.0)
        rho_ = (S * I_n).sum(1) / np.maximum((S * S).sum(1), 1e-12)
        for k in range(C.shape[0]):
            zk = Y @ C[k]
            h = (zk > 0).astype(float)
            w = rho_ * h
            A9 = (Y * w[:, None]).T @ Y
            b9 = (Y * w[:, None]).T @ I_n[:, k]
            C[k] = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
    return rho_, C


if __name__ == "__main__":
    main()
