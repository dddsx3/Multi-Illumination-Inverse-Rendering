#!/usr/bin/env python3
"""关键实验 4 · CRB 有效性检验（三层比较：逐参数 / 整体标量 / 谱层次）

设计（主智能体指令原文照抄语义）：
  估计器 A（Geometry-known）：固定真值法线/深度，只对 (ρ, C) 做 ALS。
    每步闭式线性最小二乘：给定 C 解 ρ（逐像素标量 LSQ），给定 ρ 解 C（9 维 LSQ/光）。
    从真值附近初始化。
  加噪：i.i.d. 高斯，3 个水平 = 图像峰值的 0.5% / 1% / 2%。
  次数：200-500 次/格（本实现 200，受单机算力约束，如实注记）。
  Gauge 处理：Fisher 近零子空间 N（含全局尺度 gauge），投影 P = I − Π_N；
    误差 e_t = P(θ̂ − θ)。
  三层比较：
    1) 逐参数 Var(e_k) vs [F⁺]_kk
    2) tr(Σ̂)/tr(F⁺) ≈ 1
    3) 谱层次（最重要）：log Var(u_kᵀ e) vs log(1/λ_k) 斜率 = 1

Fisher（geometry-known 版）= 实验 1 交叉验证过的 gauge_fisher_v2 口径：
  F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ （P×P，albedo 侧，消去全部 C）——
  恰是 ALS 估计的 (ρ) 参数块的有效 Fisher（profiling/ALS 的信息矩阵与 Schur 补一致，
  这是经典结果：对 nuisance 的 profiling 与线性高斯下的 Schur 补等价）。

数值细节纪律：
  - ALS 的 C 步在 ReLU 处不可微 → 用与 Fisher 同口径的 active set（h_kp）加权 LSQ；
  - 噪声水平 σ = level × I_peak（I_peak = max |I_true|）；
  - 初始化 θ₀ = θ_true + 0.01σ 扰动（"从真值附近初始化"）；
  - gauge 投影：用 F_eff 的近零特征向量（相对阈值 1e-8，预期 1 个尺度方向 δa=a）。

产物：critical_experiments/exp4_crb_validity.json
  含 log-log 谱层数据（多模态 agent 绘图：x=log10(1/λ_k), y=log10(Var(u_kᵀe))，斜率标注）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from gauge_fisher_v2 import (  # noqa: E402
    fisher_blocks, gauge_project, gauge_unit, load_scene, scene_arrays,
    schur_full, spectrum_metrics,
)

DATA = REPO / "p1" / "calibration_set" / "data"
OUT = HERE / "exp4_crb_validity.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
N_TRIALS = 200
NOISE_LEVELS = [0.005, 0.01, 0.02]
SEED = 20260905


def render(rho, Y, C, Sk=None, Hk=None):
    """I = ρ·ReLU(YCᵀ)。(P,N)"""
    S = np.maximum(Y @ C.T, 0.0)
    return rho[:, None] * S, S


def als_step_rho(I, Y, C):
    """给定 C，逐像素解 ρ_p：I_kp = ρ_p·s_kp → ρ̂_p = Σ_k s_kp·I_kp / Σ_k s_kp²（active 限 s>0）。"""
    S = np.maximum(Y @ C.T, 0.0)                # (P,N)
    num = (S * I).sum(1)
    den = (S * S).sum(1)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def als_C(I, Y, rho, C_init):
    """干净实现：给定 ρ 解全部 C_k。每光 k:
       方程 ρ_p h_kp Y_pᵀ C_k = I_kp，h_kp = 1[Y_pᵀC_k>0]（用当前 C 估计 active set，
       迭代 3 次内部 active-set 精化——与 Fisher 的 h 定义一致）。"""
    N = I.shape[1]
    C = C_init.copy()
    for _ in range(3):
        for k in range(N):
            zk = Y @ C[k]                       # (P,)
            h = (zk > 0).astype(float)
            w = rho * h                        # (P,) 权
            A = (Y * w[:, None]).T @ Y          # (9,9)
            b = (Y * w[:, None]).T @ I[:, k]    # (9,)
            # 岭正则防奇异（量级 = 微小, 不影响无偏性到一阶）
            reg = 1e-12 * (np.trace(A) + 1e-30) / 9
            C[k] = np.linalg.solve(A + reg * np.eye(9), b)
    return C


def run_scene(scene, N_lights=5, pixel_cap=400, n_trials=N_TRIALS):
    rng = np.random.default_rng(SEED + hash(scene) % 1000)
    sc = load_scene(str(DATA / scene))
    sub = list(range(N_lights))
    a, Y, C_true = scene_arrays(sc, sub, pixel_cap=pixel_cap, seed=SEED)
    P = len(a)
    # Fisher (albedo 侧有效信息): F_eff = Schur 消去 C
    bl = fisher_blocks(a, Y, C_true)
    F_eff = schur_full(bl)
    a_hat0 = gauge_unit(a)
    Fp = gauge_project(F_eff, a_hat0)
    w = np.linalg.eigvalsh(Fp)
    lam_max = max(w[-1], 1e-300)
    # 近零子空间（gauge: 相对 1e-8）
    nz = w < 1e-8 * lam_max
    d_nz = int(nz.sum())
    _, vecs = np.linalg.eigh(Fp)
    N_sub = vecs[:, :d_nz]                      # (P, d_nz) 近零特征向量
    # 投影算子作用于 e = θ̂ − θ 的自由度空间(gauge 投影后的 P-1 维 + gauge 本身)
    # 设计文档口径: e_t = P(θ̂-θ), P = I - Π_N
    # 注意 Fp 已是 gauge 投影空间; 我们把 e 直接投到 Fp 的正谱特征向量坐标上:
    pos = ~nz
    U_pos = vecs[:, pos]                        # (P, P-d_nz) 正谱特征向量

    # 真值渲染
    I_true, S_true = render(a, Y, C_true)
    I_peak = float(np.abs(I_true).max())

    results = {}
    for level in NOISE_LEVELS:
        sigma = level * I_peak
        # 存储: e 在正谱坐标下的投影 u_k^T e（每次试验一列）
        E_pos = np.zeros((U_pos.shape[1], n_trials))
        E_full = np.zeros((P, n_trials))
        for t in range(n_trials):
            noise = rng.normal(0, sigma, I_true.shape)
            I_noisy = I_true + noise
            # 初始化: 真值附近
            a0 = a * (1 + rng.normal(0, 0.01, ()))  # 标量扰动(保持 gauge 语义)
            C0 = C_true + rng.normal(0, 0.01, C_true.shape) * max(1e-3, np.abs(C_true).max())
            # ALS 迭代 (闭式交替)
            a_est = a0.copy()
            C_est = C0.copy()
            for it in range(8):
                a_est = als_step_rho(I_noisy, Y, C_est)
                a_est = np.clip(a_est, 0, None)   # 非负约束(真值域内一阶无偏)
                C_est = als_C(I_noisy, Y, a_est, C_est)
            e = a_est - a
            # gauge 处理: 投影掉近零子空间 N (在原 P 维空间)
            # 近零子空间含 δa=a 方向; U_pos 张成其正补 → e_t = U_pos U_posᵀ e
            E_pos[:, t] = U_pos.T @ e
            E_full[:, t] = e
        # ---- 三层比较 ----
        Fp_pos_eigs = w[pos]                      # λ_k (正谱,升序)
        Var_pos = E_pos.var(axis=1)               # Var(u_kᵀ e) — 特征坐标方差
        # CRB 对角 = σ²/λ_k（标准口径 Cov ≥ σ²[F⁻¹]；初版漏乘 σ² 导致比值全为 σ² 量级——bug 已修）
        crb_diag = sigma ** 2 / np.maximum(Fp_pos_eigs, 1e-300)
        # 1) 逐参数比——【可估带】λ > 1e-2·λmax（病态带另报，见下）
        keep = Fp_pos_eigs > 1e-2 * lam_max
        ratio_param = Var_pos[keep] / crb_diag[keep]
        # 病态带（收缩区）：ALS 收缩估计器可低于 CRB（CRB 仅对无偏估计成立）——如实双报告
        bad = Fp_pos_eigs <= 1e-2 * lam_max
        ratio_bad = Var_pos[bad] / crb_diag[bad] if bad.any() else np.array([np.nan])
        # 2) 整体标量（可估带口径）
        Sigma_tr = float(Var_pos[keep].sum())     # E_pos 列独立 → tr(Σ̂)=Σ Var
        Fpinv_tr = float(crb_diag[keep].sum())   # tr(σ²F⁺) 可估带
        tr_ratio = Sigma_tr / Fpinv_tr
        # 3) 谱层次 log-log 拟合（可估带：病态带的方差塌缩会污染斜率）
        x = np.log10(crb_diag)
        y = np.log10(np.maximum(Var_pos, 1e-300))
        sel = keep
        slope, intercept = np.polyfit(x[sel], y[sel], 1)
        results[str(level)] = dict(
            sigma=sigma, n_trials=n_trials,
            param_ratio_median=float(np.median(ratio_param)),
            param_ratio_p25=float(np.percentile(ratio_param, 25)),
            param_ratio_p75=float(np.percentile(ratio_param, 75)),
            shrink_band_ratio_median=float(np.nanmedian(ratio_bad)),
            shrink_band_n=int(bad.sum()),
            tr_sigma=float(Sigma_tr), tr_crb=Fpinv_tr,
            tr_ratio=float(tr_ratio),
            spectral_slope=float(slope), spectral_intercept=float(intercept),
            n_param_used=int(keep.sum()),
            lam_pos=[float(v) for v in Fp_pos_eigs],
            var_pos=[float(v) for v in Var_pos],
            crb_diag=[float(v) for v in crb_diag],
            # log-log 原始数据（多模态 agent 绘图用；x=log10(σ²/λ), y=log10(Var)）
            loglog_x=[float(v) for v in x], loglog_y=[float(v) for v in y],
            loglog_sel=[bool(v) for v in sel],
        )
        print(f"  {scene:10s} level={level:.3f} σ={sigma:.4f}: "
              f"可估带中位比={np.median(ratio_param):.3f} [p25,p75]=[{np.percentile(ratio_param,25):.2f},{np.percentile(ratio_param,75):.2f}] "
              f"| 收缩带中位比={np.nanmedian(ratio_bad):.3f}(n={bad.sum()}) "
              f"| tr(Σ)/tr(σ²F⁺)={tr_ratio:.3f} | 谱斜率(可估带)={slope:.3f}")
    return dict(scene=scene, P=P, N=N_lights, n_gauge_dirs=d_nz, results=results)


def main():
    out = {"scenes": {}, "meta": dict(n_trials=N_TRIALS, noise_levels=NOISE_LEVELS,
                                      estimator="ALS (geometry-known, closed-form alternating)",
                                      note="200 次/格受单机算力约束(设计 200-500 取下限); CRB 口径=σ²F⁺(初版漏乘σ²已修); 病态带(λ<1e-2λmax)为 ALS 收缩区——CRB 仅对无偏估计成立,收缩估计器可低于下界,双带分别报告")}
    for scene in SCENES:
        print(f"[exp4] {scene}")
        out["scenes"][scene] = run_scene(scene)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp4] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
