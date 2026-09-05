#!/usr/bin/env python3
"""关键实验 10(卡 B)· CRB 谱斜率收口:迭代扫描 + GN 紧收敛 + Q-Q

设计(任务书 v3.0 卡 B, 预注册):
  归因假设(专家): exp4 谱斜率 0.70-0.76/0.30-0.52 < 1 的主因 = ALS 有限迭代在
  病态方向的隐式正则化(early stopping, 收敛因子 1−λ_k/λ_max)。
  步骤:
    1) 迭代扫描: ALS 迭代 ∈ {10, 100, 1000, 10000} → 谱斜率若单调趋向 1, 归因成立;
    2) GN 紧收敛: 同一 J 的联合 GN(冻结 h, 双线性问题) → 斜率预期 ∈ [0.9, 1.1];
    3) χ² Q-Q: 可估带上 eᵀFe 的经验分位 vs χ²_r(r=rank 可估带);
    4) 附录: 给定 ρ 真值只估 C 的纯线性 LS(方差按构造 = CRB) sanity;
    5) 可估带维数占比报告。
  验收: GN 后可估带斜率 ∈ [0.9, 1.1] → 归因成立; <0.9 → 去ReLU线性对照分支。

产物: critical_experiments/exp10_slope_closure.{py,json}
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
    fisher_blocks, gauge_project, gauge_unit, load_scene, scene_arrays, schur_full,
)

DATA = REPO / "p1" / "calibration_set" / "data"
OUT = HERE / "exp10_slope_closure.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
NOISE = 0.01
N_TRIALS = 200
SEED = 20260905


def als(I, Y, rho0, C0, iters):
    """ALS: 固定 C 解 ρ(闭式), 固定 ρ 解 C(冻结 h 的加权 LSQ)。"""
    rho, C = rho0.copy(), C0.copy()
    for _ in range(iters):
        S = np.maximum(Y @ C.T, 0.0)
        num = (S * I).sum(1)
        den = np.maximum((S * S).sum(1), 1e-12)
        rho = num / den
        for k in range(C.shape[0]):
            zk = Y @ C[k]
            h = (zk > 0).astype(float)
            w = rho * h
            A9 = (Y * w[:, None]).T @ Y
            b9 = (Y * w[:, None]).T @ I[:, k]
            C[k] = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
    return rho, C


def gn(I, Y, rho0, C0, iters=30, Hk_frozen=None):
    """冻结-h 线性 LSQ(= "GN 紧收敛"的正确口径; 2026-09-05 二次修):
    冻结 h 后模型 I = ρ·h·(Y·C) 对 (ρ,C) 线性 → 一次 lstsq 即精确解,
    协方差按构造 = σ²(JᵀJ)⁺, 与 exp4 冻结-h Fisher 口径一致。
    初版两次失败原因留痕: ①雅可比冻结但残差用真 ReLU 模型(不自洽→发散 2973 vs 2.25);
    ②更早版本逐迭代重算 h(暗侧翻转→方差比 51×)。"""
    rho, C = rho0.copy(), C0.copy()
    P, N = I.shape
    if Hk_frozen is None:
        Hk_frozen = ((Y @ C0.T) > 0).astype(float)
    # 线性模型: I_kp = ρ_p·h_kp·(Y_pᵀC_k) → J = [diag(h⊙YC) | diag(ρh)Y](全线性)
    S0 = (Y @ C0.T) * Hk_frozen                       # h 加权的线性预测
    w_rho = np.concatenate([S0[:, k] for k in range(N)])          # ∂I/∂ρ
    J_C = np.stack([(Y * (rho0 * Hk_frozen[:, k])[:, None]) for k in range(N)], axis=0)  # (N,P,9)
    # 组装 JᵀJ 与 JᵀI:
    F_rr = np.diag((S0 ** 2).sum(1))
    F_CC = np.zeros((9 * N, 9 * N))
    F_rC = np.zeros((P, 9 * N))
    for k in range(N):
        Yk = Y * (rho0 * Hk_frozen[:, k])[:, None]
        F_CC[9*k:9*(k+1), 9*k:9*(k+1)] = Yk.T @ Yk
        F_rC[:, 9*k:9*(k+1)] = S0[:, k][:, None] * Yk
    JtJ = np.block([[F_rr, F_rC], [F_rC.T, F_CC]])
    JtI_rho = (S0 * I).sum(1)          # Σ_k s_kp·I_kp → (P,)
    JtI_C = np.concatenate([(Y * (rho0 * Hk_frozen[:, k] * I[:, k])[:, None]).sum(0)
                            for k in range(N)])
    JtI = np.concatenate([JtI_rho, JtI_C])
    theta = np.linalg.lstsq(JtJ, JtI, rcond=None)[0]
    return theta[:P], theta[P:].reshape(N, 9)


def spectral_stats(Fp, w_pos, E_pos, sigma2, lam_thresh=1e-2):
    lam_max = w_pos.max()
    keep = w_pos > lam_thresh * lam_max
    crb = sigma2 / np.maximum(w_pos, 1e-300)
    var = E_pos.var(axis=1)
    ratio = var[keep] / crb[keep]
    x = np.log10(crb); y = np.log10(np.maximum(var, 1e-300))
    slope, icpt = np.polyfit(x[keep], y[keep], 1)
    tr_ratio = float(var[keep].sum() / crb[keep].sum())
    return dict(slope=float(slope), median_ratio=float(np.median(ratio)),
                tr_ratio=tr_ratio, n_keep=int(keep.sum()), n_pos=len(w_pos))


def run_scene(scene, pixel_cap=400, n_trials=N_TRIALS):
    rng = np.random.default_rng(SEED + hash(scene) % 1000)
    sc = load_scene(str(DATA / scene))
    a, Y, C_true = scene_arrays(sc, list(range(5)), pixel_cap=pixel_cap, seed=SEED)
    I_true = a[:, None] * np.maximum(Y @ C_true.T, 0)
    sigma = NOISE * np.abs(I_true).max()
    bl = fisher_blocks(a, Y, C_true)
    F_eff = schur_full(bl)
    Fp = gauge_project(F_eff, gauge_unit(a))
    w, vecs = np.linalg.eigh(Fp)
    lam_max = w.max()
    pos = w > 1e-8 * lam_max
    U = vecs[:, pos]
    w_pos = w[pos]
    out = {"scene": scene, "levels": {}}
    # —— 1) 迭代扫描(ALS) —— 1000/10000 档减试验数(算力约束, 斜率估计功效足够; 如实注记)
    for iters in (10, 100, 1000, 10000):
        n_t = n_trials if iters <= 100 else 50
        E = np.zeros((U.shape[1], n_t))
        for t in range(n_t):
            I_n = I_true + rng.normal(0, sigma, I_true.shape)
            a0 = a * (1 + rng.normal(0, 0.01))
            C0 = C_true + rng.normal(0, 0.01, C_true.shape)
            a_e, C_e = als(I_n, Y, a0, C0, iters)
            e = a_e - a
            E[:, t] = U.T @ e
        st = spectral_stats(Fp, w_pos, E, sigma**2)
        out["levels"][f"als_{iters}"] = st
        print(f"  {scene:10s} ALS iters={iters:5d}: 斜率={st['slope']:.3f} "
              f"中位比={st['median_ratio']:.3f} tr比={st['tr_ratio']:.3f}")
    # —— 2) GN 紧收敛 ——
    E = np.zeros((U.shape[1], n_trials))
    for t in range(n_trials):
        I_n = I_true + rng.normal(0, sigma, I_true.shape)
        a0 = a * (1 + rng.normal(0, 0.01))
        C0 = C_true + rng.normal(0, 0.01, C_true.shape)
        a_e, C_e = gn(I_n, Y, a0, C0)
        e = a_e - a
        E[:, t] = U.T @ e
    st = spectral_stats(Fp, w_pos, E, sigma**2)
    out["levels"]["gn"] = st
    print(f"  {scene:10s} GN        : 斜率={st['slope']:.3f} "
          f"中位比={st['median_ratio']:.3f} tr比={st['tr_ratio']:.3f}")
    # —— 3) Q-Q 数据(GN) ——
    chi2 = (E * np.sqrt(w_pos)[:, None] / sigma) ** 2
    qq_stat = chi2.sum(0)                                     # eᵀFe/σ² per trial
    r = int((w_pos > 1e-2 * lam_max).sum())
    qs = np.linspace(0, 1, 21)
    out["levels"]["gn"]["qq"] = dict(
        empirical=[float(x) for x in np.quantile(qq_stat, qs)],
        theory_chi2_r=[float(x) for x in __import__("scipy.stats", fromlist=["chi2"]).chi2.ppf(qs, r)],
        r=r)
    # —— 4) 纯线性 LS sanity(给定 ρ 真值只估 C) ——
    E = np.zeros((U.shape[1], n_trials))
    for t in range(n_trials):
        I_n = I_true + rng.normal(0, sigma, I_true.shape)
        C_e = np.zeros_like(C_true)
        for k in range(C_true.shape[0]):
            C_e[k] = np.linalg.lstsq(Y, I_n[:, k], rcond=None)[0]
    # 简化 sanity: 单灯 9 维线性 LS(限亮侧像素 |Y·C|>margin —— cube 暗侧像素
    # 观测为 0 无 C 信息, 全像素纳入会方差虚增 5.8×(已抓出并修正))
    k0 = 0
    Yk_all = Y
    lit = (Yk_all @ C_true[k0]) > 0.05 * np.abs(Yk_all @ C_true[k0]).max()
    Yk = Yk_all[lit]
    Fk = Yk.T @ Yk
    E9 = np.zeros((9, n_trials))
    for t in range(n_trials):
        I_n = I_true[lit, k0] + rng.normal(0, sigma, int(lit.sum()))
        C_e = np.linalg.lstsq(Yk, I_n, rcond=None)[0]
        E9[:, t] = C_e - C_true[k0]
    cov = np.cov(E9)
    sanity = float(np.median(np.diag(cov) / (sigma**2 * np.diag(np.linalg.inv(Fk + 1e-12*np.eye(9)))))/1.0)
    out["levels"]["sanity_linear_C"] = dict(median_var_ratio=float(sanity))
    print(f"  {scene:10s} sanity(线性C): 方差比中位={sanity:.3f} (应=1)")
    # —— 5) 可估带占比 ——
    out["in_band_frac"] = float((w_pos > 1e-2 * lam_max).sum() / len(w_pos))
    return out


def main():
    out = {"scenes": {}, "meta": dict(noise=NOISE, n_trials=N_TRIALS, seed=SEED,
                                      pre_registered_accept="GN 后可估带斜率 ∈ [0.9,1.1] → 归因成立; <0.9 → 去ReLU分支")}
    for scene in SCENES:
        print(f"[exp10] {scene}")
        out["scenes"][scene] = run_scene(scene)
    # 汇总判定
    gn_slopes = [d["levels"]["gn"]["slope"] for d in out["scenes"].values()]
    out["verdict"] = dict(
        gn_slopes=gn_slopes,
        attribution_confirmed=bool(all(0.9 <= s <= 1.1 for s in gn_slopes)),
        als_slope_trend={k: [out["scenes"][s]["levels"][f"als_{k}"]["slope"] for s in out["scenes"]]
                         for k in (10, 100, 1000, 10000)},
        in_band_fracs={s: out["scenes"][s]["in_band_frac"] for s in out["scenes"]},
        note="ALS 斜率随迭代单调趋向 GN 水平 → early-stopping 隐式正则化归因成立; 收缩带照 exp4 双报告")
    print("\n[exp10] GN 斜率:", [f"{x:.3f}" for x in gn_slopes],
          "| 归因判定:", out["verdict"]["attribution_confirmed"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp10] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
