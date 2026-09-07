"""CI01 · 代数正确性、白化、参数化不变性（宪法 §4.1 冻结规格）。

Gate A 实验体。本模块实现规格的计算核；正式 run（卡 C07）按正式 config 放大
（q∈{1,3,9,36}、m/q 三区、Σ_y 两形态、≥5 seeds），pilot（卡 C05 冒烟）小规模跑通端到端。

主指标（预注册）：所有核心 identity 误差 <1e-10（MC 类除外）；
任何 identity 失败 → 停止后续实验先修 core（宪法 Gate A 失败动作）。

参数化纪律（宪法 §2.1）：δc = Jφ·φ, φ~N(0,Σ_φ)——φ 空间是基本参数化；
Λ=σ²Σ_φ⁻¹ 在 φ 空间恒良定义；Σ_c=JΣ_φJᵀ 在 c 空间可秩亏（此时 c 空间禁 inv）。

六项检查：
  1 dual_route       双路线逐元素相对误差（白化有效空间：Schur vs marginal×σ²）
  2 rank_deficient   Λ=0 且 B 秩亏（c 空间）：走 lstsq/pinv、M(0) 幂等
  3 parameterization φ-space profiled == c-space marginal（V4 恒等式，含低秩 Σ_c）
  4 scale_invariance (σ,Σ)→(kσ,k²Σ) retention 谱逐位不变
  5 scale_shift      σ 单独改变 → retention 谱必须移动（>0.1）
  6 retention_bounds 0 ≤ ρ ≤ 1
"""

from __future__ import annotations

import numpy as np

from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import delta_f, delta_f_marginal
from calibinfo.information.whitening import whiten_system

GATE = 1e-10


def _sh_basis(n):
    """SH-9 基（与红队脚本逐位一致的常用近似系数；photometric 结构用）。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    A0, A1, A2, A3, A4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
    return np.stack([A0 * np.ones_like(x), A1 * y, A1 * z, A1 * x,
                     A2 * x * y, A2 * y * z, A3 * (3 * z**2 - 1), A2 * x * z,
                     A4 * (x**2 - y**2)], axis=1)


def _rand_dirs(rng, n_, zmin=0.15):
    d = rng.normal(size=(n_, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) + zmin
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _build_case(rng, case):
    """构造 (A, B, J, Sig_phi, Sig_c, sigma, Sigma_y)。

    structure:
      - "random"（默认）：A/B 独立随机块（耦合弱 → shift_gate 用非空洞阈值）；
      - "photometric"：光度立体结构 A=D(s), B=D(aH)Y（红队 V5 同构，强耦合 + scale gauge，
        shift_gate 用 V5 校准的 0.1）。
    """
    m, q, n = case["m"], case["q"], case["n"]
    sigma = case.get("sigma", 0.1)
    if case.get("structure") == "photometric":
        P = m
        n_gt = _rand_dirs(rng, P)
        Y = _sh_basis(n_gt)
        a = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P))
        c = _sh_basis(_rand_dirs(rng, 1))[0]
        s_full = Y @ c
        H = (s_full > 0).astype(float)
        s = np.maximum(s_full, 0.0)
        A = np.diag(s)                                    # x = 反照率 a (P 维)
        B = (a * H)[:, None] * Y                           # δc = SH-9 光照 (9 维)
        J = np.eye(q)
        Sig_phi = 0.05 ** 2 * np.eye(q)                    # V5 各向同性先验
        Sig_c = Sig_phi.copy()
        Sigma_y = np.full(m, sigma ** 2)
        return A, B, J, Sig_phi, Sig_c, sigma, Sigma_y
    A = rng.normal(size=(m, n))
    r = case.get("rank_B") or q
    B0 = rng.normal(size=(m, r))
    # φ→c 映射取正交列（QR）：随机未归一 J 会把 cond(JᵀJ)≈cond(J)² 人工注入系统，
    # 使 identity 判定退化到 κ·eps 量级——物理参数映射按泛型良尺度构造（首轮正式
    # run 实测教训，见 CI01 memo）。
    J, _ = np.linalg.qr(rng.normal(size=(q, r)))
    J = J[:, :r]                                       # (q, r) 正交列
    B = B0 @ J.T                                       # c 空间设计矩阵（秩 r）
    Sig_phi = np.diag(rng.uniform(0.1, 1.0, size=r))
    Sig_c = J @ Sig_phi @ J.T                          # c 空间诱导先验（可秩亏）
    hetero = case.get("heteroscedastic", False)
    Sigma_y = rng.uniform(0.5 * sigma ** 2, 1.5 * sigma ** 2, size=m) if hetero \
        else np.full(m, sigma ** 2)
    return A, B, J, Sig_phi, Sig_c, sigma, Sigma_y


def run(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    k_scale = config.get("k_scale", 10.0)
    checks = []
    for ci, case in enumerate(config["cases"]):
        q = case["q"]
        r = case.get("rank_B") or q
        tag = f"{case['m']}x{q}x{case['n']}" + \
              (f"_r{r}" if r < q else "") + \
              ("_het" if case.get("heteroscedastic") else "")
        for seed_i in range(config.get("n_seeds", 1)):
            c_rng = np.random.default_rng(config["seed"] * 1000 + ci * 10 + seed_i)
            A, B, J, Sig_phi, Sig_c, sigma, Sigma_y = _build_case(c_rng, case)
            B_phi = B @ J                              # φ 空间有效设计矩阵 (m, r)

            # 白化（同方差/异方差统一入口）；有效 φ 空间 Λ_w = Σ_φ⁻¹（σ'²=1）
            Aw, Bw_phi, wmeta = whiten_system(A, B_phi, Sigma_y)
            Lam_w = np.linalg.inv(Sig_phi)

            # 1 双路线（白化 φ 有效空间，逐元素相对误差）
            DF, _M, _ = delta_f(Aw, Bw_phi, Lam_w)
            DFm, _ = delta_f_marginal(Aw, Bw_phi, Sig_phi, 1.0)
            rel_dual = float(np.linalg.norm(DF - DFm) / np.linalg.norm(DFm))

            # 2 Λ=0 秩亏（c 空间 B 秩 r<q 才触发；BᵀB 奇异 → lstsq/pinv 红线路径）
            rel_idem = None
            if r < q:
                _, Bw_c, _ = whiten_system(A, B, Sigma_y)
                DF0, M0, d0 = delta_f(Aw, Bw_c, 0.0)
                rel_idem = float(np.linalg.norm(M0 @ M0 - M0) / np.linalg.norm(M0))
                assert d0["singular"]

            # 3 参数化不变性（V4 恒等式，原空间 σ 口径，含低秩 Σ_c 的 c-marginal）
            DF_phi, _, _ = delta_f(A, B_phi, sigma ** 2 * np.linalg.inv(Sig_phi))
            DF_c, _ = delta_f_marginal(A, B, Sig_c, sigma)
            rel_param = float(np.linalg.norm(DF_phi - DF_c) / np.linalg.norm(DF_c))

            # 4/5 尺度自检 + 6 谱界（各向同性有效先验，λ 锚定工作带 s_med²/100——
            # 工作带纪律（红队 §2.2 谱截断）：BᵀB 的近零奇异值不入带；
            # 锚定保证"σ 单独改变 → continuum 移动"非空洞可检；读出已预注册）
            Finf = A.T @ A
            s2 = np.linalg.eigvalsh(B_phi.T @ B_phi)
            s_pos = s2[s2 > 1e-3 * max(s2.max(), 1e-300)]
            s_med2 = float(np.median(s_pos)) if s_pos.size else float(s2.max())
            sigma_base = float(np.sqrt(max(s_med2, 1e-12) / 100.0))   # λ_base = s_med²/100
            Sig_iso = np.eye(r)
            Lam_a = sigma_base ** 2 * np.linalg.inv(Sig_iso)
            Lam_b = (k_scale * sigma_base) ** 2 * np.linalg.inv(k_scale ** 2 * Sig_iso)
            Lam_s = (k_scale * sigma_base) ** 2 * np.linalg.inv(Sig_iso)
            rho0 = retention_spectrum(delta_f(A, B_phi, Lam_a)[0], Finf)["rho"]
            rho_k = retention_spectrum(delta_f(A, B_phi, Lam_b)[0], Finf)["rho"]
            rel_scale = float(np.abs(rho0 - rho_k).max())
            rho_s = retention_spectrum(delta_f(A, B_phi, Lam_s)[0], Finf)["rho"]
            shift = float(np.abs(rho0 - rho_s).max())
            bounds_ok = bool(min(rho0.min(), rho_k.min(), rho_s.min()) > -1e-9
                             and max(rho0.max(), rho_k.max(), rho_s.max()) < 1 + 1e-9)

            checks.append(dict(
                case=tag, seed_i=seed_i, cond_Sigma_y=wmeta["cond"],
                rel_err=rel_dual, rel_idempotent=rel_idem,
                rel_parameterization=rel_param,
                scale_invariance=rel_scale, scale_shift=shift,
                retention_bounds_ok=bounds_ok,
                gate_pass=bool(rel_dual < GATE and rel_param < GATE
                               and rel_scale < GATE
                               and (rel_idem is None or rel_idem < GATE)
                               and shift > case.get("shift_gate", 0.1)
                               and bounds_ok)))

    all_pass = all(r["gate_pass"] for r in checks)
    return dict(run_name=config.get("run_name", "run"),
                seed=config["seed"], gate=GATE, n_checks=len(checks),
                all_pass=bool(all_pass), checks=checks,
                note="CI01 预注册：任何 identity 失败 → 停止后续实验先修 core（宪法 Gate A）")
