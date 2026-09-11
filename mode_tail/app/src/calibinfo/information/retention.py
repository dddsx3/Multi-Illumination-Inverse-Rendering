"""B1 · retention：归一化 calibration-retention spectrum R(Λ)（Lemma 2）。

定义：R(Λ) = F∞^{-1/2} ΔF(Λ) F∞^{-1/2}，0 ≼ R ≼ I ⇒ 0 ≤ ρⱼ ≤ 1
（可识别子空间 range(F∞) 上）。
retention whitening：F∞^{-1/2} = 正定平方根
（eigh 构造），禁 F∞^{-1/4} 等错误形式；F∞=diag(s²) 时才可写 diag(1/s)。
ρⱼ 只作 within-scene 归一化读出，禁用于 λ⋆/跨场景逐模式比较。
绑定测试：test_information_modules.py + test_v3_retention_bounds.py / test_v5_scale.py。
"""

from __future__ import annotations

import numpy as np


def retention_spectrum_gen_eig(DeltaF, Finf):
    """独立路线 B：generalized eigenvalue（矩阵束 (ΔF, F∞)，scipy.linalg.eigh
    走 LAPACK Cholesky 归约——与白化路线的 eigh 平方根代数独立）。

    仅当 F∞ 数值正定时可用（Cholesky 前提）；F∞ 秩亏时本路线不可用，
    白化路线限制到 range(F∞) 仍是唯一读出（Lemma 2 的秩亏处置）。
    返回 (rho, ok)；ok=False 表示 F∞ 非正定、路线不可用。
    """
    from scipy.linalg import eigh as _gen_eigh
    DeltaF = np.asarray(DeltaF, float)
    Finf = np.asarray(Finf, float)
    n = Finf.shape[0]
    assert DeltaF.shape == (n, n)
    Finf = 0.5 * (Finf + Finf.T)
    try:
        rho = _gen_eigh(DeltaF, Finf, eigvals_only=True)
    except np.linalg.LinAlgError:
        return None, False
    return np.sort(rho), True


def retention_spectrum_dual(DeltaF, Finf, tol_rel=1e-12):
    """双路线交叉验证（Lemma 2 审计）：白化（正定平方根）vs generalized eig。

    F∞ 正定时两路线应逐元素一致 rel<1e-10（CI02/11 验收）；
    返回 dict(rho, dual_rel, gen_eig_available)。
    """
    out = retention_spectrum(DeltaF, Finf, tol_rel=tol_rel)
    rho_g, ok = retention_spectrum_gen_eig(DeltaF, Finf)
    dual_rel = None
    if ok and rho_g.size == out["rho"].size:
        dual_rel = float(np.abs(out["rho"] - rho_g).max()
                        / max(np.abs(rho_g).max(), 1e-300))
    out["gen_eig_available"] = bool(ok)
    out["dual_rel"] = dual_rel
    return out


def retention_spectrum(DeltaF, Finf, tol_rel=1e-12):
    """限制到可识别子空间的 retention 谱。

    参数
    ----
    DeltaF : (n, n) ΔF(Λ)（任意 Λ）
    Finf   : (n, n) F∞ = AᵀA（校准极限 Fisher）
    tol_rel: range(F∞) 截断容差（相对最大特征值）

    返回
    ----
    dict(
      rho             : (k,) R 的特征值（升序），k = rank(F∞)；名义 ∈ [0,1]
      basis           : (n, k) 可识别子空间正交基（eigh(F∞) 的正特征向量）
      modes           : (n, k) R 的特征向量（与 rho 升序一一对应；dual 坐标
                        z_j = u_jᵀ F∞^{1/2} e 用列 u_j——注意 basis 与 modes
                        是两组不同的向量，禁止混配）
      n_identifiable  : k
      cond_Finf       : F∞ 可识别部分的谱条件数
      bounds_ok       : rho ∈ [−tol, 1+tol] 的布尔自检（不抛错，由调用方决定处置）
    )
    """
    DeltaF = np.asarray(DeltaF, float)
    Finf = np.asarray(Finf, float)
    n = Finf.shape[0]
    assert DeltaF.shape == (n, n)
    w, V = np.linalg.eigh(Finf)
    keep = w > tol_rel * max(w.max(), 1e-300)
    Vk = V[:, keep]
    wk = w[keep]
    Fh = Vk @ np.diag(1.0 / np.sqrt(wk)) @ Vk.T        # 正定平方根（限制到 range）
    R = Fh @ DeltaF @ Fh
    rho, U = np.linalg.eigh(0.5 * (R + R.T))           # 升序；U 列 = retention modes
    tol = 1e-9
    return dict(rho=rho, basis=Vk, modes=U, n_identifiable=int(keep.sum()),
                cond_Finf=float(wk.max() / wk.min()),
                bounds_ok=bool(rho.min() > -tol and rho.max() < 1.0 + tol))
