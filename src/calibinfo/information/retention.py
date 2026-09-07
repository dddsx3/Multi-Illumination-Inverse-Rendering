"""B1 · retention：归一化 calibration-retention spectrum R(Λ)（Lemma 2）。

定义：R(Λ) = F∞^{-1/2} ΔF(Λ) F∞^{-1/2}，0 ≼ R ≼ I ⇒ 0 ≤ ρⱼ ≤ 1
（可识别子空间 range(F∞) 上）。
红线 RL-retention-whitening（红队报告三 攻击五）：F∞^{-1/2} = 正定平方根
（eigh 构造），禁 F∞^{-1/4} 等错误形式；F∞=diag(s²) 时才可写 diag(1/s)。
R2 修订：ρⱼ 只作 within-scene 归一化读出，禁用于 λ⋆/跨场景逐模式比较。
绑定测试：test_information_modules.py + test_v3_retention_bounds.py / test_v5_scale.py。
"""

from __future__ import annotations

import numpy as np


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
    rho = np.linalg.eigvalsh(R)
    tol = 1e-9
    return dict(rho=rho, basis=Vk, n_identifiable=int(keep.sum()),
                cond_Finf=float(wk.max() / wk.min()),
                bounds_ok=bool(rho.min() > -tol and rho.max() < 1.0 + tol))
