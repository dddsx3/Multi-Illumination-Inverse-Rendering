"""B1 · gauge：Prop 2 gauge-lifting 精确谱闭式（绝对信息单位）。

来源：V2（Proposition 2）。
闭式：aᵀΔF(λ)a = Σᵢ αᵢ² sᵢ² λ/(sᵢ²+λ)，其中 (sᵢ², V) = eig(BᵀB)、α = Vᵀc̄，
前提 = gauge 恒等式 Aa = Bc̄（调用方保证——诊断场景构造时给出）。
两端：λ→0 线性抬升（系数 Σαᵢ²）；λ→∞ 饱和 ‖Aa‖² = ‖Bc̄‖²。

（二）：λ⋆ 的正式名称是 **directional crossover precision**（数学冻结 v1.0 §29）：
"某固定 gauge direction 的 Rayleigh information 达到给定参考 floor μ 所需的
calibration precision"，存在条件 0 < μ < T（T = ‖Aa‖²）。只作 within-scene
绝对信息读出，禁止解释为普遍阈值 / 最小特征交接点 / real-world validated
predictor / 跨场景绝对可比阈值。λ⋆ 只存在于绝对信息单位；本模块输出即绝对
单位，retention（归一化）读出属 retention.py，两个度量不得混用。
绑定测试：test_information_modules.py + test_v2_gauge_closed_form.py（V 系列）。
"""

from __future__ import annotations

import numpy as np


def gauge_response(B, cbar, lam, tol_rel=1e-12):
    """gauge 方向的绝对信息谱响应（闭式）。

    参数
    ----
    B : (m, q) nuisance 设计矩阵（白化后）
    cbar : (q,) gauge nuisance 方向（满足 Aa = Bc̄）
    lam : 标量或 (k,) 数组——各向同性 Λ=λI 的 λ

    返回
    ----
    dict(
      exact      : λ 处 aᵀΔF(λ)a（标量或 (k,)）
      slope      : 小 λ 线性系数 Σαᵢ²（sᵢ²>tol 部分）
      saturation : ‖Bc̄‖² = Σαᵢ²sᵢ²（λ→∞ 极限）
      alpha      : (q,) Vᵀc̄
      eigs       : (q,) eig(BᵀB)（= sᵢ²）
    )
    """
    B = np.asarray(B, float)
    cbar = np.asarray(cbar, float)
    m, q = B.shape
    assert cbar.shape == (q,)
    eigs, V = np.linalg.eigh(B.T @ B)
    alpha = V.T @ cbar
    lam_arr = np.atleast_1d(np.asarray(lam, float))
    exact = np.array([np.sum(alpha ** 2 * eigs * l / (eigs + l)) for l in lam_arr])
    pos = eigs > tol_rel * max(eigs.max(), 1e-300)
    slope = float(np.sum((alpha ** 2)[pos]))
    saturation = float(np.sum(alpha ** 2 * eigs))
    out = dict(exact=exact if np.ndim(lam) else float(exact[0]),
               slope=slope, saturation=saturation, alpha=alpha, eigs=eigs)
    return out
