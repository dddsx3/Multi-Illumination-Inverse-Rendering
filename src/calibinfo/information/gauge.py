"""B1 · gauge：Prop 2 gauge-lifting 精确谱闭式（绝对信息单位）。

来源：红队报告三 V2（红队报告三 §8 冻结包 Proposition 2）。
闭式：aᵀΔF(λ)a = Σᵢ αᵢ² sᵢ² λ/(sᵢ²+λ)，其中 (sᵢ², V) = eig(BᵀB)、α = Vᵀc̄，
前提 = gauge 恒等式 Aa = Bc̄（调用方保证——诊断场景构造时给出）。
两端：λ→0 线性抬升（系数 Σαᵢ²）；λ→∞ 饱和 ‖Aa‖² = ‖Bc̄‖²。

R2 修订（红队报告三 攻击二）：λ⋆ 只存在于绝对信息单位；本模块输出即绝对单位，
retention（归一化）读出属 retention.py，两个度量不得混用。
绑定测试：test_information_modules.py + test_v2_gauge_closed_form.py（红队 V 系列）。
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


def mu_floor(scene_callable, tol_rel=1e-12):
    """场景本征弱模式地板的操作性定义（占位接口，CI02 实现）。

    去除 gauge 子空间后，uncalibrated/near-uncalibrated 基线中的
    最小非 gauge Rayleigh/eigen floor。CI02（卡 C10）给出具体实现并绑定预注册阈值；
    本占位仅声明接口契约，防各实验自行口径漂移。
    """
    raise NotImplementedError("μ_floor 的操作性定义在卡 C10（CI02）实现")
