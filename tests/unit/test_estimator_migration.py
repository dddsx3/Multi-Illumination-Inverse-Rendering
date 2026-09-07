"""B2 绑定测试 · 估计器迁移（exp8R v3 合格估计器侧 → estimators/joint_map.py）。

来源：REPO_MIGRATION B2 行 1/3。绑定：
  (a) 解析稀疏 Jacobian vs 中心差分 FD 闸（rel<1e-5；legacy 实测 4e-9）；
  (b) 已知答案：合成场景（已知 ρ/α/方向 + 噪声）→ ALS 数据驱动初值 + trf 多起点
      → 方向恢复 LAE 小；cost 达噪声地板量级；
  (c) 红线 #11：assert_data_driven_init —— GT 初值必须被断言拦截（exp12v3 VOID 教训）。
"""

import numpy as np
import pytest

from calibinfo.estimators.gauss_newton import (
    assert_data_driven_init,
    check_jacobian_vs_fd,
    lae,
)
from calibinfo.estimators.joint_map import (
    als_init,
    joint_map,
    joint_residual,
    jac_r_joint_sparse,
)

SEED = 20260906          # exp8R v3 协议种子


def _synthetic_obs(rng, P=200, N=3, sigma=0.01):
    """已知答案场景：朗伯模型 I = ρ·α·relu(n·l) + 噪声（与 exp8R 模型类一致）。"""
    n_gt = rng.normal(size=(P, 3))
    n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True)
    n_gt[:, 2] = np.abs(n_gt[:, 2]) + 0.15
    n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True)
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 2] = np.abs(dirs[:, 2]) + 0.15
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rho = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P))
    alphas = rng.uniform(0.8, 1.2, size=N)
    nl = np.clip(n_gt @ dirs.T, 0, None)
    I_obs = (rho[:, None] * alphas[None, :] * nl).T \
        + rng.normal(0, sigma, size=(N, P))
    return n_gt, dirs, rho, alphas, I_obs


def test_analytic_jacobian_vs_fd():
    """FD 一致性闸：解析稀疏 J 逐列 vs 中心差分，rel<1e-5。"""
    rng = np.random.default_rng(SEED)
    n_gt, dirs, rho, alphas, I_obs = _synthetic_obs(rng, P=40, N=2)
    P, N = 40, 2
    x0 = np.concatenate([rho,
                          np.column_stack([alphas, dirs[:, :2]]).ravel()])
    worst, passed = check_jacobian_vs_fd(
        lambda x: jac_r_joint_sparse(x, P, n_gt).toarray(),
        lambda x: joint_residual(x, I_obs, n_gt),
        x0)
    assert passed, f"FD 闸失败: max rel err = {worst:.2e}（红线：修 J）"
    assert worst < 1e-5


def test_known_answer_recovery():
    """已知答案：数据驱动初值（逐光 lstsq + ALS）+ 多起点 trf（cost 选优，GT 不进选择）
    → 方向 LAE < 2°、ρ·α 乘积口径相对误差 < 5%、cost 达噪声地板量级。"""
    rng = np.random.default_rng(SEED)
    n_gt, dirs, rho, alphas, I_obs = _synthetic_obs(rng)
    N, P = I_obs.shape[0], n_gt.shape[0]
    # 数据驱动方向初值：每光 lstsq(n_gt, I_k) 归一（ρ≈常数的一阶近似，禁 GT）
    dirs0 = []
    for k in range(N):
        l = np.linalg.lstsq(n_gt, I_obs[k], rcond=None)[0]
        nn = np.linalg.norm(l)
        dirs0.append(l / nn if nn > 1e-9 else np.array([0.3, 0.3, 0.9]))
    dirs0 = np.array(dirs0)
    assert not np.allclose(dirs0, dirs)
    rho0 = np.maximum(I_obs.mean(0), 1e-3)
    rho_als, dirs_als = als_init(I_obs, rho0, dirs0, n_gt)
    best = None
    for s in range(5):
        d0 = dirs_als + rng.normal(0, 0.05, dirs_als.shape) if s > 0 else dirs_als
        d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        fit = joint_map(I_obs, rho_als, d0, n_gt)
        if best is None or fit["cost"] < best["cost"]:
            best = fit
    assert lae(best["dirs"], dirs) < 2.0
    # cost 达噪声地板量级（0.5·N·P·σ² 的 3 倍内）
    assert best["cost"] < 0.5 * N * P * (0.02 ** 2) * 3
    # ρ 恢复：尺度 gauge（ρ·α 乘积规范）下乘积口径校验
    nl = np.clip(n_gt @ best["dirs"].T, 0, None)
    prod = best["rho"][:, None] * best["alphas"][None, :]
    prod_true = rho[:, None] * alphas[None, :]
    rel = np.linalg.norm(prod * nl - prod_true * nl) / np.linalg.norm(prod_true * nl)
    assert rel < 0.05


def test_redline11_gt_init_blocked():
    """红线 #11：真值初始化必须被断言拦截（exp12v3 事故的机械防线）。"""
    rng = np.random.default_rng(SEED)
    n_gt, dirs, rho, alphas, I_obs = _synthetic_obs(rng, P=30, N=2)
    with pytest.raises(AssertionError):
        assert_data_driven_init(dirs, dirs)          # GT 假扮初值 → 必须炸
    assert_data_driven_init(dirs * 0.9, dirs)        # 数据驱动扰动初值 → 放行
