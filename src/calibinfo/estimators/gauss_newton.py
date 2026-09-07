"""B2 · gauss_newton：trf 信任域求解与解析 Jacobian 的数值验证工具。

来源：exp8r v3 joint_trf 的求解器配置 + exp8r_jac_validation.py 的 FD 一致性闸
（红队/任务书红线：解析 J 逐列 vs 中心差分 rel err <1e-5 才可信，实测 4e-9）。

红线 #11 落地：assert_data_driven_init —— 多起点入口断言初值不等于 GT
（exp12v3 教训：z_als = z_true.copy() 注释写"常数场"→ GT 初始化幻象）。
"""

from __future__ import annotations

import numpy as np


def assert_data_driven_init(init, gt, name="init", atol=1e-8, rtol=1e-8):
    """红线 #11：初始化不得等于 GT（GT 只准进评分/oracle 臂）。

    exp12v3 事故：真值初始化 + LM 7 迭代"收敛" 0.44°——多起点纪律从源头断言。
    """
    if gt is None:
        return
    init = np.asarray(init, float)
    gt = np.asarray(gt, float)
    assert init.shape == gt.shape, f"{name} 与 GT 形状不一致"
    assert not np.allclose(init, gt, atol=atol, rtol=rtol), \
        f"红线 #11 违例：{name} 等于 GT（真值初始化幻象，见 exp12v3 VOID）"


def lae(dirs_est, dirs_true):
    """平均光照方向角误差（度）——评分专用（GT 只进评分）。"""
    angs = []
    for k in range(dirs_true.shape[0]):
        c = np.clip(dirs_est[k] @ dirs_true[k], -1, 1)
        angs.append(np.degrees(np.arccos(c)))
    return float(np.mean(angs))


def check_jacobian_vs_fd(jac_fn, residual_fn, x0, h=1e-7, rel_tol=1e-5):
    """FD 一致性闸：解析 J 逐列 vs 中心差分。

    来源：exp8r_jac_validation.py 纪律（解析 J 入库前必过，rel err <1e-5，legacy 实测 4e-9）。
    返回 (max_rel_err, passed)；passed=False 时调用方必须停手修 J（红线）。
    分母 = max(‖FD 列‖, ‖解析列‖)——解析列为全零的列（如 clip 边界 m0=0）同样受检。
    """
    J = np.asarray(jac_fn(x0))
    worst = 0.0
    for j in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        fd = (np.asarray(residual_fn(xp)) - np.asarray(residual_fn(xm))) / (2 * h)
        denom = max(np.linalg.norm(fd), np.linalg.norm(J[:, j]), 1e-300)
        worst = max(worst, np.linalg.norm(J[:, j] - fd) / denom)
    return worst, worst < rel_tol
