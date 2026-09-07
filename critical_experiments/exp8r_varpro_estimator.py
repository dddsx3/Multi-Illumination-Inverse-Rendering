#!/usr/bin/env python3
"""OPT-1 阶段 1 · VarPro(变量投影)估计器: exp8R v3 joint_trf 的数学等价替换
+ 已知答案单元测试(红线 #8, 先于一切)。

数学等价性声明
==============
joint_trf 残差(无权重): r[k,p] = I_obs[k,p] − ρ_p·α_k·clip(n_p·l̂_k, 0, None)
目标 cost = ½Σr²(scipy res.cost 同口径)。唯一非线性参数 = 方向 l̂_k(每光 2 DoF,
参数化 [x,y] → z=√(1−x²−y²), 与 joint_trf 完全一致)。固定方向后:
  ρ_p 逐像素闭式: ρ_p = Σ_k w α_k h I / Σ_k w (α_k h)²,  h=clip(n·l,0,None)
  α_k 逐光闭式:   α_k = Σ_p w ρ_p h I / Σ_p w ρ_p² h²
交替(ALS)收敛到剖面目标; 外层对 2N 维 [x,y] 做 GN。

【两可点 #1 · 权重口径, 留痕】
  现行 joint_trf 的 residual 实际【无权重】(a_/b_ 形参传入但只用于 diagnose_trace;
  residual=(I_sub.T−I_mod).T.ravel() 不含 w)。因此:
  - weighted=False(默认, 主口径): w≡1, 与 joint_trf 目标函数严格一致;
  - weighted=True: w=1/max(a_+b_·max(I,0),1e-6)(calibrate 噪声模型, 任务书公式
    口径), 仅作对照分支, 不进主等价判据。

【两可点 #2 · 尺度规范 (gauge), 留痕 · 实测发现】
  模型 I=ρ_p·α_k·h_kp 在 (ρ,α)→(cρ, α/c) 下严格不变 → 联合/剖面目标沿该 1 维
  规范方向完全平坦, ρ/α 各自不可辨识(仅乘积 ρ_p·α_k 可辨识)。单测实测: 从
  0.8·ρ_true 初始化, ALS 一步收敛到 (0.8ρ_t, 1.25α_t), 残差 ~1e-31(精确 0 贡献)。
  推论(阶段 2 必须遵守): VarPro 与 trf 的 x[:P](ρ 块)与 α 块【不可 raw 对照】,
  可对照的规范不变量 = 方向 / cost / LAE / 乘积矩阵 ρ_p·α_k。trf 同样有此规范
  (其 residual 同样不变), 二者在规范意义上等价。

【两可点 #3 · ALS 收敛判据读法, 留痕】
  题书 "相邻步相对 cost 降 <1e-12" 有两种读法: |Δcost|<1e-12(绝对) 或
  |Δcost|<1e-12·|cost|(相对)。本实现取【相对】: |Δcost| ≤ tol·max(|prev|,|cur|),
  cost→0 时退化为 200 步上限兜底(无害: ALS 便宜, 收敛到机器精度)。两种读法在
  真实数据 cost≥1e3 时数值一致(差因子 ~cost/1); 仅在 cost≪1(无噪声合成)时不同,
  相对读法更严。另: 数值差分求值(JAC)要求内层残差噪声 ≪ 差分信号 h·|J|~1e-6,
  故 JAC 求值附加参数变化判据 param_tol(默认 None 不启用; JAC 用 1e-12)。

【外层 Jacobian = 完整 VarPro chain-rule J, 留痕 · 实测教训】
  J 的差分列 = (r(θ+h e_j, c*(θ+h)) − r(θ−h e_j, c*(θ−h)))/(2h), 即差分时在扰动
  方向处【重新 ALS 收敛】(热启动), 自动包含 dc/dθ 链式项。若只差分固定 c 的偏
  Jacobian(缺链式项), GN 退化为线性收敛(实测: 0.05 rad 起点经 50 外层迭代仅到
  1.7e-4 rad, 不达 1e-6 rad 判据), 不可用。

h=0 边界: ALS 闭式自然 0 贡献(ρ 式分子含 h, α 式分母含 h²) → 与现行 clip 次梯度
(m0=(nl>0) 取 0)一致; GN 中心差分对 [x,y] 的导数在 h=0 像素处与解析式 [nl>0]
一致(clip 两侧导数均为 0 段)。

返回: VarProResult(与 scipy OptimizeResult 兼容, 至少 .x/.cost);
  x 布局与 joint_trf 完全一致: [ρ(P) | (α_k, x_k, y_k)×N].ravel()。

单元测试(本文件 main): 三项 (a)(b)(c), 全过打印 ALL PASS, 任一失败打印数字并
sys.exit(1)。阈值不放宽。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ----------------------------------------------------------------------------
# 几何/残差辅助
# ----------------------------------------------------------------------------
def _h(n_gt, dirs):
    """几何项 h = clip(n·l, 0, None), 形状 (P, N)。与 joint_trf 的 nl 完全同式。"""
    return np.clip(n_gt @ dirs.T, 0, None)


def _xy_to_dirs(xy):
    """[x,y](N,2) → 方向 (N,3), 与 joint_trf unpack 完全一致(含 1e-12 截断)。"""
    z = np.sqrt(np.maximum(1 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 1e-12))
    return np.column_stack([xy, z])


def _residual(I_sub, n_gt, dirs, rho, alphas):
    """残差 (P,N) 布局(joint_trf residual 的转置前形态)。"""
    h = _h(n_gt, dirs)
    I_mod = rho[:, None] * alphas[None, :] * h
    return I_sub.T - I_mod


def _profile_cost(I_sub, n_gt, dirs, rho, alphas, a_, b_, weighted=False):
    """剖面 cost = ½Σr²(与 joint_trf res.cost 同口径); weighted 加 w。"""
    r = _residual(I_sub, n_gt, dirs, rho, alphas)
    if weighted:
        w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub, 0), 1e-6)   # (N,P)
        return 0.5 * float((w * r.T ** 2).sum())
    return 0.5 * float((r ** 2).sum())


# ----------------------------------------------------------------------------
# 内层 ALS(闭式交替)
# ----------------------------------------------------------------------------
def als_closed_form(I_sub, n_gt, dirs, rho0, a_, b_, weighted=False,
                    n_iters=200, tol=1e-12, alpha_init=None, param_tol=None):
    """固定方向 dirs, ALS 闭式交替解 (ρ, α) 直至收敛。

    I_sub: (N,P); n_gt: (P,3); dirs: (N,3); rho0: (P,) ρ 初值;
    alpha_init: (N,) α 初值(热启动; None → 由 rho0 闭式一步)。
    收敛(两可点 #3 读法, 见模块 docstring):
      |Δcost| ≤ tol·max(|prev|,|cur|)(相对读法) 或
      param_tol 给定时 max 相对参数变化 ≤ param_tol 或 n_iters 用尽。
    返回 (rho, alphas, cost, n_steps); cost 与 joint_trf res.cost 同口径。
    """
    N, P = I_sub.shape
    rho = np.asarray(rho0, dtype=float).copy()
    h = _h(n_gt, dirs)                                  # (P,N)
    Icol = np.asarray(I_sub, dtype=float).T             # (P,N)
    wT = (1.0 / np.maximum(a_ + b_ * np.maximum(I_sub, 0), 1e-6)).T \
        if weighted else None                          # (P,N)

    def step_alpha(rho):
        # α_k = Σ_p w ρ_p h_kp I_kp / Σ_p w ρ_p² h_kp²
        if weighted:
            num = (wT * rho[:, None] * h * Icol).sum(0)
            den = (wT * (rho[:, None] ** 2) * h * h).sum(0)
        else:
            num = (rho[:, None] * h * Icol).sum(0)
            den = ((rho[:, None] ** 2) * h * h).sum(0)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def step_rho(alphas):
        # ρ_p = Σ_k w α_k h_kp I_kp / Σ_k w (α_k h_kp)²
        if weighted:
            num = (wT * (alphas[None, :] * h) * Icol).sum(1)
            den = (wT * (alphas[None, :] * h) ** 2).sum(1)
        else:
            num = ((alphas[None, :] * h) * Icol).sum(1)
            den = ((alphas[None, :] * h) ** 2).sum(1)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    alphas = step_alpha(rho) if alpha_init is None \
        else np.asarray(alpha_init, dtype=float).copy()

    prev_cost = np.inf
    cost = _profile_cost(I_sub, n_gt, dirs, rho, alphas, a_, b_, weighted)
    n_steps = 0
    for it in range(n_iters):
        rho_new = step_rho(alphas)
        al_new = step_alpha(rho_new)
        dr = np.max(np.abs(rho_new - rho)) / max(np.max(np.abs(rho)), 1e-300)
        da = np.max(np.abs(al_new - alphas)) / max(np.max(np.abs(alphas)), 1e-300)
        rho, alphas = rho_new, al_new
        cost = _profile_cost(I_sub, n_gt, dirs, rho, alphas, a_, b_, weighted)
        n_steps = it + 1
        done_cost = abs(prev_cost - cost) <= tol * max(abs(prev_cost), abs(cost))
        done_param = param_tol is not None and max(dr, da) <= param_tol
        if done_cost or done_param:
            break
        prev_cost = cost
    return rho, alphas, float(cost), n_steps


# ----------------------------------------------------------------------------
# 外层: 方向 GN(完整 VarPro chain-rule J, 数值中心差分 + Armijo)
# ----------------------------------------------------------------------------
@dataclass
class VarProResult:
    """与 scipy.optimize.OptimizeResult 兼容(至少 .x/.cost)。"""
    x: np.ndarray            # [ρ(P) | (α_k,x_k,y_k)×N].ravel(), 与 joint_trf 布局一致
    cost: float              # ½Σr²(与 res.cost 同口径)
    fun: np.ndarray          # 残差, 光主序 (N*P,) — 与 joint_trf 行序一致
    dirs: np.ndarray         # (N,3) 估计方向
    rho: np.ndarray          # (P,)
    alphas: np.ndarray       # (N,)
    n_outer: int
    n_fev: int               # 剖面(内层 ALS)求值次数
    als_steps_last: int
    converged: bool
    message: str


def varpro_estimate(I_sub, rho0, dirs0, n_gt, a_, b_, n_iters_outer=50,
                    weighted=False, gn_h=1e-6, als_iters=200, als_tol=1e-12,
                    verbose=False):
    """VarPro 估计器(主入口)。签名与 m.joint_trf(I_sub, rho0, dirs0, n_gt, a_, b_,
    n_iters=60) 对齐; 返回 .x/.cost, x 布局与 joint_trf 完全一致。

    外层: 方向参数 [x,y](2N 维) Gauss-Newton。
      J 列 = 数值中心差分(h=gn_h)【含链式项】: 在 θ±h·e_j 处热启动 ALS 重新收敛
      再差分残差(完整 VarPro J, 见模块 docstring 两可点 #4/教训)。
      GN 方程 (JᵀJ + λI)Δ = −Jᵀr, λ=1e-12·max(1,tr/2N) 纯数值兜底(λ→0 即纯 GN,
      不移动极小点)。
      Armijo 回溯: c1=1e-4, τ=0.5, 上限 40 步; 求值 = 内层 ALS(热启动)。
    收敛: 参数步 max|t·Δ| < 1e-12, 或 cost 平台(相对降 ≤1e-14), 或外层用尽。
    """
    N = I_sub.shape[0]
    P = n_gt.shape[0]
    I_sub = np.asarray(I_sub, dtype=float)
    rho0 = np.asarray(rho0, dtype=float)
    xy = np.asarray(dirs0, dtype=float)[:, :2].copy()    # (N,2)

    # JAC 求值用更紧的内层判据(差分卫生: 残差噪声须 ≪ h·|J|~1e-6; 两可点 #3)
    JAC_ITERS, JAC_TOL, JAC_PTOL = 1000, 1e-12, 1e-12

    n_fev = 0

    def profile(xyv, warm):
        nonlocal n_fev
        n_fev += 1
        d = _xy_to_dirs(xyv)
        if warm is None:
            rho, al, cost, st = als_closed_form(
                I_sub, n_gt, d, rho0, a_, b_, weighted,
                n_iters=als_iters, tol=als_tol)
        elif warm == "jac":
            # JAC 求值: 冷启动 rho0 + 参数判据(与 ± 两侧同源, 避免热启动不对称)
            rho, al, cost, st = als_closed_form(
                I_sub, n_gt, d, rho0, a_, b_, weighted,
                n_iters=JAC_ITERS, tol=JAC_TOL, param_tol=JAC_PTOL)
        else:
            rho_w, al_w = warm
            rho, al, cost, st = als_closed_form(
                I_sub, n_gt, d, rho_w, a_, b_, weighted,
                n_iters=als_iters, tol=als_tol, alpha_init=al_w)
        return cost, rho, al, d, st

    cost, rho, alphas, dirs, als_steps = profile(xy, None)

    converged = False
    message = "max outer iters reached"
    n_outer = 0
    for it in range(n_iters_outer):
        n_outer = it + 1
        # ---- 完整 VarPro J(中心差分, ±处 ALS 收敛含链式项) ----
        r_base = _residual(I_sub, n_gt, dirs, rho, alphas).T.ravel()  # (N*P,)
        cols = []
        for j in range(2 * N):
            xyp = xy.reshape(-1).copy(); xyp[j] += gn_h
            xym = xy.reshape(-1).copy(); xym[j] -= gn_h
            _, rp_, al_p, d_p, _ = profile(xyp.reshape(N, 2), "jac")
            _, rm_, al_m, d_m, _ = profile(xym.reshape(N, 2), "jac")
            rp = _residual(I_sub, n_gt, d_p, rp_, al_p).T.ravel()
            rm = _residual(I_sub, n_gt, d_m, rm_, al_m).T.ravel()
            cols.append((rp - rm) / (2 * gn_h))
        J = np.column_stack(cols)                       # (N*P, 2N)
        g = J.T @ r_base                                # 剖面梯度
        H = J.T @ J
        lam = 1e-12 * max(1.0, float(np.trace(H)) / max(2 * N, 1))
        try:
            delta = np.linalg.solve(H + lam * np.eye(2 * N), -g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(H, g, rcond=None)[0]
        step = delta.reshape(-1)

        # ---- Armijo 回溯 ----
        t = 1.0
        accepted = False
        c_prev = cost
        for _ in range(40):
            xy_new = np.clip(xy.reshape(-1) + t * step, -0.999, 0.999)
            c_new, rho_n, al_n, d_n, st_n = profile(xy_new.reshape(N, 2), (rho, alphas))
            if c_new <= cost + 1e-4 * t * float(step @ g):
                xy = xy_new.reshape(N, 2)
                rho, alphas, dirs, cost, als_steps = rho_n, al_n, d_n, c_new, st_n
                accepted = True
                break
            t *= 0.5
        if not accepted:
            converged = True
            message = "Armijo exhausted (local minimum)"
            break
        if np.max(np.abs(t * step)) < 1e-12:
            converged = True
            message = "parameter step < 1e-12"
            break
        if c_prev - cost <= 1e-14 * max(1.0, abs(cost)):
            converged = True
            message = "cost plateau (rel drop <= 1e-14)"
            break
        if verbose:
            print(f"    [outer {it+1}] cost={cost:.6e} |g|={np.linalg.norm(g):.3e} "
                  f"t={t:.3g} als_last={als_steps}")

    x = np.concatenate([rho, np.column_stack([alphas, xy]).ravel()])
    return VarProResult(
        x=x, cost=float(cost),
        fun=_residual(I_sub, n_gt, dirs, rho, alphas).T.ravel(),
        dirs=_xy_to_dirs(xy), rho=rho, alphas=alphas,
        n_outer=n_outer, n_fev=n_fev, als_steps_last=als_steps,
        converged=converged, message=message)


# ----------------------------------------------------------------------------
# 1a · 已知答案单测(红线 #8)
# ----------------------------------------------------------------------------
def _dense_rho_lstsq(I_sub, n_gt, dirs, alphas, w=None):
    """独立对照: 逐像素 1 参数稠密 lstsq 解 ρ_p(给定 α; w=None 无权)。"""
    h = _h(n_gt, dirs)
    out = np.empty(n_gt.shape[0])
    for p in range(n_gt.shape[0]):
        A = (alphas * h[p])[:, None]                    # (N,1)
        b = I_sub.T[p].copy()
        if w is not None:
            s = np.sqrt(w[:, p])
            A = A * s[:, None]; b = b * s
        out[p] = np.linalg.lstsq(A, b, rcond=None)[0][0]
    return out


def _dense_alpha_lstsq(I_sub, n_gt, dirs, rho, w=None):
    """独立对照: 逐光 1 参数稠密 lstsq 解 α_k(给定 ρ)。"""
    h = _h(n_gt, dirs)
    out = np.empty(I_sub.shape[0])
    for k in range(I_sub.shape[0]):
        A = (rho * h[:, k])[:, None]                    # (P,1)
        b = I_sub.T[:, k].copy()
        if w is not None:
            s = np.sqrt(w[k])
            A = A * s[:, None]; b = b * s
        out[k] = np.linalg.lstsq(A, b, rcond=None)[0][0]
    return out


def _angle_rad(d_est, d_true):
    """逐光夹角(rad)。"""
    return [float(np.arccos(np.clip(d_est[k] @ d_true[k], -1, 1)))
            for k in range(d_true.shape[0])]


def test_known_answer(seed=20260907, P=50, N=3):
    """已知答案单测, 三项, 阈值不放宽。

    (a) 固定真方向, 无噪声正演: ALS 闭式交替 1 步精确恢复 ρ/α(rel err<1e-10;
        与稠密(加权)lstsq 对照; 无权 + 加权两分支都验)。
        注(两可点 #2): 模型有尺度规范 (ρ,α)→(cρ,α/c), "恢复 ρ/α" 仅当初值位于
        真值规范时良定义 → (a) 从 ρ_true 初始化(检验闭式代数本身);
        规范现象另行在 (a4) 显式记录(乘积恢复 + 残差 0)。
    (b) 全变体(真方向 + 恢复的 ρ/α)重构残差 < 1e-12。
    (c) 外层 GN 从 0.05 rad 扰动起点恢复方向 < 1e-6 rad(与真值夹角, 逐光最大)。
    """
    print("=" * 74)
    print(f"[1a] 已知答案单测: 合成 P={P}, N={N}, seed={seed}")
    print("=" * 74)
    rng = np.random.default_rng(seed)

    def draw_hemisphere(size):
        v = rng.normal(size=(size, 3))
        v[:, 2] = np.abs(v[:, 2]) + 0.5
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # 拒绝采样保证可辨识性(留痕): 每像素 ≥1 光照亮(否则 ρ_p 不可辨识, 闭式得 0,
    # 已知答案测试不成立); 每光 ≥1 照亮像素(α 可辨识)。分布仍为"随机 z>0"。
    for _ in range(1000):
        n_gt = draw_hemisphere(P)
        dirs_true = draw_hemisphere(N)
        h0 = n_gt @ dirs_true.T
        if (h0 > 0).sum(1).min() >= 1 and (h0 > 0).sum(0).min() >= 1:
            break
    else:
        print("  [INC] 采样失败(1000 次拒绝)"); sys.exit(1)

    rho_true = rng.uniform(0.05, 0.5, P)
    alpha_true = rng.uniform(0.5, 2.0, N)
    h_true = np.clip(n_gt @ dirs_true.T, 0, None)             # (P,N)
    I_syn = rho_true[:, None] * alpha_true[None, :] * h_true   # (P,N)
    I_sub = I_syn.T                                           # (N,P) joint_trf 输入布局

    ok_all = True

    # ---------------- (a) 固定真方向, ALS 1 步精确恢复 ----------------
    # 无噪声 → (ρ_true, α_true) 是联合最优(cost 0); 从 ρ_true 初始化, 一步交替
    # 后 ρ/α 均应达机器精度(闭式代数正确性检验)。
    rho_a, al_a, cost_a, steps_a = als_closed_form(
        I_sub, n_gt, dirs_true, rho_true, 1.0, 0.0, weighted=False, n_iters=1)
    rel_rho = float(np.max(np.abs(rho_a - rho_true) / np.abs(rho_true)))
    rel_alpha = float(np.max(np.abs(al_a - alpha_true) / np.abs(alpha_true)))
    # 稠密 lstsq 对照(独立实现路径)
    rho_ref = _dense_rho_lstsq(I_sub, n_gt, dirs_true, alpha_true)
    al_ref = _dense_alpha_lstsq(I_sub, n_gt, dirs_true, rho_true)
    d_rho_ref = float(np.max(np.abs(rho_a - rho_ref)))
    d_alpha_ref = float(np.max(np.abs(al_a - al_ref)))
    ok_a = rel_rho < 1e-10 and rel_alpha < 1e-10 \
        and d_rho_ref < 1e-10 and d_alpha_ref < 1e-10
    ok_all &= ok_a
    print(f"(a1) ALS 1 步 @ 真方向(无权): steps={steps_a} cost={cost_a:.3e}")
    print(f"     rel_err(ρ)={rel_rho:.3e}  rel_err(α)={rel_alpha:.3e}  (阈值 1e-10)")
    print(f"     vs 稠密 lstsq: max|ρ−ρ_ref|={d_rho_ref:.3e}  max|α−α_ref|={d_alpha_ref:.3e}  (阈值 1e-10)")
    print(f"     → {'PASS' if ok_a else 'FAIL'}")

    # 加权分支(噪声权重非平凡 a=0.37, b=1.3): 同样应精确恢复(噪声 0 → 最优不变)
    aw, bw = 0.37, 1.3
    rho_aw, al_aw, cost_aw, _ = als_closed_form(
        I_sub, n_gt, dirs_true, rho_true, aw, bw, weighted=True, n_iters=1)
    rel_rho_w = float(np.max(np.abs(rho_aw - rho_true) / np.abs(rho_true)))
    rel_alpha_w = float(np.max(np.abs(al_aw - alpha_true) / np.abs(alpha_true)))
    w_full = 1.0 / np.maximum(aw + bw * np.maximum(I_sub, 0), 1e-6)   # (N,P)
    rho_ref_w = _dense_rho_lstsq(I_sub, n_gt, dirs_true, alpha_true, w=w_full)
    al_ref_w = _dense_alpha_lstsq(I_sub, n_gt, dirs_true, rho_true, w=w_full)
    d_rho_w = float(np.max(np.abs(rho_aw - rho_ref_w)))
    d_alpha_w = float(np.max(np.abs(al_aw - al_ref_w)))
    ok_aw = rel_rho_w < 1e-10 and rel_alpha_w < 1e-10 \
        and d_rho_w < 1e-10 and d_alpha_w < 1e-10
    ok_all &= ok_aw
    print(f"(a2) 加权分支(a={aw},b={bw}): rel_err(ρ)={rel_rho_w:.3e} rel_err(α)={rel_alpha_w:.3e}")
    print(f"     vs 稠密加权 lstsq: |ρ−ρ_ref|={d_rho_w:.3e} |α−α_ref|={d_alpha_w:.3e}  (阈值 1e-10)")
    print(f"     → {'PASS' if ok_aw else 'FAIL'}")

    # (a4) 尺度规范现象显式记录(两可点 #2): 偏离规范初值 → 收敛到规范等价点,
    # 乘积/残差精确, ρ/α 各自偏移(预期 ~0.8/1.25 因子)。
    rho_g, al_g, cost_g, steps_g = als_closed_form(
        I_sub, n_gt, dirs_true, rho_true * 0.8, 1.0, 0.0, weighted=False)
    prod_true = rho_true[:, None] * alpha_true[None, :]
    prod_g = rho_g[:, None] * al_g[None, :]
    rel_prod = float(np.max(np.abs(prod_g - prod_true) / np.maximum(np.abs(prod_true), 1e-300)))
    resid_g = float(np.max(np.abs(I_sub.T - prod_g * h_true)))
    rho_shift = float(np.median(rho_g / rho_true))
    alpha_shift = float(np.median(al_g / alpha_true))
    print(f"(a4) 规范留痕: 0.8·ρ 初始化 → 收敛 (ρ/ρ_t 中位={rho_shift:.4f}, "
          f"α/α_t 中位={alpha_shift:.4f}); 乘积 rel_err={rel_prod:.3e} "
          f"重构残差={resid_g:.3e} (规范现象, 非 FAIL 项)")

    # ---------------- (b) 全变体重构残差 ----------------
    ok_b = resid_g < 1e-12
    ok_all &= ok_b
    print(f"(b)  全变体重构(真方向+恢复 ρ/α): max|I−I_recon|={resid_g:.3e} "
          f"(阈值 1e-12) → {'PASS' if ok_b else 'FAIL'}")

    # ---------------- (c) 外层 GN 从 0.05 rad 扰动恢复方向 ----------------
    rng_p = np.random.default_rng(seed + 1)
    dirs0 = np.empty_like(dirs_true)
    for k in range(N):
        v = rng_p.normal(size=3)
        v = v - (v @ dirs_true[k]) * dirs_true[k]       # 切向
        v /= np.linalg.norm(v)
        ang = 0.05
        dirs0[k] = np.cos(ang) * dirs_true[k] + np.sin(ang) * v
    res = varpro_estimate(I_sub, rho_true * 0.8, dirs0, n_gt, 1.0, 0.0,
                          n_iters_outer=50)
    angs = _angle_rad(res.dirs, dirs_true)
    ang_max = max(angs); ang_mean = float(np.mean(angs))
    ok_c = ang_max < 1e-6
    ok_all &= ok_c
    print(f"(c)  GN 恢复方向(0.05 rad 扰动起点): max 夹角={ang_max:.3e} rad "
          f"(mean={ang_mean:.3e}) (阈值 1e-6 rad) → {'PASS' if ok_c else 'FAIL'}")
    print(f"     外层 {res.n_outer} 迭代({res.message}), 剖面求值 {res.n_fev} 次, "
          f"末次 ALS {res.als_steps_last} 步, cost={res.cost:.3e}")

    print("-" * 74)
    if ok_all:
        print("ALL PASS")
        return True
    print(f"FAILED: a1={'PASS' if rel_rho < 1e-10 and rel_alpha < 1e-10 else 'FAIL'} "
          f"a2={'PASS' if ok_aw else 'FAIL'} b={'PASS' if ok_b else 'FAIL'} "
          f"c={'PASS' if ok_c else 'FAIL'} — 数字见上, 不放宽阈值")
    sys.exit(1)


def main():
    test_known_answer()


if __name__ == "__main__":
    main()
