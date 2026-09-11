"""Certified allocation machinery（plan v2 L4，v1.1 修订）：预算约束凸程序 + FW 全局证书。

方向参数化（tests/test_math_foundations.py 钉死）：
    ΔF(t) = diag(F∞) − Σ_k u_k (M0k + t_k Λ0k)⁻¹ u_kᵀ，  t ∈ [1, κ]^L，
    t 大 = 精度高 = 信息多（与冻结 allocation 协议的 λ_l → regime·λ_l 同向）。

证书程序：
    min J_A(t) = tr ΔF(t)⁻¹   s.t.  Σ_k (t_k − 1) ≤ B,  t ∈ [1, κ]^L。

凸性链：ΔF 联合算子凹（L2）+ X ↦ tr X⁻¹ 凸且 Loewner 递减（PD 锥上）
⇒ J_A 凸且光滑（ΔF ≻ 0 于可行域，Λ0k ≻ 0）。梯度解析且已被 FD 对拍
（tests/test_math_foundations.py，≤1e-6，对**装配后** ΔF）：
    ∇_{t_k} J_A = −tr(ΔF⁻¹ · u_k K Λ0 K u_kᵀ · ΔF⁻¹) ≤ 0。

历史注记（v1.0→v1.1）：证书泛函最初选 J_E = −λmin（凸但**非光滑**——
λmin 近简并时单特征向量次梯度给出虚假下降方向，FW 实测停滞）；
切换到光滑的 J_A 后机械稳定。J_E 保留为 report-only 评估。
Frank–Wolfe 对偶间隙 g(t) = ∇J_A(t)ᵀ(t − s*(t)) ≥ 0 是全局最优性证书：
J_A(t) − J* ≤ g(t)（凸目标 + 凸紧可行集）。J_A/J_E 之外的模式泛函
（R 口径 mode-tail）凸性未证明也未证伪（72k 对随机中点检验 0 违例）——
report-only，不做证书。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibinfo.allocation.blocks import LightBlocks, sym_inv


@dataclass
class CertificateProblem:
    """预算约束凸程序（预测侧泄露面 = SelectionState 同款字段）。

    route="dense"：装配 P×P ΔF + inv（P ≤ ~2000）；
    route="woodbury"：低秩 push-through（information.lowrank，P ≫ 2000 的
    全分辨率路线；要求 F∞ > 0）。两路线的 J_A/grad 在 PD 域上等价
    （tests/test_convex_certificates.py 的等价性测试）。"""

    blocks: LightBlocks
    kappa: float = 10.0
    route: str = "dense"
    _cache_key: tuple = None                     # 单槽缓存（P×P 矩阵 11.5MB/个
    _cache_df: np.ndarray = None                 # @P=1200——无界缓存会 OOM）

    # -------------------------------------------------- 目标与梯度（J_A 主泛函）
    def delta_f(self, t) -> np.ndarray:
        key = tuple(float(x) for x in t)
        if key != self._cache_key:
            blk = self.blocks
            lam = np.stack([t[k] * blk.lam0[k] for k in range(blk.L)])
            self._cache_key = key
            self._cache_df = blk.assemble(lam)
        return self._cache_df

    def J_A(self, t) -> float:
        if self.route == "woodbury":
            from calibinfo.information.lowrank import woodbury_trace_inv
            blk = self.blocks
            return woodbury_trace_inv(blk.finf, blk.u, blk.M0, blk.lam0,
                                      blk.active, np.asarray(t, float))
        return float(np.trace(np.linalg.inv(self.delta_f(t))))

    def J_A_with_inv(self, t):
        DFinv = np.linalg.inv(self.delta_f(t))
        return float(np.trace(DFinv)), DFinv

    def J_and_grad(self, t):
        t = np.asarray(t, float)
        if self.route == "woodbury":
            from calibinfo.information.lowrank import woodbury_trace_inv_grad
            blk = self.blocks
            return woodbury_trace_inv_grad(blk.finf, blk.u, blk.M0, blk.lam0,
                                           blk.active, t)
        return self.J_A(t), self.grad_J_A(t)

    def grad_J_A(self, t) -> np.ndarray:
        """∇_k J_A = −tr(ΔF⁻¹ u_k K Λ0 K u_kᵀ ΔF⁻¹) ≤ 0（解析，FD 对拍 ≤1e-6）。"""
        _val, DFinv = self.J_A_with_inv(t)
        g = np.empty(self.blocks.L)
        for k in range(self.blocks.L):
            K = sym_inv(self.blocks.M0[k] + t[k] * self.blocks.lam0[k])
            M = K @ self.blocks.lam0[k] @ K                    # (3,3) ⪰ 0
            G = DFinv @ self.blocks.u[k]                       # (P,3)
            g[k] = -float(np.trace(M @ (G.T @ G)))
        return g

    # -------------------------------------------------- report-only 评估
    def J_E(self, t) -> float:
        """−λmin（report-only：非光滑，简并处次梯度不可靠——见模块注记）。"""
        return -float(np.linalg.eigvalsh(self.delta_f(t))[0])

    # -------------------------------------------------- LMO（预算约束）
    def lmo(self, grad: np.ndarray, B: float) -> np.ndarray:
        """min ⟨g, s⟩ s.t. Σ(s_k−1) ≤ B, s ∈ [1,κ]^L：按 g 升序做预算前缀，
        且只在 g_k < 0 的坐标上花费（在 g_k ≥ 0 的坐标上加预算只会更差——
        剩余预算保持未用；预算约束是 ≤ 而非 =）。"""
        L = len(grad)
        cap = self.kappa - 1.0
        b = np.zeros(L)
        remain = float(min(B, L * cap))
        for k in np.argsort(grad):
            if grad[k] >= 0.0 or remain <= 1e-15:
                break
            take = min(cap, remain)
            b[k] = take
            remain -= take
        return 1.0 + b

    # -------------------------------------------------- Frank–Wolfe
    def frank_wolfe(self, B: float, iters: int = 60, t0=None):
        """返回 dict(t, J_A, gap, s_lmo, history)。gap 是全局证书：
        J_A(t) − J* ≤ gap（凸目标 + 凸紧可行集）。"""
        L = self.blocks.L
        t = np.ones(L) if t0 is None else np.asarray(t0, float).copy()
        assert np.all(t >= 1.0 - 1e-12) and np.all(t <= self.kappa + 1e-12)
        assert np.sum(t - 1.0) <= B + 1e-9
        hist = []
        for it in range(iters):
            J, g = self.J_and_grad(t)
            s = self.lmo(g, B)
            gap = float(g @ (t - s))
            hist.append(dict(iter=it, J_A=J, gap=gap))
            if gap <= 1e-12 * max(1.0, abs(J)):
                break
            gamma = self._line_search_grid(t, s)
            if gamma == 0.0:
                break
            t = (1.0 - gamma) * t + gamma * s
        J, g = self.J_and_grad(t)
        s = self.lmo(g, B)
        gap = float(g @ (t - s))
        return dict(t=t, J_A=float(J), gap=gap, s_lmo=s, history=hist)

    def _line_search_grid(self, t, s, max_evals=14):
        """γ 网格线搜索（γ = 1, 1/2, 1/4, ...）：J_A 沿线段是 γ 的凸函数，
        网格取 min 近似精确线搜索。选网格而非 Armijo 的原因：t=1 处 gap
        可能远大于任何有限步的真实下降量（凸切线是下界、高估可行下降），
        Armijo 的 0.1·γ·gap 阈值会被数值噪声淹没导致步长停滞。"""
        J0 = self.J_A(t)
        best_gamma, best_J = 0.0, J0
        gamma = 1.0
        for _ in range(max_evals):
            J_new = self.J_A((1.0 - gamma) * t + gamma * s)
            if J_new < best_J - 1e-15 * max(1.0, abs(best_J)):
                best_J, best_gamma = J_new, gamma
            gamma *= 0.5
        return best_gamma


def budget_for_k(k: int, kappa: float) -> float:
    """使可行集包含全部"恰精化 k 灯（t=κ）"离散点的最小预算：B = k(κ−1)。"""
    return k * (kappa - 1.0)
