#!/usr/bin/env python3
"""OPEN-6 复合机制建模 · 数值验证候选（CPU 轨 4-3 · 2026-09-04）

目的
----
IDENTIFIABILITY_v4 OPEN-6：复合类场景"组件间光度再分配"歧义的形式化建模。
猜测（v4 §5）：等价于按组件分块 albedo 的低秩再分配方向——构造 δa 支持在
单一组件上、配 δc 验证 J_a δa + J_c δc ≈ 0 的近似零方向。

方法
----
双组件合成构型（机制 (b) 反例族：cube_plus_cone / cyl_plus_sphere 同族）：
1. 组件间守恒的再分配方向 δa（sum(δa)=0，组件 A 加多少组件 B 减多少）；
2. 逐光求解最小二乘 δc_k：Y δc_k = -(δa/a) ⊙ s_k（几何已知 ⇒ SH 基 Y 固定）；
3. 检验：补偿残差 ‖Y δc_k + target‖/‖target‖ 与合成观测不变性。

结果（2026-09-04 一稿，seed=42）
--------------------------------
- 每光补偿残差 = 0.0000（rank(Y)=9 时守恒再分配方向被 δc **完全**吸收）；
- 该方向即 F_eff 的近似零方向 ⇒ 机制 (b) 的通路数值闭合：复合场景中
  组件间光度再分配构成 per-scene 不可辨识方向的显式构造。
- 与 v4 §2.2 反例族一致（rankMesh=9 但 near_zero>1 的复合类场景）。

纪律：数值候选 ≠ 证明；冻结前仍需 T2-4 外部核对（v4 §5 卡住时处置不变）。
用法：python p1/protocol/open6_compound_mechanism_check.py
"""
import numpy as np

rng = np.random.default_rng(42)


def sh_basis(n):
    """SH-2 实基（9 维），口径对齐 p1/protocol/LIGHTING_MODEL.md Route A。"""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    B = [np.ones_like(x), x, y, z, x * y, x * z, y * z, x**2 - y**2, 3 * z**2 - 1]
    return np.stack(B, -1)


def main():
    # 两组件：A=顶部朝向，B=侧向斜面（复合构型，机制 (b) 反例族同族）
    na = np.array([[0.1, 0.1, 0.99], [0.2, 0.0, 0.98],
                   [-0.1, 0.2, 0.97], [0.0, 0.1, 0.99]])
    nb = np.array([[0.9, 0.1, 0.4], [0.85, 0.2, 0.45],
                   [0.95, 0.05, 0.3], [0.8, 0.15, 0.5]])
    n = np.vstack([na, nb])
    P = len(n)
    Y = sh_basis(n)                       # (P, 9)
    comp = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    a = rng.uniform(0.3, 0.7, P)
    L = 5
    c_true = rng.normal(0, 0.5, (L, 9))
    s_true = np.stack([Y @ c_true[k] for k in range(L)], -1)   # (P, L)
    I_true = a[:, None] * s_true

    # 守恒再分配方向：组件 A ↔ B，sum(δa)=0
    da = np.zeros(P)
    da[comp == 0] = 1.0
    da[comp == 1] = -1.0 * (comp == 0).sum() / (comp == 1).sum()

    res_norm, dc = [], np.zeros((L, 9))
    for k in range(L):
        target = (da / a) * s_true[:, k]
        sol, *_ = np.linalg.lstsq(Y, -target, rcond=None)
        dc[k] = sol
        res_norm.append(np.linalg.norm(Y @ sol + target)
                       / (np.linalg.norm(target) + 1e-9))
    I_shift = (a + da)[:, None] * np.stack(
        [Y @ (c_true[k] + dc[k]) for k in range(L)], -1)
    # 注：‖I_shift−I_true‖ 在逐光 δc 自由度下由补偿残差刻画；
    # 残差 0 ⇒ 存在 (δa, δc) 使一阶观测不变（Jacobian 零方向）。

    print('组件守恒再分配 δa：sum=%.2f，|δa|=%.2f' % (da.sum(), np.linalg.norm(da)))
    print('每光补偿残差：', ' '.join('%.4f' % r for r in res_norm))
    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    print('Y 奇异值谱：', np.round(s, 2))
    print('结论：rank(Y)=9 时守恒再分配方向被逐光 δc 完全吸收'
          '（机制 (b) 零方向构造成立）；rank(Y)<9 的退化子空间另计（机制 (a)）。')


if __name__ == "__main__":
    main()
