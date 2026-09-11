"""CI04 · 受控真实 corruption。

协议（估计器侧 corruption——真实数据的受控注入形态，跑前预注册）：
  真值 = GT 灯位/强度（OpenIllumination light_pos.npy，校准 GT 齐备——
  这是选它做 CI04 的原因）；数据固定不重渲染；corruption 注入**估计器的
  假设校准** δc~N(0,Σ_c)，即理论量化的"校准不确定度"本体（HCRB 因果链）。

  nominal（corruption 前，GT/nominal 计算，禁反调）：
    1. 标定朗伯 PS（GT 灯已知，逐像素 (ρ̂,n̂) 交替最小二乘，数据驱动初值）；
    2. 线性化资产：A_k=D(ŝ_k)、B_k=ρ̂h·[∂/∂logI, ∂/∂θ, ∂/∂ψ]（物理单位 φ 空间）；
    3. 白化（异方差 (a,b)）→ 每灯 Λ'_k=Σ_φ⁻¹ → Σ_k delta_f 逐灯和（V1b 恒等式，
       block-diag nuisance 每灯独立 3 维）→ ΔF(Σ_c) 与基线 F∞=Σ_k D(ŝ)²；
    4. 预测退化：弱 5 模式（R = F∞^{-1/2}ΔF F∞^{-1/2} 广义 retention 谱基）
       deg_pred_j = 1/ρ_j ≥ 1（ρ_j = 广义 retention 特征值；禁止写成普通
       特征值之比 λ_j(F∞)/λ_j(ΔF)——两矩阵不共享特征向量，见冻结 v1.0 §17）。
  empirical（每 level × seed）：
    5. 注入 δc（方向旋转 + 强度对数扰动，物理单位）→ 假设校准被污染；
    6. 同一估计器（固定 n̂ 的白化逐像素 GLS）重估 ρ̃ → 误差 e=ρ̃−ρ̂ 投影到弱模式；
    7. 对照臂 δc=0（同 seeds）给经验基线方差 → deg_emp_j = var_j/var0_j。
  Main check (H5): Spearman + bootstrap 95% CI between deg_pred and deg_emp
    + 一个全局仿射尺度；Σ_c_eff 敏感性带记入 summary。

dev/test 纪律：pilot 只用 development 8 对象（修协议/bug，禁动判据）；
正式 run（C16）冻结 test 对象清单后才跑全量。

数学冻结 v1.0 correctness gate（M0-1/M0-2）：noise_fit 与模式投影各有
legacy（冻结基准逐位复现口径，默认）与 corrected 两个 convention——
NominalScene(noise_fit_convention=...) 与 predicted_degradation(mode_coordinate=...)
显式选择；A/B/C/D factorial 复算见 experiments/openillumination_factorial.py。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from calibinfo.datasets.openillumination import load_object
from calibinfo.information.schur import delta_f
from calibinfo.models.corruption import CorruptionGenerator, rotate_dirs

N_ANALYSIS_LIGHTS = 48
N_MODES = 5
N_PIXEL_SUB = 1200


# ---------------------------------------------------------------- nominal 标定
def calibrated_ps(I, dirs, iters=25, w_floor=1e-3):
    """GT 灯已知的逐像素朗伯 PS（交替最小二乘；数据驱动初值 n=(0,0,1)）。

    I: (L,P) 灰度；dirs: (L,3)。返回 (rho (P,), n (P,3))。
    """
    L, P = I.shape
    n = np.tile(np.array([0.0, 0.0, 1.0]), (P, 1))
    rho = np.maximum(I.mean(0), 1e-6)
    for _ in range(iters):
        ndl = np.clip(n @ dirs.T, 0, None)                   # (P,L)
        rho = (ndl * I.T).sum(1) / np.maximum((ndl * ndl).sum(1), 1e-12)
        rho = np.maximum(rho, 1e-6)
        # n 更新：逐像素 LSQ（act 灯上 I/ρ ≈ n·d）
        act = (I.T > 1e-3) & (ndl > 0)                       # (P,L)
        n = np.zeros((P, 3))
        for p in range(P):
            m_ = act[p]
            if m_.sum() >= 3:
                Dk = dirs[m_]
                bk = I[m_, p] / rho[p]
                v = np.linalg.lstsq(Dk, bk, rcond=None)[0]
                nv = np.linalg.norm(v)
                n[p] = v / nv if nv > 1e-9 else np.array([0.0, 0.0, 1.0])
            else:
                n[p] = np.array([0.0, 0.0, 1.0])
    return rho, n


def noise_fit(I, s_hat, convention="legacy"):
    """异方差噪声模型 (a,b)：Var ≈ a + b·I（亮区拟合）。

    convention（数学冻结 v1.0 M0-1 correctness gate）：
      - "legacy"（默认）：冻结基准的历史行为——np.polyfit(x, y, 1) 返回
        [slope, intercept]，旧实现按 (a, b) 顺序接收（a←slope、b←intercept，
        系数顺序错位）。保留它仅为逐位复现冻结产物（severity 重算门、
        allocation 配对锚点都依赖）；论文最终口径不使用。
      - "corrected"：a=intercept、b=slope，与模型 Var ≈ a + b·I 一致。

    退化守卫在两种 convention 下判定**同一对象**（slope < 0 → 回退平坦噪声
    (1e-4, 0)），不引入任何新 clipping/robust 规则。
    """
    resid = I - s_hat
    sel = s_hat.ravel() > 0.05
    if sel.sum() < 200:
        return 1e-4, 0.0
    if convention not in ("legacy", "corrected"):
        raise ValueError(f"未知 noise_fit convention: {convention!r}")
    slope, intercept = np.polyfit(s_hat.ravel()[sel],
                                  (resid.ravel() ** 2)[sel], 1)
    if slope < 0:
        return 1e-4, 0.0
    if convention == "legacy":
        return float(slope), float(intercept)   # 历史错位顺序（a←slope）
    return float(intercept), float(slope)       # corrected：(a, b) 按模型定义


def _tangents(dirs):
    """每灯切向正交基 (t1,t2)：与 exp8S diagnose 同定义。"""
    t1 = np.zeros_like(dirs)
    t2 = np.zeros_like(dirs)
    for k, d in enumerate(dirs):
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(d, ref)
        v /= np.linalg.norm(v)
        t1[k] = v
        t2[k] = np.cross(d, v)
    return t1, t2


class NominalScene:
    """单对象 nominal 资产（corruption 前一次性构造）。

    noise_fit_convention: "legacy"（默认，冻结基准逐位复现口径）或
    "corrected"（M0-1 修复：a=intercept、b=slope）。convention 只改噪声拟合
    系数顺序 → w / Finf_diag / 白化链全部随之变化；冻结产物（severity 重算门、
    allocation 锚点）依赖 legacy，factorial rerun（M0-3）显式传 corrected。
    """

    def __init__(self, obj, rng, noise_fit_convention="legacy", selections=None):
        img = obj["images"][..., 0]                          # 灰度 (142,H,W)
        mask = obj["mask"]
        I_all = img[:, mask]                                  # (142, P_full)
        dirs_all = obj["light_directions"]
        # 固定分析子集（跨 level/seed 复用；预注册）。
        # selections=(sel, pidx) 显式给入时不消耗 rng（factorial 双场景复用
        # 同一冻结子集）；默认按原 run 消耗 rng 流。
        if selections is None:
            sel = np.sort(rng.choice(142, N_ANALYSIS_LIGHTS, replace=False))
            P_full = I_all.shape[1]
            pidx = np.sort(rng.choice(P_full, min(P_full, N_PIXEL_SUB),
                                      replace=False))
        else:
            sel, pidx = selections
        self.sel = sel
        self.pidx = pidx
        self.dirs = dirs_all[self.sel]
        I = I_all[self.sel][:, pidx]
        self.I = I
        rho, n = calibrated_ps(I, self.dirs)
        self.rho, self.n = rho, n
        ndl = np.clip(n @ self.dirs.T, 0, None)              # (P,L)
        self.s_hat = ndl.T                                   # (L,P)
        self.h = (self.s_hat > 0).astype(float)
        self.a, self.b = noise_fit(I, self.s_hat * rho[None, :],
                                   convention=noise_fit_convention)
        self.w = 1.0 / np.maximum(self.a + self.b * I, 1e-6)  # (L,P) 白化权重
        # φ 空间 Jacobian（物理单位）：列 [∂/∂logI, ∂/∂θ, ∂/∂ψ]
        t1, t2 = _tangents(self.dirs)
        self.B_phi = np.stack([
            self.s_hat * rho[None, :],                       # ∂I/∂logI = ρ·s  (L,P)
            (n @ t1.T).T * rho[None, :] * self.h,            # ∂I/∂θ  (L,P)
            (n @ t2.T).T * rho[None, :] * self.h,            # ∂I/∂ψ  (L,P)
        ], axis=-1)                                          # (L,P,3)
        # F∞（白化）：Σ_k w_kp·ŝ_kp² → 对角（逐像素）
        self.Finf_diag = (self.w * self.s_hat ** 2).sum(0)   # (P,)

    # ------------------------------------------------ 理论侧
    def delta_f_injected(self, sig_phi_diag):
        """Σ_c(物理单位) → 白化逐灯 delta_f 求和（V1b：block-diag nuisance）。

        sig_phi_diag: (3,) 或 (3,3) φ 空间方差（log²I, rad², rad²）。
        返回 ΔF (P,P) 与弱模式退化比（相对 F∞ 基线）。
        """
        P = self.I.shape[1]
        L = self.s_hat.shape[0]
        sig = (np.diag(np.asarray(sig_phi_diag, float))
               if np.ndim(sig_phi_diag) == 1
               else np.asarray(sig_phi_diag, float))
        Lam = np.linalg.inv(sig)                            # σ'²=1（白化后）
        DF = np.zeros((P, P))
        for k in range(L):
            A_k = np.diag(np.sqrt(self.w[k]) * self.s_hat[k])             # 白化 A
            B_k = np.sqrt(self.w[k])[:, None] * self.B_phi[k]            # 白化 B (P,3)
            DFk, _, _ = delta_f(A_k, B_k, Lam)
            DF += DFk
        return DF

    def predicted_degradation(self, sig_phi_diag, mode_coordinate="legacy"):
        sig = (np.diag(np.asarray(sig_phi_diag, float))
               if np.ndim(sig_phi_diag) == 1
               else np.asarray(sig_phi_diag, float))
        DF = self.delta_f_injected(sig)
        Fd = np.diag(self.Finf_diag)
        # 弱模式 = F∞^{-1/2} 白化谱的最小特征方向（可识别子空间 = 全 P，对角正定）
        Finv_h = np.diag(1.0 / np.sqrt(self.Finf_diag))
        R = Finv_h @ DF @ Finv_h
        rho_R, V = np.linalg.eigh(R)                          # 升序（最弱保留在前）
        weak = list(range(min(N_MODES, P := len(rho_R))))
        pred_deg = 1.0 / rho_R[weak]                          # 1/ρ_j（广义 retention 谱）
        # 经验误差投影算子（M0-2 correctness gate）：
        #   legacy:  E @ U          ——冻结基准的历史（近似）坐标；
        #   dual:    E @ F∞^{1/2} U ——严格 1/ρ_j 定理对应的 normalized dual
        #                             坐标 z_j = u_jᵀ F∞^{1/2} e（行向量实现；
        #                             gauge 对齐 P 作用在 E 侧，F∞^{1/2} 在此并入算子）。
        if mode_coordinate == "legacy":
            W = V[:, weak]
        elif mode_coordinate == "dual":
            W = np.diag(np.sqrt(self.Finf_diag)) @ V[:, weak]
        else:
            raise ValueError(f"未知 mode coordinate: {mode_coordinate!r}")
        return pred_deg, W

    @classmethod
    def _from_arrays(cls, base, I_new):
        """以 base 的资产（n̂/ρ̂/ŝ/w）+ 新观测 I 构造轻量副本（对照臂用）。"""
        obj = object.__new__(cls)
        for k in ("dirs", "rho", "n", "s_hat", "h", "a", "b", "w", "B_phi",
                  "Finf_diag", "sel", "pidx"):
            setattr(obj, k, getattr(base, k))
        obj.I = I_new
        return obj

    # ------------------------------------------------ empirical 侧
    def estimate_albedo(self, dirs_assumed, gains):
        """固定 n̂ 的白化逐像素 GLS（corrupted 假设校准）→ ρ̃ (P,)。"""
        s = np.clip(self.n @ dirs_assumed.T, 0, None).T       # (L,P)
        g = np.broadcast_to(np.asarray(gains, float).reshape(-1, 1),
                            s.shape) if np.ndim(gains) == 1 else gains
        model = s * g                                         # gains (L,) 每灯一档
        num = (self.w * model * self.I).sum(0)
        den = (self.w * model * model).sum(0)
        return num / np.maximum(den, 1e-12)


def run(config, run_dir):
    """CI04 主入口。

    config 可选键（缺省 = 冻结基准的 legacy 口径，冻结 config 不携带这两个键
    ——修正路径由 factorial rerun（M0-3）显式选择）：
      noise_fit_convention: "legacy" | "corrected"（M0-1 polyfit 系数顺序）
      mode_coordinate:      "legacy" | "dual"（M0-2 normalized dual 坐标投影）
    """
    rng = np.random.default_rng(config["seed"])
    data_root = config["data_root"]
    objects = config.get("objects") or json.loads(
        (Path(config.get("data_meta", "D:/data/OpenIllumination_meta"))
         / "dev_selection.json").read_text(encoding="utf-8"))["selection"]
    objects = objects[:config.get("max_objects", 8)]
    levels = config["levels"]
    seeds_per_level = config["seeds_per_level"]
    ctype = config.get("corruption_type", "joint")
    noise_conv = config.get("noise_fit_convention", "legacy")
    mode_coord = config.get("mode_coordinate", "legacy")

    rows = []
    for obj_name in objects:
        obj = load_object(data_root, obj_name,
                          data_meta=config.get("data_meta"))
        scen = NominalScene(obj, rng, noise_fit_convention=noise_conv)
        V_cache = {}
        for level in levels:
            gen = CorruptionGenerator(ctype, level)
            sig_phi = gen.sigma_phi_diag()                   # (3,3) 物理单位
            pred_deg, W = scen.predicted_degradation(np.diag(sig_phi),
                                                     mode_coordinate=mode_coord)
            V_cache[level] = W
            E = np.empty((seeds_per_level, scen.I.shape[1]))
            E0 = np.empty((seeds_per_level, scen.I.shape[1]))
            resid_pool = (scen.I - scen.s_hat * scen.rho[None, :]).ravel()
            for s_i in range(seeds_per_level):
                s_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + s_i * 7)
                d2, gains = gen.apply(s_rng, scen.dirs)
                rho_t = scen.estimate_albedo(d2, gains)
                E[s_i] = rho_t - scen.rho
                # 对照臂 = 残差 bootstrap 重抽（同量噪声、零 corruption）：
                # I* = ŝ·ρ̂ + ε*（ε* 有放回），同估计器重估——corruption 的"额外伤害"
                # 才是分母（δc=0 解析恒等解会让对照方差为 0 → 除法爆炸，pilot 首轮教训）
                I_star = (scen.s_hat * scen.rho[None, :]
                          + s_rng.choice(resid_pool, size=scen.I.shape,
                                         replace=True))
                sc_b = NominalScene._from_arrays(scen, I_star)
                rho_c = sc_b.estimate_albedo(scen.dirs, np.ones(len(scen.dirs)))
                E0[s_i] = rho_c - scen.rho
            # 模式投影 + 经验退化（相对 δc=0 对照臂方差）
            # 尺度 gauge 对齐（CI04：允许一个全局仿射尺度；ρ·gain 双解的已知
            # 自由度，估计后对齐——不做此步，强度维 corruption 会被 gauge 吸收成
            # "全 1 方向脉冲"假模式。预注册于 C15 pilot 首轮教训）
            for _E in (E, E0):
                s_scale = (_E * scen.rho).sum() / (scen.rho ** 2).sum()
                _E -= s_scale * scen.rho
            proj = E @ W
            proj0 = E0 @ W
            emp_deg = proj.var(0, ddof=1) / np.maximum(proj0.var(0, ddof=1), 1e-30)
            rows.append(dict(object=obj_name, level=float(level), ctype=ctype,
                             pred_deg=pred_deg.tolist(),
                             emp_deg=emp_deg.tolist(),
                             n_seeds=seeds_per_level,
                             noise_fit_convention=noise_conv,
                             mode_coordinate=mode_coord))

    # ---- Main check: Spearman + global affine scale + bootstrap CI ----
    preds = np.concatenate([np.asarray(r["pred_deg"]) for r in rows])
    emps = np.concatenate([np.asarray(r["emp_deg"]) for r in rows])
    rho_sp = float(spearmanr(preds, emps).statistic)
    # 全局仿射尺度（log-log 最小二乘斜率，截距固定 0 口径）
    ok = (preds > 0) & (emps > 0)
    slope = float(np.sum(np.log(preds[ok]) * np.log(emps[ok]))
                  / np.sum(np.log(preds[ok]) ** 2))
    bs = []
    n_all = len(preds)
    b_rng = np.random.default_rng(config["seed"] + 1)
    for _ in range(config.get("bootstrap", 1000)):
        idx = b_rng.integers(0, n_all, n_all)
        bs.append(spearmanr(preds[idx], emps[idx]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    gate_rho = config["gate_spearman_min"]
    return dict(run_name=config.get("run_name", "pilot"), seed=config["seed"],
                n_rows=len(rows), n_points=int(n_all),
                spearman=rho_sp, spearman_ci95=[float(lo), float(hi)],
                affine_slope_loglog=slope,
                gate_spearman_min=gate_rho,
                gate_pass=bool(rho_sp > gate_rho and lo > 0),
                rows=rows,
                note="估计器侧 corruption 协议：真值=GT 灯位，注入 N(0,Σ_c) 到估计器"
                     "假设校准；预测=ΔF(Σ_c) 弱模式退化比（corruption 前 nominal 计算）；"
                     "empirical=同估计器 corrupted 重估 vs δc=0 对照臂")
