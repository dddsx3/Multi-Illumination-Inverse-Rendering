#!/usr/bin/env python3
"""卡S' 机制审计(协议允许的"下一轮裁决"证据): trace 负显著的方向/简化疑点定位。

方法(全部确定性, 与云端子集一一对齐):
  - 子集抽样完全复刻主循环: r2=crc32(name)种子抽 3 光 ×50; rng3=SEED 抽像素(含守卫跳过)
  - 对齐自检: 本机重算的 sigma_min 必须与云端 rows 的 sigma_min 一致(≤1e-9)
  - 真值几何(GT): rho_full(全光朗伯校准), α=1, l̂=dirs_sub 真值, w=噪声模型(a+bI)⁻¹
  - 变体A 严格版: 全耦合 F(ρ, 3N) → 精确 Schur 消 ρ(F_ρρ 精确对角) → 全局投影 α 维
    → 每光 2×2 方向块 → trace=(F11+F22)/det(reliable 与云端实现同口径)
  - 变体B 简化版@真值: 云端 diagnose_trace 的逐像素对角简化, 但输入换成真值(排除"拟
    合值污染")
  - 变体C 信息量方向: 变体A 每光块取 det/(F11+F22)(trace 的倒数口径)
  - 每变体对云端 LAE 求 Spearman(n=50) 并计数显著正/负
输出: exp8s_oracle_schur_audit.json
"""
from __future__ import annotations
import json
import time
import zlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp8r_diligent_discrimination_v3 as m

HERE = Path(__file__).resolve().parent
ROOT = Path("D:/data/DiLiGenT/pmsData")
N3_SAMPLES = 50
SEED = m.SEED
CLOUD_DIR = Path("D:/MIR_Archive_20260829/_cloud_results_9obj")


def audit_object(name):
    d = ROOT / name
    dirs, n_gt, I_norm, _mask = m.load_object(d)
    rho_full, a_, b_, _lam = m.calibrate(n_gt, dirs, I_norm)      # 全光 GT 校准
    r2 = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 1000)
    sels = [r2.choice(96, 3, replace=False) for _ in range(N3_SAMPLES)]

    cloud_rows = json.loads((CLOUD_DIR / f"exp8r_per_object_{name}.json")
                            .read_text(encoding="utf-8"))["rows"]
    cloud_n3 = [r for r in cloud_rows if r["N"] == 3]
    laes = np.array([r["lae"] for r in cloud_n3])
    sigs = np.array([r["sigma_min"] for r in cloud_n3])

    out = {"n": len(laes), "sigma_align_max_err": 0.0,
           "variants": {}}
    kept = 0
    A_all, B_all, C_all, sig_my = [], [], [], []
    for sel in sels:
        dirs_sub = dirs[sel]
        I_sub = I_norm[sel]
        n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
        keep = m.pixel_guard(I_sub, n_masked)
        if keep.sum() < 100:
            continue
        idx_k = np.where(keep)[0]
        if len(idx_k) < 100:
            continue
        rng3 = np.random.default_rng(SEED)
        sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k) // 8, 2000)),
                                      replace=False))
        if kept >= len(laes):
            break
        n_k = n_gt[sub_idx]      # (P,3)
        I_k = I_sub[:, sub_idx]  # (3,P)
        sig_my.append(float(np.linalg.svd(dirs_sub, compute_uv=False)[-1]))
        # ---- GT Fisher(α=1) ----
        nl = np.clip(n_k @ dirs_sub.T, 0, None)          # (P,3)
        h = (nl > 0).astype(float)
        w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_k, 0), 1e-6)   # (3,P)
        rho = rho_full[sub_idx]
        # 切向基(与 diagnose_trace 同定义)
        t1s = np.zeros((3, 3)); t2s = np.zeros((3, 3))
        for k in range(3):
            ref = np.array([0., 0., 1.]) if abs(dirs_sub[k, 2]) < 0.9 else np.array([1., 0., 0.])
            t1 = np.cross(dirs_sub[k], ref); t1 /= np.linalg.norm(t1)
            t1s[k] = t1; t2s[k] = np.cross(dirs_sub[k], t1)
        # 变体A: 严格全耦合 Schur
        # 参数序: [α0..2, t1_0,t2_0, t1_1,t2_1, t1_2,t2_2] (9=3N, N=3)
        n_l = 9
        Fpp = (((h * nl)**2) * w.T).sum(1)                              # (P,)  α=1
        Fpp = np.maximum(Fpp, 1e-300)
        F_ll = np.zeros((n_l, n_l))
        F_lr = np.zeros((n_l, len(sub_idx)))
        # 每光残差只依赖本光参数; ρ 交叉在像素级
        for k in range(3):
            wt = w[k] * h[:, k]
            # dI/dα = ρ nl ;  dI/dt1 = ρ (n·t1) ; dI/dt2 = ρ (n·t2)
            g_a = rho * nl[:, k]
            g_1 = rho * (n_k @ t1s[k])
            g_2 = rho * (n_k @ t2s[k])
            F_ll[3 * k + 0, 3 * k + 0] = (wt * g_a * g_a).sum()
            F_ll[3 * k + 1, 3 * k + 1] = (wt * g_1 * g_1).sum()
            F_ll[3 * k + 2, 3 * k + 2] = (wt * g_2 * g_2).sum()
            F_ll[3 * k + 0, 3 * k + 1] = F_ll[3 * k + 1, 3 * k + 0] = (wt * g_a * g_1).sum()
            F_ll[3 * k + 0, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 0] = (wt * g_a * g_2).sum()
            F_ll[3 * k + 1, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 1] = (wt * g_1 * g_2).sum()
            F_lr[3 * k + 0] = wt * nl[:, k] * g_a
            F_lr[3 * k + 1] = wt * nl[:, k] * g_1
            F_lr[3 * k + 2] = wt * nl[:, k] * g_2
        S = F_ll - F_lr @ (F_lr.T / Fpp[:, None])                       # 精确 Schur(消ρ)
        alpha_idx = [0, 3, 6]
        dir_idx = [1, 2, 4, 5, 7, 8]
        S_aa = S[np.ix_(alpha_idx, alpha_idx)]
        S_dir = S[np.ix_(dir_idx, dir_idx)] - \
            S[np.ix_(dir_idx, alpha_idx)] @ np.linalg.pinv(S_aa) @ S[np.ix_(alpha_idx, dir_idx)]
        trA = 0.0
        for k in range(3):
            blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
            det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
            if det > 0:
                trA += (blk[0, 0] + blk[1, 1]) / det
        # 变体B: 简化式@真值(云端 diagnose_trace 公式, 输入换真值)
        trB = 0.0
        for k in range(3):
            Fpk = (h[:, k] * nl[:, k])**2 * w[k]
            g1 = rho * h[:, k] * (n_k @ t1s[k])
            g2 = rho * h[:, k] * (n_k @ t2s[k])
            F11 = (w[k] * g1 * g1).sum(); F22 = (w[k] * g2 * g2).sum(); F12 = (w[k] * g1 * g2).sum()
            c1 = (w[k] * g1 * (h[:, k] * nl[:, k])).sum()
            c2 = (w[k] * g2 * (h[:, k] * nl[:, k])).sum()
            Fpp_all = 0.0
            for j in range(3):
                Fpp_all += ((h[:, j] * nl[:, j])**2 * w[j]).sum()
            F11 -= c1 * c1 / max(Fpp_all, 1e-300)
            F22 -= c2 * c2 / max(Fpp_all, 1e-300)
            F12 -= c1 * c2 / max(Fpp_all, 1e-300)
            det = F11 * F22 - F12 * F12
            if det > 0:
                trB += (F11 + F22) / det
        # 变体C: 信息量方向(A 的 det/(F11+F22) 口径)
        trC = 0.0
        for k in range(3):
            blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
            det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
            ssum = blk[0, 0] + blk[1, 1]
            if det > 0 and ssum > 0:
                trC += det / ssum
        A_all.append(trA); B_all.append(trB); C_all.append(trC)
        kept += 1

    out["sigma_align_max_err"] = float(np.abs(np.array(sig_my) - sigs).max())
    for tag, vv in [("A_strict_schur_oracle", A_all), ("B_simplified_oracle", B_all),
                    ("C_info_direction_oracle", C_all)]:
        v = np.array(vv)
        r, p = spearmanr(v, laes)
        out["variants"][tag] = dict(rho=float(r), p=float(p),
                                    range=[float(v.min()), float(v.max())])
    r_s, p_s = spearmanr(np.array(sig_my), laes)
    out["sigma_min_vs_lae"] = dict(rho=float(r_s), p=float(p_s))
    return out


def main():
    names = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
             "gobletPNG", "pot1PNG", "pot2PNG", "readingPNG"]
    res = {}
    t0 = time.time()
    for name in names:
        try:
            res[name] = audit_object(name)
            print(f"[audit] {name}: σ对齐误差={res[name]['sigma_align_max_err']:.2e} | "
                  f"A ρ={res[name]['variants']['A_strict_schur_oracle']['rho']:+.3f} "
                  f"(p={res[name]['variants']['A_strict_schur_oracle']['p']:.2e}) | "
                  f"B ρ={res[name]['variants']['B_simplified_oracle']['rho']:+.3f} "
                  f"(p={res[name]['variants']['B_simplified_oracle']['p']:.2e}) | "
                  f"C ρ={res[name]['variants']['C_info_direction_oracle']['rho']:+.3f} "
                  f"(p={res[name]['variants']['C_info_direction_oracle']['p']:.2e})",
                  flush=True)
        except Exception as e:
            res[name] = dict(error=str(e))
            print(f"[audit] {name} FAILED: {e}", flush=True)
    out = dict(meta=dict(seed=SEED, n3=N3_SAMPLES, time_s=time.time() - t0,
                         note="真值几何; 与云端子集 sigma_min 对齐自检; 判据不动, 裁决证据"),
               objects=res)
    (HERE / "exp8s_oracle_schur_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 汇总: 各变体显著正/负计数
    print("\n[audit] 汇总(显著= p<0.05):")
    for tag in ["A_strict_schur_oracle", "B_simplified_oracle", "C_info_direction_oracle"]:
        pos = neg = 0
        for name in names:
            v = res[name].get("variants", {}).get(tag)
            if not v:
                continue
            if v["p"] < 0.05:
                if v["rho"] > 0:
                    pos += 1
                else:
                    neg += 1
        print(f"  {tag}: 显著正 {pos} | 显著负 {neg}")


if __name__ == "__main__":
    main()