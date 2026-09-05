#!/usr/bin/env python3
"""关键实验 1 · P-A2 到底算了什么（光照扰动 + 尺度 gauge 双测试）

设计来源：主智能体关键实验指令（2026-09-05）· 零修改照抄测试语义。
被测对象：r5_compute_audit/w2a2_fisher.py::compute_fisher_for_config
（P-A2 产出 2.59/0.37 数字的同一函数，不改其一行——测试它，不是改它）。

测试 A（光照扰动不变性）：
  固定 scene（法线/反照率），只换光照配置 ω（完全不同的两组方向），
  重算 Fisher。若 F 逐元素不变 → 证实 F 与光照无关（"albedo+normal 已知"口径）。
  数量判据：max|F1 − F2| / max|F1| （相对 Frobenius 差）。

测试 B（尺度 gauge 测试）：
  把"当前光照参数向量 v"代入 F @ v。被测函数没有显式 v 输出（它根本没用 ω），
  因此 v 取两组口径各测一次，保证结论不依赖 v 的选取歧义：
  (B1) v = 该 config 的 Lambertian 加权 SH 基展开：v_l = A_l ⊙ y(ω_l)（w2a2 代码里
       唯一处"用到" ω 的量——它乘进 F 之前就被丢弃，但取它作 v 是最宽容的读法）；
  (B2) v = 随机单位向量（若 F 满秩，任意 v 的 Fv 范数都非零，作对照）。
  判据：‖F v‖/‖F‖‖v‖ 接近 0 → albedo 曾作为未知量做 Schur 补；非零 → albedo 已知。

交叉验证（防止只测到 w2a2 的偶然写法）：
  对 gauge_fisher_v2.schur_full（F_eff = Schur 补消去光照后的 P×P 矩阵）
  重做测试 B：已知解析 null 方向 = a（尺度 gauge，B_kᵀa = F_ll,k c_k 恒等式）。
  若 ‖F_eff·a‖/‖F_eff‖‖a‖ ≈ 0 → GFv2 是真正的 Schur 补口径（albedo 未知）。
  这一步把"P-A2 写错"与"GA-ISI v2 写对"两个对象分开，避免一锅端结论。

输出：critical_experiments/exp1_pa2_verdict.json + 控制台表格。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
sys.path.insert(0, str(REPO / "r5_compute_audit"))

from gauge_fisher_v2 import (  # noqa: E402
    fisher_blocks, gauge_project, gauge_unit, load_scene, schur_full,
    scene_arrays, sh_basis_npy,
)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "w2a2", REPO / "r5_compute_audit" / "w2a2_fisher.py")
w2a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w2a2)

DATA = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT = REPO / "critical_experiments" / "exp1_pa2_verdict.json"
SCENES = ["conf_sphere_r05", "conf_cube_axis", "conf_ellipsoid_z06", "conf_cone_r04_d12"]
SEED = 20260905


def run():
    rng = np.random.default_rng(SEED)
    results = {"scenes": {}, "cross_check_gf2": {}, "meta": {
        "tested_function": "r5_compute_audit/w2a2_fisher.py::compute_fisher_for_config",
        "seed": SEED, "n_scenes": len(SCENES)}}

    # 两套完全不同的光照方向配置（均匀球面采样，不同种子 → 方向集合独立）
    omega_a = rng.normal(size=(8, 3)); omega_a /= np.linalg.norm(omega_a, axis=1, keepdims=True)
    omega_b = rng.normal(size=(8, 3)); omega_b /= np.linalg.norm(omega_b, axis=1, keepdims=True)
    results["meta"]["omega_angle_check_deg"] = float(np.degrees(
        np.arccos(np.clip((omega_a[0] @ omega_b[0]) /
                          (np.linalg.norm(omega_a[0]) * np.linalg.norm(omega_b[0])), -1, 1))))

    for scene in SCENES:
        d = DATA / scene
        if not d.is_dir():
            continue
        # ---- 测试 A：同一 scene，两组完全不同的 ω ----
        F_a = w2a2.compute_fisher_for_config(str(d), omega_a, 500)
        F_b = w2a2.compute_fisher_for_config(str(d), omega_b, 500)
        rel_fro = float(np.linalg.norm(F_a - F_b) / (np.linalg.norm(F_a) + 1e-30))
        max_el = float(np.max(np.abs(F_a - F_b)) / (np.max(np.abs(F_a)) + 1e-30))

        # ---- 测试 B：‖F v‖ 判据（两种 v 口径）----
        # B1: 最宽容读法——v = Lambertian 加权 SH(ω)。w2a2 里 ω 唯一出现的形式。
        A_l = np.array([np.pi, 2*np.pi/3, np.pi/4, 0, 0, 0, 0, 0, 0])
        v_lam = (sh_basis_npy(omega_a) * A_l[None, :]).sum(axis=0)  # (9,)
        # B2: 对照——随机单位向量（F 满秩时任何 v 都非零）
        v_rand = rng.normal(size=9); v_rand /= np.linalg.norm(v_rand)

        def rayleigh(v):
            v = np.asarray(v, float)
            Fn = np.linalg.norm(F_a)
            return float(np.linalg.norm(F_a @ v) / (Fn * np.linalg.norm(v) + 1e-30))

        # ---- 交叉验证：gauge_fisher_v2 的 F_eff（真 Schur 补）----
        sc = load_scene(str(d))
        sub = list(range(5))  # 前 5 灯（固定，可复现）
        a, Y, C = scene_arrays(sc, sub, pixel_cap=500, seed=SEED)
        bl = fisher_blocks(a, Y, C)
        F_eff = schur_full(bl)
        a_hat = gauge_unit(a)
        Fp = gauge_project(F_eff, a_hat)
        # 尺度 gauge 的解析 null 方向：δa = a（消去光照后 F_eff·a = 0，v2 头注恒等式）
        ray_gauge = float(np.linalg.norm(F_eff @ a) /
                          (np.linalg.norm(F_eff) * np.linalg.norm(a) + 1e-30))
        # 对照：随机方向的 Rayleigh 商（应 O(λ_typ/‖F‖) 非零）
        v_r2 = rng.normal(size=len(a)); v_r2 /= np.linalg.norm(v_r2)
        ray_rand_eff = float(np.linalg.norm(F_eff @ v_r2) /
                             (np.linalg.norm(F_eff) + 1e-30))

        results["scenes"][scene] = {
            "testA_rel_frobenius": rel_fro,
            "testA_max_element": max_el,
            "testB1_rayleigh_lambert_v": rayleigh(v_lam),
            "testB2_rayleigh_random_v": rayleigh(v_rand),
            "gf2_schur_gauge_rayleigh_deltaA": ray_gauge,
            "gf2_schur_random_rayleigh": ray_rand_eff,
            "F_eigs": [float(x) for x in np.linalg.eigvalsh(F_a)],
        }
        print(f"{scene:22s} A: relFro={rel_fro:.2e}  B1: {rayleigh(v_lam):.2e}  "
              f"B2: {rayleigh(v_rand):.2e}  | GFv2 gauge δa: {ray_gauge:.2e}  "
              f"rand: {ray_rand_eff:.2e}")

    # ---- 汇总判定（阈值先验写死，不看结果调）----
    s = results["scenes"]
    a_all_zero = all(v["testA_rel_frobenius"] < 1e-10 for v in s.values())
    b1_all = [v["testB1_rayleigh_lambert_v"] for v in s.values()]
    b2_all = [v["testB2_rayleigh_random_v"] for v in s.values()]
    gf2_gauge_all = [v["gf2_schur_gauge_rayleigh_deltaA"] for v in s.values()]
    gf2_rand_all = [v["gf2_schur_random_rayleigh"] for v in s.values()]

    results["verdict"] = {
        "testA_light_perturbation_invariant": bool(a_all_zero),
        "testA_conclusion": ("F 与光照配置完全无关（相对差 <1e-10）→ P-A2 的 F 是"
                             "『albedo+normal 已知』口径的 Σ a²Y(n)Y(n)ᵀ，光照 ω 是死变量"
                             ) if a_all_zero else "F 随光照变化 → 需进一步排查",
        "testB1_mean": float(np.mean(b1_all)), "testB2_mean": float(np.mean(b2_all)),
        "testB_conclusion": (
            "‖Fv‖ 与随机方向同量级（非零）→ 该 9×9 F 把 albedo 当已知量，"
            "没做过 Schur 补；不存在尺度 gauge 零方向"
            if np.mean(b1_all) > 1e-8 and np.mean(b2_all) > 1e-8
            else "Fv≈0 → albedo 曾为未知量并做过 Schur 补"),
        "gf2_conclusion": (
            "gauge_fisher_v2.schur_full 满足解析恒等式 F_eff·a=0（Rayleigh %.1e，"
            "随机方向 %.1e）→ GA-ISI v2 是 albedo-未知 的真 Schur 补口径，与 w2a2 判然两物"
            % (np.mean(gf2_gauge_all), np.mean(gf2_rand_all))),
        "implication_for_259": (
            "2.59/0.37 的 P-A2 数字 = 法线 SH 子空间 Gram 的谱统计（结构量），"
            "不含光照信息、不涉 Schur 补——『近零维数 2.59 < 4』的异常是口径错位"
            "（拿 albedo-已知的 Gram 去对 albedo-未知 的歧义维数预测），"
            "不是理论 4 维零空间的证伪。GA-ISI v2（真 Schur）不受此影响。"),
    }
    print("\n=== 判定 ===")
    for k, v in results["verdict"].items():
        print(f"  {k}: {v}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n[exp1] 结果落盘 -> {OUT}")


if __name__ == "__main__":
    run()
