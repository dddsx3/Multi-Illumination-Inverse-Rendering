"""P-RADIUS · 真实数据线性化有效半径（十问 Q3 / 评估 action 2）。

问题：以名义标定处的雅可比 Bφ 构造的线性化预测（ΔF 与 pred_deg），
在多大的腐蚀水平 ℓ 上仍与经验估计误差的二阶矩一致？
偏离 {2×, 10×} 的交叉水平 = **线性化有效半径**（逐物体报告）。

口径（与 ci04 完全一致）：
  - 场景：P=1200 子采样、48 active 灯、corrected 口径、场景 rng 同冻结；
  - 腐蚀：joint（方向旋转 + 强度缩放），水平 ℓ = CorruptionGenerator("joint", ℓ)
    的物理单位（σ_logI = ℓ、σ_deg = ℓ 度）——预注册网格 [0.05 ... 8.0]；
  - 估计器：固定 n̂ 白化逐像素 GLS，腐蚀后的 d2/g 注入（同冻结协议）；
  - 残差：e = ρ̃ − ρ̂，gauge 对齐（同冻结）；经验二阶矩 = 逐 seed 的
    e @ W_dual 平方（bottom-5 dual 子空间能量）的均值 E(ℓ)；
  - 线性化预测：pred_deg_j(ℓ) = 1/ρ_j(Σ_φ(ℓ))（ΔF 在 ℓ 处的广义保留谱），
    Σ_φ(ℓ) = ℓ² Σ_φ(1)（腐蚀水平平方缩放进精度）。

度量：dev(ℓ) = E(ℓ) / E_ref，其中 E_ref 为最小水平 {0.05} 的能量——
线性化生效时 dev ∝ (ℓ/0.05)²（理论标度）；偏离该标度 = 线性化失效。
半径 = dev 首次超过 {2×dev(0.05), 10×dev(0.05)} 的最小 ℓ。

输出：results/magnitude/linearization_radius.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.datasets.openillumination import load_object            # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.openillumination_validation import NominalScene       # noqa: E402

LEVELS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
SEEDS = 20
CROSSINGS = (2.0, 10.0)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def run(config_path=REPO / "configs/linearization_radius.yaml",
        out_path=REPO / "results/magnitude/linearization_radius.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    levels = [float(lv) for lv in cfg["levels"]]
    seeds = int(cfg["seeds"])
    ref_level = float(cfg.get("reference_level", levels[0]))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    objects = {}
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        rows_obj = []
        for level in levels:
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            # 线性化预测：ΔF(ℓ) 的保留谱（Σ_φ = level 的物理单位方差）
            _deg, W_dual = scen.predicted_degradation(
                gen.sigma_phi_diag(), mode_coordinate="dual")
            rho_pred = np.sort(_deg)
            # 经验：seeds 个腐蚀实现（方向 + 强度，联合），gauge 对齐
            energies = np.empty(seeds)
            for s_i in range(seeds):
                rng = np.random.default_rng(
                    [20260916, 4242, obj_idx, int(level * 1000), s_i])
                from calibinfo.allocation.corruption import raw_innovations
                raw = raw_innovations(rng, len(scen.sel))
                scales = np.ones(len(scen.sel))
                from calibinfo.allocation.corruption import apply_scaled_corruption
                d2, g = apply_scaled_corruption(
                    scen.dirs, sig_logI, sig_rad, scales, raw)
                rho_t = scen.estimate_albedo(d2, g)
                e = rho_t - scen.rho
                sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                e = e - sg * scen.rho
                energies[s_i] = float(np.sum((e @ W_dual) ** 2))
            E_mean = float(energies.mean())
            rows_obj.append(dict(level=level, E_mean=E_mean,
                                 pred_deg_bottom5=[float(x) for x in
                                                   1.0 / rho_pred[:5]],
                                 n_seeds=seeds))
        # 参考能量（最小水平）
        E_ref = rows_obj[0]["E_mean"]
        for r in rows_obj:
            r["dev_from_ref"] = r["E_mean"] / E_ref if E_ref > 0 else float("nan")
        # 半径：dev 首次超过 {2, 10} 的最小 ℓ
        radius = {}
        for thr in CROSSINGS:
            hit = [r["level"] for r in rows_obj
                   if r["dev_from_ref"] > thr]
            radius[f"cross_{thr:g}x"] = (min(hit) if hit else None)
        # 理论标度检查：dev(ℓ) ≈ (ℓ/ℓ_ref)²（线性化预言）
        objects[obj_name] = dict(rows=rows_obj, radius=radius,
                                 E_ref=E_ref)
        med_all = [r["dev_from_ref"] for r in rows_obj]
        print(f"[radius] {obj_name}: dev @0.05={med_all[0]:.2f} "
              f"@1.0={med_all[5]:.2f} @8.0={med_all[-1]:.2f}; "
              f"radius2x={radius['cross_2x']}, radius10x={radius['cross_10x']}",
              flush=True)

    rads2 = [v["radius"]["cross_2x"] for v in objects.values()
             if v["radius"]["cross_2x"] is not None]
    rads10 = [v["radius"]["cross_10x"] for v in objects.values()
              if v["radius"]["cross_10x"] is not None]
    summary = dict(
        gate="P-RADIUS: linearization validity radius on real data",
        analysis_status="linearization_radius_v1",
        levels=levels, seeds=seeds,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        radius_2x=dict(median=float(np.median(rads2)) if rads2 else None,
                       n_crossed=len(rads2),
                       per_object={k: v["radius"]["cross_2x"]
                                   for k, v in objects.items()}),
        radius_10x=dict(median=float(np.median(rads10)) if rads10 else None,
                        n_crossed=len(rads10),
                        per_object={k: v["radius"]["cross_10x"]
                                    for k, v in objects.items()}),
        objects=objects,
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(), seeds=seeds),
        note="Linearization validity radius: for each corruption level ℓ, "
             "E(ℓ) = mean over seeds of the bottom-5 dual-coordinate energy "
             "of the gauge-aligned residual; dev(ℓ) = E(ℓ)/E(ℓ_min). "
             "The linearized theory predicts dev ∝ (ℓ/ℓ_min)²; deviation "
             "from that scaling = linearization breakdown. Radius = first "
             "level where dev crosses 2× / 10× the reference; per object. "
             "No sign-based gate; all levels reported.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    print(f"[radius] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/linearization_radius.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/magnitude/linearization_radius.json"))
    args = ap.parse_args()
    run(args.config, args.out)
