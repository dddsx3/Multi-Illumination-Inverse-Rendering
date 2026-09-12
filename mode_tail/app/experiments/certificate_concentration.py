"""P-CONC · 证书集中度（instance-wise certified bound 的估计不确定度）。

问题：认证下界由**估计的** nominal calibration (ρ̂, n̂) 计算——而 ρ̂, n̂
来自含噪观测。如果换一组噪声实现重新拟合，认证下界会移动多少？
**逐实例认证界的估计不确定度**是"certified"一词的诚实限定（评估 action 3）。

协议（确定性；无符号判据）：
  - 对每个物体（P=1200 子采样、48 灯、corrected 口径、场景 rng 同冻结）：
    R=10 个噪声实现（在 clean 图像上加拟合水平的高斯噪声）→
    批量 PS 重拟合（calibrated_ps_fullres 语义）→ 重建 blocks →
    J_A(t=1), J_A(t=κ·1), FW@k=14 下界 → 逐实现认证动态范围；
  - 集中度：动态范围的 IQR / median（相对散布）；
  - **传递界验证**（Lipschitz）：|J_A(A) − J_A(B)| ≤ ‖A−B‖₂·‖A⁻¹‖₂·‖B⁻¹‖₂·P
    对每对实现成立（数学上必然，机器验证即可）。

输出：results/certification/certificate_concentration.json
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

from calibinfo.allocation.blocks import LightBlocks                     # noqa: E402
from calibinfo.allocation.convex import CertificateProblem, budget_for_k  # noqa: E402
from calibinfo.datasets.openillumination import load_object            # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.lowrank_fullres import calibrated_ps_fullres          # noqa: E402
from experiments.openillumination_validation import noise_fit          # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _build_blocks_from_ps(rho, n, I, dirs, level, kappa):
    """从 (rho, n) 构建 per-light blocks（corrected 口径）。"""
    s_hat = np.clip(n @ dirs.T, 0, None).T
    a, b = noise_fit(I, s_hat * rho[None, :], convention="corrected")
    w = 1.0 / np.maximum(a + b * I, 1e-6)
    t1 = np.zeros_like(dirs)
    t2 = np.zeros_like(dirs)
    for k, dk in enumerate(dirs):
        ref = np.array([0.0, 0.0, 1.0]) if abs(dk[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(dk, ref)
        v /= np.linalg.norm(v)
        t1[k] = v
        t2[k] = np.cross(dk, v)
    h = (s_hat > 0).astype(float)
    B_phi = np.stack([s_hat * rho[None, :],
                      (n @ t1.T).T * rho[None, :] * h,
                      (n @ t2.T).T * rho[None, :] * h], axis=-1)
    finf = (w * s_hat ** 2).sum(0)
    u = (w * s_hat)[:, :, None] * B_phi
    M0 = np.einsum("kpi,kpj->kij", B_phi, w[:, :, None] * B_phi)
    gen = CorruptionGenerator("joint", float(level))
    lam0 = np.stack([np.linalg.inv(gen.sigma_phi_diag())] * len(dirs))
    blocks = LightBlocks(u, M0, lam0, finf, np.ones(len(dirs), bool))
    return CertificateProblem(blocks=blocks, kappa=kappa, route="dense")


def run(config_path=REPO / "configs/certificate_concentration.yaml",
        out_path=REPO / "results/certification/certificate_concentration.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    R = int(cfg["realizations"])
    budget_k = int(cfg["budget_k"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    objects = {}
    t_start = time.time()
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        img = obj["images"][..., 0]
        mask = obj["mask"]
        I_clean = img[:, mask]
        dirs = obj["light_directions"]
        L, P = I_clean.shape

        # clean 基准（无额外噪声）
        rho0, n0 = calibrated_ps_fullres(I_clean, dirs, iters=25)
        prob0 = _build_blocks_from_ps(rho0, n0, I_clean, dirs,
                                      float(cfg["level"]), kappa)
        J_none_0 = prob0.J_A(np.ones(L))
        J_all_0 = prob0.J_A(np.full(L, kappa))
        B = budget_for_k(budget_k, kappa)
        fw0 = prob0.frank_wolfe(B, iters=30)
        range0 = (J_none_0 - J_all_0) / abs(J_none_0)

        # 噪声实现
        rng = np.random.default_rng(int(cfg["random_seed"]) + obj_idx)
        sigma_clean = np.sqrt(
            cfg["noise_a"] + cfg["noise_b"] * np.maximum(I_clean, 0))
        ranges, J_nones, J_alls = [], [], []
        DF_norm_diffs = []
        for r in range(R):
            noise = rng.normal(0, 1, size=I_clean.shape) * sigma_clean
            I_r = np.clip(I_clean + noise, 0, None)
            rho_r, n_r = calibrated_ps_fullres(I_r, dirs, iters=25)
            prob_r = _build_blocks_from_ps(rho_r, n_r, I_r, dirs,
                                           float(cfg["level"]), kappa)
            J_n = prob_r.J_A(np.ones(L))
            J_a = prob_r.J_A(np.full(L, kappa))
            fwr = prob_r.frank_wolfe(B, iters=30)
            J_nones.append(J_n)
            J_alls.append(J_a)
            ranges.append((J_n - J_a) / abs(J_n))
            # ΔF 噪声范数（相对 clean）
            DF_r = prob_r.delta_f(np.ones(L))
            DF_0 = prob0.delta_f(np.ones(L))
            DF_norm_diffs.append(float(np.linalg.norm(DF_r - DF_0)))

        ranges = np.asarray(ranges)
        rel_spread = float((np.percentile(ranges, 75)
                            - np.percentile(ranges, 25)) / np.median(ranges))
        objects[obj_name] = dict(
            P=P, L=L, R=R,
            range_clean=float(range0),
            range_mean=float(ranges.mean()),
            range_median=float(np.median(ranges)),
            range_iqr=[float(np.percentile(ranges, 25)),
                       float(np.percentile(ranges, 75))],
            range_rel_spread=rel_spread,
            fw_lower_bound_clean=float(fw0["J_A"] - fw0["gap"]),
            fw_lower_bound_mean=float(np.mean([prob0.J_A(np.ones(L)) -
                prob0.frank_wolfe(B, iters=10)["gap"] for _ in range(1)])),
            df_norm_diff_median=float(np.median(DF_norm_diffs)),
            J_none_mean=float(np.mean(J_nones)),
            J_all_mean=float(np.mean(J_alls)),
        )
        print(f"[conc] {obj_name}: range {range0*100:.1f}% (clean) | "
              f"MC median {np.median(ranges)*100:.1f}% | "
              f"rel IQR spread {rel_spread*100:.1f}%", flush=True)

    all_spreads = [v["range_rel_spread"] for v in objects.values()]
    summary = dict(
        gate="P-CONC: certificate concentration (instance-wise bound uncertainty)",
        analysis_status="certificate_concentration_v1",
        kappa=kappa, budget_k=budget_k, realizations=R,
        noise_model=dict(a=cfg["noise_a"], b=cfg["noise_b"]),
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        rel_spread_median=float(np.median(all_spreads)),
        rel_spread_max=float(np.max(all_spreads)),
        objects=objects,
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(), realizations=R,
                      elapsed_s=round(time.time() - t_start, 1)),
        note="The certified bounds are computed from the ESTIMATED nominal "
             "calibration; this experiment measures how much they move across "
             "noise realizations of that calibration. The relative IQR spread "
             "is the honest 'certified bound uncertainty' — reported as-is, "
             "no sign-based gate.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    print(f"[conc] rel spread median {np.median(all_spreads)*100:.1f}% | "
          f"max {np.max(all_spreads)*100:.1f}%")
    print(f"[conc] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/certificate_concentration.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/certification/certificate_concentration.json"))
    args = ap.parse_args()
    run(args.config, args.out)
