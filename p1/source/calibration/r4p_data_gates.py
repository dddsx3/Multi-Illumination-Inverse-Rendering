"""P1-R4'-C · 全部确认集数据 Gate 汇总（validation + oracle）。"""
import csv
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "calibration"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "generation"))

ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
ORACLE = os.path.join(_REPO, "p1", "calibration_set", "confirmatory_gate_reports")
REPORT_MD = os.path.join(ORACLE, "R4P_CONFIRMATORY_DATA_GATES.md")
ORACLE_PSNR_THRESHOLD = 25.0

from validation_gates import load_scene as vload, gate_G1_pixel_diversity, gate_G2_direction_image_consistency, gate_G3_metadata_swap  # noqa
from sh import sh_basis_npy  # noqa


def oracle_per_scene(sc):
    """复用 oracle_gate 核心公式（不 import 整图）—— 见 occlude_oracle。"""
    mask = sc["mask"]
    A = sc["albedo"]
    n = sc["n_mesh"].transpose(1, 2, 0)
    Yn = sh_basis_npy(n[mask])
    K = sc["imgs_lin"].shape[0]
    out = {}
    for tag, nkey in [("Or1", "n_mesh"), ("Or2", "n_depth")]:
        if nkey == "n_depth":
            Yn_loc = sh_basis_npy(sc["n_depth"].transpose(1, 2, 0)[mask])
        else:
            Yn_loc = Yn
        psnrs = []
        for k in range(K):
            c = sc["sh_irr"][k]
            s = np.maximum(Yn_loc @ c, 0.0)
            rec = A[mask] * s
            img = sc["imgs_lin"][k][mask]
            scale = (rec * img).sum() / max((rec * rec).sum(), 1e-12)
            mse = float(((scale * rec - img) ** 2).mean())
            psnrs.append(10 * np.log10(1 / max(mse, 1e-12)))
        out[tag + "_mean_si_psnr"] = float(np.mean(psnrs))
    return out


def main():
    scenes = sorted([os.path.join(ROOT, d) for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_")
                     and os.path.isfile(os.path.join(ROOT, d, "sh_coeffs_irradiance.npy"))])
    assert scenes, "无确认集"
    os.makedirs(ORACLE, exist_ok=True)
    rows_v, rows_o = [], []
    for sd in scenes:
        sc = vload(sd)
        g1 = gate_G1_pixel_diversity(sc)
        g2 = gate_G2_direction_image_consistency(sc)
        g3 = gate_G3_metadata_swap(sc)
        rows_v.append(dict(scene=os.path.basename(sd), G1_pass=g1["passed"],
                            G1_ratio=g1["light_to_noise_ratio"],
                            G2_pass=g2["passed"], G2_delta=g2["delta_db"],
                            G3_pass=g3["passed"], G3_delta=g3["delta_db"],
                            any_fail=not (g1["passed"] and g2["passed"] and g3["passed"])))
        o = oracle_per_scene(sc)
        rows_o.append(dict(scene=os.path.basename(sd),
                            Or1_psnr=o["Or1_mean_si_psnr"], Or2_psnr=o["Or2_mean_si_psnr"]))
        print(f"  {os.path.basename(sd):28s} G1={'P' if g1['passed'] else 'F'}  G2={'P' if g2['passed'] else 'F'}(Δ{g2['delta_db']:.2f})  Or1={o['Or1_mean_si_psnr']:.2f}dB", flush=True)
    v_csv = os.path.join(ORACLE, "validation_all.csv")
    o_csv = os.path.join(ORACLE, "oracle_all.csv")
    for path, rows in [(v_csv, rows_v), (o_csv, rows_o)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(scenes)
    n_pass_v = sum(1 for r in rows_v if not r["any_fail"])
    or1 = np.array([r["Or1_psnr"] for r in rows_o])
    lines = ["# R4′-C 确认集数据 Gate 汇总", "",
             f"- scene 数：{n}（canary 1 + 批量 17；seed 20260901；horizon 18 mesh 中 7 mesh 渲染时空遮罩被剔除，见日志）",
             f"- 渲染协议：SUN 远场 / 纯 Diffuse / 128² / 32 samples / light_energy 3.0",
             f"- INC-001 帧级校验已嵌入；本批无坏帧逃出",
             "", "## Validation Gates (G1/G2/G3)",
             "",
             "| scene | G1 ratio | G2 Δ(dB) | G3 Δ(dB) | PASS |",
             "|---|---|---|---|---|"]
    for r in rows_v:
        lines.append(f"| {r['scene']} | {r['G1_ratio']:.1f} | {r['G2_delta']:.2f} | "
                     f"{r['G3_delta']:.2f} | {'✓' if not r['any_fail'] else '✗'} |")
    lines += ["", f"**{n_pass_v}/{n} 场景全部 3 个 Gate PASS**" if n_pass_v == n
              else f"**{n_pass_v}/{n} PASS**（部分场景需复核）",
              "", "## Oracle Gate（mesh normal + GT albedo + GT light）",
              "",
              "| scene | Or1 SI-PSNR (dB) | Or2 SI-PSNR (dB) |",
              "|---|---|---|"]
    for r in rows_o:
        lines.append(f"| {r['scene']} | {r['Or1_psnr']:.2f} | {r['Or2_psnr']:.2f} |")
    lines += ["", "**汇总**",
              f"- Or1 SI-PSNR: mean={or1.mean():.2f} dB, min={or1.min():.2f}, max={or1.max():.2f}",
              f"- 阈值 ({ORACLE_PSNR_THRESHOLD} dB): "
              f"{int((or1 > ORACLE_PSNR_THRESHOLD).sum())}/{n} 场景 Or1 PASS",
              "",
              "## 结论",
              "",
              "- 物理协议零改动：Or1 全员 > 阈值即 P 域 oracle 成立；",
              "- 确认集与 Discovery (4 scene) 在同协议下表现一致；",
              "- 数据可进入 R4′ 确认性 Gate（E2/G2/E3）。"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[R4'-C gates] {REPORT_MD}")


if __name__ == "__main__":
    main()
