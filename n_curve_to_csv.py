"""N 曲线 JSON → CSV / xlsx 转换（论文附录格式）

读取：
- eval_output/n_curve_synth_v3/n_curve_agg.json
- eval_diligent/n_curve/diligent_n_curve.json

输出：
- report_assets/n_curve_synth.csv
- report_assets/n_curve_diligent.csv
- report_assets/n_curve_combined.csv（双轨并列）
- report_assets/n_curve_synth.xlsx（仅当 openpyxl 可用时）
"""
import csv
import json
import os
import sys

import numpy as np


def load_synth():
    p = "eval_output/n_curve_synth_v3/n_curve_agg.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_diligent():
    p = "eval_diligent/n_curve/diligent_n_curve.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def write_synth_csv(agg, out):
    if agg is None:
        return
    ns = sorted([int(k) for k in agg.keys()])
    keys = ["image_psnr", "image_ssim",
            "albedo_mse", "albedo_mae", "albedo_si_mae",
            "normal_mae_deg", "normal_median_deg",
            "normal_acc_11_25", "normal_acc_22_5", "normal_acc_30",
            "depth_rmse", "depth_mae", "depth_si_rmse"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N"] + [f"{k}_mean" for k in keys] + [f"{k}_std" for k in keys])
        for N in ns:
            row = [N]
            for k in keys:
                if k in agg[str(N)]:
                    row.append(f"{agg[str(N)][k]['mean']:.6f}")
                else:
                    row.append("")
            for k in keys:
                if k in agg[str(N)]:
                    row.append(f"{agg[str(N)][k]['std']:.6f}")
                else:
                    row.append("")
            w.writerow(row)
    print(f"OK -> {out}")


def write_diligent_csv(diligent, out):
    if diligent is None:
        return
    nc = diligent.get("n_curve", {})
    ns = sorted([int(k) for k in nc.keys()])
    keys = ["mae_mean", "mae_std", "median_mean", "acc_11_25_mean", "n_objects", "n_subsets"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N"] + keys)
        for N in ns:
            row = [N]
            for k in keys:
                row.append(nc[str(N)].get(k, ""))
            w.writerow(row)
    print(f"OK -> {out}")


def write_combined_csv(synth, diligent, out):
    if synth is None and diligent is None:
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "N",
            "synth_v3_normal_mae_mean", "synth_v3_normal_mae_std",
            "synth_v3_albedo_si_mae_mean", "synth_v3_albedo_si_mae_std",
            "synth_v3_image_psnr_mean", "synth_v3_image_psnr_std",
            "diligent_MAE_mean", "diligent_MAE_std",
            "diligent_acc11.25_mean",
        ])
        # 用 union 的 N 值
        ns = set()
        if synth:
            ns.update(int(k) for k in synth.keys())
        if diligent:
            nc = diligent.get("n_curve", {})
            ns.update(int(k) for k in nc.keys())
        for N in sorted(ns):
            row = [N]
            if synth and str(N) in synth:
                row += [
                    f"{synth[str(N)].get('normal_mae_deg', {}).get('mean', float('nan')):.4f}",
                    f"{synth[str(N)].get('normal_mae_deg', {}).get('std', float('nan')):.4f}",
                    f"{synth[str(N)].get('albedo_si_mae', {}).get('mean', float('nan')):.4f}",
                    f"{synth[str(N)].get('albedo_si_mae', {}).get('std', float('nan')):.4f}",
                    f"{synth[str(N)].get('image_psnr', {}).get('mean', float('nan')):.4f}",
                    f"{synth[str(N)].get('image_psnr', {}).get('std', float('nan')):.4f}",
                ]
            else:
                row += [""] * 6
            if diligent and str(N) in diligent.get("n_curve", {}):
                d = diligent["n_curve"][str(N)]
                row += [
                    f"{d.get('mae_mean', float('nan')):.4f}",
                    f"{d.get('mae_std', float('nan')):.4f}",
                    f"{d.get('acc_11_25_mean', float('nan')):.4f}",
                ]
            else:
                row += [""] * 3
            w.writerow(row)
    print(f"OK -> {out}")


def main():
    os.makedirs("report_assets", exist_ok=True)
    synth = load_synth()
    diligent = load_diligent()
    if synth is None and diligent is None:
        print("[FAIL] no input found")
        sys.exit(1)
    write_synth_csv(synth, "report_assets/n_curve_synth.csv")
    write_diligent_csv(diligent, "report_assets/n_curve_diligent.csv")
    write_combined_csv(synth, diligent, "report_assets/n_curve_combined.csv")

    # 尝试生成 xlsx
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "synth_v3"
        if synth is not None:
            ws.append(["N"] + sorted(synth[next(iter(synth))].keys()))
            for N in sorted(synth.keys(), key=int):
                ws.append([N] + [synth[N][k].get("mean", "") for k in synth[N].keys()])
        wb.save("report_assets/n_curve_synth.xlsx")
        print("OK -> report_assets/n_curve_synth.xlsx")
    except ImportError:
        print("[skip] openpyxl not available, xlsx not generated")


if __name__ == "__main__":
    main()
