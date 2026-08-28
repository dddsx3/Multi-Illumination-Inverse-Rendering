#!/usr/bin/env python3
"""T2.7 报告资产管线（PRE-2A）：全部臂完成后「填空出图」。

输入（均已入库的 D3 原始数据）：
  eval_output/*/eval_summary.json      各臂冻结 test 13 项指标
  ../logs/<run_id>/tfevents            各臂训练期标量（val/total、val/metric_*）
  eval_output/n_curve_raw.json         N 敏感性原始曲线（eval_n_curve.py 产出）

输出（report_assets/）：
  comparison_matrix.md / .xlsx / .csv  论文级对比矩阵（最佳值加粗）
  curve_<arm>.png                      训练曲线（val loss + 3 个关键指标随 epoch）
  n_curve.png                          N 收敛曲线（存在 n_curve_raw.json 时）

设计约束：只读既有产物重新渲染，不做任何数值再计算（T5.2 图源可追溯）。
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE / "eval_output"
LOG_ROOT = HERE.parent / "logs"

# 展示顺序与标签；缺臂自动跳过（矩阵随回传进度逐步补齐）
ARMS = [
    ("p2_t26a_test_phase1recovered", "Phase1-recovered ckpt（参考行）"),
    ("p2_r0_v3gray_test",            "R0 对照臂（原 U-Net, gray）"),
    ("p2_t22_f_n5gray_test",         "F-N5-gray（FusionUNet 主交付）"),
    ("p2_t22_f_n5rgb_test",          "F-N5-RGB（双模态链路, v2 best）"),
    ("p2_t22_f_n5rgb_v2_test",       "v2 best（v3 test 重测, INC-0012 物理断言）"),
    ("p2_t23_f_physcon_test",        "F-physcon（softplus 物理约束）"),
    ("p2_t25_f_resA_test",           "F-resA（残差关闭）"),
    ("p2_t25_f_resA_v2",             "F-resA v2 重测（INC-0012 物理断言）"),
    ("p2_t25_f_resC_test",           "F-resC（残差容量 32）"),
    ("p2_t25_f_albOff_test",         "F-albOff（逐光照反照率关）"),
    ("p2_t25_f_albOff_v2",           "F-albOff v2 重测（INC-0012 物理断言）"),
    ("p2_t25_f_noFiLM_test",         "F-noFiLM（FiLM 调制关闭，判别实验 b）"),
    ("p2_t25_f_lowSmooth_test",      "F-lowSmooth（albedo_smooth=1.0，判别实验 c）"),
]
HIGHER_BETTER = {"image_psnr", "image_ssim",
                 "normal_acc_11_25", "normal_acc_22_5", "normal_acc_30"}


def load_summaries():
    rows = []
    for dirname, label in ARMS:
        p = EVAL_ROOT / dirname / "eval_summary.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        ms = s["metrics_mean_std"]
        # INC-0012: 物理断言摘要入矩阵
        phys = s.get("physical_assertions", {})
        if phys:
            for pk, pv in phys.items():
                ms[f"phys_{pk}"] = pv
        rows.append({
            "dir": dirname, "label": label,
            "arch": s.get("architecture", "?"), "modality": s.get("modality", "?"),
            "metrics": ms,
        })
    return rows


def build_matrix(rows):
    keys = sorted({k for r in rows for k in r["metrics"]})
    # INC-0012: 把 phys_* 物理断言项加入（albedo/depth violation ratio，越低越好）
    HIGHER_BETTER_LOCAL = HIGHER_BETTER | {
        # 物理断言 violation 越低越好（0% 是最优）
    }
    best = {}
    for k in keys:
        vals = [(r["metrics"][k]["mean"], i) for i, r in enumerate(rows)
                if k in r["metrics"]]
        if not vals:
            continue
        # 物理断言 violation ratio 用 min
        is_lower_better = (k in HIGHER_BETTER_LOCAL) or k.startswith("phys_")
        bi = (max(vals) if k in HIGHER_BETTER_LOCAL else min(vals))[1]
        best[k] = bi
    return keys, best


def fmt(v):
    return f"{v['mean']:.4f} ± {v['std']:.4f}"


def write_md(rows, keys, best, out):
    lines = ["# Phase 2 冻结 test 对比矩阵（13 项指标，mean ± std）", "",
             f"场景数：124（splits/synthetic_v3.json 冻结划分）；"
             f"生成：make_report_assets.py（图源可追溯至 eval_output/*/eval_summary.json）",
             ""]
    header = "| 臂 | 架构/模态 | " + " | ".join(keys) + " |"
    lines += [header,
              "|" + "---|" * (len(keys) + 2)]
    for i, r in enumerate(rows):
        cells = []
        for k in keys:
            cell = fmt(r["metrics"][k]) if k in r["metrics"] else "—"
            if best.get(k) == i:
                cell = f"**{cell}**"
            cells.append(cell)
        arch = f"{r['arch']}/{r['modality']}"
        lines.append(f"| {r['label']} | {arch} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_xlsx(rows, keys, out_xlsx):
    import pandas as pd
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame(
            [[r["label"]] + [fmt(r["metrics"].get(k)) if k in r["metrics"] else ""
                             for k in keys] for r in rows],
            columns=["臂"] + keys).to_excel(xw, sheet_name="matrix", index=False)
        for sheet, field in (("raw_mean", "mean"), ("raw_std", "std")):
            pd.DataFrame(
                [[r["label"]] + [r["metrics"][k][field] if k in r["metrics"] else None
                                 for k in keys] for r in rows],
                columns=["臂"] + keys).to_excel(xw, sheet_name=sheet, index=False)


def find_log_dirs(eval_dirname):
    """eval 目录名 -> 训练日志目录（处理 R0 的 v3gray/gray 命名差异与日期后缀）。"""
    base = eval_dirname[:-len("_test")] if eval_dirname.endswith("_test") else eval_dirname
    prefixes = {base}
    if "_v3gray" in base:
        prefixes.add(base.replace("_v3gray", "_gray"))
    found = []
    for d in LOG_ROOT.iterdir():
        if d.is_dir() and any(d.name.startswith(p) for p in prefixes):
            found.append(d)
    return found


CURVE_TAGS = [("val/total", "val loss"),
              ("val/metric_image_psnr", "PSNR (dB)"),
              ("val/metric_normal_mae_deg", "normal MAE (deg)"),
              ("val/metric_albedo_si_mae", "albedo si-MAE")]


def plot_curves(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    made = []
    for r in rows:
        log_dirs = find_log_dirs(r["dir"])
        series = {t: [] for t, _ in CURVE_TAGS}
        for ld in log_dirs:
            ea = EventAccumulator(str(ld), size_guidance={"scalars": 0})
            ea.Reload()
            for t, _ in CURVE_TAGS:
                if t in ea.Tags()["scalars"]:
                    series[t] += [(e.step, e.value)
                                  for e in ea.Scalars(t)]
        if not any(series.values()):
            print(f"[skip] {r['dir']}: 无 TB 标量")
            continue
        fig, axes = plt.subplots(1, len(CURVE_TAGS), figsize=(4 * len(CURVE_TAGS), 3))
        for ax, (t, name) in zip(axes, CURVE_TAGS):
            pts = sorted(set(series[t]))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("epoch", fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(r["label"], fontsize=10)
        fig.tight_layout()
        png = out_dir / f"curve_{r['dir'].replace('_test', '')}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        made.append(png.name)
        print(f"[curve] {png.name}")
    return made


def plot_n_curve(out_dir):
    src = EVAL_ROOT / "n_curve_raw.json"
    if not src.is_file():
        print("[skip] n_curve_raw.json 不存在（待各臂完成后由 eval_n_curve.py 产出）")
        return None
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    data = json.loads(src.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ns = sorted(int(k) for k in data)
    ax.errorbar(ns, [data[str(n)]["mae"] for n in ns],
                yerr=[data[str(n)].get("std", 0) for n in ns], marker="o")
    ax.set_xlabel("# lights N"); ax.set_ylabel("normal MAE (deg)")
    ax.grid(alpha=0.3); ax.set_title("N-agnostic inference convergence")
    fig.tight_layout()
    (out_dir / "n_curve.png").savefig(fig, dpi=140)
    plt.close(fig)
    print("[curve] n_curve.png")
    return "n_curve.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "report_assets"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    rows = load_summaries()
    if not rows:
        print("[FAIL] eval_output 下没有任何 eval_summary.json")
        sys.exit(1)
    print(f"[load] {len(rows)} 臂:", ", ".join(r["dir"] for r in rows))

    keys, best = build_matrix(rows)
    md = out_dir / "comparison_matrix.md"
    write_md(rows, keys, best, md)
    print(f"[md] {md}")
    try:
        write_xlsx(rows, keys, out_dir / "comparison_matrix.xlsx")
        print("[xlsx] comparison_matrix.xlsx")
    except Exception as e:
        print(f"[warn] xlsx 失败（{e}），md/csv 仍在")
    (out_dir / "comparison_matrix.csv").write_text(
        "\n".join(["arm," + ",".join(keys)] +
                  [r["label"] + "," + ",".join(
                      f"{r['metrics'][k]['mean']:.6f}" if k in r["metrics"] else ""
                      for k in keys) for r in rows]), encoding="utf-8")
    print("[csv] comparison_matrix.csv")

    plot_curves(rows, out_dir)
    plot_n_curve(out_dir)
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
