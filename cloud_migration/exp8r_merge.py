#!/usr/bin/env python3
"""exp8R v3 云并行结果合并 + 终判定(本地跑, 秒级)

用法: 把 10 个 exp8r_per_object_<name>.json 收集到同目录后:
  python exp8r_merge.py
输出: exp8r_diligent_discrimination_v3.json(与单进程全量版同构)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp8r_diligent_discrimination_v3.json"


def main():
    per = sorted(f for f in HERE.glob("exp8r_per_object_*.json") if ".partial" not in f.name)
    if not per:
        print("未找到 exp8r_per_object_*.json")
        sys.exit(1)
    objects = {}
    for f in per:
        d = json.loads(f.read_text(encoding="utf-8"))
        name = f.stem.replace("exp8r_per_object_", "")
        objects[name] = d
    n3_stats = {k: v["spearman_n3"] for k, v in objects.items() if "spearman_n3" in v}
    sig_pos = sum(1 for v in n3_stats.values() if v["p"] < 0.05 and v["rho"] > 0)
    sig_all = sum(1 for v in n3_stats.values() if v["p"] < 0.05)
    ps = [v["p"] for v in n3_stats.values()]
    meta_p = float(1 - chi2.cdf(-2 * sum(np.log(max(p, 1e-300)) for p in ps), 2 * len(ps))) if ps else float('nan')
    out = dict(
        objects={k: dict(rows=v["rows"], lambert_resid=v["lambert_resid"],
                         outlier_frac=v["outlier_frac"], spearman_n3=v["spearman_n3"])
                 for k, v in objects.items()},
        meta=dict(protocol="v3 geometry-known joint trf, cloud-parallel run",
                  per_object_files=[f.name for f in per]),
        verdict=dict(
            n_objects=len(n3_stats), n_significant_positive=sig_pos,
            n_significant_total=sig_all, meta_p=meta_p,
            per_object={k: dict(rho=round(v['rho'], 3), p=round(v['p'], 5),
                                lambert=round(objects[k]['lambert_resid'], 3))
                        for k, v in n3_stats.items()},
            acceptance="≥6/10 物体 LAE 显著正(p<0.05) → 支柱③真实侧成立",
            result=("支柱③真实侧成立" if sig_pos >= 6 else
                    f"混合/负结果: {sig_pos} 显著正 / {sig_all - sig_pos} 显著负(交下一轮裁决)")))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[merge] {len(per)} 物体 | 显著正 {sig_pos}/{len(n3_stats)} | meta-p={meta_p:.3e}")
    print("  →", out["verdict"]["result"])
    print(f"[merge] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
