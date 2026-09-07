#!/usr/bin/env bash
# ==============================================================================
# show_progress.sh · exp8R v3 云并行进度总览
#
# 用法(运行期间另开一个终端):
#   watch -n 15 bash cloud_migration/show_progress.sh
#   (在仓库根直接: watch -n 15 bash show_progress.sh)
#
# 显示: 每物体 DONE/RUN/PENDING、已完成子集计数(共110子集/物体)、
#       每物体自估 ETA、整体墙钟 ETA(取各运行物体 ETA 最大者)。
#       逐物体明细在 logs/<物体>.log 的 [P] 行。
# ==============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 2
PY_BIN="${PYTHON:-python3}"
if [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
fi

"$PY_BIN" - <<'EOF'
import json, os, re, time

NOW = time.time()
OBJECTS = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
           "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG"]
print("=" * 74)
print(" exp8R v3 云并行进度  (刷新: %s)" % time.strftime("%H:%M:%S"))
print("=" * 74)
n_done = n_run = 0
etas = []
for o in OBJECTS:
    jf = "exp8r_per_object_%s.json" % o
    lf = "logs/%s.log" % o
    if os.path.exists(jf):
        try:
            d = json.load(open(jf, encoding="utf-8"))
            sp = d.get("spearman_n3") or {}
            rho = sp.get("rho", float("nan"))
            pv = sp.get("p", float("nan"))
            print("  DONE    %-12s N=3 rho=%+.3f (p=%.4f) | 朗伯残差 %s"
                  % (o, rho, pv, d.get("lambert_resid", "?")))
        except Exception as e:
            print("  DONE    %-12s (json 读取失败: %s)" % (o, e))
        n_done += 1
        continue
    if os.path.exists(lf):
        lines = []
        try:
            with open(lf, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "[P] " in line:
                        lines.append(line.strip())
        except Exception:
            pass
        stale = NOW - os.path.getmtime(lf)
        last = lines[-1].replace("      [P] ", "") if lines else "(第1子集进行中)"
        print("  RUN     %-12s %s   [日志更新 %.0fs 前]" % (o, last, stale))
        n_run += 1
        if lines:
            m = re.search(r"eta_obj=([0-9]+)s", lines[-1])
            if m:
                etas.append(NOW + int(m.group(1)))
        continue
    print("  PENDING %-12s" % o)
n_pend = len(OBJECTS) - n_done - n_run
print("-" * 74)
print("  完成 %d/%d | 运行中 %d | 排队 %d" % (n_done, len(OBJECTS), n_run, n_pend))
if etas:
    eta_wall = max(etas) - NOW
    print("  整体墙钟 ETA ≈ %.1f h(约 %s 收口)" % (eta_wall / 3600,
          time.strftime("%m-%d %H:%M", time.localtime(max(etas)))))
elif n_run:
    print("  整体墙钟 ETA: 待各运行物体完成第 1 个子集后自动出现")
print("=" * 74)
EOF