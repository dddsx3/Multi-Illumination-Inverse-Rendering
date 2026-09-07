#!/usr/bin/env bash
# ==============================================================================
# exp8R v3 · 云实例一键运行入口(Ubuntu Linux)
#
# 用法(在云实例上, 一条命令全量完成):
#   bash run_exp8r_cloud.sh
#
# 可选环境变量:
#   DATA_ROOT      DiLiGenT pmsData 根目录(默认: /data/DiLiGenT/pmsData;
#                  若不存在则自动从 ./pmsData 或 ~/DiLiGenT/pmsData 探测)
#   WORKDIR        工作目录(默认: 脚本所在目录)
#   PARALLEL       并行物体数(默认: 物体数与 CPU 核数的较小值)
#   REPO_URL       Git 仓库(默认下方 DEFAULT_REPO)
#   BRANCH         分支(默认 main)
# ==============================================================================
set -euo pipefail

DEFAULT_REPO="https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git"
BRANCH="${BRANCH:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
DATA_ROOT="${DATA_ROOT:-}"

log()  { echo -e "\033[1;34m[exp8R-cloud]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[exp8R-cloud]\033[0m $*"; }
fail() { echo -e "\033[1;31m[exp8R-cloud][FATAL]\033[0m $*"; exit 1; }

cd "$WORKDIR"

# ------------------------------------------------------------------------------
# 1. 代码获取: 若当前目录没有主脚本, 从 GitHub 克隆
# ------------------------------------------------------------------------------
MAIN_SCRIPT="exp8r_diligent_discrimination_v3.py"
if [[ ! -f "$MAIN_SCRIPT" ]]; then
    log "当前目录无主脚本 → 从 GitHub 克隆仓库($BRANCH 分支)…"
    REPO_URL="${REPO_URL:-$DEFAULT_REPO}"
    if [[ ! -d .git ]]; then
        git clone --depth 1 -b "$BRANCH" "$REPO_URL" repo_tmp || fail "git clone 失败(检查网络/代理)"
        cp repo_tmp/cloud_migration/*.py repo_tmp/cloud_migration/*.md . 2>/dev/null || true
        # 也把主脚本从 critical_experiments 带上(开发态副本)
        cp repo_tmp/critical_experiments/$MAIN_SCRIPT . 2>/dev/null || true
        rm -rf repo_tmp
    fi
    [[ -f "$MAIN_SCRIPT" ]] || fail "克隆后仍未找到 $MAIN_SCRIPT — 请检查仓库分支"
    ok "代码就绪"
fi

# ------------------------------------------------------------------------------
# 2. 环境检测与自动配置(Python 3.10+ / numpy / scipy / pillow)
# ------------------------------------------------------------------------------
PY_BIN="${PYTHON:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    log "未找到 python3 → 自动安装…"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-python3-dev python3-venv || \
        sudo apt-get install -y -qq python3 python3-dev python3-venv || fail "apt 安装 python3 失败"
    else
        fail "非 Ubuntu/Debian 环境且无 python3 — 请手工安装后重试"
    fi
    PY_BIN=python3
fi

log "Python: $($PY_BIN --version)"

NEED_INSTALL=0
"$PY_BIN" - <<'EOF' 2>/dev/null || NEED_INSTALL=1
import numpy, scipy, PIL
EOF

if [[ "$NEED_INSTALL" == "1" ]]; then
    log "缺 numpy/scipy/pillow → 自动安装(优先 venv, 避免污染系统)…"
    if [[ ! -d .venv ]]; then
        "$PY_BIN" -m venv .venv || fail "创建 venv 失败"
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet numpy scipy pillow || fail "pip 安装依赖失败"
    ok "依赖安装完成(venv: .venv/)"
else
    log "依赖已齐(numpy/scipy/pillow)"
    # 若之前建过 venv 则沿用(断点续跑一致性)
    if [[ -d .venv ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
fi

# ------------------------------------------------------------------------------
# 3. 数据探测: pmsData 根目录
# ------------------------------------------------------------------------------
if [[ -z "$DATA_ROOT" ]]; then
    for cand in "/data/DiLiGenT/pmsData" "$WORKDIR/pmsData" "$HOME/DiLiGenT/pmsData" "$HOME/pmsData"; do
        if [[ -d "$cand/ballPNG" ]]; then DATA_ROOT="$cand"; break; fi
    done
fi
# DiLiGenT 数据(约 936MB)不在本仓库中(gitignore data/), 也无可靠的免注册直链
# ——官方页面 http://sites.google.com/site/diligentdt/ 提供下载, 但直链随托管方更新,
# 禁止硬编码未验证 URL(项目数字卫生纪律)。脚本提供三个可靠路径:
if [[ -n "${DATA_AUTO_DL:-}" ]]; then
    # 路径 3(进阶): 用户自行验证过的直链 → DATA_AUTO_DL=<url> 自动下载解压
    log "DATA_AUTO_DL 已设置 → 下载数据…"
    DL_DIR="${WORKDIR}/DiLiGenT_download"
    mkdir -p "$DL_DIR"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$DL_DIR/pmsData_dl" "$DATA_AUTO_DL"
    else
        curl -L --progress-bar -o "$DL_DIR/pmsData_dl" "$DATA_AUTO_DL"
    fi
    (cd "$DL_DIR" && { unzip -q pmsData_dl || tar xzf pmsData_dl; } ) || true
    for cand in "$DL_DIR"/*/pmsData "$DL_DIR"/pmsData "$DL_DIR"; do
        if [[ -d "$cand/ballPNG" ]]; then DATA_ROOT="$cand"; break; fi
    done
fi
[[ -n "$DATA_ROOT" && -d "$DATA_ROOT/ballPNG" ]] || fail "未找到 DiLiGenT pmsData(含 ballPNG)。
  数据获取(任选其一后重跑本脚本):
  1. 本地上传(最可靠, 数据已在 Windows 机的 D:/data/DiLiGenT/ 共 936MB):
       scp -r D:/data/DiLiGenT/pmsData <云用户>@<云IP>:~/DiLiGenT/pmsData
       然后: DATA_ROOT=~/DiLiGenT/pmsData bash run_exp8r_cloud.sh
       (或直接放到 ~/DiLiGenT/pmsData / /data/DiLiGenT/pmsData, 脚本自动探测)
  2. 官方下载(约 936MB): http://sites.google.com/site/diligentdt/
     下载解压后同上设置 DATA_ROOT
  3. 已验证直链: DATA_AUTO_DL=<真实URL> bash run_exp8r_cloud.sh(自动下载解压)"
ok "数据根: $DATA_ROOT"
ok "数据根: $DATA_ROOT"

# ------------------------------------------------------------------------------
# 4. 断点续跑: 已有逐物体 json 的物体自动跳过(--only 只跑缺失者)
# ------------------------------------------------------------------------------
OBJECTS=(ball bear buddha cat cow goblet harvest pot1 pot2 reading)
TODO=()
DONE=()
for o in "${OBJECTS[@]}"; do
    if [[ -f "exp8r_per_object_${o}PNG.json" ]]; then
        DONE+=("${o}PNG")
    else
        TODO+=("${o}PNG")
    fi
done
log "已完成: ${DONE[*]:-无} | 待跑: ${TODO[*]:-无}"

# ------------------------------------------------------------------------------
# 5. 并行执行(一物体一进程; 默认并行度 = min(物体数, CPU 核数))
# ------------------------------------------------------------------------------
NCPU="$(nproc)"
N_TODO="${#TODO[@]}"
PARALLEL="${PARALLEL:-$(( N_TODO < NCPU ? N_TODO : NCPU ))}"
[[ "$N_TODO" == "0" ]] && { ok "全部 10 物体已存在, 跳过计算"; } || {
    log "开始并行计算: ${N_TODO} 物体 × 并行度 ${PARALLEL}(CPU ${NCPU} 核)…"
    mkdir -p logs
    PIDS=()
    for o in "${TODO[@]}"; do
        ( "$PY_BIN" -u "$MAIN_SCRIPT" --only "$o" --data_root "$DATA_ROOT" \
            > "logs/${o}.log" 2>&1 ) &
        PIDS+=($!)
        # 简易并行池: 达到并行度时等待任一完成
        while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do
            wait -n 2>/dev/null || true
        done
    done
    wait
    # 逐个检查退出与产物
    FAILS=()
    for o in "${TODO[@]}"; do
        if [[ ! -f "exp8r_per_object_${o}PNG.json" ]]; then
            FAILS+=("$o")
        fi
    done
    if [[ ${#FAILS[@]} -gt 0 ]]; then
        for o in "${FAILS[@]}"; do
            echo "---- $o 失败, 日志尾部 ----"
            tail -n 15 "logs/${o}.log" || true
        done
        fail "以下物体未产出结果: ${FAILS[*]}(看 logs/*.log; 修复后重跑本脚本将自动续跑)"
    fi
    ok "全部物体计算完成"
}

# ------------------------------------------------------------------------------
# 6. 合并 + 终判定 + 打包
# ------------------------------------------------------------------------------
log "合并逐物体结果并计算终判定…"
"$PY_BIN" exp8r_merge.py || fail "合并失败"

# 结果打包: 主结果单文件 + 逐物体明细 + 日志 → zip
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTZIP="exp8r_v3_results_${STAMP}.zip"
"$PY_BIN" - "$OUTZIP" <<'EOF'
import zipfile, glob, os, sys
outzip = sys.argv[1]
files = (glob.glob("exp8r_diligent_discrimination_v3.json")
         + sorted(glob.glob("exp8r_per_object_*.json"))
         + sorted(glob.glob("logs/*.log")))
with zipfile.ZipFile(outzip, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, arcname=os.path.basename(f))
print(f"打包 {len(files)} 个文件 -> {outzip}")
EOF
ok "结果包: $OUTZIP(主结果单文件 exp8r_diligent_discrimination_v3.json + 逐物体明细 + 运行日志)"

# ------------------------------------------------------------------------------
# 7. 打印终判定摘要
# ------------------------------------------------------------------------------
"$PY_BIN" - <<'EOF'
import json
d = json.load(open("exp8r_diligent_discrimination_v3.json", encoding="utf-8"))
v = d["verdict"]
print()
print("=" * 66)
print("  exp8R v3 终判定")
print("=" * 66)
print(f"  物体数: {v['n_objects']} | 显著正: {v['n_significant_positive']}"
      f" | 显著负: {v['n_significant_total'] - v['n_significant_positive']}"
      f" | meta-p: {v['meta_p']:.3e}")
for k, x in v.get("per_object", {}).items():
    print(f"    {k:12s} ρ={x['rho']:+.3f} (p={x['p']:.5f}) | 朗伯残差 {x['lambert']}")
print()
print(f"  → {v['result']}")
print("=" * 66)
EOF

ok "全部完成。回传: $OUTZIP + 逐物体 json(云结果回收后本地跑 exp8r_merge.py 复核)"
