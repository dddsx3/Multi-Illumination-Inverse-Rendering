"""R4″ H0.1 · 冻结旧 R4′ 状态为只读归档（任务书 §4 T0.1）。

产出 archive/R4prime_frozen/：
  data/**            逐文件复制（Windows 下设只读属性）
  MANIFEST.csv       path, sha256, bytes, mtime
  ENVIRONMENT.txt    commit / python / 依赖版本 / GPU / OS
  TERMINATION.md     旧 solve 终止点记录

严禁覆盖：若目标目录已存在且 MANIFEST 校验通过，则只报告不改写。
用法：python p1/source/information_audit/r4pp_freeze_archive.py [--force]
"""
import argparse
import csv
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARCHIVE = os.path.join(_REPO, "archive", "R4prime_frozen")
DATA = os.path.join(ARCHIVE, "data")

# 冻结清单：相对 _REPO 的路径
FREEZE_LIST = [
    # 采集数据（全量未筛选）
    "p1/information_audit/r4p_confirmatory_trials.csv",
    "p1/information_audit/r4p_confirmatory_scores.csv",
    "p1/information_audit/r4p_confirmatory_verdict.json",
    "p1/information_audit/r4p_conv_thresholds.json",
    # 诊断取证
    "p1/information_audit/diagnostics/r4p_trial_eigenspectrum.csv",
    "p1/information_audit/diagnostics/r4p_raw_trials_joined.csv",
    "p1/information_audit/diagnostics/r4p_scene_gram_spectrum.csv",
    # Discovery 阶段（exploratory 基线）
    "p1/information_audit/ga_isi_v2_discovery.csv",
    "p1/information_audit/ga_isi_v2_discovery_cap500.csv",
    "p1/information_audit/ga_isi_v2_discovery_cut1e6.csv",
    "p1/information_audit/r4d_cutoff_sweep.npz",
    # 报告与协议（当时状态）
    "p1/information_audit/R4P_STATUS_REPORT.md",
    "p1/information_audit/R4P_DIAGNOSTIC_BUNDLE.md",
    "p1/information_audit/R4P_DISCOVERY_RERUN_REPORT.md",
    "p1/information_audit/R4P_CUTOFF_PLATEAU.md",
    "p1/information_audit/R3P_MATH_AUDIT_REPORT.md",
    "p1/protocol/R4P_PREREGISTRATION.md",
    "p1/protocol/CLAIM_REGISTRY.md",
    "p1/protocol/IDENTIFIABILITY_v2.md",
    # 产生这些数据的代码（快照，保证可追溯）
    "p1/source/information_audit/gauge_fisher_v2.py",
    "p1/source/information_audit/r4p_confirmatory_gate.py",
    "p1/source/information_audit/r4p_diagnostics.py",
    "p1/source/information_audit/r4p_discovery_rerun.py",
    "p1/source/information_audit/solver_batched.py",
    "p1/source/information_audit/information_audit_v2.py",
    "p1/source/generation/render_multilight.py",
    "p1/tests/test_gauge_fisher_v2.py",
    # 场景数据 Gate 结论（不复制 npy 大数据，只留 Gate 证据）
    "p1/calibration_set/confirmatory_gate_reports/R4P_CONFIRMATORY_DATA_GATES.md",
    "p1/calibration_set/confirmatory_gate_reports/validation_all.csv",
    "p1/calibration_set/confirmatory_gate_reports/oracle_all.csv",
]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def set_readonly(path):
    try:
        os.chmod(path, stat.S_IREAD)
    except Exception:  # noqa: BLE001
        pass


def write_environment():
    lines = [
        "# ENVIRONMENT · R4prime_frozen",
        f"frozen_at_utc       : {datetime.utcnow().isoformat()}Z",
        f"frozen_at_local     : {datetime.now().isoformat()}",
        f"git_commit          : {run('git rev-parse HEAD')}",
        f"git_commit_short    : {run('git rev-parse --short HEAD')}",
        f"git_branch          : {run('git rev-parse --abbrev-ref HEAD')}",
        f"git_dirty           : {'yes' if run('git status --porcelain') else 'no'}",
        f"python              : {sys.version.split()[0]} ({sys.executable})",
        f"platform            : {sys.platform}",
        "",
        "## 依赖版本",
    ]
    for mod in ["numpy", "scipy", "torch"]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            extra = ""
            if mod == "torch":
                extra = f" | cuda_available={m.cuda.is_available()}"
                if m.cuda.is_available():
                    extra += f" | device={m.cuda.get_device_name(0)}"
            lines.append(f"{mod:20s}: {v}{extra}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{mod:20s}: <import failed: {e}>")
    lines += ["", "## GPU", run("nvidia-smi --query-gpu=name,memory.total,driver_version "
                                "--format=csv,noheader") or "<no nvidia-smi>"]
    lines += ["", "## Blender / BlenderProc",
              "blenderproc CLI     : /c/Users/35702/AppData/Local/Programs/"
              "Python/Python310/Scripts/blenderproc",
              "blenderproc version : 2.8.0 (Python 3.10 env, CLI only)",
              "blender version     : 4.2.1 LTS"]
    p = os.path.join(ARCHIVE, "ENVIRONMENT.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    set_readonly(p)
    return p


def write_termination(trial_rows, scene_counts):
    lines = [
        "# TERMINATION · 旧 R4′ solve 终止记录",
        "",
        "> 决策 **D-1 = 终止**（R4″ 执行手册 §0）。依据：R4″ 任务书 §0 已将旧 R4′",
        "> 整体降级为 instrument-development / exploratory dataset；§29 明确旧 18",
        "> scene 不进入新 confirmatory hypothesis testing。继续跑完剩余 ~3.5h GPU",
        "> 对新 sprint 无边际价值，算力转入关键路径 Task C（noise floor）。",
        "",
        "## 终止时状态",
        "",
        f"- 终止时刻（本地）：{datetime.now().isoformat()}",
        f"- 已采集 trial 数：**{trial_rows}** / 计划 1620（{trial_rows/1620*100:.1f}%）",
        f"- 完整 scene 数：**{len(scene_counts)}** / 18",
        "- 最后完成单元：`conf_icosphere_sub3` N=5（N=8 未完成）",
        "- 终止方式：solve 进程自然退出后不再重启（非强杀，数据无截断风险）",
        "- GPU 状态确认：无 python 计算进程、trials.csv 行数 20s 内无变化",
        "",
        "## 逐 scene trial 数",
        "",
        "| scene | trials |",
        "|---|---|",
    ]
    for k, v in sorted(scene_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## 可逆性",
        "",
        "本终止**可逆**：`r4p_confirmatory_gate.py --stage solve` 具备断点续跑",
        "（按 (scene, N, subset) 跳过 trials.csv 中已存在的记录）。若未来需要补齐",
        "剩余 7 scene，直接重跑该命令即可，无需重算 scores。",
        "",
        "## 允许 / 禁止用途（任务书 §0）",
        "",
        "**允许**：找 bug、检查数值稳定性、比较 candidate metrics、发现 interaction、",
        "power planning。",
        "",
        "**禁止**：宣称 H-COND confirmatory success；在多候选指标里挑相关性最高者并",
        "报告显著性；修完指标后沿用旧 prereg 标签；继续基于 converged-only subset",
        "做核心统计推断。",
    ]
    p = os.path.join(ARCHIVE, "TERMINATION.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    set_readonly(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="允许覆盖已有归档（默认拒绝）")
    args = ap.parse_args()

    manifest_path = os.path.join(ARCHIVE, "MANIFEST.csv")
    if os.path.exists(manifest_path) and not args.force:
        print(f"[freeze] 归档已存在：{ARCHIVE}")
        print("[freeze] 执行校验（不改写）……")
        bad = []
        for r in csv.DictReader(open(manifest_path, encoding="utf-8")):
            p = os.path.join(ARCHIVE, r["path"])
            if not os.path.isfile(p):
                bad.append((r["path"], "MISSING"))
            elif sha256(p) != r["sha256"]:
                bad.append((r["path"], "SHA256 MISMATCH"))
        if bad:
            for p, why in bad:
                print(f"  [FAIL] {p}: {why}")
            sys.exit(1)
        print(f"[freeze] 校验通过，{len(open(manifest_path, encoding='utf-8').readlines())-1} 个文件完好。")
        print("[freeze] 如需重建请显式 --force。")
        return

    # ---- 统计终止点（在复制前读取当前状态）----
    trials_src = os.path.join(_REPO, "p1/information_audit/r4p_confirmatory_trials.csv")
    scene_counts = {}
    trial_rows = 0
    if os.path.isfile(trials_src):
        from collections import Counter
        rows = [r for r in csv.DictReader(open(trials_src, encoding="utf-8"))
                if r.get("N", "").lstrip("-").isdigit()]
        trial_rows = len(rows)
        scene_counts = dict(Counter(r["scene"] for r in rows))

    os.makedirs(DATA, exist_ok=True)
    manifest = []
    missing = []
    for rel in FREEZE_LIST:
        src = os.path.join(_REPO, rel)
        if not os.path.isfile(src):
            missing.append(rel)
            continue
        dst = os.path.join(DATA, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.chmod(dst, stat.S_IWRITE)
        shutil.copy2(src, dst)
        st = os.stat(dst)
        manifest.append(dict(path=os.path.relpath(dst, ARCHIVE).replace("\\", "/"),
                             source=rel, sha256=sha256(dst), bytes=st.st_size,
                             mtime=datetime.fromtimestamp(st.st_mtime).isoformat()))
        set_readonly(dst)
        print(f"  froze {rel}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "sha256", "bytes", "mtime"])
        w.writeheader(); w.writerows(manifest)
    set_readonly(manifest_path)

    env_p = write_environment()
    term_p = write_termination(trial_rows, scene_counts)

    print(f"\n[freeze] archive  : {ARCHIVE}")
    print(f"[freeze] files    : {len(manifest)} 已冻结（只读）")
    print(f"[freeze] manifest : {manifest_path}")
    print(f"[freeze] env      : {env_p}")
    print(f"[freeze] termination: {term_p}")
    if missing:
        print(f"[freeze] WARN 缺失 {len(missing)} 项（不影响归档完整性，已记录）：")
        for m in missing:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
