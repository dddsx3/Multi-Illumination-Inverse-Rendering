#!/usr/bin/env python3
"""
Phase 2 云端自主训练编排器（跨平台：Windows/Linux 通用）

设计目标（任务书 v2.0 §1.1 + 用户云端运行要求）：
  1. 断点续训：每个 arm 拆成 SEGMENT_EPOCHS 的子任务；每 epoch 落盘
     checkpoint（trainer 内建），子任务结束/被杀后，下次自动从最新
     checkpoint_epoch_NNNN.pth --resume 热重启。
  2. 自动检测最新 checkpoint：扫描 checkpoints/{run_id}/ 取最大 N。
  3. 串行触发顺序：R0 -> F-N5-gray -> F-N5-RGB -> F-physcon ->
     F-resA -> F-resC -> F-albOff；每 arm 完成后立即跑冻结 test 评估。
  4. 幂等：progress.json 记录各 arm 状态；已完成的 arm 自动跳过。
  5. 被杀安全：任何时刻 kill 编排器或训练进程，重跑本脚本即可续。

用法:
  python run_phase2_all.py                # 全部 arm
  python run_phase2_all.py --only R0,F-N5-gray   # 指定 arm
  python run_phase2_all.py --status       # 只看进度
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRESS = HERE / "progress.json"
MANIFEST = HERE / "splits" / "synthetic_v3.json"

# 数据集与产物根目录（相对仓库；云端解压后保持结构即可）
# 2026-08-25 起云端计划取消，七臂全部转本地串行执行；P2_* 环境变量仍可覆盖
DATA_ROOT = os.environ.get("P2_DATA_ROOT", r"D:/data/synthetic_v3")
CKPT_ROOT = os.environ.get("P2_CKPT_ROOT", str(HERE.parent / "checkpoints"))
LOG_ROOT = os.environ.get("P2_LOG_ROOT", str(HERE.parent / "logs"))
EVAL_ROOT = os.environ.get("P2_EVAL_ROOT", str(HERE.parent / "eval_output"))

SEGMENT_EPOCHS = int(os.environ.get("P2_SEGMENT_EPOCHS", "10"))
TOTAL_EPOCHS = int(os.environ.get("P2_TOTAL_EPOCHS", "100"))
IMG_H = int(os.environ.get("P2_IMG_H", "256"))
IMG_W = int(os.environ.get("P2_IMG_W", "256"))
BATCH = int(os.environ.get("P2_BATCH", "8"))
THREADS = os.environ.get("P2_THREADS", "")

# arm 定义: (run_id, 训练额外参数, 冻结 test 评估额外参数)
# 已完成臂（本地人工执行，不在本表）：
#   p2_r0_gray_20260825   —— R0 对照臂，评估 eval_output/p2_r0_v3gray_test
#   p2_t22_f_n5gray_20260825 —— F-N5-gray 主交付臂，评估 eval_output/p2_t22_f_n5gray_test
# 消融臂评估旗标与训练旗标一一对应；架构/模态之外的变体参数
# （sh_constraint/res_hidden/residual_off）无法从权重形状推断，必须显式传。
ARMS = [
    ("p2_t22_f_n5rgb", ["--model", "fusion", "--modality", "rgb"],
     ["--model", "fusion", "--modality", "rgb"]),
    ("p2_t23_f_physcon", ["--model", "fusion", "--modality", "gray",
                          "--sh_constraint", "softplus"],
     ["--model", "fusion", "--sh_constraint", "softplus"]),
    ("p2_t25_f_resA", ["--model", "fusion", "--modality", "gray",
                       "--residual_off"],
     ["--model", "fusion", "--residual_off"]),
    ("p2_t25_f_resC", ["--model", "fusion", "--modality", "gray",
                       "--res_hidden", "32"],
     ["--model", "fusion", "--res_hidden", "32"]),
    ("p2_t25_f_albOff", ["--model", "fusion", "--modality", "gray",
                         "--no_per_light_albedo"],
     ["--model", "fusion"]),
]


def latest_checkpoint(run_id):
    d = Path(CKPT_ROOT) / run_id
    if not d.is_dir():
        return None, -1
    best = None
    for p in d.glob("checkpoint_epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if m:
            n = int(m.group(1))
            if best is None or n > best:
                best, bp = n, p
    if best is None and (d / "latest_model.pth").exists():
        return None, -1          # 只有 latest 无法确定 epoch => 视为需重扫
    return (str(bp) if best is not None else None), (best if best is not None else -1)


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {}


def save_progress(p):
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=1),
                        encoding="utf-8")


def run_cmd(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    return r.returncode


def eval_arm(run_id, extra=None):
    cmd = [sys.executable, "-u", "evaluate_model.py",
           "--checkpoint", str(Path(CKPT_ROOT) / run_id / "best_model.pth"),
           "--data_root", DATA_ROOT,
           "--split", "test",
           "--split_manifest", str(MANIFEST),
           "--batch_size", "4",
           "--out_dir", str(Path(EVAL_ROOT) / f"{run_id}_test")]
    if extra:
        cmd += extra
    return run_cmd(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的 run_id 子集")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    progress = load_progress()
    if args.status:
        for k, v in sorted(progress.items()):
            print(k, "->", v)
        return

    selected = ARMS if not args.only else [
        a for a in ARMS if a[0] in args.only.split(",")]

    for run_id, train_extra, eval_extra in selected:
        st = progress.get(run_id, {})
        if st.get("status") == "done":
            print(f"[skip] {run_id} 已完成")
            continue

        ck_dir = Path(CKPT_ROOT) / run_id
        epochs_done = 0
        latest, n_latest = latest_checkpoint(run_id)
        if latest:
            epochs_done = n_latest + 1
            print(f"[resume] {run_id}: 最新 checkpoint epoch={n_latest}")

        while epochs_done < TOTAL_EPOCHS:
            target = min(epochs_done + SEGMENT_EPOCHS, TOTAL_EPOCHS)
            seg_epochs = target - epochs_done
            cmd = [sys.executable, "-u", "main.py", "--mode", "train",
                   "--data_root", DATA_ROOT,
                   "--total_epochs", str(target),
                   "--stage1_epochs", "30", "--stage2_epochs", "30",
                   "--batch_size", str(BATCH), "--image_size", str(IMG_H), str(IMG_W),
                   "--num_lights", "5", "--device", "cuda",
                   "--use_amp", "--amp_dtype", "bf16",
                   "--split_manifest", str(MANIFEST),
                   "--run_id", run_id,
                   "--checkpoint_dir", str(ck_dir),
                   "--log_dir", str(Path(LOG_ROOT) / run_id),
                   "--viz_dir", str(Path(HERE.parent) / "visualizations" / run_id),
                   ] + train_extra
            if latest:
                cmd += ["--resume", "--checkpoint", latest]

            print(f"[segment] {run_id}: epochs {epochs_done}->{target}", flush=True)
            rc = run_cmd(cmd)

            new_latest, n_new = latest_checkpoint(run_id)
            if new_latest is None:
                print(f"[abort] {run_id} 本段未产出 checkpoint (rc={rc})，跳到下一段前重试一次")
            else:
                epochs_done = n_new + 1
            latest = new_latest

            progress[run_id] = {
                "status": "training",
                "epochs_done": epochs_done,
                "segments": st.get("segments", 0) + 1 if False else
                            progress.get(run_id, {}).get("segments", 0) + 1,
            }
            save_progress(progress)

            if epochs_done >= TOTAL_EPOCHS:
                break

        # 冻结 test 评估
        print(f"[eval] {run_id}")
        rc = eval_arm(run_id, extra=eval_extra)
        progress[run_id] = {"status": "done" if rc == 0 else "eval_failed",
                            "epochs_done": TOTAL_EPOCHS}
        save_progress(progress)
        print(f"[arm done] {run_id} eval_rc={rc}")

    print("ALL ARMS COMPLETE")


if __name__ == "__main__":
    main()
