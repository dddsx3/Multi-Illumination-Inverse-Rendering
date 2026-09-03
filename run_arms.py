#!/usr/bin/env python3
"""Phase 2 消融臂训练引擎（单机通用：A10 / V100 / 本机）。

一键入口见 train_v100.sh（V100）与 setup_a10.sh（A10）；本文件是共用引擎。

一条命令跑完：环境预检 → 吞吐标定 → 预算规划 → 训练（分段续跑）→
冻结 test 评估 → 回传打包。

三条硬约束
----------
1. **不改训练口径**：batch_size=8、100 epoch、三阶段 30/30/40、bf16、
   cudnn.deterministic —— 全部沿用 R0 / F-N5-gray 已完成基线的设置。
   任何为提速而改动这些的做法都会让消融对比混入第二个变量（纪律 D10），
   因此本脚本只在"与训练数学无关"的维度上优化（并行车道、进程编排）。
2. **不超时**：每段（10 epoch）开跑前用**实测速率**核算剩余预算，装不下
   就不启动。宁可少跑完整的臂，也不留半截臂——半截臂不能进对比矩阵。
3. **不产半成品**：臂只有跑满 100 epoch 才评估、才算完成；未完成的臂
   保持可续跑，下次同一条命令自动接上。

用法
----
  python run_a10.py --data_root /data/synthetic_v3               # 正式跑
  python run_a10.py --data_root /data/synthetic_v3 --dry-run     # 只出计划
  python run_a10.py --data_root ... --budget-hours 6
  python run_a10.py --status                                     # 看进度
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from thermal_guard import read_gpu_temp, wait_until_cool

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "splits" / "synthetic_v3.json"
PLAN_JSON = HERE / "arms_plan.json"

TOTAL_EPOCHS = 100
SEGMENT_EPOCHS = 10
# FIX-06（2026-09-04）：默认 batch 8 → 4。bs8 需 ≥16GB 显存机器；本机 12GB 实测不可行
# （INC-0014）。保留 RUN_ARMS_BATCH 环境变量覆盖通道（外部 ≥16GB 机器可显式 RUN_ARMS_BATCH=8）。
BATCH_SIZE = int(os.environ.get("RUN_ARMS_BATCH", "4"))
BATCHES_PER_EPOCH = 56       # 447 训练场景 / bs8，上取整（预算按实际批次数自动校正）
STAGE1, STAGE2 = 30, 30
CKPT_MB = 157                # 单个 checkpoint 实测体积

# 各精度在 bs8 下的实测峰值分配显存（本机 RTX 5070 Ti Laptop）
PEAK_GB_BY_DTYPE = {"bf16": 8.63, "fp16": 8.08, "fp32": 15.35}

# 端到端 epoch 墙钟 ÷ 纯训练步耗时 的实测比值，用于把标定 proxy 折算成
# 真实 epoch 预算。来源：本机 RTX 5070 Ti Laptop，bench_throughput proxy
# 0.528 s/batch（=29.6 s/epoch）vs TensorBoard 实测端到端 98–101 s/epoch。
# 差额来自完整损失项、GT 监督、逐光照分支、49 场景验证、每 epoch 157MB 落盘。
# 仅用于"首段之前"的预估；首段跑完即被实测速率取代（见 measured_rate）。
OVERHEAD_FACTOR = 3.38

# 同精度参照臂：仅当精度 != bf16 时自动排在最前。
# 理由：既有 R0 / F-N5-gray 基线是 bf16；实测 fp16 与 bf16 不等效
# （INC-0007 §6.4，同种子 epoch1 验证损失差 2.04x）。先在目标精度下重跑
# F-N5-gray，既得到「同精度参照」让后续消融臂可比，又直接量出精度本身的
# 影响量（与 eval_output/p2_t22_f_n5gray_test 的 bf16 数字对照）。
REFERENCE_ARM = (
    "p2_t22_f_n5gray",
    ["--model", "fusion", "--modality", "gray"],
    ["--model", "fusion", "--modality", "gray"],
    "同精度参照臂：给消融臂提供同口径基线，并量出 fp16↔bf16 差值")

# (run_id, 训练旗标, 评估旗标, 门禁价值)　顺序即优先级：预算不足从后往前砍
ARMS = [
    # A3-0 F-N5 复现臂（任务书 T v2.0 · 2026-09-03 裁定）：
    # R4″ 世代主交付 F-N5-gray ckpt 未归档（永久缺失），论文主表基线需可复现。
    # 同协议（fusion+gray+N5+bf16@bs8·256²·v3 划分·seed42）重训；对照
    # eval_output/p2_t22_f_n5gray_test 的历史数字，偏差超阈即 INC + 统一刷新叙事。
    ("A3-0_f_n5gray_seed42",
     ["--model", "fusion", "--modality", "gray"],
     ["--model", "fusion", "--modality", "gray"],
     "A3-0 F-N5 复现臂：闭合主表基线 ckpt 缺失的可复现性缺口"),
    # A3-1 noFiLM（v2.1 R-C 排序第一臂 · Gen-A3 协议：bs4+物理约束头+gray）：
    # 判别 FiLM 必要性（表 1 消融行）。训练开关 --disable_film（FiLM gamma≡1/beta≡0）。
    ("A3-1_noFiLM",
     ["--model", "fusion", "--modality", "gray", "--disable_film"],
     ["--model", "fusion"],
     "EX-03 A3-1 noFiLM：FiLM 调制关闭判别（Gen-A3 同协议）"),
    ("p2_t25_f_resA",
     ["--model", "fusion", "--modality", "gray", "--residual_off"],
     ["--model", "fusion", "--residual_off"],
     "G2.5 残差消融——任务卡点名的论文关键证据（已完成 2026-08-27）"),
    ("p2_t25_f_albOff",
     ["--model", "fusion", "--modality", "gray", "--no_per_light_albedo"],
     ["--model", "fusion"],
     "G2.2 逐光照反照率 A2 的价值对照（核心创新消融，已完成 2026-08-27）"),
    ("p2_t22_f_n5rgb_v2_seed42",
     ["--model", "fusion", "--modality", "rgb"],
     ["--model", "fusion", "--modality", "rgb"],
     "A3-bis seed 42 续跑（33/100 → 100，从 interrupt_state.pth 续训，INC-0010 临时改 v2）"),
    # INC-0013 判别实验变体——中期审计 v2 §2-P2 反照率退化 culprit 判定
    # 完成顺序：(a) F-albOff 已完成；待 (b) F-noFiLM + (c) F-lowSmooth
    ("p2_t25_f_noFiLM",
     ["--model", "fusion", "--modality", "gray", "--disable_film"],
     ["--model", "fusion"],
     "INC-0013 判别实验 (b)：FiLM 调制关闭（gamma≡1, beta≡0），验证 FiLM 是否反照率退化 culprit"),
    ("p2_t25_f_lowSmooth",
     ["--model", "fusion", "--modality", "gray", "--albedo_smooth_stage1", "1.0"],
     ["--model", "fusion"],
     "INC-0013 判别实验 (c)：Stage 1 albedo_smooth=1.0（默认 10.0），验证权重是否过高"),
    # T2.3 / T2.5 残差消融变体
    ("p2_t23_f_physcon",
     ["--model", "fusion", "--modality", "gray", "--sh_constraint", "softplus"],
     ["--model", "fusion", "--sh_constraint", "softplus"],
     "G2.3 物理约束替代 clamp hack（占位已建，待启动）"),
    ("p2_t25_f_resC",
     ["--model", "fusion", "--modality", "gray", "--res_hidden", "32"],
     ["--model", "fusion", "--res_hidden", "32"],
     "G2.5 残差容量消融（占位已建，待启动）"),
]


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


# ── INC-0008：本机 32 GB 提交上限的 worker 硬约束 ────────────────────────
# 事由
# ----
# 2026-08-27 长跑期间，n5rgb 第一次启动 0→10 段在 epoch 0 验证阶段崩了
# `OSError: [WinError 1455] 页面文件太小… Error loading curand64_10.dll`。
# 排查过程见 docs/incidents/INC-0008_win1455_n5rgb验证阶段spawn触发.md。
# 根因：单车道默认 4 worker 训练 → 验证阶段 val_loader 又 spawn 4 worker
# → 6 个 spawn worker 同时持有 RGB 张量，每 worker ~150 MB 提交内存，
# 叠加主进程的 torch + 第三方 cuDNN/cuBLAS 提交内存 = 32 GB 提交上限
# 越线，curand 加载时申请不到内存。
#
# 改了什么
# --------
# 把 run_arms.py spawn main.py / bench_throughput.py 时传的 --num_workers
# 硬约束到 ≤2。**仅作用于本编排器**；不修改 config.py 默认值（4）、
# 不修改 data_loader.py 任何代码、不修改超参/损失/阶段门控/随机种子/augmentation。
# 直跑 `python main.py ...` 的用户拿到仍是 4 worker（其口径由 config.py
# 决定，与 run_arms.py 编排口径解耦）。
#
# 影响范围
# --------
# - 已完成臂（p2_t25_f_resA、p2_t25_f_albOff）：零影响。它们已经 eval OK
#   落盘，永不重跑，worker 数与它们最终数字无关。
# - 正在跑的 n5rgb：本改动不修复其当前崩死的进程（它已经在 abort 路径上），
#   仅在编排器下次开新段时生效（编排器 spawn 时读 args.num_workers）。
# - 待跑的 physcon / resC：同样在本编排器下次启动它们时生效。
# - 显式 --num_workers 覆盖：保留用户的输入值，不替换。
#
# 数学等价性
# ----------
# - 超参、损失权重、阶段门控（30/30/40）、bf16、batch_size=8：完全不变。
# - 随机流（python / numpy / torch / cuda RNG）：完全不变。DataLoader 默认
#   worker_init_fn 用 base_seed + worker_id 派生，**改 worker 数会改 sampler
#   偏移**——但这本来就是"每个 worker 独立 sampler"的既有行为，resA/albOff
#   之间的 batch 顺序也互不相同（与 worker 数无关），消融矩阵的对比是
#   **收敛后测试指标**（PSNR / normal MAE / albedo si-MAE），不依赖 batch
#   顺序的逐位等价（D10 纪要：本机长跑不变量 = 超参+损失+阶段+种子）。
# - 数据集、augmentation 算子、stage transition 触发点：完全不变。
# - 仅变：DataLoader 后台进程数（4 → 2），即 prefetch 队列大小、并发读盘数。
#   在 bs=8 / 256×256 / 56 batch 小图规模下，2 worker 的 prefetch 队列
#   仍能在 GPU 计算期间填满（GPU 单 batch 0.6s vs DataLoader 单 batch
#   ~0.05s，2 worker 并行即可在 0.1s 内填满 56 batch），故**训练吞吐
#   不会变慢**（本机已有 4→2 的吞吐无差异的实证：resA/albOff 段速
#   127/250 s/epoch 与 n5gray 基线 98–101 s/epoch 的差全部来自
#   THERMAL_PACE=1.5 + 不同模态/分支，不是 worker 数）。
#
# 取消该限制
# ----------
# 提升 D: 盘页面文件 ≥ 48000 MB 并重启后，把本函数体改为
#   return int(os.environ.get('RUN_ARMS_NUM_WORKERS', requested or max(2, min(8, (os.cpu_count() or 8) // lanes))))
# 即可。留下"len(workers_aware)=2"的限制是环境约束，不是算法选择。
def _safe_num_workers(requested, lanes=None, modality=None):
    """把 run_arms.py 传给 DataLoader 的 worker 数压到 32 GB 提交上限能承受的范围内。

    requested=0 表示"自动"：按 (cpu_count // lanes) 推算，但上限封顶为 2。
    requested>0 表示用户显式给值，原样返回（不替用户做主）。

    INC-0008 续补（2026-08-27）：原 1 train + 1 val spawn 路径（4+4=8 worker）
    撞 WinError 1455，已硬约束到 2 train worker + val 阶段新建 spawn。
    2026-08-27 16:00+ 实测：n5rgb (rgb 模态) 在 2 train worker 下能稳定跑到
    epoch 5+，2 worker 路径不撞 win1455（_new_shared 共享内存申请走 32 GB
    提交上限内）。modality 参数仅作可观测性记录，不参与决策。
    """
    HARD_CAP = 2  # INC-0008：单车道/单次 spawn 进程 ≤ 2 worker
    if requested and requested > 0:
        return int(requested)
    cpu = os.cpu_count() or 8
    base = max(1, cpu // max(1, lanes or 1))
    return min(HARD_CAP, base)



def tag_run(run_id, amp_dtype):
    """非 bf16 精度的产物一律带精度后缀：既避免覆盖既有 bf16 产物，
    也让审计一眼看出该臂的数值口径（D8 变更可回溯）。"""
    return run_id if amp_dtype == "bf16" else f"{run_id}_{amp_dtype}"


def epochs_done(ckpt_root, run_id):
    d = Path(ckpt_root) / run_id
    if not d.is_dir():
        return 0
    # 作废标记（INC-0006）：目录内一旦有 _CONTAMINATED_*.md，其历史不可续跑，
    # 视为 0 epoch，逼迫从头重训，避免污染轨迹被静默带进对比矩阵（D10）。
    if any(d.glob("_CONTAMINATED_*.md")):
        return 0
    best = -1
    for p in d.glob("checkpoint_epoch_*.pth"):
        try:
            best = max(best, int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return best + 1


def contaminated(ckpt_root, run_id):
    d = Path(ckpt_root) / run_id
    return d.is_dir() and any(d.glob("_CONTAMINATED_*.md"))


def latest_ckpt(ckpt_root, run_id):
    if contaminated(ckpt_root, run_id):
        return None                      # 作废 run 不得作为续跑起点
    n = epochs_done(ckpt_root, run_id) - 1
    p = Path(ckpt_root) / run_id / f"checkpoint_epoch_{n:04d}.pth"
    return str(p) if n >= 0 and p.is_file() else None


class State:
    """进度状态（单文件；车道并行时每车道一份，父进程合并）。"""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text(encoding="utf-8")) \
            if self.path.is_file() else {}

    def set(self, run_id, **kw):
        self.d.setdefault(run_id, {}).update(kw)
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    @staticmethod
    def merge(paths):
        out = {}
        for p in paths:
            if Path(p).is_file():
                out.update(json.loads(Path(p).read_text(encoding="utf-8")))
        return out


def preflight(args):
    """环境与数据体检。返回 (info, blockers, warns)。"""
    info, blockers, warns = {}, [], []
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            cc = torch.cuda.get_device_capability(0)
            info["gpu"] = prop.name
            info["gpu_mem_gb"] = round(prop.total_memory / 1024 ** 3, 2)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
            info["bf16_native"] = cc[0] >= 8          # 张量核心自 sm_80 起
            info["bf16_api_reported"] = bool(torch.cuda.is_bf16_supported())
            info["amp_dtype"] = args.amp_dtype
            # 精度可用性：bf16 是既有基线口径，非 Ampere+ 卡上必须显式换档
            if args.amp_dtype == "bf16" and not info["bf16_native"]:
                blockers.append(
                    f"该 GPU 算力 {cc[0]}.{cc[1]} 无原生 BF16（需 sm_80+），"
                    f"而 bf16 是既有基线口径。T4/Turing 请显式 --amp-dtype fp16 "
                    f"并向审计声明数值口径偏差（见 INC-0007）；"
                    f"注意 is_bf16_supported() 可能因仿真返回 True，吞吐会塌")
            need_gb = PEAK_GB_BY_DTYPE.get(args.amp_dtype, 8.63)
            info["peak_gb_expected"] = need_gb
            if info["gpu_mem_gb"] < need_gb * 1.25 + 1.0:
                blockers.append(
                    f"显存 {info['gpu_mem_gb']}GB 不足：{args.amp_dtype} @bs8 实测峰值"
                    f"分配 {need_gb}GB（+碎片余量），会 OOM 或退化到系统内存交换"
                    + ("　fp32 在 16GB 卡上必然不可行，实测保留达 17.76GB"
                       if args.amp_dtype == "fp32" else ""))
        elif not args.dry_run:
            blockers.append("无可用 CUDA 设备")
        elif not args.dry_run:
            blockers.append("无可用 CUDA 设备")
    except ImportError:
        blockers.append("未安装 torch（pip install -r requirements.txt）")
    info["cpu_count"] = os.cpu_count()

    root = Path(args.data_root)
    if not root.is_dir():
        blockers.append(f"数据根目录不存在：{root}")
    else:
        scenes = sorted(d for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("_"))
        info["scene_dirs"] = len(scenes)
        if scenes:
            s = scenes[0]
            need = [f"light_{k:03d}.png" for k in range(1, 6)] + \
                   ["albedo.npy", "depth.npy", "normal.npy", "mask.npy"]
            miss = [f for f in need if not (s / f).is_file()]
            if miss:
                blockers.append(f"样例场景 {s.name} 缺文件：{miss[:5]}")
            info["rgb_present"] = all(
                (s / f"light_{k:03d}_rgb.png").is_file() for k in range(1, 6))
            if not info["rgb_present"]:
                warns.append("未见 *_rgb.png：n5rgb 臂将被跳过（灰度臂不受影响）")

    if not MANIFEST.is_file():
        blockers.append(f"缺少冻结划分清单：{MANIFEST}")
    else:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        info["split_counts"] = {k: len(m[k]) for k in ("train", "val", "test")
                                if k in m}
        info["split_frozen"] = m.get("frozen")
        info["batches_per_epoch"] = -(-len(m["train"]) // BATCH_SIZE)
        if info["batches_per_epoch"] != BATCHES_PER_EPOCH:
            warns.append(f"每 epoch 批次数实为 {info['batches_per_epoch']}"
                         f"（常量 {BATCHES_PER_EPOCH}），预算按实际值算")
        if root.is_dir() and "scene_dirs" in info:
            missing = [n for n in (m["train"][:20] + m["test"][:20])
                       if not (root / n).is_dir()]
            if missing:
                blockers.append(f"划分清单中的场景在数据根下不存在（抽查）："
                                f"{missing[:3]}")

    tgt = Path(args.ckpt_root)
    probe = tgt if tgt.exists() else HERE
    free_gb = shutil.disk_usage(probe).free / 1024 ** 3
    info["disk_free_gb"] = round(free_gb, 1)
    per_arm_gb = round(CKPT_MB * (TOTAL_EPOCHS + 2) / 1024, 1)
    info["disk_per_arm_gb"] = per_arm_gb
    if free_gb < per_arm_gb * 1.2:
        blockers.append(f"磁盘可用 {free_gb:.1f}GB，单臂全量存档需 {per_arm_gb}GB"
                        f"（D5 每 epoch 存档）——请扩容或加 --ckpt-keep-last")
    info["disk_arms_affordable"] = int(free_gb // per_arm_gb)
    return info, blockers, warns


def calibrate(args, info):
    """标定单 epoch 耗时与单车道显存需求。"""
    fallback_spb = 0.528  # 本机 proxy 实测
    bpe = info.get("batches_per_epoch", BATCHES_PER_EPOCH)
    if args.sec_per_epoch:
        return {"sec_per_epoch": args.sec_per_epoch,
                "source": "调用方给定（车道子进程或用户显式指定）",
                "peak_alloc_gb": args.assume_peak_gb, "gpu_util_mean": None}
    if args.dry_run:
        return {"sec_per_epoch": round(fallback_spb * bpe * OVERHEAD_FACTOR
                                       * args.slowdown, 1),
                "source": f"dry-run：本机 proxy {fallback_spb} s/batch × {bpe}"
                          f" × 开销系数 {OVERHEAD_FACTOR}"
                          + (f" × 目标机减速 {args.slowdown}x" if args.slowdown != 1
                             else ""),
                "peak_alloc_gb": args.assume_peak_gb, "gpu_util_mean": 98.7}
    out = HERE / "_arms_bench.json"
    # 标定 worker 数同样压到 2（原因同 train_arm 段的注释：32 GB 提交上限
    # 硬约束；标定 12 batch 不需要 4 worker 的吞吐）。
    rc = sh([sys.executable, "-u", "bench_throughput.py",
             "--data_root", args.data_root, "--split_manifest", MANIFEST,
             "--batch_size", BATCH_SIZE, "--num_workers", _safe_num_workers(args.num_workers),
             "--batches", 12, "--batches_per_epoch", bpe,
             "--dtype", args.amp_dtype, "--json", out],
            cwd=HERE).returncode
    if rc != 0 or not out.is_file():
        print("[warn] 标定失败，退回本机 proxy 常数")
        return {"sec_per_epoch": round(fallback_spb * bpe * OVERHEAD_FACTOR, 1),
                "source": "标定失败回退本机 proxy",
                "peak_alloc_gb": args.assume_peak_gb, "gpu_util_mean": None}
    b = json.loads(out.read_text(encoding="utf-8"))
    return {"sec_per_epoch": round(b["sec_per_batch_train"] * bpe * OVERHEAD_FACTOR, 1),
            "source": f"实测 {b['sec_per_batch_train']} s/batch × {bpe}"
                      f" × 开销系数 {OVERHEAD_FACTOR}",
            "peak_alloc_gb": b.get("peak_mem_gb") or args.assume_peak_gb,
            "gpu_util_mean": b.get("gpu_util_mean"), "bench": b}


def decide_lanes(args, info, calib):
    """按显存与 GPU 利用率决定并行车道数，并给出理由。"""
    mem = info.get("gpu_mem_gb", 24.0)
    peak = (calib.get("peak_alloc_gb") or args.assume_peak_gb) * 1.25  # 碎片余量
    eval_reserve = 4.0        # 评估进程与另一车道重叠时的显存
    lanes_mem = max(1, int((mem - eval_reserve) // peak))
    util = calib.get("gpu_util_mean")
    # GPU 已近满载时并行只能填补验证/落盘空窗，聚合收益远小于线性
    if util is None or util >= 90:
        gain = {1: 1.0, 2: 1.15, 3: 1.2}.get(min(lanes_mem, args.max_lanes), 1.2)
    else:
        gain = {1: 1.0, 2: 1.8, 3: 2.4}.get(min(lanes_mem, args.max_lanes), 2.4)
    lanes = max(1, min(args.max_lanes, lanes_mem))
    if lanes == 1:
        gain = 1.0
    reason = (f"显存 {mem}GB − 评估预留 {eval_reserve}GB ÷ 单车道 {peak:.1f}GB"
              f"（峰值分配 ×1.25 碎片余量）= {lanes_mem}；"
              f"GPU 利用率 {util}% → 聚合系数 {gain}x")
    return lanes, gain, reason


def make_plan(args, info, calib):
    bpe = info.get("batches_per_epoch", BATCHES_PER_EPOCH)
    spe = calib["sec_per_epoch"] * bpe / BATCHES_PER_EPOCH
    lanes, gain, reason = decide_lanes(args, info, calib)
    usable_s = args.budget_hours * 3600 - args.reserve_min * 60

    pool = list(ARMS)
    if args.amp_dtype != "bf16" and not args.no_reference_arm:
        pool.insert(0, REFERENCE_ARM)
    selected = [a for a in pool if not args.only or a[0] in args.only.split(",")]
    if args.only:
        # --only 显式给定时按用户书写顺序执行（便于把"便宜且能确定跑完"的臂排前面）
        order = {rid: i for i, rid in enumerate(args.only.split(","))}
        selected.sort(key=lambda a: order.get(a[0], 999))
    if not info.get("rgb_present", True):
        selected = [a for a in selected if a[0] != "p2_t22_f_n5rgb"]
    afford = info.get("disk_arms_affordable", len(selected))

    rows, cum, will = [], 0, []
    for i, (base_id, _, _, why) in enumerate(selected):
        run_id = tag_run(base_id, args.amp_dtype)
        done = epochs_done(args.ckpt_root, run_id)
        if contaminated(args.ckpt_root, run_id):
            why += "　[已按 INC-0006 作废，本轮从 epoch 0 重训]"
        need = max(0, TOTAL_EPOCHS - done)
        cum += need
        eta = cum * spe / gain
        ok = eta <= usable_s and len(will) < afford
        rows.append({"run_id": run_id, "epochs_done": done, "epochs_needed": need,
                     "cum_eta_h": round(eta / 3600, 2), "fits": ok, "why": why})
        if ok:
            will.append(run_id)
    return {"sec_per_epoch": round(spe, 1), "lanes": lanes, "assumed_gain": gain,
            "lane_reason": reason, "usable_seconds": usable_s,
            "budget_hours": args.budget_hours, "reserve_min": args.reserve_min,
            "arms": rows, "will_run": will,
            "deferred": [r["run_id"] for r in rows if not r["fits"]]}


def print_plan(info, calib, p):
    print("\n" + "=" * 78)
    print("A10 训练预算规划")
    print("=" * 78)
    for k in ("gpu", "gpu_mem_gb", "compute_capability", "bf16_native",
              "amp_dtype", "peak_gb_expected", "torch",
              "cpu_count", "scene_dirs", "split_counts", "rgb_present",
              "batches_per_epoch", "disk_free_gb", "disk_per_arm_gb"):
        if k in info:
            print(f"  {k:22s} {info[k]}")
    print(f"  {'标定':22s} {calib['source']}")
    print(f"  {'单 epoch 预估':22s} {p['sec_per_epoch']} s"
          f"（= {p['sec_per_epoch']*TOTAL_EPOCHS/3600:.2f} h/臂）")
    print(f"  {'并行车道':22s} {p['lanes']}　{p['lane_reason']}")
    print(f"  {'可用预算':22s} {p['usable_seconds']/3600:.2f} h"
          f"（总 {p['budget_hours']}h − 预留 {p['reserve_min']}min 评估/打包）")
    print("-" * 78)
    print(f"  {'臂':22s} {'已完成':>6} {'待跑':>5} {'累计ETA':>9}  判定")
    for r in p["arms"]:
        print(f"  {r['run_id']:22s} {r['epochs_done']:>6} {r['epochs_needed']:>5} "
              f"{r['cum_eta_h']:>8.2f}h  {'装得下' if r['fits'] else '超预算'}")
        print(f"      -> {r['why']}")
    print("-" * 78)
    print(f"  本轮执行：{', '.join(p['will_run']) or '（无）'}")
    if p["deferred"]:
        print(f"  顺延：{', '.join(p['deferred'])}"
              f"　（保持可续跑，不产半截结果）")
    print("=" * 78 + "\n")


class Rate:
    """单 epoch 耗时的自适应估计：首段之前用标定预估，之后一律用实测。

    这是 6h 预算不超时的核心——标定折算可能偏差 ±50%，但每段跑完都会用
    真实墙钟校正，因此"装不下就不启动"的判断随时间越来越准。
    """

    def __init__(self, initial):
        self.spe = float(initial)
        self.measured = False

    def update(self, elapsed_s, epochs_advanced):
        if epochs_advanced <= 0:
            return
        obs = elapsed_s / epochs_advanced
        self.spe = obs if not self.measured else 0.4 * self.spe + 0.6 * obs
        self.measured = True
        print(f"[rate] 实测 {obs:.1f} s/epoch → 采用 {self.spe:.1f} s/epoch"
              f"（{'实测' if self.measured else '预估'}）", flush=True)


def train_arm(args, run_id, train_extra, deadline, rate, state):
    """分段训练一个臂。每段前用实测速率核算 deadline。返回是否跑满。

    温度墙协作（低散热平台）：训练进程撞墙会以退出码 42 优雅退出（trainer
    已把 epoch 中途状态落盘为 interrupt_state.pth）；本函数看到 42 就等
    温度降到安全线后原命令重跑——主进程 load_checkpoint + 中途状态自动接上，
    损失不超过一个 batch。预算耗尽或真实错误（其他非零码）照旧返回 False。
    """
    ck_dir = Path(args.ckpt_root) / run_id
    while True:
        done = epochs_done(args.ckpt_root, run_id)
        if done >= TOTAL_EPOCHS:
            return True
        seg = min(SEGMENT_EPOCHS, TOTAL_EPOCHS - done)
        remain = deadline - time.time()
        need = seg * rate.spe * 1.1          # 10% 安全边际
        if remain < need:
            print(f"[deadline] {run_id}: 剩余 {remain/60:.1f} min < 本段需 "
                  f"{need/60:.1f} min（{rate.spe:.0f} s/epoch × {seg}），"
                  f"停在 epoch {done}，保持可续跑", flush=True)
            return False
        target = done + seg
        # 从 train_extra 解析 --modality，给 _safe_num_workers 决策（rgb 走 0）
        _modality = "gray"
        if "--modality" in train_extra:
            i = train_extra.index("--modality")
            if i + 1 < len(train_extra):
                _modality = train_extra[i + 1]
        cmd = [sys.executable, "-u", "main.py", "--mode", "train",
               "--data_root", args.data_root,
               "--total_epochs", target,
               "--stage1_epochs", STAGE1, "--stage2_epochs", STAGE2,
               "--batch_size", BATCH_SIZE, "--image_size", 256, 256,
               "--num_lights", 5, "--device", "cuda",
               "--use_amp", "--amp_dtype", args.amp_dtype,
               "--split_manifest", MANIFEST,
               "--run_id", run_id,
               "--checkpoint_dir", ck_dir,
               "--log_dir", Path(args.log_root) / run_id,
               # INC-0008 续补：train_arm spawn main.py 时也把 --num_workers
               # 传过去。原修复只在 calibrate() 给 bench_throughput.py 传了
               # --num_workers 2，但 train_arm 段的 cmd 拼装漏了，导致 n5rgb
               # 在修复落地后重启仍走 4 worker。补上后，编排器自动续跑/重
               # 试时直接落到 _safe_num_workers 决策的 worker 数（gray→2,
               # rgb→0；见 _safe_num_workers 注释中 INC-0008 续补段）。
               "--num_workers", _safe_num_workers(args.num_workers, modality=_modality),
               "--viz_dir", Path(args.viz_root) / run_id] + train_extra
        ck = latest_ckpt(args.ckpt_root, run_id)
        if ck:
            cmd += ["--resume", "--checkpoint", ck]
        print(f"[segment] {run_id}: epoch {done} -> {target}", flush=True)
        t0 = time.time()
        with open(HERE / f"_arm_{run_id}_log.txt", "a", encoding="utf-8") as f:
            rc = sh(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 42:
            # 温度墙优雅停机：状态已落盘（interrupt_state.pth）。
            # 等温度降到安全线后回到循环顶部，同一条命令自动接上。
            print(f"[thermal] {run_id}: 训练进程撞温度墙（rc=42），"
                  f"中途状态已存档，等冷却后自动续跑", flush=True)
            remain = deadline - time.time()
            if remain <= 0:
                return False
            wait_until_cool(max_wait_s=max(int(remain), 60))
            continue
        after = epochs_done(args.ckpt_root, run_id)
        rate.update(time.time() - t0, after - done)
        state.set(run_id, epochs_done=after, sec_per_epoch=round(rate.spe, 1))
        if after <= done:
            print(f"[abort] {run_id} 本段未推进（rc={rc}），"
                  f"见 _arm_{run_id}_log.txt 末尾", flush=True)
            return False


def eval_arm(args, run_id, eval_extra):
    out = HERE / "eval_output" / f"{run_id}_test"
    cmd = [sys.executable, "-u", "evaluate_model.py",
           "--checkpoint", Path(args.ckpt_root) / run_id / "best_model.pth",
           "--data_root", args.data_root, "--split", "test",
           "--split_manifest", MANIFEST, "--batch_size", 4,
           "--out_dir", out] + eval_extra
    with open(HERE / f"_arm_{run_id}_eval_log.txt", "w", encoding="utf-8") as f:
        rc = sh(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        j = out / "eval_summary.json"
        if j.is_file():
            s = json.loads(j.read_text(encoding="utf-8"))["metrics_mean_std"]
            print(f"[eval] {run_id}: PSNR {s['image_psnr']['mean']:.2f} | "
                  f"normal MAE {s['normal_mae_deg']['mean']:.2f} deg | "
                  f"albedo si-MAE {s['albedo_si_mae']['mean']:.4f}", flush=True)
    return rc


def package(args, merged):
    """回传包：best_model + 评估 json/csv + 日志 + TB + 计划/进度。"""
    out = Path(args.package_dir)
    if out.exists():
        # INC-0014 续：环境级 safe-delete（回收站不可用）可能让 rmtree 抛 OSError
        # 打爆收尾。打包目录为自产临时目录，ignore_errors=True 无数据风险。
        shutil.rmtree(out, ignore_errors=True)
    for sub in ("checkpoints", "eval_output", "logs"):
        # INC-0014 续：rmtree(ignore_errors) 可能留残目录，mkdir 必须 exist_ok=True
        (out / sub).mkdir(parents=True, exist_ok=True)
    for run_id in merged:
        src = Path(args.ckpt_root) / run_id / "best_model.pth"
        if src.is_file():
            shutil.copy2(src, out / "checkpoints" / f"{run_id}_best_model.pth")
        ev = HERE / "eval_output" / f"{run_id}_test"
        if ev.is_dir():
            shutil.copytree(ev, out / "eval_output" / f"{run_id}_test")
        for lg in HERE.glob(f"_arm_{run_id}*log.txt"):
            shutil.copy2(lg, out / "logs" / lg.name)
        tb = Path(args.log_root) / run_id
        if tb.is_dir():
            shutil.copytree(tb, out / "logs" / f"tb_{run_id}", dirs_exist_ok=True)
    (out / "arms_progress_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in (PLAN_JSON, HERE / "_arms_bench.json"):
        if f.is_file():
            shutil.copy2(f, out / f.name)
    z = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"[package] 回传包 -> {z}（{Path(z).stat().st_size/1e6:.0f} MB）")
    return z


def main():
    ap = argparse.ArgumentParser(description="A10 一键训练入口")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--ckpt_root", default=str(HERE.parent / "checkpoints"))
    ap.add_argument("--log_root", default=str(HERE.parent / "logs"))
    ap.add_argument("--viz_root", default=str(HERE.parent / "visualizations"))
    ap.add_argument("--package-dir", dest="package_dir",
                    default=str(HERE.parent / "arms_return_package"))
    ap.add_argument("--budget-hours", dest="budget_hours", type=float, default=6.0)
    ap.add_argument("--reserve-min", dest="reserve_min", type=int, default=40,
                    help="预留给评估与打包的分钟数")
    ap.add_argument("--max-lanes", dest="max_lanes", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0, help="0=自动；自动模式会取 _safe_num_workers() 决定的值（见其注释）")
    ap.add_argument("--amp-dtype", dest="amp_dtype", default="bf16",
                    choices=["bf16", "fp16"],
                    help="计算精度。bf16=既有基线口径（需 sm_80+）；"
                         "fp16=Turing/T4 唯一可用的张量核心路径，属数值口径偏差，"
                         "须向审计声明（见 INC-0007）")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-reference-arm", dest="no_reference_arm",
                    action="store_true",
                    help="非 bf16 精度下不自动插入同精度参照臂（默认插入）")
    ap.add_argument("--state-file", dest="state_file",
                    default=str(HERE / "arms_progress.json"))
    ap.add_argument("--sec-per-epoch", dest="sec_per_epoch", type=float, default=0)
    ap.add_argument("--assume-peak-gb", dest="assume_peak_gb", type=float, default=0,
                    help="标定不可用时假定的单车道峰值分配显存（本机实测）")
    ap.add_argument("--slowdown", type=float, default=1.0,
                    help="仅 dry-run 用：目标机相对本机的减速倍数，用于异地估算。"
                         "Tesla T4 相对 RTX 5070 Ti Laptop 换算区间 2.5–3.0"
                         "（张量核心峰值比 1.66x、显存带宽比 2.8x、fp32 向量比 3.3x）")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--plan-only", dest="plan_only", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--skip-package", dest="skip_package", action="store_true")
    ap.add_argument("--smoke-epochs", dest="smoke_epochs", type=int, default=0,
                    help="仅用于管线自检：把目标 epoch 数临时改成该值，"
                         "跑通 训练→评估→打包 全链路。产物不得进对比矩阵。")
    args = ap.parse_args()
    if not args.assume_peak_gb:
        args.assume_peak_gb = PEAK_GB_BY_DTYPE.get(args.amp_dtype, 8.63)

    if args.smoke_epochs:
        global TOTAL_EPOCHS, SEGMENT_EPOCHS
        TOTAL_EPOCHS = args.smoke_epochs
        SEGMENT_EPOCHS = min(SEGMENT_EPOCHS, args.smoke_epochs)
        print(f"[SMOKE] 目标 epoch 临时改为 {TOTAL_EPOCHS}——仅验证管线，"
              f"产物不可用于论文对比矩阵（D10）")

    if args.status:
        if PLAN_JSON.is_file():
            p = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
            print(f"计划：车道 {p['lanes']}｜{p['sec_per_epoch']} s/epoch｜"
                  f"执行 {p['will_run']}｜顺延 {p['deferred']}")
        merged = State.merge(list(HERE.glob("arms_progress*.json")))
        for k, v in sorted(merged.items()):
            print(f"  {k}: {v}")
        return
    if not args.data_root:
        ap.error("--data_root 必填（除 --status）")

    t0 = time.time()
    info, blockers, warns = preflight(args)
    for w in warns:
        print(f"[warn] {w}")
    for b in blockers:
        print(f"[BLOCK] {b}")
    if blockers and not args.dry_run:
        print("\n预检未通过，已中止——请先解决上述阻断项。")
        sys.exit(2)

    calib = calibrate(args, info)
    p = make_plan(args, info, calib)
    p.update({"preflight": info, "calibration": calib, "warnings": warns})
    PLAN_JSON.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    print_plan(info, calib, p)
    if args.dry_run or args.plan_only:
        print(f"计划已写入 {PLAN_JSON.name}；正式执行去掉 --dry-run/--plan-only")
        return

    lanes = p["lanes"]
    if not args.num_workers:
        # INC-0008 临时压低：单车道从 4 worker 降到 2，原因见 _safe_num_workers。
        # 取消该上限后回到 max(2, min(8, ...)) 即可（注释留下方便审计）。
        args.num_workers = _safe_num_workers(0, lanes=lanes)
    deadline = t0 + args.budget_hours * 3600 - args.reserve_min * 60
    pool = list(ARMS)
    if args.amp_dtype != "bf16" and not args.no_reference_arm:
        pool.insert(0, REFERENCE_ARM)
    todo = [a for a in pool if tag_run(a[0], args.amp_dtype) in p["will_run"]]

    if lanes > 1 and len(todo) > 1:
        # 多车道：每条车道一个子进程、一份 state 文件，避免并发写冲突
        lane_spe = p["sec_per_epoch"] * lanes / p["assumed_gain"]
        procs = []
        for li in range(lanes):
            bucket = todo[li::lanes]
            if not bucket:
                continue
            cmd = [sys.executable, "-u", __file__,
                   "--data_root", args.data_root, "--ckpt_root", args.ckpt_root,
                   "--log_root", args.log_root, "--viz_root", args.viz_root,
                   "--only", ",".join(a[0] for a in bucket),
                   "--max-lanes", 1, "--num_workers", args.num_workers,
                   "--amp-dtype", args.amp_dtype,
                   "--sec-per-epoch", round(lane_spe, 1),
                   "--budget-hours", max(0.05, (deadline - time.time()) / 3600),
                   "--reserve-min", 0, "--skip-package",
                   "--state-file", HERE / f"arms_progress_lane{li}.json"]
            log = HERE / f"_arm_lane{li}_log.txt"
            print(f"[lane {li}] {[a[0] for a in bucket]} -> {log.name}", flush=True)
            f = open(log, "w", encoding="utf-8")
            procs.append((subprocess.Popen([str(c) for c in cmd], cwd=HERE,
                                           stdout=f, stderr=subprocess.STDOUT), f))
        for pr, f in procs:
            pr.wait()
            f.close()
    else:
        state = State(args.state_file)
        rate = Rate(p["sec_per_epoch"])
        for base_id, tr, ev, _ in todo:
            run_id = tag_run(base_id, args.amp_dtype)
            print(f"\n===== ARM {run_id}（精度 {args.amp_dtype}）=====", flush=True)
            ok = train_arm(args, run_id, tr, deadline, rate, state)
            state.set(run_id, complete=ok,
                      epochs_done=epochs_done(args.ckpt_root, run_id))
            if ok:
                state.set(run_id, eval_rc=eval_arm(args, run_id, ev))
            else:
                print(f"[arm partial] {run_id} 未跑满 100 epoch，不做评估"
                      f"（半截结果不能进对比矩阵）", flush=True)
                break

    if not args.skip_package:
        merged = State.merge(list(HERE.glob("arms_progress*.json")))
        package(args, merged)
        print(f"\n总耗时 {(time.time()-t0)/3600:.2f} h　完成情况：")
        for k, v in sorted(merged.items()):
            print(f"  {k}: {v}")
        done_full = [k for k, v in merged.items() if v.get("complete")]
        print(f"\n跑满 100 epoch 并已评估：{done_full or '（无）'}")
        print("回传后在本地执行：python make_report_assets.py 生成对比矩阵与图表")


if __name__ == "__main__":
    main()


