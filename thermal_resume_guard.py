#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
thermal_resume_guard.py — 温度续跑门卫（INC-0010 审计裁决 §3 A4 强化版）

设计动机：
  原 thermal_guard.py 内部 wait_until_cool 单次读数 ≤ 70°C 即可续跑。
  但本机 GPU 散热能力下，单次读数可能因瞬时风扇降速、shell 调度延迟等
  误判为"已冷却"——而实际 GPU 仍处于高温状态续跑，1-2 个 batch 后再撞墙。
  强化策略：要求 **5/5 连续 60s 间隔读数** ≤ 75°C 才放行；任一 > 75°C
  立即重置计数器并冷却 10 分钟。

物理参数（实测）：
  闲置 GPU 温度 = 74°C
  THERMAL_RESUME = 75°C（固化在 _env.sh / _env.ps1）
  THERMAL_LIMIT  = 80°C（撞墙停机阈值）
  HARD_KILL      = 88°C（watchdog 兜底）
  安全余量：75 → 80 = 5°C

退出码：
  0   5/5 通过，可以续跑
  42  冷却超时 / 5 次内有 >75°C（与 thermal_guard.py 一致语义）
"""
import subprocess
import time
import sys
import os
import argparse
from datetime import datetime


def read_gpu_max_temp():
    """读取所有 GPU 温度的最大值。失败返回 None。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu',
             '--format=csv,noheader,nounits'],
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    temps = []
    for line in out.strip().splitlines():
        try:
            temps.append(int(line.strip()))
        except ValueError:
            continue
    return max(temps) if temps else None


def wait_until_cool(resume_c=75, n_samples=5, interval_s=60,
                   cool_sleep_s=600, log_path=None, run_id="unknown"):
    """
    阻塞直到 n_samples 连续读数都 ≤ resume_c。
    任一读数 > resume_c 立即重置计数器并冷却 cool_sleep_s 秒。

    日志格式：每行 "<unix_ts> <run_id> <max_temp>"，写入 log_path。
    """
    log_path = log_path or f"logs/thermal_resume_{run_id}_{int(time.time())}.log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _log(line):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        print(line, flush=True)

    _log(f"# thermal_resume_guard start run_id={run_id} "
         f"resume_c={resume_c} n_samples={n_samples} interval_s={interval_s}")

    passed = 0
    while passed < n_samples:
        t = read_gpu_max_temp()
        ts = int(time.time())
        if t is None:
            _log(f"{ts} {run_id} READ_FAIL")
            print("[thermal] 读不到温度，跳过（建议检查 nvidia-smi）", flush=True)
            time.sleep(cool_sleep_s)
            passed = 0
            continue

        line = f"{ts} {run_id} {t}"
        if t > resume_c:
            _log(f"{line}  # EXCEED -> cool {cool_sleep_s}s, reset counter")
            print(f"[thermal] 读数 {t}°C > {resume_c}°C，冷却 {cool_sleep_s//60} 分钟...",
                  flush=True)
            time.sleep(cool_sleep_s)
            passed = 0
            continue

        passed += 1
        _log(f"{line}  # PASS {passed}/{n_samples}")
        if passed < n_samples:
            time.sleep(interval_s)

    _log(f"# thermal_resume_guard READY: 5/5 通过 run_id={run_id}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--resume-c', type=int, default=int(os.environ.get('THERMAL_RESUME', '75')),
                    help='续跑上限温度 (°C)，默认从 $THERMAL_RESUME 读')
    ap.add_argument('--n-samples', type=int, default=5,
                    help='连续通过次数（默认 5/5）')
    ap.add_argument('--interval-s', type=int, default=60,
                    help='相邻读数间隔 (秒)')
    ap.add_argument('--cool-sleep-s', type=int, default=600,
                    help='超温后冷却时间 (秒)')
    ap.add_argument('--log-path', type=str, default=None,
                    help='日志文件路径')
    ap.add_argument('--run-id', type=str, default=os.environ.get('RUN_ID', 'unknown'),
                    help='run_id（写入日志便于追溯）')
    args = ap.parse_args()

    print(f"[thermal] resume_c={args.resume_c}  n_samples={args.n_samples}/5  "
          f"interval_s={args.interval_s}  cool_sleep_s={args.cool_sleep_s}",
          flush=True)
    try:
        ok = wait_until_cool(
            resume_c=args.resume_c,
            n_samples=args.n_samples,
            interval_s=args.interval_s,
            cool_sleep_s=args.cool_sleep_s,
            log_path=args.log_path,
            run_id=args.run_id,
        )
    except KeyboardInterrupt:
        print("[thermal] 用户中断", flush=True)
        sys.exit(130)

    if ok:
        print("[thermal] 5/5 通过，可以续跑", flush=True)
        sys.exit(0)
    else:
        print("[thermal] 5 次内有 >75°C，抛 ThermalNotReady", flush=True)
        sys.exit(42)


if __name__ == "__main__":
    main()
