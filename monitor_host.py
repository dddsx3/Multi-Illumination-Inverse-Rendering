"""
monitor_host.py · 运行中主机资源监控（INC-0014 后新增，观测/报警用）

周期采样物理内存空闲、页面文件空闲、GPU 温度，写心跳日志。
主机内存的"强制执行"由 trainer 内的 check_host_memory（每 10 batch 熔断、
rc=42 存档续跑）承担；本脚本为独立观测哨兵：持续低于硬地板时打 CRITICAL。

用法:
  python monitor_host.py --interval 3 --log docs/incidents/INC-0014_host_monitor.log
"""

import argparse
import time

import runtime_safety as rs

CRIT_PHYS_GB = 0.4
CRIT_PF_GB = 1.5
CRIT_STRIKES = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    def out(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if a.log:
            with open(a.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    strikes = 0
    while True:
        mem = rs.host_memory()
        gpu = rs.gpu_info()
        phys = mem["avail_phys_gb"] if mem else float("nan")
        pf = mem["avail_pagefile_gb"] if mem else float("nan")
        temp = gpu["temp_c"] if gpu else None
        if phys < CRIT_PHYS_GB or pf < CRIT_PF_GB:
            strikes += 1
            out(f"[CRITICAL {strikes}/{CRIT_STRIKES}] phys={phys:.2f}GB "
                f"pf={pf:.2f}GB temp={temp}——训练内存熔断将在 ~10 batch 内触发；"
                f"如无反应请人工检查")
        else:
            strikes = 0
            if strikes == 0:
                out(f"[ok] phys={phys:.2f}GB pf={pf:.2f}GB temp={temp}°C")
        if strikes >= CRIT_STRIKES:
            out("[ALERT] 主机资源持续低于硬地板。训练侧熔断未及时生效，"
                "请人工介入（kill python / 关闭占用程序）。")
            strikes = 0
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
