"""
runtime_safety.py · 主机资源安全层（2026-09-03 INC-0014 后新增）

职责：
1. 运行前自检（preflight_train）：物理内存/页面文件(commit)/GPU 温度/VRAM 需求/
   残留 python 进程/磁盘，输出 BLOCK(2)/WARN(1)/PASS(0)。
2. 运行中熔断（check_host_memory）：训练循环内每 N batch 调一次，
   主机内存越界抛 MemoryStop —— trainer 捕获后按"温度墙同款"路径存档
   interrupt_state 并以 rc=42 退出，编排器续跑（无损或最多丢失若干 batch）。

设计原则（INC-0014 复盘）：
- 只读测量优先，绝不分配大块内存做"探测"（空分配通过 ≠ 真实训练通过，反而
  消耗 commit）。VRAM 可行性用"实测崩溃表"硬编码，不用实时空分配试探。
- 所有阈值集中在此文件常量，便于审计与调整。

Windows 实现说明：用 ctypes 调 GlobalMemoryStatusEx 读 ullAvailPhys /
ullTotalPageFile / ullAvailPageFile，避免依赖 psutil。
"""

import ctypes
import os
import shutil
import subprocess
import sys

# ---- 阈值（INC-0014 复盘后定） -------------------------------------------------
# 本机硬事实（2026-09-03 实测）：物理内存仅 ~16GB，开机即被系统/应用占 ~11.5GB；
# 页面文件 C+D 共 ~48GB。bs8+bf16+fusion+renderer@256² 在 12GB 5070Ti Laptop 上
# 实测正向即 CUDA OOM（INC-0014 时间线）——不可行，禁止启动。
PHYS_FREE_BLOCK_GB = 3.0          # 物理内存空闲 < 此值 -> BLOCK
PHYS_FREE_WARN_GB  = 5.0          # < 此值 -> WARN
PAGEFILE_FREE_BLOCK_GB = 6.0      # 页面文件(commit 余量) < 此值 -> BLOCK
PAGEFILE_FREE_WARN_GB  = 10.0     # < 此值 -> WARN
GPU_IDLE_TEMP_BLOCK_C = 80        # 空闲即 >= 此温度 -> BLOCK（热预算不足）
GPU_IDLE_TEMP_WARN_C  = 70        # 空闲 >= 此温度 -> WARN
DISK_FREE_BLOCK_GB = 1.5          # 数据/ckpt 盘可用空间下限
# VRAM 可行性表：键 (amp_dtype, batch) -> (需要GB, 结论)。来源：INC-0014 实测
# （bs8 正向/backward 多次 CUDA OOM 与 cuDNN workspace 分配失败；bs4 单 epoch
# 完整跑通 训练+验证）。bs4 上限取 6.5GB 留足余量。
VRAM_NEED_GB = {
    ("bf16", 4): 6.5,
    ("bf16", 8): None,   # None = 已实测不可行，直接 BLOCK
}
VRAM_SAFETY_HEADROOM = 1.25       # free_vram >= need * 1.25 + 0.5


class MemoryStop(Exception):
    """主机内存越界：训练循环内抛出，触发与温度墙相同的优雅停机/续跑通道。"""

    def __init__(self, avail_phys_gb, avail_pf_gb, phys_floor_gb, pf_floor_gb):
        self.avail_phys_gb = avail_phys_gb
        self.avail_pf_gb = avail_pf_gb
        super().__init__(
            f"主机内存告警：物理空闲 {avail_phys_gb:.2f}GB(<{phys_floor_gb}GB) "
            f"或页面文件空闲 {avail_pf_gb:.2f}GB(<{pf_floor_gb}GB)")


# ---- 测量 ----------------------------------------------------------------------
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_memory():
    """返回 dict：total_phys_gb / avail_phys_gb / avail_pagefile_gb。"""
    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return {
            "total_phys_gb": st.ullTotalPhys / 2 ** 30,
            "avail_phys_gb": st.ullAvailPhys / 2 ** 30,
            "avail_pagefile_gb": st.ullAvailPageFile / 2 ** 30,
        }
    except Exception:
        return None


def gpu_info():
    """读 GPU 名称/总显存/已用/温度（nvidia-smi）。失败返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        name, total_mib, used_mib, temp_c = [x.strip() for x in
                                             out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "total_vram_gb": int(total_mib) / 1024,
                "used_vram_gb": int(used_mib) / 1024, "temp_c": int(temp_c)}
    except Exception:
        return None


def python_processes():
    """返回本机 python.exe 进程 PID 列表（排除自身，用于单实例互斥）。"""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=15)
        me = os.getpid()
        pids = []
        for line in out.stdout.splitlines():
            if "python.exe" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    pid = int(parts[1])
                    if pid != me:
                        pids.append(pid)
        return pids
    except Exception:
        return None


# ---- 运行前闸门 ------------------------------------------------------------------
def preflight_train(amp_dtype="bf16", batch=4, allow_stale=False,
                    log=print, prog="main"):
    """运行前自检。返回 (rc, items)；rc: 0=PASS 1=WARN 2=BLOCK。

    items: [(level, msg), ...]
    """
    items = []
    rc = 0

    def add(level, msg):
        nonlocal rc
        items.append((level, msg))
        if level == "BLOCK":
            rc = 2
        elif level == "WARN" and rc == 0:
            rc = 1

    mem = host_memory()
    if mem is None:
        add("BLOCK", "无法读取主机内存（GlobalMemoryStatusEx 失败），禁止启动")
    else:
        if mem["avail_phys_gb"] < PHYS_FREE_BLOCK_GB:
            add("BLOCK", f"物理内存空闲 {mem['avail_phys_gb']:.2f}GB < "
                         f"{PHYS_FREE_BLOCK_GB}GB 下限——启动可能导致系统停摆")
        elif mem["avail_phys_gb"] < PHYS_FREE_WARN_GB:
            add("WARN", f"物理内存空闲 {mem['avail_phys_gb']:.2f}GB（<{PHYS_FREE_WARN_GB}GB），"
                        f"建议先关闭其他程序")
        if mem["avail_pagefile_gb"] < PAGEFILE_FREE_BLOCK_GB:
            add("BLOCK", f"页面文件空闲 {mem['avail_pagefile_gb']:.2f}GB < "
                         f"{PAGEFILE_FREE_BLOCK_GB}GB（commit 余量不足，WinError 1455 风险）")
        elif mem["avail_pagefile_gb"] < PAGEFILE_FREE_WARN_GB:
            add("WARN", f"页面文件空闲 {mem['avail_pagefile_gb']:.2f}GB（<{PAGEFILE_FREE_WARN_GB}GB）")

    gpu = gpu_info()
    if gpu is None:
        add("BLOCK", "无法读取 GPU（nvidia-smi 失败）")
    else:
        if gpu["temp_c"] >= GPU_IDLE_TEMP_BLOCK_C:
            add("BLOCK", f"GPU 空闲温度 {gpu['temp_c']}°C ≥ {GPU_IDLE_TEMP_BLOCK_C}°C，"
                         f"热预算不足，先散热再启动")
        elif gpu["temp_c"] >= GPU_IDLE_TEMP_WARN_C:
            add("WARN", f"GPU 空闲温度 {gpu['temp_c']}°C（≥{GPU_IDLE_TEMP_WARN_C}°C），"
                        f"训练将频繁撞热墙，属低散热平台常态")
        need = VRAM_NEED_GB.get((amp_dtype, batch))
        free_vram = gpu["total_vram_gb"] - gpu["used_vram_gb"]
        if need is None:
            add("BLOCK", f"VRAM 可行性表：({amp_dtype}, bs{batch}) 实测不可行"
                         f"（INC-0014），请降配到 bs4")
        else:
            require = need * VRAM_SAFETY_HEADROOM + 0.5
            if free_vram < require:
                add("BLOCK", f"显存空闲 {free_vram:.2f}GB < 需 {require:.2f}GB"
                             f"（bs{batch} @ {amp_dtype} 估算 {need}GB）")
            else:
                log(f"[safety] VRAM 可行性：({amp_dtype}, bs{batch}) ok "
                    f"(需≈{need}GB, 空闲 {free_vram:.2f}GB)")

    if not allow_stale:
        pids = python_processes()
        if pids:
            add("BLOCK", f"检测到残留 python 进程 {pids}——禁止并发训练，先清理"
                         f"（或 --allow-stale 明确放行）")

    disk = shutil.disk_usage(os.path.dirname(os.path.abspath(".")) or ".")
    free_disk = disk.free / 2 ** 30
    if free_disk < DISK_FREE_BLOCK_GB:
        add("BLOCK", f"磁盘可用 {free_disk:.1f}GB < {DISK_FREE_BLOCK_GB}GB")
    else:
        log(f"[safety] 磁盘可用 {free_disk:.1f}GB ok")

    for level, msg in items:
        log(f"[safety][{level}] {msg}")
    return rc, items


def check_host_memory(phys_floor_gb=1.5, pf_floor_gb=2.0):
    """运行中熔断：越界抛 MemoryStop。供 trainer 训练循环周期性调用。"""
    mem = host_memory()
    if mem is None:
        return
    if mem["avail_phys_gb"] < phys_floor_gb or mem["avail_pagefile_gb"] < pf_floor_gb:
        raise MemoryStop(mem["avail_phys_gb"], mem["avail_pagefile_gb"],
                         phys_floor_gb, pf_floor_gb)


if __name__ == "__main__":
    # CLI 自测：python runtime_safety.py
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--allow-stale", action="store_true")
    a = ap.parse_args()
    rc, _ = preflight_train(amp_dtype=a.amp, batch=a.batch,
                            allow_stale=a.allow_stale)
    sys.exit(rc)
