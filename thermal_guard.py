"""温度墙守卫（低散热平台的进程内热保护）。

背景：本机 RTX 5070 Ti Laptop 散热能力退化，功耗墙被拉到 123.57 W，
持续负载数分钟即冲到 85 °C+，已导致两次整机热保护关机
（见 docs/本机长跑前置清单.md §1.2）。

原有防线是外部看门狗 `thermal_supervisor.ps1`：巡检到高温直接
`Stop-Process -Force`。它保住了机器，但代价是**当前 epoch 的进度全丢**
（进程被硬杀，只剩上一个 epoch 的 checkpoint）。在"跑几分钟停一次"的
喘息循环下，这个代价会吃掉大部分算力。

本模块把热保护搬进训练进程：批次间轮询温度，越线时抛 `ThermalStop`，
由训练器**先把完整状态落盘再退出**，于是热停机的损失从"一个 epoch"
降到"一个 batch"。硬杀看门狗保留为兜底（阈值抬到本守卫之上）。

环境变量
--------
  THERMAL_LIMIT   停机阈值 °C，默认 80（越线即存档退出）。
                  实测 3s 轮询下温度读数会滞后 ~3–4°C（见 _smoke 记录：阈值 82
                  时读到 86 才停下），故主防线取 80，峰值压在 ~84 以内，
                  与兜底硬杀阈值拉开距离。
  THERMAL_RESUME  重启安全温度 °C，默认 70（由编排器使用）
  THERMAL_POLL    轮询间隔 s，默认 3.0（批次间节流，避免频繁起进程）
  THERMAL_GUARD   置 0 关闭守卫（默认开启；无 nvidia-smi 时自动降级为关闭）

只读温度、只决定"停不停"，不触碰任何超参与随机流，不影响 D10 单变量口径。
"""
import os
import shutil
import subprocess
import time


class ThermalStop(Exception):
    """温度越过停机阈值。携带触发温度，供上层写停机原因。"""

    def __init__(self, temp_c, limit_c):
        self.temp_c = temp_c
        self.limit_c = limit_c
        super().__init__(f"GPU 温度 {temp_c}°C ≥ 停机阈值 {limit_c}°C")


def read_gpu_temp(timeout=5.0):
    """读取 GPU 温度（°C）。读不到返回 None，绝不抛异常打断训练。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def wait_until_cool(resume_c=None, poll_s=30, log=print, max_wait_s=None):
    """阻塞直到温度回落到安全线。返回最终温度（读不到温度时返回 None）。"""
    if resume_c is None:
        resume_c = int(float(os.environ.get("THERMAL_RESUME", "70")))
    t0 = time.time()
    while True:
        t = read_gpu_temp()
        if t is None:
            log("[thermal] 读不到温度，跳过冷却等待")
            return None
        if t <= resume_c:
            log(f"[thermal] 已冷却至 {t}°C ≤ {resume_c}°C，可以续跑"
                f"（等待 {time.time() - t0:.0f}s）")
            return t
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            log(f"[thermal] 冷却超时（{max_wait_s}s），当前 {t}°C，仍未达 {resume_c}°C")
            return t
        log(f"[thermal] 冷却中 t={t}°C > {resume_c}°C，等待 {poll_s}s")
        time.sleep(poll_s)


class ThermalGuard:
    """批次间温度巡检。`poll()` 越线时抛 ThermalStop，否则返回当前温度或 None。"""

    def __init__(self, limit_c=82, poll_interval_s=3.0, enabled=True, log=print):
        self.limit_c = int(limit_c)
        self.poll_interval_s = float(poll_interval_s)
        self.log = log
        self._last_poll = 0.0
        self.last_temp = None
        self.max_temp = None
        self.enabled = bool(enabled) and shutil.which("nvidia-smi") is not None
        if bool(enabled) and not self.enabled:
            self.log("[thermal] 未找到 nvidia-smi，温度墙守卫关闭")

    @classmethod
    def from_env(cls, log=print):
        return cls(limit_c=int(float(os.environ.get("THERMAL_LIMIT", "80"))),
                   poll_interval_s=float(os.environ.get("THERMAL_POLL", "3.0")),
                   enabled=os.environ.get("THERMAL_GUARD", "1") not in ("0", "false", "False"),
                   log=log)

    @property
    def resume_c(self):
        return int(float(os.environ.get("THERMAL_RESUME", "70")))

    def poll(self, force=False):
        """按间隔节流地读一次温度。

        Raises:
            ThermalStop: 温度 ≥ limit_c
        """
        if not self.enabled:
            return None
        now = time.time()
        if not force and now - self._last_poll < self.poll_interval_s:
            return None
        self._last_poll = now
        t = read_gpu_temp()
        if t is None:
            return None
        self.last_temp = t
        self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        if t >= self.limit_c:
            raise ThermalStop(t, self.limit_c)
        return t
