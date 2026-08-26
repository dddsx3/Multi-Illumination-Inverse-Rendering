"""
训练稳定性守卫（INC-0001 防复发机制，验收条件 C3 落地）。

三层防线：
  1. 非有限损失：跳过该 batch；连续 nan_streak_limit 次 -> 抛 RuntimeError 终止
  2. 预裁剪梯度范数 > warn_threshold（默认 1e3）：预警 + TensorBoard 标量
  3. 预裁剪梯度范数 > abort_threshold（默认 1e4）：立即抛 RuntimeError 终止

设计为纯逻辑类（torch 张量进、决策/异常出，TensorBoard 通过回调挂钩），
便于在不构造完整训练器的情况下做行为级单元测试（任务书 T2.1 G2.1）。

阈值依据见 docs/design/t2_1_params.md：
  - nan_streak_limit 默认 10（与指导书"连续 3 个"的折中收敛值，理由见文档）
  - warn 1e3 / abort 1e4 对应 run1 事故中 9.8e3 的早期预警缺口
"""
import math

import torch


class StabilityGuard:
    def __init__(self, nan_streak_limit=10, warn_threshold=1e3,
                 abort_threshold=1e4, on_abort=None, tb_scalar=None):
        """
        Args:
            nan_streak_limit: 连续非有限损失达到该值即终止
            warn_threshold: 梯度范数预警阈值（仅记录，不中断）
            abort_threshold: 梯度范数硬停机阈值（预裁剪值）
            on_abort: 终止前回调（如 TensorBoard flush）
            tb_scalar: 回调 fn(name, value, step)，用于写 TensorBoard 标量
        """
        self.nan_streak_limit = int(nan_streak_limit)
        self.warn_threshold = float(warn_threshold)
        self.abort_threshold = float(abort_threshold)
        self.on_abort = on_abort
        self.tb_scalar = tb_scalar
        self.nan_streak = 0
        self.warn_count = 0
        self.total_nan_skips = 0
        # INC-0007：fp16 + GradScaler 专用计数。缩放因子标定期的偶发溢出是
        # 正常现象（scaler 会跳过该次更新并下调 scale），不能按发散处理；
        # 但连续溢出说明 scale 无法收敛，仍需停机。
        self.scaler_overflow_streak = 0
        self.total_scaler_overflows = 0

    def note_scaler_overflow(self) -> None:
        """记录一次 GradScaler 溢出（非有限梯度）。

        Raises:
            RuntimeError: 连续溢出达到 nan_streak_limit（缩放因子无法收敛）
        """
        self.scaler_overflow_streak += 1
        self.total_scaler_overflows += 1
        if self.tb_scalar is not None:
            try:
                self.tb_scalar('stability/scaler_overflow_streak',
                               self.scaler_overflow_streak,
                               self.total_scaler_overflows)
            except Exception:
                pass
        if self.scaler_overflow_streak >= self.nan_streak_limit:
            if self.on_abort is not None:
                try:
                    self.on_abort()
                except Exception:
                    pass
            raise RuntimeError(
                f"连续 {self.scaler_overflow_streak} 个 batch 的 fp16 梯度溢出，"
                "GradScaler 缩放因子无法收敛，判定发散并停机。"
                "建议：改用 bf16（需 sm_80+ GPU）或关闭 --use_amp")

    def note_scaler_ok(self) -> None:
        """一次有限梯度的正常更新，重置溢出连击。"""
        self.scaler_overflow_streak = 0

    def check_loss(self, total_loss: torch.Tensor) -> bool:
        """检查损失是否有限。

        Returns:
            True  —— 损失有限，继续正常反向/更新流程
            False —— 非有限损失，调用方应 zero_grad 并跳过本 batch

        Raises:
            RuntimeError: 连续非有限次数达到 nan_streak_limit（发散判定）
        """
        if torch.isfinite(total_loss):
            self.nan_streak = 0
            return True
        self.nan_streak += 1
        self.total_nan_skips += 1
        if self.nan_streak >= self.nan_streak_limit:
            if self.on_abort is not None:
                try:
                    self.on_abort()
                except Exception:
                    pass
            raise RuntimeError(
                f"连续 {self.nan_streak} 个 batch 出现非有限损失，"
                "判定训练发散并自动停机"
                "（参见 docs/incidents/INC-0001）。"
                "建议：关闭 --use_amp / 降低 albedo_smooth / "
                "降低学习率")
        return False

    def check_grad_norm(self, grad_norm) -> str:
        """检查（预裁剪）梯度范数。

        Returns:
            "ok"   —— 低于预警阈值
            "warn" —— 位于 (warn_threshold, abort_threshold]，仅记录不中断

        Raises:
            RuntimeError: 范数非有限或超过 abort_threshold
        """
        g = float(grad_norm)
        if not math.isfinite(g):
            if self.on_abort is not None:
                try:
                    self.on_abort()
                except Exception:
                    pass
            raise RuntimeError(
                f"\u68af\u5ea6\u8303\u6570\u975e\u6709\u9650\uff08{g}\uff09\uff0c"
                "\u786c\u505c\u673a\uff08C3 \u9632\u590d\u53d1\u673a\u5236\uff09")
        if g > self.abort_threshold:
            if self.on_abort is not None:
                try:
                    self.on_abort()
                except Exception:
                    pass
            raise RuntimeError(
                f"\u9884\u88c1\u526a\u68af\u5ea6\u8303\u6570 {g:.1f} > "
                f"{self.abort_threshold:.0e}\uff0c\u4f18\u5316\u666f\u89c2\u5df2\u75c5\u6001\uff0c"
                "\u786c\u505c\u673a\uff08C3 \u9632\u590d\u53d1\u673a\u5236\uff09")
        if g > self.warn_threshold:
            self.warn_count += 1
            if self.tb_scalar is not None:
                self.tb_scalar("grad_norm_exceed_1e3", g)
            return "warn"
        return "ok"
