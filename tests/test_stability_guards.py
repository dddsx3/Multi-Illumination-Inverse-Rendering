"""
T2.1 / G2.1 稳定性守卫单元测试（任务书四用例 + 集成冒烟）

运行: python tests/test_stability_guards.py
断言的是行为（异常/状态/返回值），不是打印内容。

用例映射（任务书 T2.1 第 3 步）:
  用例1 连续坏批达阈值 -> 终止(RuntimeError) + on_abort 回调触发
  用例2 梯度范数 > abort_threshold(1e4) -> 硬停机 RuntimeError
  用例3 梯度范数 1e3-1e4 -> 只预警(warn 返回值) + TensorBoard 标量回调写入
        且不抛异常
  用例4 坏批后恢复有限损失 -> nan_streak 计数清零、继续返回 True

另含一个最小集成测试：真实 InverseRenderTrainer + 注入 NaN 的 mock dataloader，
在 GPU 上验证守卫经 train_epoch 全链路生效（RuntimeError 冒出）。
"""
import os
import sys
import tempfile
import tempfile

tempfile = tempfile  # 显式引用，避免 lint 折叠

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stability import StabilityGuard


class TBRecorder:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars.setdefault(name, []).append((step, value))

    def flush(self):
        pass


def make_guard(limit=10, warn=1e3, abort=1e4):
    tb = TBRecorder()
    aborted = {"called": False}
    def on_abort():
        aborted["called"] = True
    g = StabilityGuard(nan_streak_limit=limit, warn_threshold=warn,
                       abort_threshold=abort, on_abort=on_abort,
                       tb_scalar=lambda n, v: tb.add_scalar(n, v, len(tb.scalars.get(n, []))))
    return g, tb, aborted


def case1_nan_streak_abort():
    """连续 >= limit 个非有限损失 -> RuntimeError + on_abort 触发"""
    g, tb, aborted = make_guard(limit=10)
    raised = False
    for i in range(10):
        try:
            cont = g.check_loss(torch.tensor(float("nan")))
            assert cont is False, "非有限损失应返回 skip"
        except RuntimeError as e:
            raised = True
            assert "停机" in str(e) or "nan" in str(e).lower()
            break
    assert raised, "第 10 次连续 NaN 应终止训练"
    assert aborted["called"], "终止前应调用 on_abort（flush）"
    print("case1 PASS: 连续10次NaN -> RuntimeError + on_abort")


def case2_grad_abort():
    """梯度范数 > 1e4 -> RuntimeError 硬停机"""
    g, tb, _ = make_guard()
    try:
        g.check_grad_norm(torch.tensor(2.0e4))
        raise AssertionError("2e4 应触发硬停机")
    except RuntimeError as e:
        assert "1e4" in str(e) or "C3" in str(e)
    # 边界内不触发
    r = g.check_grad_norm(torch.tensor(5.0e3))
    assert r == "warn", r
    print("case2 PASS: grad>1e4 -> 硬停机; 5e3 -> 仅 warn")


def case3_warn_tier_records_tb():
    """1e3-1e4 区间 -> warn 返回 + TB 标量写入，且不抛异常"""
    g, tb, _ = make_guard()
    r = g.check_grad_norm(torch.tensor(2.0e3))
    assert r == "warn"
    assert any(k == "grad_norm_exceed_1e3" for k in tb.scalars), "TB 标量未写入"
    ok = g.check_grad_norm(torch.tensor(50.0))
    assert ok == "ok"
    print("case3 PASS: 1e3-1e4 仅预警且 TB 标量记录")


def case4_recovery_resets_streak():
    """坏批后恢复有限损失 -> 计数清零、继续训练（防误杀）"""
    g, tb, _ = make_guard(limit=5)
    # 4 次坏批（未达 limit=5）
    for _ in range(4):
        assert g.check_loss(torch.tensor(float("nan"))) is False
    assert g.nan_streak == 4
    # 恢复有限损失
    assert g.check_loss(torch.tensor(0.5)) is True
    assert g.nan_streak == 0, "恢复后计数应清零"
    # 再来 4 次坏批也不应触发终止（说明计数确实清零了）
    for _ in range(4):
        g.check_loss(torch.tensor(float("nan")))
    print("case4 PASS: 恢复清零防误杀")


def integration_real_trainer_poison():
    """集成：真实 InverseRenderTrainer + NaN mock dataloader -> RuntimeError 冒出，
    且此前有健康 checkpoint 在盘（先健康 batch 存档再注毒）。"""
    from unet_model import IntrinsicUNet
    from physics_renderer import PhysicsRenderer
    from residual_modules import HierarchicalResidual
    from loss_functions import LossCalculator
    from trainer import InverseRenderTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntrinsicUNet(num_images=5, base_channels=16)
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=True, num_images=5,
                                    feature_channels=16)

    H = W = 64
    healthy = torch.rand(2, 5, H, W)
    poison = torch.full((2, 5, H, W), float("nan"))

    # 批序列：2 健康 -> 之后全为毒（守卫应在 streak 内终止）
    batches = [(healthy.clone(), None, f"s{i}") for i in range(2)] \
              + [(poison.clone(), None, f"p{i}") for i in range(20)]
    class FakeLoader:
        def __init__(self, batches): self.batches = batches
        def __iter__(self): return iter(self.batches)
        def __len__(self): return len(self.batches)

    cfg = {
        "num_lights": 5, "image_size": [H, W], "batch_size": 2,
        "total_epochs": 1, "learning_rate": 1e-4, "weight_decay": 0,
        "stage1_epochs": 30, "stage2_epochs": 30, "base_channels": 16,
        "scheduler": "none", "grad_clip": 1.0,
        "log_dir": tempfile.mkdtemp(),
        "checkpoint_dir": tempfile.mkdtemp(),
        "vis_dir": tempfile.mkdtemp(),
        "log_interval": 1000, "tensorboard_interval": 1000,
        "val_interval": 1000, "vis_interval": 1000, "save_interval": 1000,
        "use_amp": False, "amp_dtype": "bfloat16",
        "nan_abort_streak": 5,   # 注入测试按该显式值触发
        "grad_norm_warn_threshold": 1e3, "grad_norm_abort_threshold": 1e4,
    }
    trainer = InverseRenderTrainer(
        model=model, renderer=renderer, residual=residual,
        train_loader=FakeLoader(batches), val_loader=FakeLoader([]),
        config=cfg)

    # 先存一个“最后健康 checkpoint”到盘（模拟健康阶段产物）
    trainer.save_checkpoint(epoch=0, val_loss=0.5, is_best=False)
    import glob
    ckpts_before = glob.glob(os.path.join(cfg["checkpoint_dir"], "*.pth"))
    assert ckpts_before, "健康 checkpoint 未落盘"

    raised = False
    try:
        trainer.train_epoch()
    except RuntimeError as e:
        raised = ("停机" in str(e)) or ("C3" in str(e))
    assert raised, "注入毒 batch 后守卫应经 train_epoch 终止训练"
    print("integration PASS: NaN 注入 -> 守卫经全链路终止训练")


if __name__ == "__main__":
    case1_nan_streak_abort()
    case2_grad_abort()
    case3_warn_tier_records_tb()
    case4_recovery_resets_streak()
    print("--- 单元用例全部通过，进入真实训练器集成测试 ---")
    integration_real_trainer_poison()
    print("ALL STABILITY GUARD TESTS PASSED")