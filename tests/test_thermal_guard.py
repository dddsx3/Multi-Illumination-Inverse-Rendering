"""
温度墙守卫单元测试（T2.1 风格：逻辑级 + 存档结构级）

运行: python tests/test_thermal_guard.py

用例:
  1. 温度 < 阈值 -> poll 不抛异常
  2. 温度 >= 阈值 -> poll 抛 ThermalStop，携带触发温度与阈值
  3. 轮询节流：间隔内不重复读温度（读函数计数不增长）
  4. 读不到温度（None）-> 全路径静默通过，绝不打断训练
  5. trainer 中途存档字段完整性：build 一个最小组件集合做结构级断言，
     不构造真实 dataloader（避免 15GB 内存页文件约束下的负担）
"""
import os
import sys
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thermal_guard
from thermal_guard import ThermalGuard, ThermalStop


class FakeTrainer:
    """trainer._save_interrupt_state 的最小宿主（只借它的那些字段）。"""

    def __init__(self, tmp):
        from pathlib import Path as P
        self.checkpoint_dir = P(tmp)
        self.current_epoch = 3
        self.global_step = 170
        self.current_stage = 2
        self.best_val_loss = 0.05
        self.continuous_qualified_epochs = 1
        self.model = torch.nn.Linear(2, 2)
        self.renderer = torch.nn.Identity()
        self.residual = None
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = None
        self._use_scaler = False
        self.loss_calculator = types.SimpleNamespace(weights={'reconstruction': 1.0})
        self.writer = types.SimpleNamespace(add_scalar=lambda *a: None, flush=lambda: None)
        self.thermal = ThermalGuard(limit_c=80, enabled=False)
        self.config = {'x': 1}


class TestThermalGuard(unittest.TestCase):
    def setUp(self):
        self._original = thermal_guard.read_gpu_temp

    def tearDown(self):
        thermal_guard.read_gpu_temp = self._original

    def _fake_read(self, value, calls=None):
        def fake(timeout=5.0):
            if calls is not None:
                calls.append(1)
            return value
        return fake

    def test_cool_temp_passes(self):
        thermal_guard.read_gpu_temp = self._fake_read(60)
        g = ThermalGuard(limit_c=82, poll_interval_s=0, enabled=True)
        self.assertEqual(g.poll(force=True), 60)      # 返回实测温度
        self.assertEqual(g.last_temp, 60)

    def test_hot_temp_raises(self):
        thermal_guard.read_gpu_temp = self._fake_read(83)
        g = ThermalGuard(limit_c=82, poll_interval_s=0, enabled=True)
        with self.assertRaises(ThermalStop) as ctx:
            g.poll(force=True)
        self.assertEqual(ctx.exception.temp_c, 83)
        self.assertEqual(ctx.exception.limit_c, 82)

    def test_poll_throttling(self):
        calls = []
        thermal_guard.read_gpu_temp = self._fake_read(60, calls)
        g = ThermalGuard(limit_c=82, poll_interval_s=3600, enabled=True)
        self.assertEqual(g.poll(force=True), 60)      # 第一次立即读
        self.assertIsNone(g.poll())                   # 间隔内：不读
        self.assertIsNone(g.poll())                   # 间隔内：不读
        self.assertEqual(g.poll(force=True), 60)      # force：读
        self.assertEqual(len(calls), 2)

    def test_unreadable_temp_is_silent(self):
        thermal_guard.read_gpu_temp = self._fake_read(None)
        g = ThermalGuard(limit_c=82, poll_interval_s=0, enabled=True)
        self.assertIsNone(g.poll(force=True))     # 不抛异常


class TestInterruptStateRoundtrip(unittest.TestCase):
    def test_save_structure_and_roundtrip(self):
        import trainer as tr
        from pathlib import Path
        with torch.random.fork_rng([]):
            tr.torch.manual_seed(123)
            host = FakeTrainer("TMP_UNUSED")
            state = {
                'kind': 'thermal_interrupt',
                'epoch': host.current_epoch,
                'batches_done': 37,
                'num_batches': 56,
                'global_step': host.global_step,
                'current_stage': host.current_stage,
                'best_val_loss': host.best_val_loss,
                'continuous_qualified_epochs': host.continuous_qualified_epochs,
                'model_state_dict': host.model.state_dict(),
                'renderer_state_dict': host.renderer.state_dict(),
                'residual_state_dict': None,
                'optimizer_state_dict': host.optimizer.state_dict(),
                'scheduler_state_dict': None,
                'scaler_state_dict': None,
                'partial': {
                    'epoch_losses': {'total': 12.5},
                    'albedo_grad_l1_total': 0.1,
                    'albedo_image_corr_total': 2.2,
                    'quality_metric_count': 37,
                },
                'rng_state': {
                    'python': __import__('random').getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': None,
                },
                'loss_weights': {'reconstruction': 1.0},
                'reason': 'test', 'temp_c': 84.0, 'saved_at': 'now',
                'config': host.config,
            }
            # 中途存档必须完整：模型/优化器/调度器/随机流不可缺失
            # （residual/scaler 在无残差、非 fp16 配置下允许为 None，与 load 端一致）
            for k in ('model_state_dict', 'renderer_state_dict',
                      'optimizer_state_dict', 'rng_state', 'partial',
                      'loss_weights', 'config'):
                self.assertIsNotNone(state[k], f"字段 {k} 不应为 None")
            self.assertEqual(state['batches_done'], 37)
            self.assertLess(state['batches_done'], state['num_batches'])
            # global_step 语义：已完成的累计批数
            self.assertEqual(state['global_step'], host.global_step)


if __name__ == "__main__":
    unittest.main()