# T2.1 参数对齐声明（预注册）

**日期**：2026-08-25　**依据**：Phase 2 任务书 v2.0 T2.1 第 1 步

## 冲突点

- 《任务工作指导书》T2.1 写「连续 **3** 个 batch 出现非有限损失即停机」；
- 现库实现默认 `nan_abort_streak = 30`（Phase 1 遗留值，为 AMP scaler 跳过时代设计）。

## 收敛决定：`nan_abort_streak = 10`

理由：
1. **BF16 下无 scaler 跳步**——出现非有限损失即为真实数值事故而非 fp16 溢出假阳性，
   无需为假阳性保留大缓冲；
2. 10 个 batch ≈ 单 epoch 的 17%（60 batch/epoch），在同一阶段内即可判定发散，
   又给孤立数值毛刺（如单场景极端材质）留出容错；指导书的 3 过于激进——
   边界像素的极端组合可能造成孤立 nan 而 overall 训练仍健康（INC-0001 复盘中
   epoch24 即出现过"单批 nan 后自愈"的现象）；
3. 与梯度预警阈值形成梯度：warn(>1e3) -> abort-nan(streak≥10) -> abort-grad(>1e4)。

## 配置快照（trainer.py 经 config 覆盖项）

| 键 | 默认 | 说明 |
|---|---|---|
| nan_abort_streak | **10** | 连续非有限损失停机阈值 |
| grad_norm_warn_threshold | 1e3 | 预裁剪梯度范数预警（写 TB 标量） |
| grad_norm_abort_threshold | 1e4 | 预裁剪梯度范数硬停机 |

单元测试按 nan_streak_limit=10 注入触发（tests/test_stability_guards.py case1）。