# INC-0006 断点续跑重做上一个 epoch（off-by-one）+ n5rgb 臂数据污染定性

**日期**：2026-08-26　**发现人**：执行者（构建者）　**状态**：已修复并验证
**关联**：INC-0005（同一次"恢复语义"排查中发现的第二个缺陷）

---

## 1. 缺陷

`trainer.load_checkpoint` 原实现 `self.current_epoch = checkpoint['epoch']`，
而 checkpoint 里存的是**已完成**的 epoch 编号；`train()` 的循环
`for epoch in range(self.current_epoch, total_epochs)` 于是从该 epoch 重新开始。

后果三条：

1. **每次续跑重做一个 epoch**，10-epoch 分段编排下浪费约 10% 算力；
2. **覆盖已归档的 checkpoint**（同名 `checkpoint_epoch_00NN.pth` 被重写），
   破坏 D3/D8 的产物可追溯性；
3. 与 INC-0005 叠加时，被重做的那个 epoch 是在错误的损失权重下训练的，
   等于用坏数据覆盖好数据。

## 2. 证据

- `_a10_p2_t22_f_n5rgb_log.txt`：`检查点已加载 …checkpoint_epoch_0063.pth` /
  `Epoch: 63` 紧跟 `开始Epoch 63`——从已完成的 epoch 重跑；
- 同一文件修复后：`已完成 Epoch: 64　续跑起点: 65` 紧跟 `开始Epoch 65`，
  并新产出 `checkpoint_epoch_0065.pth`，未覆盖 0064；
- `checkpoint_epoch_0063.pth` 的 val 值在污染期被改写（0.04596 → 0.03352），
  即"归档件被重写"的直接物证。

## 3. 修复

```python
self.current_epoch = checkpoint['epoch'] + 1      # 从下一个 epoch 起跑
self.current_stage = self._get_current_stage()    # 阶段按起跑 epoch 重算
self._update_loss_weights()                       # 见 INC-0005
```

阶段改为**按起跑 epoch 重算**而非沿用 checkpoint 里的值：checkpoint 存的是
已完成 epoch 的阶段，跨阶段边界续跑（如 ckpt=29 → 起跑 30）时会差一个阶段。
静态验证：`(0,29)->1`、`(30,59)->2`、`(60,99)->3`；续跑起点映射
`29->30/阶段2`、`59->60/阶段3`、`63->64/阶段3`、`9->10/阶段1` 全部通过。

## 4. INC-0005 的触发条件（本次排查才精确定位）

INC-0005 的表现不是"每次续跑都错"，而是：

> `load_checkpoint` 把 `current_stage` 设成 checkpoint 里的值后，训练循环的
> `if new_stage != self.current_stage` 判断恰好**相等**，于是本该在此刻应用的
> 阶段权重更新被跳过，权重表停留在 `__init__` 的阶段 1。

因此：

| 续跑场景 | 是否污染 | 原因 |
|---|---|---|
| 段内起点在阶段 1（epoch < 30） | 否 | 阶段 1 权重恰好就是 `__init__` 的值 |
| 段跨阶段边界（如起点 29，段内 30 处切换） | 否 | epoch 30 触发真实切换，权重被正确刷新 |
| **段内起点在阶段 2 或 3 内部** | **是** | 判断相等 → 整段用阶段 1 权重 + 残差冻结 |

## 5. 受影响产物的定性

| 臂 | 结论 | 依据 |
|---|---|---|
| `p2_r0_gray_20260825`（R0 对照） | **干净** | 唯一一次续跑从 epoch 22 起（阶段 1 内），权重恰好一致 |
| `p2_t22_f_n5gray_20260825`（主交付） | **干净** | 单进程连续 100 epoch，总耗时 9981.81 s，无续跑 |
| `p2_t22_f_n5rgb`（双模态） | **污染，判作废** | 见下 |

**n5rgb 污染范围**：TensorBoard `val/total` 曲线在 epoch 40 与 50 两个分段
边界处出现跃升并长期高位（epoch 39 = 0.0632 → 40 = 0.1111，41–59 维持
0.09–0.11），而 epoch 60（阶段 3 边界，切换正常触发）立刻回落至 0.0494。
与灰度孪生臂同期 0.037–0.063 对比，可判定 **epoch 40–59 共 20 个 epoch 在
阶段 1 权重下训练**（应为阶段 2），另有若干 epoch 被 off-by-one 重写。

**处置**：该 run 不作为任何对比结论的来源，也**不作为 A10 续跑的起点**——
带着 20 个错误 epoch 的历史续跑到 100，权重轨迹无法与其它臂同口径比较
（纪律 D10）。若需要双模态臂，须从 epoch 0 重训。目录内已放置
`_CONTAMINATED_INC0006.md` 标记，避免后续误用。

## 6. 未尽事项

- [ ] 恢复语义单测入库：构造 stage2/stage3 断点 → load → 断言起跑 epoch、
      阶段号、权重表、`residual.requires_grad` 四项（补强 G2.1）；
- [ ] 若审计要求双模态臂进入 Phase 2 矩阵，需另行申请 ~3 h 算力从零重训。

## 7. 可借鉴失败点

"从 checkpoint 恢复的状态"与"由 epoch 推导的状态"必须择一为权威源，混用
会让守卫失效于最需要它的场合。本例中两者叠加，恰好使"检测到切换才更新"
这一优化在续跑路径上永久失效——**依赖差分触发的状态更新，在恢复路径上一定要
补一次全量同步**。
