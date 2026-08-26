# INC-0007 fp16 混合精度路径自始不可用（缺 unscale_）+ T4 回退陷阱

**日期**：2026-08-26　**发现人**：执行者（构建者）　**状态**：已修复并冒烟验证
**触发场景**：为评估「A10 6h → Tesla T4 18h」置换方案而首次真正运行 fp16 路径

---

## 1. 缺陷一：GradScaler 路径缺 `unscale_`，训练在第 1 个 batch 即停机

`--amp_dtype fp16` 路径的原实现：

```python
scaler.scale(total_loss).backward()
grad_norm = clip_grad_norm_(params, 1.0)   # ← 此时梯度仍被放大 2^16 倍
stability.check_grad_norm(grad_norm)       # ← 非有限 → 硬停机
```

`GradScaler` 默认把损失放大 65536 倍以保住 fp16 的小梯度，**裁剪与检查必须
先 `unscale_` 还原真实尺度**。原实现直接在放大后的梯度上操作，后果：

1. 裁剪阈值 1.0 作用在 65536 倍的梯度上，等于把梯度压成原来的 1/65536——
   即使不停机，训练也等于没在学；
2. 放大后的梯度极易超出 fp16 表示范围变成 inf/nan，`check_grad_norm` 判为
   非有限 → 抛 `RuntimeError: 梯度范数非有限（nan），硬停机`。

**实测**：T4 置换评估中首次运行 `--amp_dtype fp16`，第 1 个 batch 即停机
（`_smoke_fp16_log.txt` 首版）。该路径此前从未被执行过——Phase 1 T1.5 起
全部训练都走 bf16，`_train_fp32_log.txt` 为 0 字节，T2.1 回归冒烟也只覆盖
fp32/bf16 两档。

## 2. 缺陷二：非 Ampere GPU 的 bf16 自动回退会踩进缺陷一

`trainer._setup_optimizer` 原用 `torch.cuda.is_bf16_supported()` 判定，
不支持则**静默回退 fp16**。两个问题：

- 该 API 在 Turing（sm_75，如 T4）上可能因「仿真支持」返回 True，于是
  bf16 会以极低吞吐运行而不报警；
- 若返回 False 则回退到缺陷一的 fp16 路径 → 一上机就崩。

也就是说：**把这套代码直接丢到 T4 上，无论走哪个分支都会浪费租用时段。**

## 3. 修复

| 位置 | 修复 |
|---|---|
| `trainer.py` 反向传播段 | `scaler.scale(loss).backward()` 后立即 `scaler.unscale_(optimizer)`，再裁剪与检查 |
| `trainer.py` 守卫分支 | fp16 路径下非有限梯度改为「跳过该 batch + 计数」，交给 `scaler.step` 自行跳过并下调 scale；**bf16/fp32 路径的硬停机语义完全不变** |
| `stability.py` | 新增 `note_scaler_overflow()` / `note_scaler_ok()`：连续溢出达 `nan_streak_limit`（10）仍停机，避免「scale 无法收敛」被无限容忍 |
| `trainer.py` bf16 判定 | 改为按算力判定（`cc >= 8.0` 才算原生 BF16），非原生时打印显式声明再回退 fp16 |
| `trainer.py` GradScaler 构造 | 用 `torch.amp.GradScaler('cuda')` 新式 API，旧版自动回退 |

## 4. 验证

- **单测**：溢出连击未达限不抛、成功一次即重置、连续达限抛「无法收敛」；
  并回归确认 bf16/fp32 路径的非有限梯度仍立即硬停机。
- **冒烟**（fusion/gray，2 epoch 跨阶段 1→2，bs8）：

| 精度 | epoch0 val | epoch1 val | 溢出次数 | 结果 |
|---|---|---|---|---|
| fp16（修复后） | 1.0342 | 0.2013 | 0 | 正常收敛，无停机 |
| bf16（同配置对照） | 见 `_smoke_bf16_ref_log.txt` | — | — | 基线口径 |

## 5. 对 T4 置换方案的结论（详见 `docs/A10_6h_作战手册.md` §6）

修复后 fp16 可用，但**仍不足以在 18h 内跑完全部 5 臂**：换算需 30–46h
（最乐观的纯张量核心比值也需 23.6h）。fp32 则双重不可行——bs8 显存需
15.35GB 分配 / 17.76GB 保留，超出 T4 的 16GB；且 T4 fp32 算力仅 8.1 TFLOPS，
是其 fp16 张量路径的 1/8。

## 6. 未尽事项

- [ ] 若最终采用 T4/fp16，需向审计声明**数值口径偏差**（既有 R0 / F-N5-gray
      基线为 bf16），并附 fp16↔bf16 的短程曲线对照作为可比性证据；
- [ ] T2.1 回归冒烟矩阵补 fp16 一档（当前只有 fp32/bf16），防止该路径再次
      长期无人执行。

## 7. 可借鉴失败点

**没跑过的代码路径等于不存在。** `--amp_dtype fp16` 作为 CLI 选项存在了整个
Phase 1–2，还被写成「不支持 bf16 时自动回退」的兜底路径，却从未被执行——
真正需要它的那一刻（换到 Turing 卡）就是它暴雷的那一刻。凡是宣称「兜底」
的分支，必须进回归冒烟矩阵。
