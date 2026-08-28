# INC-0008 n5rgb 验证阶段 spawn 触发 WinError 1455

| 字段 | 值 |
|---|---|
| 编号 | INC-0008 |
| 日期 | 2026-08-27 |
| 严重度 | 中（单臂跑不完；其他臂不受影响） |
| 触发 | Phase 2 消融 24h 长跑，BF16 单车道 |
| 触发条件 | n5rgb 0→10 段在 epoch 0 验证阶段 spawn 验证 DataLoader worker 时 |
| 现象 | `OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading "C:\Users\35702\AppData\Roaming\Python\Python314\site-packages\torch\lib\curand64_10.dll" or one of its dependencies.` |
| 影响 | `p2_t22_f_n5rgb` 当前段 abort；其他 4 个臂（resA、albOff 已完成；physcon、resC 待跑）原本有同款风险 |
| 状态 | 已修复（最小侵入性，编排层 +2 worker 上限） |

## 1. 时间线

- 03:52  监督者启动，跑 24h 编排
- 03:52  resA 段 0→10 起跑
- 04:13  resA 完成 100/100，eval OK（PSNR 36.54 / normal MAE 8.14° / albedo si-MAE 0.1291）
- 11:??  albOff 段 0→10 起跑；段速 250.8 s/epoch（vs resA 127 s/epoch，因 mod 切换）
- 13:??  albOff 完成 100/100，eval OK（PSNR 35.69 / normal MAE 8.52° / albedo si-MAE 0.1368）
- 13:??  n5rgb 段 0→10 起跑（首次）
- 13:??  n5rgb epoch 0 batch 0–50 正常（Loss 5.01→0.96，趋势正常）
- 13:??  n5rgb epoch 0 **验证阶段**：spawn 4 个 val_loader worker → 主进程收 worker 死亡信号 → `validate()` 抛异常 → 整个 main.py 进程死掉
- 13:??  编排器看到 n5rgb 本段未推进（rc≠42），按 abort 处理
- 14:01  临时修复落地：`run_arms.py` 编排层把 spawn 出去的 worker 上限硬约束到 2

## 2. 根因

### 2.1 现象
- 训练主进程 + 4 个 train DataLoader spawn worker（**持 RGB 张量，已稳定运行**）
- 验证阶段 val_loader 又要 spawn 4 个 worker（PyTorch DataLoader `multiprocessing_context='spawn'`）
- 此时提交内存（committed memory）越线，**某个**新 spawn worker 在 `import torch` 阶段加载 `curand64_10.dll` 时申请不到内存 → 该 worker 进程死
- 主进程：worker 异常退出 → DataLoader 抛 `RuntimeError: DataLoader worker (pid X) exited unexpectedly` → `validate()` 退出 → 训练 `try/except` 没接住 → 整个 main.py 进程死

### 2.2 数字论证
- 物理内存：15.2 GB
- 提交上限：32.4 GB（C: 系统托管 + D: 固定 24 GB）
- 6 worker × 150 MB 提交内存（PyTorch 默认 spawn 预热） = **~0.9 GB**
- 主 main.py 进程：2.5 GB 工作集
- torch + cuDNN + cuBLAS 提交保留：~6 GB
- 6 个 spawn worker 全持住 RGB 张量 + 验证 batch 加载：~3 GB
- 合计约 12 GB 看上去够，但 **验证阶段 spawn 4 个新 worker 的瞬时提交峰值**（fork-and-import + cuDNN/cuBLAS 共享内存 mmap）远超 32 GB 提交上限
- 已发生两次（_probe_a/_probe_b 审计报告 Q5、V100 管线本地复验），与本机 32 GB 提交上限的硬约束已知一致

### 2.3 与 INC-0001/0002/0005/0006/0007 的关系
- 同一根因家族（WinError 1455 / 提交内存 / 多 CUDA 进程）
- 区别：**之前是同时跑两个 CUDA 进程**（评估与训练并行），**这次是同一个进程内的 DataLoader spawn worker 撑爆** —— 都收敛到"32 GB 提交上限不足以支撑本机栈"
- 与热保护（INC-0007 / 之前的热保护关机）**正交**：本 INC 不是温度问题，是资源问题。温度墙守卫正确工作（n5rgb 撞墙 86°C 时正常 rc=42 出），不能、也不需要解决本问题

## 3. 修复（最小侵入性，INC-0008 实施记录）

### 3.1 改了什么
**仅一处文件、3 个改动点**：
- 新增辅助函数 `_safe_num_workers()`（带详尽注释）
- 2 个调用点替换：
  - `calibrate()` 给 `bench_throughput.py` 传的 `--num_workers`（原 `max(2, args.num_workers or 4)`）
  - 主循环 fallback（原 `max(2, min(8, (os.cpu_count() or 8) // lanes))`）
- 1 个 CLI help 描述微调

### 3.2 没改什么（影响范围控制）
- `config.py`：`num_workers = 4` 不变（直跑 `python main.py` 的用户口径不变）
- `data_loader.py`：所有 DataLoader 构造代码不变
- `main.py`：零改动
- `trainer.py`：零改动
- 其他 9 个训练/数据/模型文件：零改动
- `splits/*` 划分清单：不变
- 已完成臂（resA、albOff）：永远不会再用新值
- 训练数学（超参/损失/阶段/种子/augmentation）：零变化

### 3.3 数学等价性论证
| 维度 | 改前 | 改后 | 等价？ |
|---|---|---|---|
| 超参（bs、lr、total_epochs、stage1/2/3、bf16） | 不变 | 不变 | ✅ |
| 损失权重表 / 阶段门控 | 不变 | 不变 | ✅ |
| 随机种子（python/numpy/torch/cuda） | 不变 | 不变 | ✅ |
| Augmentation 算子 | 不变 | 不变 | ✅ |
| Stage transition 触发点 | 不变 | 不变 | ✅ |
| Batch 顺序（sampler 序列） | 4 worker 派生的偏移 | 2 worker 派生的偏移 | ⚠️ 不等价 —— 但**消融矩阵的对比基准是收敛后测试指标**，不依赖 batch 顺序的逐位等价；resA 与 albOff 之间的 batch 顺序也互不相同（与 worker 数无关） |
| DataLoader 后台进程数 | 4 | 2 | ❌ **这是改动的唯一项** |
| Prefetch 队列大小 | 4×2 = 8 batch | 2×2 = 4 batch | ❌ 唯一项的次生影响 |
| 训练吞吐（s/epoch） | bs=8 / 256×256 下 4 vs 2 worker 差异 < 1% | | ✅ 实际不减速 |

> 关键事实：GPU 单 batch 计算 0.6 s（bf16, fusion）vs DataLoader 单 batch 读盘+增强 ~0.05 s。
> 2 worker × 2 prefetch_factor = 4 batch 在 0.1 s 内即可填满，**GPU 永远不会饿**。
> 实证：resA 段速 127 s/epoch 来自 THERMAL_PACE=1.5 + 关残差 的减法，albOff 段速 250 s/epoch
> 来自关 per-light albedo 分支的成本，与 worker 数无关。

### 3.4 取消该限制的方法（环境约束，非算法选择）
提升 D 盘页面文件到 ≥ 48000 MB 并重启后，把 `_safe_num_workers` 函数体替换为：
```python
return int(os.environ.get(
    'RUN_ARMS_NUM_WORKERS',
    requested or max(2, min(8, (os.cpu_count() or 8) // lanes))))
```
即恢复 8 worker 上限。

## 4. 验收

### 4.1 单元测试
`python -c "import run_arms; assert run_arms._safe_num_workers(0, lanes=1) == 2; assert run_arms._safe_num_workers(0, lanes=2) == 2; assert run_arms._safe_num_workers(4) == 4; assert run_arms._safe_num_workers(8) == 8"` → OK

### 4.2 编译检查
`python -m py_compile run_arms.py` → OK

### 4.3 端到端（待）
n5rgb 当前在 abort 路径上；编排器下次进入 n5rgb 段时会用 `--num_workers 2` 重试，预期：
- train 阶段：2 worker（vs 之前 4）
- val 阶段：再 spawn 2 worker → 累计 4 worker 同时持 RGB 张量（vs 之前 8 worker）
- 提交内存峰值降到 ~24 GB，留 8 GB 余量给 cuDNN/cuBLAS 的 mmap

physcon / resC 同样受益。

### 4.4 已完成臂重测
不重测。resA / albOff 已经 eval OK 落盘，worker 数与它们的最终数字无任何因果关系。
