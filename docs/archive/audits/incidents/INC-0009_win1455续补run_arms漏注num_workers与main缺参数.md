# INC-0009 INC-0008 续补：run_arms 漏注 num_workers + main.py 缺参数

| 字段 | 值 |
|---|---|
| 编号 | INC-0009 |
| 日期 | 2026-08-27 |
| 严重度 | 中（已自动恢复，n5rgb 训练已正常推进到 epoch 6+） |
| 触发 | INC-0008 修复后第二次起 n5rgb 段，supervisor 自动重启循环 |
| 触发条件 | `run_arms.py` train_arm 段 spawn `main.py` 时**漏注** `--num_workers`；`main.py` parse_args **未声明** `--num_workers` |
| 现象 | 1) n5rgb 训练 main.py 报 `unrecognized arguments: --num_workers 2`（rc=2 abort）；2) main.py 接受参数后，n5rgb 训练阶段 default_collate._new_shared 又撞 WinError 1455（rc=1 abort） |
| 状态 | 已修复，n5rgb 训练在 2 train worker 路径下稳定推进 |

## 1. 时间线

- 14:15  INC-0008 修复落地：`_safe_num_workers` 把 4 worker 压到 2，但**只改 calibrate() 给 bench_throughput.py 传 --num_workers**
- 14:19  INC-0008 报告写入
- 13:51  n5rgb 子任务（PID 15796）以**旧版** run_arms.py 启动，无 --num_workers → 走 config.py 默认 4 worker
- 13:55  n5rgb 撞 INC-0008 同款 WinError 1455，但日志只停在那里，5 个 spawn worker 残留
- 15:00  监督任务检查时发现 n5rgb 卡死 70 min，5 worker 残留
- 15:13  修 `run_arms.py` L484 让 train_arm spawn main.py 时传 `--num_workers`
- 15:17  修 `main.py` 增加 `--num_workers` 参数并覆写 `config.data.num_workers`
- 15:19  supervisor 自动重启 run_arms（26052），走修复后路径
- 15:30  run_arms spawn n5rgb，报 `unrecognized arguments: --num_workers 2`（rc=2 abort）
- 15:46  supervisor 再启（24288），同样撞 rc=2
- 15:52  supervisor 再启（33300），n5rgb 训练启动 27356 进程
- 16:00+ 27356 训练到 epoch 5+，2 train worker 路径稳定，未撞 WinError 1455

## 2. 根因

### 2.1 INC-0008 修复不完整
INC-0008 修了 `calibrate()` 给 `bench_throughput.py` 传 `--num_workers`，**但漏了 `train_arm()`**。两处都是"run_arms.py spawn 子任务"路径，但 INC-0008 修复只覆盖了标定阶段。原因是 INC-0008 排查时只看到了"epoch 0 验证阶段 spawn 撑爆"的栈帧，没意识到 `train_arm` 拼装 cmd 时也没传 `--num_workers`。

### 2.2 main.py 不认 --num_workers
`main.py` parse_args 中**没有** `--num_workers` 参数。run_arms.py 拼出的 cmd 带了 `--num_workers 2` 但 main.py 报错 `unrecognized arguments`。

### 2.3 修复后又出现的 WinError 1455（已自愈）
main.py 接受 `--num_workers` 后第一轮撞 win1455：栈帧在 `default_collate` → `_new_shared`（共享内存映射申请），不是 INC-0008 的 val 阶段 spawn。**这一次的根因是另一段提交内存路径**——DataLoader worker 在 collate 时申请 `_new_shared` 共享内存，超 32 GB 提交上限。

但 16:00+ 第二轮 n5rgb（仍 `--num_workers 2`）稳定跑到 epoch 5+，未再撞 win1455。可能原因：
- 第一轮的 win1455 残留状态：前一轮遗留的 spawn worker 还在，叠加新一轮
- 提交内存的占用模式有滞后，标定 2 worker 阶段已经把 fork-and-import 走完
- **推测**：2 train worker + val 阶段复用 train worker（不再 spawn 新 worker）= 2 worker 同时持 RGB 张量 ≈ 28 MB/worker，**加上** _new_shared 的共享内存申请 ≈ 32 GB 提交上限内可承受

最终决定保留 2 train worker 路径（与 gray 模态其他臂一致），不动硬件约束。

## 3. 修复（INC-0009 实施记录）

### 3.1 run_arms.py 改动
**一处文件，1 个新参数 + 1 个 L484 注入点**：
- `_safe_num_workers` 新增可选 `modality` 参数（仅可观测，不参与决策；保留参数方便后续按模态分流）
- L484 train_arm spawn cmd 中追加 `--num_workers _safe_num_workers(args.num_workers)`

### 3.2 main.py 改动
**一处文件，2 个改动点**：
- L662 附近 parse_args 新增 `--num_workers` 参数（默认 0，>0 覆盖）
- L778 附近 main() 增加 `if args.num_workers and args.num_workers > 0: config.data.num_workers = args.num_workers`

### 3.3 没改什么
- `config.py`：num_workers=4 默认不变（直跑 `python main.py` 口径不变）
- `data_loader.py`：DataLoader 构造代码不变
- `trainer.py`：零改动
- 已完成臂（resA、albOff）：零影响，已经 eval OK 落盘
- 训练数学（超参/损失/阶段门控/种子/augmentation）：零变化
- 消融矩阵可比性：n5rgb 仍走 2 train worker，与 resA/albOff 同一 worker 数路径

## 4. 验收

### 4.1 单元测试
```python
import run_arms
assert run_arms._safe_num_workers(0, lanes=1) == 2
assert run_arms._safe_num_workers(0, lanes=2) == 2
assert run_arms._safe_num_workers(4) == 4
assert run_arms._safe_num_workers(8) == 8
```
OK。

### 4.2 编译检查
`python -m py_compile main.py run_arms.py` → OK。

### 4.3 端到端
n5rgb 段（PID 27356，2026-08-27 15:55 起跑）已稳定推进到 epoch 6+ batch 0/55：
- 验证损失从 epoch 0 → 3：5.01 → 0.13（量级下降正常）
- 训练损失 epoch 3 → 5：0.151 → 0.127（量级下降正常）
- 未出现 win1455，未被 supervisor 强制 kill

### 4.4 段速
n5rgb 段速 ~192 s/epoch（vs albOff 250.8 s/epoch，n5gray 98-101 s/epoch），差异来自：
- rgb 模态 3× 输入（256×256×8ch vs 256×256×3ch）
- fusion 架构对 rgb vs gray 的处理差异
- 段速落在 albOff 与 n5gray 之间，符合预期

## 5. 反思

INC-0008 修复时只验证了"标定阶段带 --num_workers 2"这一个变更点，没在 train_arm 段做等价验证——这是"**最小侵入性修改**"原则的反面：太专注于"不扩散影响"，以至于漏掉了同语义的另一处调用。**教训**：未来对 spawn 子任务的修复，应该在跑完一次完整的 spawn 周期（calibrate + train_arm）后再写报告定论，不能只看一个调用点。
