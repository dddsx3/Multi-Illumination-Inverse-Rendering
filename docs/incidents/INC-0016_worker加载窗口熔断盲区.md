# INC-0016 · worker 加载窗口熔断盲区（epoch 0 启动段 pf 尖峰无护栏）

> 编号顺延：INC-0015 之后（INC-0013 为预留判别实验编号，未占用事故序列）。
> 日期：2026-09-04 · 严重度：**P0（22:3x 升级重开——EX-05 epoch 0 spawn 死锁再现，§6）** · 状态：**REOPENED（FIX-08-5 内存类加固对死锁形态无效；处置 = nw=0 绕开 spawn 通道）**
> 关联：INC-0008（WinError 1455 · spawn worker commit 尖峰）· INC-0014（系统停摆冷重启 · pf 耗尽主导信号）· 复盘 §四 · commit c59bce2（熔断主信号改 pf<1.5GB）
> 开立依据：任务书 T v2.2 条例 FIX-08-5 / G2（审计报告_20260904）。

---

## 1. 问题描述（盲区定义）

运行中熔断 `check_host_memory()`（pf<1.5GB 主信号，c59bce2）**只在训练 batch 循环内生效**，且按 10-batch 间隔巡检。而主机内存最陡的曲线发生在 **epoch 0 的加载窗口**：

- cuFFT/cuDNN/cuBLAS DLL 映射 + spawn worker 各自复制 Python 运行时（INC-0008 证据链：每 worker ~0.15GB commit、瞬时谷值 0.3–0.6GB）；
- `train_loader` 与 `val_loader` 的 worker 先后启动，叠加主进程大页锁内存；
- 该窗口内**没有任何熔断巡检**——第一次 `check_host_memory()` 在 batch 10 才执行。若启动时 pf 已在低水位，加载尖峰可在巡检前把 commit 推到上限，触发 WinError 1455 或（更糟）INC-0014 同款系统级冻结。

## 2. 观测证据（疑似案例，归因不唯一）

A3-1_noFiLM 夜间段（2026-09-04 00:32–02:39，logs/A3-1_noFiLM tfevents 时间戳重建）：

- 00:32 首个 epoch tfevents 落盘后，出现 **4895s + 7928s 两段长停**（≈3.6h，epoch 0–17 区间），非温度墙节奏（热墙停机为分钟级冷却循环）；
- 主机当时 pf 总 42GB/可用 ~22GB，phys 16GB 总/空闲 1.4–4.4GB 波动（_arm_A3-1_run.log 运行前自检段 04:05 实测 phys 4.38GB WARN）；
- 无 Python traceback、无 rc42 存档记录 → 符合"加载窗口无熔断护栏"特征（也符合外部干扰，如用户态休眠/杀软全盘扫描；**本次无法定证，故 P1 防再犯而非 P0 事故**）。

账本/锚点影响已回填：docs/gpu_ledger.md A3-1 行（有效 8.1h/墙钟 10.3h，"训练窗口 approximate"）；CALIBRATION.md 09-04 行注明两段长停。

## 3. 加固措施（FIX-08-5 三件套 + 断言，本 INC 落地）

| # | 措施 | 位置 | 状态 |
|---|---|---|---|
| 1 | **启动前置断言**：可用 pf < 3GB → 拒绝启动（BLOCK，提示关闭大内存应用/扩页面文件）；dry-run 放行但 WARN | `run_arms.py preflight()` | ✅ 落地 |
| 2 | **num_workers 自动档收紧 2→1**（Windows spawn + commit 尖峰权衡，显式传参仍原样尊重） | `run_arms.py _safe_num_workers()` HARD_CAP | ✅ 落地 |
| 3 | **epoch 0 预热窗口巡检加密**：前 10 batch 每 2 batch 查一次 pf/phys，其后恢复 10-batch 间隔 | `trainer.py` 训练循环 | ✅ 落地 |
| 4 | 三指纹启动即落盘（run 名确定后、训练循环前，含 HEAD/config/manifest）——为事后归因提供启动时刻资源/代码快照 | `run_arms.py record_fingerprints()`（FIX-08-4） | ✅ 落地 |

## 4. 残余风险与验收

- 前置断言读的是**启动时刻** pf；若训练中途第三方应用（浏览器/杀毒扫描）吃掉 commit，仍靠运行中熔断 pf<1.5GB 兜底（rc42 存档续跑）。断言不覆盖"启动后外因"，属已知接受。
- **闭合判据**：~~EX-04 干净即闭合~~ **作废——见 §6 再现记录**。原判据：EX-04（A3-1b lowSmooth）启动日志含三指纹行 + preflight pf 断言读数 + 全程无加载窗口长停即可闭合；若再现同款无痕长停 → 升 P0 重开归因。EX-04 本身全干净（100ep 零中断，273.7 s/epoch），但 EX-05 启动时**同类挂死再现**（§6）→ 按原判据升 P0 重开归因。

## 6. 再现记录（2026-09-04 22:3x · P0 重开）· A3-2_seed123 epoch 0 DataLoader spawn 死锁

**现象链（实证，全部当场采集）**：
- 21:46:42 启动 main.py（run_arms 自动三指纹行正常，HEAD=bab573a）；21:46:47 DataLoader spawn worker（PID 16800）；
- epoch 0 **未产出任何 batch**：无 checkpoint、tfevents 1094 字节后停写（27+ 分钟）、viz epoch_0000 空目录；
- **GPU 空载锁定**：utilization 0%、功耗 8.36 W（训练态应 60–100W+）、显存 6.2GB 被持有——CUDA 上下文已建但无 kernel 提交；
- **进程证据**：主进程 23096 CPU 时间 4 分钟仅 +0.05s（空转）；worker 16800 的 **25 线程全部 Wait(UserRequest)**——worker 等主进程消费管道、主进程等 worker 首批数据，双向死锁；
- 熔断（pf<1.5GB / phys<0.25GB）全程未触发——**本次不是内存耗尽**，是 IPC 死锁：死锁点在 DataLoader 迭代器取数阶段，**batch 循环未进入**，FIX-08-5 的预热加密巡检（epoch 0 前 10 batch 每 2 batch 查）一次都没执行到。**加固对死锁形态无效**——盲区比 §1 预估的"加载窗口内存尖峰"更深：spawn 通道本身可死锁。
- 处置：22:30 kill 主进程+worker（PowerShell Stop-Process）；run_arms 检测 rc=4294967295、`[abort] 本段未推进`，正确不评估；残留 tfevents/viz/ckpt 空目录清理；证据日志归档 `INC-0016_A3-2_spawn_deadlock_run_log.txt`。

**与 A3-1 凌晨两段长停的关系**：形态一致性上升（都在 epoch 0 加载窗口、都无熔断记录、都非热墙），但仍不能排除当时另有外因——本此有进程级证据（线程全 Wait + GPU 8W），归因收敛到 **Windows spawn DataLoader 的管道死锁**（INC-0008/0009 家族的第三种表现：0008=1455 崩溃、0009=参数漏传、本次=无告警死锁）。

**P0 处置（立即生效）**：
1. EX-05 重启以 `--num_workers 0`（单进程加载，绕开 spawn 通道；吞吐影响可控——bs4 数据集小、前两臂 1 worker 时 GPU 每 batch 0.6s 远大于读盘）；
2. `_safe_num_workers` 自动档封顶再降 2→**0**（本机默认零 spawn；16GB+ 机器可显式传参解除）；
3. 若 nw=0 下 epoch 0 仍卡（>10 min 无 batch）→ 说明非 spawn 通道问题，升层排查（数据盘/杀软钩子），INC 重开。

## 7. 时间线（更新）

- 2026-09-04 凌晨：A3-1 夜间段两段长停（本 INC §2 观测对象）
- 2026-09-04 12:2x：HANDOFF_20260904 §7.3 提示加载窗口风险（"worker 加载 cufft 时 commit 尖峰"）
- 2026-09-04 晚：审计裁定 G2 + 任务书 v2.2 FIX-08-5 开立本 INC；同批落地四项加固并冒烟
- 2026-09-04 21:46–22:30：EX-04 干净完成后，EX-05（A3-2 seed123）epoch 0 spawn 死锁再现（§6）→ **升 P0 重开归因，nw=0 处置**
