# INC-0016 · worker 加载窗口熔断盲区（epoch 0 启动段 pf 尖峰无护栏）

> 编号顺延：INC-0015 之后（INC-0013 为预留判别实验编号，未占用事故序列）。
> 日期：2026-09-04 · 严重度：P1（防再犯级，非事故级——A3-1 两次长停归因疑似项）· 状态：**OPEN（FIX-08-5 加固已落地，待 EX-04 实战验证后闭合）**
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
- **闭合判据**：EX-04（A3-1b lowSmooth）启动日志含三指纹行 + preflight pf 断言读数 + 全程无加载窗口长停（或长停时有明确外因记录），即可闭合本 INC；若再现同款无痕长停 → 升 P0 重开归因（考虑加 Windows 事件日志拉取脚本）。

## 5. 时间线

- 2026-09-04 凌晨：A3-1 夜间段两段长停（本 INC §2 观测对象）
- 2026-09-04 12:2x：HANDOFF_20260904 §7.3 提示加载窗口风险（"worker 加载 cufft 时 commit 尖峰"）
- 2026-09-04 晚：审计裁定 G2 + 任务书 v2.2 FIX-08-5 开立本 INC；同批落地四项加固并冒烟
