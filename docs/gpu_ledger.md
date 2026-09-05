# GPU 算力账本（gpu_ledger）

> 归属：任务书 T v2.0 OP-4 · 表头：`日期|run|预算(h)|实际(h)|状态|备注` · 每周日 21:00 加周汇总行。
> 情景基线（T0-4 裁定后定）：情景 A 20-25h/周 | 情景 B 8-12h/周。

| 日期 | run | 预算(h) | 实际(h) | 状态 | 备注 |
|---|---|---|---|---|---|
| 2026-09-03 | A3-0_f_n5gray_seed42（含冒烟） | 11.0 | ~8.9（13:34→23:0x 完成） | complete | INC-0014 后 bs4 降配；100/100 epoch + eval_rc=0（scene 级主表见 *_test_v2_scenelevel，INC-0015）|
| 2026-09-04 | FIX-01..07 批次 | 0 GPU | 0 | **完成** | 强制修复书全绿；tag gate-FIX-20260904 已推送（fb6a95e）；EX-01/02 已闭合（INC-0015 scene 级口径）|
| 2026-09-04 | 世代变更行 Gen-A3 内 → A3-1 noFiLM | — | — | 登记 | 理由=判别 FiLM 必要性（v2.1 R-C）；训练代码与 A3-0 同语义（FIX-06 默认 bs4 一致；INC-0015 仅改评估链路 batch=1，不影响训练）；相关 commit=FIX-06/INC-0015(eval-only)/run_arms A3-1 臂 |
| 2026-09-04 | A3-1_noFiLM（EX-03，含冒烟+eval） | ~10（估） | **~8.1 有效 / 10.3 墙钟** | complete | 训练窗口 2026-09-04 00:32→10:48（approximate，从 logs/A3-1_noFiLM tfevents 时间戳重建：00:32 首epoch、02:39 ep16 ckpt、04:05 run_arms 续跑、10:48 ep99 ckpt+eval 10:49）；epoch 0–17 段有两段长停（4895s+7928s≈3.6h，原因未记录——疑似加载窗口熔断盲区，INC-0016 在案）；实际 epoch 速度 **287.8–292 s/epoch**（夜间+白天连续段一致，无昼夜分速证据）——**显著快于白天锚点 370.6s（bench proxy 保守值），CALIBRATION 已以 292s 为新基准锚点**（R-D 双锚点裁定：不分昼夜单锚点）；eval_rc=0 + RUN_CARD 已入库 |
| 2026-09-04 | FIX-08 批次（R1–R7+G1–G3 清理加固） | 0 GPU | 0 | **完成** | 残留清理（README图注/R-C/清点表/EX-03 报告，85e8813）+ 账本 A3-1 回填（fde2f5b）+ diligent RUN_CARD hash 补算（d3880cd）+ run_arms 自动三指纹+时间字段 冒烟PASS（c6e354e）+ INC-0016 开立与三件套加固 冒烟PASS（8875962）；**tag gate-FIX08-20260904 已推送** → 进入 EX-04 |
| 2026-09-04 | A3-1b_lowSmooth（EX-04，含冒烟 3ep） | 12（预算） | **7.68（13:31→21:07 墙钟）** | complete | 100/100 epoch 零中断零 rc42（FIX-08-5 加固后首个生产 run，INC-0016 加载窗口未复现）；273.7 s/epoch 实测；**RUN_CARD 自动三指纹首例**（HEAD=ba6ab76 启动即落盘）；判据全 PASS（normal 差 −1.69°≤2.0 / si-MAE 0.0558≤0.065 / phys 0%）；观测 **albedo range 0.168→0.363 压缩恢复改善，INC-0013(c) 闭环**；披露 PSNR −9.67dB；S-06·A3-1b 已登记 → 下一步 EX-05 seed123 |
| 2026-09-04 | A3-2_seed123 第①次启动（EX-05 尝试一） | — | **0.75（21:46→22:30 诊断+kill）** | aborted | **INC-0016 再现升 P0**：epoch 0 DataLoader spawn 死锁（worker 25 线程全 Wait、GPU 8W 空转、熔断/预热巡检未及进入 batch 循环）→ 人工 kill，零 GPU 有效计算，无产物；证据归档 INC-0016_A3-2_spawn_deadlock_run_log.txt |
| 2026-09-04 | A3-2_seed123 第②次启动（nw=0 尝试） | — | **~0.4（22:44→22:5x kill）** | aborted | 同款死锁再现——根因定位：main.py:804 把 `--num_workers 0` 当"未传"哨兵 → config 默认 4 spawn。INC-0009 镜像 bug，修复 6727ef9（哨兵 default=0→None） |
| 2026-09-04→05 | A3-2_seed123 第③次启动（修复后 nw=0） | 12（预算） | **7.6（23:02→06:38 夜跑）** | complete | 100/100 epoch 零中断零 rc42（INC-0016 根因修复 6727ef9 后首个全程 run，死锁未复现）；272 s/epoch；自动三指纹 HEAD=6727ef9；eval_rc=0；**seed 噪声带第一点：normal 差 −4.10°（超 2.0° 预期带，上报主智能体复核）**、si-MAE 差 −0.00014、PSNR +6.86dB、median 8.18° vs 8.41°；S-06·A3-2_seed123 已登记；EX-05 验收报告在案（3-seed 终判待 seed2024） |

## T0-4 · 情景裁定（2026-09-03，草案 → G0 裁决书确认）

**裁定：情景 A（20-25h/周）——以 T0-1 演练 + 今日实测为依据（任务书 T0-4 允许路径）。**

依据行（可追溯）：
- 今日单臂实测：13:34 启动 → 预计 ~22:00 完成（≈8.4h 单日连续 GPU），净速 289.5 s/epoch（bs4）；热墙 74–79°C 循环下 run_arms 自动 rc42→冷却→续跑，**无人值守成立**（见 _arm_A3-0_run.log、INC-0014_host_monitor.log）。
- 若按 OP-3 夜跑窗口（20:30 起 6h/夜 × 4 夜）+ 白天片段补充 → 20–25h/周可达成。
- 历史 run_arms 日志无完整归档（G-8），故按任务书允许的"T0-1 实测"路径取代历史日志路径。

**回退条件（写死）**：若连续两周实际周用量 < 12h（无人值守窗口受限 / 环境散热不允许），
在周汇总行标注并回退情景 B（8-12h/周）；情景变更需在 G0 后首次周汇总写明依据。
