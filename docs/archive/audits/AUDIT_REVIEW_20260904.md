# AUDIT_REVIEW · 全项目复盘与交接（2026-09-04 凌晨 01:45 快照）

> 用途：供作者后续审计 + 可能的 demo 交付。所有数字以入库 eval json 为源（R6 纪律）。
> 仓库：Multi-Illumination-Inverse-Rendering · 分支 main（本地 ahead 显示异常为环境假象，服务端已同步至最新）

---

## 一、当前状态总览

| 项 | 状态 |
|---|---|
| A3-0（Gen-A3 主臂）| ✅ 100/100 epoch 完成；**scene 级主表**已定（INC-0015 校准） |
| 评估口径 | ✅ **scene 级（batch=1）为唯一口径**（evaluate_model 已修）；旧 batch 池化数字作废 |
| EX-01 n_curve | ✅ Gen-A3 冻结版（极差 0.017°，N_min=1） |
| EX-02 DiLiGenT | ✅ 40.41°（S-04 回填） |
| EX-03 A3-1 noFiLM | ⏸️ **epoch 7/100 后因页面文件耗尽(1455)保护性终止**，ckpt 完好可续（见 §四） |
| 门禁 | ✅ gate-FIX-20260904 已打并推送 |
| 文档/账本/指纹 | ✅ CLAIM_CARDS/G0 定稿/CALIBRATION/gpu_ledger/RUN_CARD 三指纹/REGISTRY v0.7 |

## 二、主线时间线（09-03 → 09-04）

1. **安全层**：INC-0014（系统停摆复盘：热墙 + 主机内存熔断 + run_safe_arms + 自动续跑闭环）。
2. **阶段 0 文档**：T0-1 标定、T0-2 资产清点一稿、T0-3 叙事卡 S-01~S-05、T0-4 情景裁定 A。
3. **理论轨**：IDENTIFIABILITY_v4 草案（T2-1/T2-2/OPEN-5 数值一稿：两类机制口径）。
4. **FIX 批次（gate-FIX-20260904）**：README 世代双行 + 0.0532 除名；CLAIM_CARDS 世代对齐；
   REGISTRY v0.7；G0 定稿；RUN_CARD 三指纹；run_arms 默认 bs4；v2.1 附录入库。
5. **EX-01 → INC-0015（本夜最重要）**：n_curve 重测平坦，但 N=5 与主表差 4.6° → 根因=
   **evaluate_model batch 池化 + per_scene 错标 + albedo 归一化污染** → 作者裁决选项 A
   （scene 级唯一口径）→ 修复 + 重跑 + 全文档刷新 → **CLOSED**。
6. **EX-02**：DiLiGenT zero-shot 40.41°（无漂移）。
7. **EX-03**：A3-1 noFiLM 冒烟通过 → 生产启动 → 凌晨 01:42 页面文件耗尽 1455 → 保护性终止。

## 三、关键冻结数字（scene 级口径 · demo 直接用这些）

- **Gen-A3 主表**（A3-0，124 场景）：normal MAE **14.887°±12.62**｜PSNR **32.54±7.53 dB**｜
  albedo si-MAE **0.0543±0.045**｜物理违规 **0.0000%**（`eval_output/A3-0_f_n5gray_seed42_test_v2_scenelevel/`）
- **N 曲线（S-02 冻结）**：N1..N5 = 14.875/14.870/14.880/14.886/14.887° → 极差 0.017°，N_min=1
  （`eval_output/A3-0_f_n5gray_seed42_n_curve/`）
- **DiLiGenT（S-04）**：40.41°（10 物体 36.4–46.1）
- **历史世代（reference-only，batch 口径注记）**：F-N5 gray 7.792°/36.09/0.1279；rgb v2 8.177°/37.25/0.1304；R0 10.66°
- **判据包（v2.1 R-F/R-G 定稿）**：albedo 保护 ≤ **0.065**（1.2×0.0543）；判据 1 参照系 = Gen-A3 n_curve 曲线

**⚠️ 审计/演示红线**：旧 batch 池化数字（10.30°/0.1482）已作废——任何对外材料一律引用上述 scene 级数字并标注口径；禁止无世代/口径标注引用历史数字。

## 四、A3-1 noFiLM 事件（需早间续跑决策）

- 冒烟 3-epoch ✅（loss 2.60→0.70）。
- 生产：epoch 7/100（loss ≈0.15，收敛正常）时 **pf(commit) 降至 0.25–0.5GB** → DataLoader worker
  加载 torch/cufft 触发 WinError 1455 → 训练侧熔断（检查间隔 10 batch）未及时介入 worker 加载窗口。
- 处置：夜间无人值守下保护性 kill（0 python，GPU 释放，phys 回 2.98GB）。
- **续跑建议（早间执行）**：`run_arms` 从 checkpoint_epoch_0006 无缝续跑；**num_workers 降为 1**（
  减 commit 尖峰）；启动前确认 pf 余量 ≥3GB；若 pf 仍紧张 → 关闭大内存应用或改白天有人值守段跑。

## 五、提交链与标签（全部已推送 origin/main）

- 代表性 commit：`c59bce2`(安全调优) · `ad9a713`(A3-1 臂) · `2c97b5f`(INC-0015 A 落地) ·
  `16e3056`(EX-02) · `0f8d4ba`(EX-01) · `fb6a95e`(FIX-07) · `68855b4`(FIX-06 bs4) 等
- 标签：`gate-FIX-20260904`（新）｜`phase1-complete`｜`r4prime-frozen`

## 六、队列与待办（按 v2.1）

1. EX-03 续跑 A3-1（早间，nw=1）→ 验收（vs A3-0 差 ≤2.0°/albedo ≤0.03，超阈 INC+上报）→ RUN_CARD/S-06。
2. EX-04 A3-1b lowSmooth → EX-05 A3-2 3-seed → EX-06 A3-4 MeanPool → EX-07 A3-3 FW（A1 设计 T1-1~6 并行 CPU）。
3. G0 裁决书（09-13）需收尾（草案待起草）。
4. 历史世代数字若需与 Gen-A3 同口径比较 → 同口径化属 P1 批次（当前仅 reference-only 使用）。

## 七、风险与纪律备忘（防再犯）

- 本环境文件删除会被路由回收站且常失败 → 代码/脚本不依赖"删除成功"。
- 物理内存夜间偏紧；commit(pf) 是停摆主信号 → 已把熔断主信号改为 pf<1.5GB（phys 仅近零触发）；
  但 worker 加载窗口的瞬时 commit 尖峰仍可能绕开 10-batch 检查 → **nw 控制在 0–1 + 启动前 pf 检查**。
- GPU 禁止多任务并行；训练只走 run_safe_arms；禁裸跑。
- 评估/对外口径唯一 = scene 级（evaluate_model 已是 batch=1）。

*复盘快照 · 2026-09-04 01:45 · 后续以 git log + 各 INC/FIX/EX 文档为准。*
