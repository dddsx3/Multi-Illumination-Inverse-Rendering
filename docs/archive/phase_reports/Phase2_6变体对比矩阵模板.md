# Phase 2 6 变体对比矩阵模板

> **目的**：T2.5 / T2.7 消融矩阵的"骨架"——等所有变体训完后填数。
> **当前状态**：3 变体已完成（v2 / albOff / resA），3 变体待训（physcon / resC / noFiLM / lowSmooth）
> **关联**：`docs/Phase2_结论草稿.md` + `T_ARM_续做清单.md` + `report_assets/comparison_matrix_v2.csv`

---

## 1. 变体列表（含 INC-0012 物理约束重训版）

| # | 变体 | 训练旗标 | 状态 | 评估预期 |
|---|---|---|---|---|
| 1 | **R0** (Phase 1) | --model fusion --modality gray | ✅ 已交付 | Phase 1 v2 test 13 项 |
| 2 | **F-N5-gray** (Phase 2) | --model fusion --modality gray | ✅ 已交付 | 13 项 + 物理断言 |
| 3 | **v2 best (n5rgb)** | --model fusion --modality rgb | ✅ 已交付 | 13 项 + 物理断言 + N 曲线 |
| 4 | **v2 + INC-0012 重训** | +Sigmoid/Softplus 物理约束 | ⏳ 待启动 | 重训后 13+2 项对比 v2 |
| 5 | **F-albOff** | +--no_per_light_albedo | ✅ 已交付 | ΔA 价值消融 |
| 6 | **F-resA** | +--residual_off | ✅ 已交付 | 残差价值消融 |
| 7 | **F-physcon** | +--sh_constraint softplus | ⏳ 待启动 | T2.3 G2.3 |
| 8 | **F-resC** | +--res_hidden 32 | ⏳ 待启动 | 残差容量消融 |
| 9 | **F-noFiLM** | +--disable_film | ⏳ 待启动 | 判别实验 (b) |
| 10 | **F-lowSmooth** | +--albedo_smooth_stage1 1.0 | ⏳ 待启动 | 判别实验 (c) |

> **D11 公平对比纪律**：所有 Phase 2 内变体对比统一 v3 test 124 场景（与 Phase 1 v2 test 127 场景**禁止混合**）。
> **D12 冒烟隔离**：冒烟产物（如 T-PHYS）禁止进入对比矩阵。

---

## 2. 13+2 项指标矩阵（待填数）

下表每格 = "mean ± std"（"-" 表示未训）。13 项 = Phase 1 基线 13 项；+2 = INC-0012 物理断言（albedo/depth violation ratio）。

| 指标 | R0 | F-N5-gray | v2 best | v2+INC-0012 | albOff | resA | physcon | resC | noFiLM | lowSmooth |
|---|---|---|---|---|---|---|---|---|---|---|
| image_psnr ↑ | – | – | **37.25** | – | 35.69 | 36.54 | – | – | – | – |
| image_ssim ↑ | – | – | – | – | – | – | – | – | – | – |
| albedo_mse ↓ | – | – | – | – | – | – | – | – | – | – |
| albedo_mae ↓ | – | – | – | – | – | – | – | – | – | – |
| **albedo_si_mae ↓** | – | 0.128 | **0.0532** | – | – | – | – | – | – | – |
| normal_mae_deg ↓ | – | 7.79 | 8.18 | – | 8.52 | 8.14 | – | – | – | – |
| normal_median_deg ↓ | – | – | – | – | – | – | – | – | – | – |
| normal_acc_11_25 ↑ | – | – | – | – | – | – | – | – | – | – |
| normal_acc_22_5 ↑ | – | – | – | – | – | – | – | – | – | – |
| normal_acc_30 ↑ | – | – | – | – | – | – | – | – | – | – |
| depth_mae ↓ | – | – | – | – | – | – | – | – | – | – |
| depth_rmse ↓ | – | – | 0.3554 | – | – | – | – | – | – | – |
| depth_rmse_aligned ↓ | – | – | – | – | – | – | – | – | – | – |
| **+ albedo_violation_ratio ↓** | – | – | 0% (冒烟) | – | – | – | – | – | – | – |
| **+ depth_violation_ratio ↓** | – | – | 0% (冒烟) | – | – | – | – | – | – | – |

**待填项**：
- R0 / F-N5-gray / albOff / resA 列：用 B1 任务产物（INC-0012 物理断言）补全 +2 项
- v2 best 列：补全 +2 项（已有冒烟 0%，正式评估待 A3-bis 完）
- v2+INC-0012 列：等重训后填数
- physcon / resC / noFiLM / lowSmooth 列：等 T-ARM 训完填数

---

## 3. N 敏感性矩阵（已有 + 待扩展）

| 轨 | N 值 | 已有模型 | 待测模型 |
|---|---|---|---|
| 合成 v3 | {1, 2, 3, 4, 5} | v2 best (N 曲线已测) | albOff / resA（B2 任务）|
| DiLiGenT | {1, 2, 3, 5, 7, 10, 15} | v2 best (N 曲线已测) | albOff / resA（B3 任务）|

---

## 4. 判别实验 (a)(b)(c) culprit 判定表

| 实验 | 假设 | 变体 | 当前状态 | 预期结果 |
|---|---|---|---|---|
| (a) F-albOff | ΔA 分支与共享反照率头梯度竞争 | p2_t25_f_albOff | ✅ 已训（100 epoch, PSNR 35.69）| 若反照率恢复 → ΔA 是 culprit（**未恢复**，albedo_si_mae 仍 0.0532 量级）|
| (b) F-noFiLM | FiLM 调制干扰 bottleneck | p2_t25_f_noFiLM | ⏳ 待训 | 若反照率恢复 → FiLM 是 culprit |
| (c) F-lowSmooth | albedo_smooth=10.0 权重过高 | p2_t25_f_lowSmooth | ⏳ 待训 | 若反照率恢复 → 权重是 culprit |

**判定规则**：
- 单一变体反照率恢复 → 单一 culprit
- 多变体同时恢复 → 叠加实验（noFiLM+lowSmooth）判定主次
- 全部不恢复 → 当前优化空间已饱和，INC-0012 物理约束是主因

---

## 5. 消融结论草稿（T2.7 收尾用）

| 升级 | 收益 / 无收益 | 物理解释 |
|---|---|---|
| RGB vs 灰度模态 | **+5.2% PSNR**（v2 best 37.25 vs F-N5-gray 35.69 量级）| RGB 携带更多色彩信息，反照率估计更准 |
| FusionUNet vs 纯 U-Net | TBD（v2 vs R0 严格对比需统一 test 集）| 注意力融合模块的多光照聚合优势 |
| Sigmoid/Softplus 物理约束 | TBD（重训后验证）| 截断"平坦 shading + 平均图像 albedo"欺骗路径 |
| 残差模块 | TBD（F-resA vs v2 严格对比）| 非朗伯效应建模 |
| ΔA 分支 | TBD（F-albOff vs v2）| 逐光照反照率调制 |
| T2.3 softplus | TBD（F-physcon vs v2）| SH 物理约束替代 clamp hack |
| T2.5 res_hidden 32 | TBD（F-resC vs F-resA）| 残差容量减半 |
| FiLM 调制 | TBD（F-noFiLM vs v2）| 瓶颈层注意力调制 |
| albedo_smooth 1.0 vs 10.0 | TBD（F-lowSmooth vs v2）| 反照率平滑正则强度 |

---

## 6. 复现命令（D8 纪律）

每变体训练命令（D8 复现纪律）：
```bash
python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 8 \
    --max-lanes 1 --only <run_id> --amp-dtype bf16 --skip-package
```

每变体评估命令：
```bash
python -u evaluate_model.py --checkpoint <path> --data_root D:/data/synthetic_v3 \
    --split test --split_manifest splits/synthetic_v3.json --out_dir <out_dir>
```

---

## 7. 引用

- 顶层设计 v2.1 T2.5 任务卡：`文档类材料/顶层设计-任务工作指导书 (1).md` §4
- INC-0010 A6 13 项指标：`docs/incidents/INC-0010_A6_13项指标_执行版.md`
- N 曲线报告：`docs/design/t2_5_n_sensitivity_report.md`
- 阶段 2 结论草稿：`docs/Phase2_结论草稿.md`
- T-ARM 续做清单：`T_ARM_续做清单.md`

---

*本模板由 2026-08-28 23:00 阶段决策落地，等所有变体训完后填数。*
