# Phase 2 验收报告（初稿 · 2026-08-28 23:00 后期）

> **依据**：《独立审计与门槛验收规程》v2.1 §10.1 模板
> **阶段**：Phase 2（架构升级与消融）
> **当前状态**：⚠️ **阶段收口未达**——T-ARM 4 变体（physcon / resC / noFiLM / lowSmooth）+ A3-bis seed 123/2024 续训均待启动；G2.1/G2.4 已 PASS，其他 5 门禁部分 PASS 或未 PASS
> **建议**：本报告作为"现状盘点 + 缺口清单 + 下一位 agent 接力模板"，等 T-ARM 全部完成后扩展为正式验收报告

---

## 1. 环境

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU（11.94 GB 显存，bf16 native）
- 驱动：CUDA 12.8 + torch 2.12.0.dev20260303+cu128
- Python：3.14
- 操作系统：Windows 10.0.26200 x64
- 温度墙：THERMAL_RESUME=75 / THERMAL_LIMIT=80 / THERMAL_PACE=2.0（_env.sh 固化）
- 进程：seed 42 续跑中（占用 GPU 0，~80°C 温度节律），其他 0 训练进程

---

## 2. 复现路径

### 2.1 已交付 3 臂（按 handoff §0 一致）
- `p2_t25_f_resA` 100 epoch（PSNR 36.54 dB，2026-08-27 完成）
- `p2_t25_f_albOff` 100 epoch（PSNR 35.69 dB，2026-08-27 完成）
- `p2_t22_f_n5rgb_v2` 100 epoch（PSNR 37.25 dB，val=0.0159，2026-08-28 完成）

### 2.2 A3-bis seed 42 续跑（进行中）
- 命令：`run_arms.py --only p2_t22_f_n5rgb_v2_seed42 --amp-dtype bf16 --skip-package`
- 进度：epoch 33 → 当前 49/100（第 2 段 47→57）
- 状态：温度墙守卫工作正常，预计 3-4h 完跑

### 2.3 T-PHYS 冒烟（已完成）
- `python _smoke_phys_constraints.py --smoke_epochs 3 --albedo_smooth 10.0 --tag albedo10`
- `python _smoke_phys_constraints.py --smoke_epochs 3 --albedo_smooth 1.0 --tag albedo1`
- 产物：`_SMOKE_phys_constraints/{albedo10,albedo1}/*.json`

### 2.4 N 曲线双轨（已完成）
- 合成 v3：`python eval_n_curve.py --ns "1,2,3,4,5" --subsets_per_n 3`
- DiLiGenT：`python evaluate_diligent.py --n_curve_ns "1,2,3,5,7,10,15" --num_lights_subsets 3`
- 出图：`python plot_n_curve.py`
- 转 CSV：`python n_curve_to_csv.py`

---

## 3. 数据（按 G2.6 冻结 test 集）

### 3.1 划分文件
- `splits/synthetic_v2.json`：train 456 + val 50 + test 127 = 633 场景
- `splits/synthetic_v3.json`：train 447 + val 49 + test 124 = 620 场景
- v2/v3 划分差异：89 个场景从 v2 train 移到 v3 test（C4 审计发现，D11 公平对比影响）

### 3.2 13 项指标（v2 best, A6 §2）
- image_psnr ↑：**37.25 dB**（v2）
- normal_mae_deg ↓：8.18°
- albedo_si_mae ↓：0.0532
- depth_rmse ↓：0.3554
- 完整 13 项见 `docs/incidents/INC-0010_A6_13项指标_执行版.md`

### 3.3 +2 物理断言（INC-0012, A6 §9.1）
- albedo_violation_ratio ↓：**0.0000%**（T-PHYS 3 epoch × 2 组）
- depth_violation_ratio ↓：**0.0000%**

### 3.4 N 曲线
- 合成 v3：N∈{1..5}，normal_mae 极差 0.030°（< 0.3%）
- DiLiGenT：N∈{1,2,3,5,7,10,15}，MAE 极差 0.33°（< 1%）
- **N_min = 1**（双轨均不退化）

---

## 4. 公平性声明（D11 强制）

| 类别 | 声明 |
|---|---|
| Phase 1 vs Phase 2 | ⚠️ **不同 test 集**（v2 test 127 ≠ v3 test 124）—— 对比时**必须**分别标注 |
| Phase 2 内变体对比 | ✓ 统一 v3 test 124 场景（v2 / albOff / resA / physcon / resC / noFiLM / lowSmooth）|
| 真实世界（Phase 3 启动前）| 不混 Phase 2 test 集 |
| 残差对比 | F-resA / F-resC / F-v2 统一 test + 残差残置/容量变体；F-v2 vs F-resA residual_off 状态明确 |
| 模态对比 | RGB（v2）vs gray（F-N5-gray）注：因 test 集不同，**不能直接对比**；需 R0 复测统一 test 集 |
| N 曲线 | ✓ v2 best 同模型多 N 评估，无跨模型问题 |
| 冒烟 vs 全量 | ✓ T-PHYS 产物物理隔离（D12），禁止入对比矩阵 |

---

## 5. 异常记录（本阶段全部 INC 编号与关闭状态）

| INC | 状态 | 描述 |
|---|---|---|
| INC-0010 | ⚠️ 进行中 | 数学底层三重偏差，v2 best 已修；A3-bis 3-seed 抖动 + 13 项指标 v2+3-seed 综合版待 A3-bis 完跑后重写 |
| INC-0011 | ✅ 已关闭 | F-resA 单 run 反转方法论沉淀（决策 3）|
| INC-0012 | ✅ 已关闭 | 物理约束补建（决策 1）+ 二次风险验证（albedo_smooth=10.0 维持）|

---

## 6. 交付物清单

### 6.1 训练产物
- ✅ p2_t25_f_resA 100 ckpt
- ✅ p2_t25_f_albOff 100 ckpt
- ✅ p2_t22_f_n5rgb_v2 100 ckpt
- ⏳ p2_t22_f_n5rgb_v2_seed42 33/100 + 续跑中
- ⏳ p2_t22_f_n5rgb_v2_seed123/2024 待启动
- ⏳ p2_t23_f_physcon / p2_t25_f_resC / p2_t25_f_noFiLM / p2_t25_f_lowSmooth 待启动

### 6.2 评估产物
- ✅ `eval_output/p2_t22_f_n5rgb_v2_test/eval_summary.json`（v2 best 124 test 场景）
- ✅ `eval_output/p2_t25_f_albOff_test/eval_summary.json`（albOff 124 test 场景）
- ✅ `eval_output/p2_t25_f_resA_test/eval_summary.json`（resA 124 test 场景）
- ✅ `eval_output/n_curve_synth_v3/n_curve_{raw,agg}.json`（1860 推理）
- ✅ `eval_diligent/n_curve/diligent_n_curve.json`（210 推理）
- ✅ `eval_diligent/diligent_results.json`（v2 单 N=5 等距对照）

### 6.3 报告
- ✅ `docs/incidents/INC-0010_数学底层三重偏差与编排器续跑冲突.md` §8 链注
- ✅ `docs/incidents/INC-0010_A6_13项指标_执行版.md` §9 增补
- ✅ `docs/incidents/INC-0011_F-resA单run反转与seed噪声影响.md`
- ✅ `docs/incidents/INC-0012_物理约束补建与albedo_smooth二次风险验证.md`
- ✅ `docs/design/t2_2_design.md` §10 物理约束加注
- ✅ `docs/design/t2_5_n_sensitivity_report.md` N 曲线报告
- ✅ `docs/design/splits_audit_c4.md` 划分审计 + D11 隐患
- ✅ `docs/design/paper_constraints_audit_c3.md` §10 约束核对
- ✅ `docs/Phase2_结论草稿.md` 论文骨架
- ✅ `docs/Phase2_6变体对比矩阵模板.md` 矩阵模板
- ✅ `T_ARM_续做清单.md` 下一位 agent 接力
- ✅ `CHANGES_20260828_后期.md` 修改文件清单
- ✅ `D14_判据充分性纪律_候选.md` 候选纪律

### 6.4 图表
- ✅ `report_assets/comparison_matrix.{md,csv,xlsx}`（已有）
- ✅ `report_assets/curve_*.png`（已有）
- ✅ `report_assets/n_curve_{synth,diligent,combined}.png`（新）

### 6.5 代码
- ✅ `fusion_unet.py`（INC-0012 物理约束 + A1 noFiLM 开关）
- ✅ `evaluate_model.py`（assert_physical + --num_lights N + B1 边界修复）
- ✅ `evaluate_diligent.py`（n_curve 模式 + modality 探测）
- ✅ `eval_n_curve.py`（import 修正 + fusion 探测 + luma 重建目标）
- ✅ `main.py`（--disable_film / --albedo_smooth_stage1 旗标）
- ✅ `run_arms.py`（ARMS 列表加 4 变体 + 状态标注）
- ✅ `plot_n_curve.py`（CJK 字体 + combined 视图）
- ✅ `n_curve_to_csv.py`（CSV / xlsx 转换）
- ✅ `_smoke_phys_constraints.py`（T-PHYS 冒烟脚本）
- ✅ `make_report_assets.py`（ARMS 列表扩展 + 物理断言集成）

---

## 7. 结论（自评）

| 判定 | 理由 |
|---|---|
| ⚠️ **有条件通过 / 建议** | 主体达标（v2 best 交付 + 物理约束补建 + N 曲线双轨 + 文档链注）|
| 阻塞 | T-ARM 4 变体未训（physcon / resC / noFiLM / lowSmooth）+ A3-bis seed 123/2024 未启动 |
| 建议 | 等 T-ARM 续训完成（约 22-30h 串行温度节律）+ A3-bis 3-seed 完跑（约 11-16h）后扩展本报告为正式版 |

**当前不申请 Phase 3 放行**——需 G2.3 / G2.5 / G2.7 全部 PASS 后启动 Phase 3（真实世界验证）。

---

## 8. 引用

- 顶层设计 v2.1：`文档类材料/顶层设计-任务工作指导书 (1).md`
- 独立审计规程 v2.1：`独立审计执行/顶层设计-独立审计与门槛验收规程 (1).md`
- 中期审计 v2：`文档类材料/Phase2中期审计-问题发现与改进方向 (1).md`
- Handoff 文档：`repo/HANDOFF_20260828.md`
- 续做清单：`T_ARM_续做清单.md`

---

*本初稿由 2026-08-28 23:00 阶段决策落地，等 T-ARM 4 变体训完后扩展为正式版。*
