# CLAIM_CARDS · 共享事实叙事卡

> 归属：任务书 T v2.0 条例 T0-3 / OP-2 · 本文件每张共享事实卡一个条目。
> 格式（OP-2）：`【S-编号】一句话 | 数值 | 来源文件(相对路径) | 复现命令 | 口径(允许/禁用的措辞) | 登记日期`
> 口径栏照抄 CLAIM_REGISTRY（v0.7 实测裁决段）禁词约束；每产出/登记一次 S 资产，10 分钟内追加并提交。
> 审计注记：CLAIM_REGISTRY.md 文件头版本号仍写 v0.6（正文已含 v0.7 裁决段、git 提交亦标 v0.7）——版本头待主智能体确认后统一，本文件不擅自修改"宪法"。

## 卡片区

- 【S-01】合成 v3 test（124 场景）主结果——历史世代基准，A3-0 复现中
  - 一句话：固定 N=5 线性域多光照（几何已知）下，fusion 前向逆渲染在合成 v3 test 124 场景上达到 normal MAE 7.792° / PSNR 36.09 dB / albedo si-MAE 0.1279（F-N5-gray 世代）。
  - 数值：normal MAE 7.792°；PSNR 36.0905；albedo si-MAE 0.1279（README §4.1 的 albedo 0.0532 系跨臂混标，见 docs/G0_资产清点表.md）
  - 来源文件：eval_output/p2_t22_f_n5gray_test/eval_summary.json
  - 复现命令：python evaluate_model.py --checkpoint ckpt/A3-0_f_n5gray_seed42.pt --data_root D:/data/synthetic_v3 --split test --split_manifest splits/synthetic_v3.json --out_dir eval_output/A3-0_f_n5gray_seed42_test（A3-0 bs4 世代产出后按同协议重评）
  - 口径：允许——normal MAE/PSNR/albedo si-MAE，口径与 CLAIM_REGISTRY System Claim 一致（known geometry、feed-forward、fixed-N、linear-domain）；**禁止**——joint recoverability 类、把 bs8 历史世代与 A3-0 bs4 世代数字直接混比而不注明 batch 口径、将 DiLiGenT 迁移量级与合成数字并列为主结果。
  - 登记日期：2026-09-03（A3-0 完成后数值终稿）

- 【S-02】合成 v3 N 敏感性曲线——N_min=1
  - 一句话：合成 v3 test 124 场景 × N=1–5 × 3 随机子集，normal MAE 极差 0.030°（<0.3%）——N=1 不退化。
  - 数值：MAE 极差 0.030°；相对 <0.3%
  - 来源文件：eval_output/n_curve_synth_v3/{n_curve_agg.json, n_curve_raw.json}；解读报告 docs/design/t2_5_n_sensitivity_report.md
  - 复现命令：python eval_n_curve.py --checkpoint ckpt/<run>.pt --ns "1,2,3,4,5" --subsets_per_n 3
  - 口径：允许——"subset-sensitivity saturation / relative subset-sensitivity"；**禁止**——noise-floor saturation / hits the noise floor / render noise floor / N curve is projection of conditioning（禁词替换为 solver-repeat noise / repeatability floor）。
  - 登记日期：2026-09-03

- 【S-03】物理断言违规率 0%
  - 一句话：已训 3 臂（albOff/resA 等）物理约束断言违规率 0.0000%（Sigmoid/Softplus 约束，INC-0012）。
  - 数值：phys_albedo_violation_ratio=0.0；phys_depth_violation_ratio=0.0（3 臂局部证据）
  - 来源文件：eval_output/p2_t25_f_albOff_v2/eval_summary.json、p2_t25_f_resA_v2/…、p2_t25_f_albOff_n_curve/…（字段含 phys_*_violation_ratio）
  - 复现命令：python evaluate_model.py …（输出含 physical 断言字段）；A3 世代每 run 复核后追加行
  - 口径：允许写"物理约束违规率 0.0000%（本世代实测）"；**禁止**——在 A3 世代未逐 run 复核前将旧世代数字写为全局结论。
  - 登记日期：2026-09-03

- 【S-04】DiLiGenT 迁移量级（合成→真实零样本）~40°
  - 一句话：DiLiGenT 10 物体零样本迁移 normal MAE 中位约 40°（球 47°/熊 40°/佛 41°/…；README N=5 子集零样本 39.41°），任务书 §B 25° 门槛未达——如实作为迁移困难披露，不作主 claim。
  - 数值：中位 ~40°（W2-B.1 cell-1 baseline，R4″ 复用）；N=5 零样本 39.41°
  - 来源文件：eval_diligent/diligent_results.json（R4″ 实测）；README §4.1；r5_compute_audit/W2B1_cell1_baseline.md
  - 复现命令：python evaluate_diligent.py --checkpoint ckpt/<run>.pt …（零样本协议；A3-0 完成后复核）
  - 口径：允许——标"zero-shot / reference / 固定子集"；**禁止**——将 40° 表述为达标性能、把 25° 门槛说成已达成、与 matched 重训数字混淆（matched 表须另列 reference-only）。
  - 登记日期：2026-09-03

- 【S-05】GSIQ 排名稳定性（albedo 不敏感 + held-out 保持）与 GBR 主导性
  - 一句话：GSIQ（Gauge-Schur Information Quality）排名在 albedo 绝对量变化下稳定、在 held-out scene 保持；任意残差优先沿 GBR 群方向展开。
  - 数值：Q1 ρ(O,A)=0.99997（in-domain，P=500）；Q3 held-out median ρ=1.0000 / min 0.9762（24 cell）；W2-A.1 GBR 重建误差 0.39 vs RANDOM 1.00（差值 +0.61）
  - 来源文件：r5/r5_p1_albedo_ablation.csv、r5/r5_p2_heldout.csv、r5_compute_audit/raw_profile/a_track_p_a1_gbr.csv；裁决：r5_compute_audit/decision_reports/W2A1_P_A1_GBR_Verdict.md
  - 复现命令：python r5_compute_audit/w2a1_gbr_proj.py（GBR）；Q1/Q3 数据源脚本见 r5_compute_audit/README_操作表.md
  - 口径：允许——"GSIQ 作为 F_eff 的 ill-conditioning audit / rank stability under albedo & scene variation / 提供 identifiability diagnostic 工具"；**禁止**——"measures absolute information"、未升级前使用 "predicts reconstruction quality" / "enables subset selection" / "outperforms random"（C3 升级路径 + D FAIL 判定：selection 假说已否）。
  - 登记日期：2026-09-03

- 【S-06 / S-07 / S-09】A3 世代新卡（占位）：随 A3-0~A3-5 各 run 评估产出登记（S-06 主结果新世代 / S-07 GSIQ 定义与口径标注 / S-09 FW 融合）。A4-1 归因图须引用 S-07 卡口径（GSIQ 定义 + 不衡量 absolute information）。

## 禁词自检（T0-3 验收）

已对照 CLAIM_REGISTRY 字面禁词清单逐卡扫描本文件：无 noise-floor / joint recoverability / render noise floor / N-curve-as-projection / selection-method 升级措辞命中。新增卡一律沿用。

*状态：S-01~S-05 建卡完成（2026-09-03）；A3-0 完成后回填 S-01/S-04 终稿与 S-06。*
