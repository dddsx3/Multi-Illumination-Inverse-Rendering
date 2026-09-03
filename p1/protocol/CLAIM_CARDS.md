# CLAIM_CARDS · 共享事实叙事卡

> 归属：任务书 T v2.0 条例 T0-3 / OP-2 · 本文件每张共享事实卡一个条目。
> 格式（OP-2）：`【S-编号】一句话 | 数值 | 来源文件(相对路径) | 复现命令 | 口径(允许/禁用的措辞) | 登记日期`
> 口径栏照抄 CLAIM_REGISTRY（v0.7 实测裁决段）禁词约束；每产出/登记一次 S 资产，10 分钟内追加并提交。
> 注记：CLAIM_REGISTRY 版本头已随 FIX-03（2026-09-04）统一为 v0.7；本文件世代口径以 Gen-A3 为准（FIX-02）。

## 卡片区

- 【S-01】合成 v3 test（124 场景）主结果——Gen-A3 世代（FIX-02 终稿回填）
  - 一句话：固定 N=5 线性域多光照（几何已知、物理约束 clamp 头、bs4）下，Gen-A3 复现臂（A3-0）在合成 v3 test 124 场景上达到 normal MAE 10.30° / PSNR 32.55 dB / albedo si-MAE 0.1482 / 物理违规率 0.0000%。
  - 数值（Gen-A3 主行）：normal MAE 10.3039±3.7507°；PSNR 32.546±3.499 dB；albedo si-MAE 0.14819±0.04458；depth_rmse_aligned 0.3377±0.0636
  - 来源文件：eval_output/A3-0_f_n5gray_seed42_test/eval_summary.json
  - 复现命令：python evaluate_model.py --checkpoint ckpt/A3-0_f_n5gray_seed42.pt --data_root D:/data/synthetic_v3 --split test --split_manifest splits/synthetic_v3.json --out_dir eval_output/A3-0_f_n5gray_seed42_test
  - reference-only 对照（历史世代 bs8、pre-constraint、ckpt 永久缺失，禁作对比基准）：F-N5 gray 7.792°/36.09/0.1279（p2_t22_f_n5gray_test）；F-N5 rgb v2 8.177°/37.25/0.1304（p2_t22_f_n5rgb_v2_test）；R0 gray 10.66°/36.04/0.0548（p2_r0_v3gray_test）
  - 口径：允许——"Gen-A3 世代（bs4 + 物理约束）"标注后引用本卡主行；**禁止**——无世代标注直接引用历史数字、把 bs8 历史世代与 Gen-A3 数字同格混比、joint recoverability 类措辞。
  - 登记日期：2026-09-03（数值终稿回填 FIX-02 2026-09-04）

- 【S-02】合成 v3 N 敏感性曲线——N_min=1（**世代注记：待 EX-01 重测定稿**）
  - 一句话：合成 v3 test 124 场景 × N=1–5 × 3 随机子集，normal MAE 极差 0.030°（<0.3%）——N=1 不退化。
  - 数值：MAE 极差 0.030°；相对 <0.3%（**数值出自旧消融世代 resA/albOff n_curve，协议与 Gen-A3 主臂不可比；EX-01 重测前禁止作为 N_min=1 的当前世代引用源**）
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
  - 数值：Gen-A3（A3-0，EX-02 重测 2026-09-04）= **MAE 40.41°**（median 40.18；10 物体 36.36–46.07，ball 46.07/bear 40.10/buddha 40.71）；历史世代 N=5 零样本 = 39.41°（R4″/W2-B.1，中位 ~40°）
  - 来源文件：eval_output/A3-0_f_n5gray_seed42_diligent/diligent_results.json（Gen-A3）；eval_diligent/diligent_results.json（历史）
  - 复现命令：python evaluate_diligent.py --root D:/data/DiLiGenT/pmsData --checkpoint ckpt/A3-0_f_n5gray_seed42.pt --num_lights 5 --num_lights_subsets 3 --out_dir eval_output/<run>_diligent
  - **世代注记（EX-02 已回填）**：Gen-A3 40.41° vs 历史 39.41°——同量级（+1.0° 无实质漂移），与"合成→真实迁移困难（~40°）"叙事一致；仍仅作 reference，不作达标 claim。
  - 口径：允许——标"zero-shot / reference / 固定子集"；**禁止**——将 40° 表述为达标性能、把 25° 门槛说成已达成、与 matched 重训数字混淆（matched 表须另列 reference-only）。
  - 登记日期：2026-09-03（Gen-A3 回填 2026-09-04 EX-02）

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
