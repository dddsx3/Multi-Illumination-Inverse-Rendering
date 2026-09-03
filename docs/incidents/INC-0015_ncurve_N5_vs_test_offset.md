# INC-0015 · EX-01 n_curve N=5 与 test 评估同 ckpt 数值偏移（14.89° vs 10.30°）

> 日期：2026-09-04 · 严重度：🟠（判据 1 参照系 & S-02 冻结前必须校准）· 状态：待作者裁决
> 关联：EX-01（v2.1 §2）、S-02、R-F/R-G 判据包回填

## 时间线
- EX-01 完成：A3-0（Gen-A3）best_model 对 synthetic_v3 test 124 场景 × N∈{1..5} 重测 n_curve（eval_n_curve.py 默认协议），产物 `eval_output/A3-0_f_n5gray_seed42_n_curve/{n_curve_agg,n_curve_raw}.json`。
- 分支判定（EX-01 验收判据"平坦"）：N1 14.875 → N5 14.887，**极差 0.017° ≤ 0.5°**，曲线高度平坦，N_min=1 叙事在 n_curve 协议下保留 ✓。

## 异常证据
- 同一 ckpt 的 N=5（全 5 光子集，输入应与 test 评估一致）聚合 normal MAE：**n_curve=14.887±12.60** vs **test eval=10.304±3.75**（eval_output/A3-0_f_n5gray_seed42_test/eval_summary.json），偏移 +4.58°；median 9.12 vs 2.93 亦系统性偏移。
- n_curve per-subset 重尾（mean≫median）提示存在高分位 scene 大误差；test 评估 std 仅 3.75。

## 候选成因（未验证，供裁决方向）
1. eval_n_curve 对 N<5 用 `sel = combo + [combo[0]]*(num_images−N)` 重复补光喂入 5 输入模型——N=5 时无重复，理论应与 test 等价，故偏移更像 loader/协议级差异；
2. 两脚本对 GT normal/深度或遮罩、或 albedo/图像重投影（recon_target）的取法差异（需 diff data_loader 调用与 compute_all 入参）；
3. 聚合口径：n_curve 以 per-(scene,subset) 等权平均；test 为 per-scene 单次——N=5 均单子集，口径相同，排除。

## 处置（执行方已做 / 待裁决）
- ✅ EX-01 产物与 RUN_CARD 补测行入库（commit 见 git log）。
- ✅ S-02/判据 1 参照系：**暂缓以 n_curve 绝对值为冻结数字**；曲线平坦性结论可用于 N_min 叙事，但 R-F 判据 1 参照系需与主表同口径。
- 待作者裁决：(a) 校准 eval_n_curve 协议至与 evaluate_model 一致后重跑（~1–2h GPU）；(b) 接受 n_curve 自成体系但判据 1 参照系改用"主表 test 单次数字 + 平坦性引 n_curve"，绝对级差在 S-02 注记。
- 推荐 (a)：判据包/主图 1 需要口径自洽。

*2026-09-04 · 未校准前 S-02 数值行保持"待定"状态。*
