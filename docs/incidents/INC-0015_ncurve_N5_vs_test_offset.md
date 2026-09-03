# INC-0015 · EX-01 n_curve N=5 与 test 评估同 ckpt 数值偏移（14.89° vs 10.30°）

> 日期：2026-09-04 · 严重度：🟠（判据 1 参照系 & S-02 冻结前必须校准）· 状态：INC-15-1 根因已定位（batch 池化 vs scene 级）；裁决(2026-09-04)=选(a)校准后重跑
> 关联：EX-01（v2.1 §2）、S-02、R-F/R-G 判据包回填

## 时间线
- EX-01 完成：A3-0（Gen-A3）best_model 对 synthetic_v3 test 124 场景 × N∈{1..5} 重测 n_curve（eval_n_curve.py 默认协议），产物 `eval_output/A3-0_f_n5gray_seed42_n_curve/{n_curve_agg,n_curve_raw}.json`。
- 分支判定（EX-01 验收判据"平坦"）：N1 14.875 → N5 14.887，**极差 0.017° ≤ 0.5°**，曲线高度平坦，N_min=1 叙事在 n_curve 协议下保留 ✓。

## 异常证据
- 同一 ckpt 的 N=5（全 5 光子集，输入应与 test 评估一致）聚合 normal MAE：**n_curve=14.887±12.60** vs **test eval=10.304±3.75**（eval_output/A3-0_f_n5gray_seed42_test/eval_summary.json），偏移 +4.58°；median 9.12 vs 2.93 亦系统性偏移。
- n_curve per-subset 重尾（mean≫median）提示存在高分位 scene 大误差；test 评估 std 仅 3.75。

## 根因定位（INC-15-1 静态 diff + 数据对照，2026-09-04）

**主因（证据强）——evaluate_model 的 batch 级聚合 + per_scene 名义错标**：
- test `per_scene_metrics.csv` 的 normal_mae_deg 按每 4 个相邻 scene 一组完全同值
  （9.013×4、8.000×4、12.615×4…）：evaluate_model 以 batch>1 推理，`compute_all` 返回
  **batch 级标量**，第 331–341 行把该标量复制给 batch 内每个 scene → per_scene 行名不副实；
- n_curve 逐 scene `compute_all`（batch=1）→ 单 scene 级数值。两者不可比：
  - std：test 3.75（batch 内像素池化抹平 scene 间方差）vs n_curve 12.6（scene 级）；
  - 相关性 Pearson(n_curve N5, test per_scene) = **0.365**（池化 vs scene 级被稀释）；
  - 均值 +4.6° 与 albedo si-MAE（test 0.148 vs n_curve 0.054）疑似 albedo 的
    scale-invariant 归一化在 batch 池化下被跨 scene 污染（si-MAE 须逐 scene 归一）。

**链路 diff 清单（其余项均一致）**：加载（同 MultiLightingDataset is_training=False）✓；
GT normal/mask 源同（normal.npy/mask.npy）✓；前向与 renderer/residual stage3 同 ✓；
metric 函数同 compute_all —— **唯一实质差异 = 推理批量（batch>1 池化+复制 vs scene 级）**。

**影响面**：历史与 A3-0 的 test 主表数字均为 batch 池化口径（非 scene 级）；凡涉及
"per_scene 级"使用（n_curve 判据 1 参照系、逐 scene 分析）须以 scene 级口径为准。

## 升级：albdeo 归一化证据——主表口径本身存疑（需作者裁决，2026-09-04 EX-02 后）

- 同 ckpt N=5：n_curve 的 **albedo si-MAE=0.0543**（逐 scene 归一）vs A3-0 test **0.1482**
  （evaluate_model batch 池化）。scale-invariant 指标**必须逐 scene 归一**，batch 内跨 scene
  池化会系统性污染 → test 的 albedo 数字可能被 batch 放大（≈2.7×）。
- 推论：evaluate_model 的 batch 池化 + per_scene 复制不仅影响 n_curve 对齐，**可能使
  A3-0 主表（10.30°/0.1482，FIX-01/02 已发布）的"scene 级含义"不成立**；normal MAE
  scene 级真实值或接近 n_curve 的 ~14.9°（未确认，需 batch=1 A/B）。

**因此 INC-15-2 的修复方向不能单方面执行**——"把 n_curve 对齐到 evaluate_model"会把
scene 级口径拖回已知有缺陷的 batch 池化。必须作者裁决口径基准：
- **选项 A（推荐）**：以 **scene 级（batch=1）**为唯一评估口径——修 evaluate_model
  （batch=1 + per_scene 真值），重跑 A3-0 test（~10min）→ 主表数字可能更新
  （albedo→~0.054、normal MAE 待测），README/S-01/判据包同步刷新；
- **选项 B**：维持 batch 池化为主表口径（视为"全像素加权平均"），n_curve 改为同口径
  （batch 池化 + per_scene 复制），albedo 行加注"batch 池化口径"——接受 albedo 0.15 级别；
- 无论 A/B：EX-02（DiLiGenT 40.41°）已闭合不受影响；EX-03 待本 INC 闭合后启动。

*2026-09-04 · INC-15-1 完成（主因定位）+ 口径基准升级待裁决 · INC-15-2 暂停等裁决。*
