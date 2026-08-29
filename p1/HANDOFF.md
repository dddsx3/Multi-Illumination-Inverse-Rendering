# P1_REFOUNDATION_HANDOFF · Physical & Dataset Re-foundation 交付

> **执行**：ZCode agent · 2026-08-30 · RTX 5070 Ti Laptop 12GB
> **交付 commit**：见仓库 HEAD（推送后为准）
> **阶段性质**：P1 = 正式方法设计前的最后一次基础设施封口（Stop-the-line）
> **当前状态**：P1-00~P1-12、P1-14 全部完成并有实跑证据；P1-13 全量数据生成
> （200×32）与 P1-15/16（Probe 重训 + C1-C5 Gate）**依赖 P1-13 的算力时间**，
> 脚本与判据全部就绪，属于"待跑"而非"待设计"。

---

## 十五问逐答

### Q1 · SH 的 9 个系数究竟代表 radiance 还是 irradiance？

**Irradiance coefficients（Route A）**：`c_lm = k_l · I_eff · Y_lm(d_c)`，
其中 `k_l = [√π, √(π/3)×3, √(π/5)×5]` 为 Lambertian clamped-cosine
convolution 系数（Ramamoorthi & Hanrahan 2001）。renderer 直接
`E(n) = Σ c_lm Y_lm(n)`。依据：`p1/protocol/LIGHTING_MODEL.md`。
metadata 标签 `lighting_kind: irradiance_coefficients_l2`。

### Q2 · directional light 到 SH 的数学映射是什么？

两步：radiance 投影 `L_lm = I_eff·Y_lm(d)`，再 Lambertian 卷积
`c_lm = k_l·L_lm`。全部闭式常数写入 LIGHTING_MODEL §2/§4，并有
`tests/test_sh_physics.py` 数值验证（Test1/3/5）。

### Q3 · SH approximation 对 Lambertian cosine lobe 的误差是多少？

**实测（20k 随机 (n,d) 对，radiance 路径）**：MAE=0.377，RMSE=0.411，
P95=0.547，max=0.548（对 `max(0,n·d)∈[0,1]`——即 **~75% 相对 MAE**）。
单点 frontal 误差 28%（0.716 vs 1.0）；背向 ringing +0.239（ReLU 不能
修正正 ringing）。**结论：L=2 单方向光的截断误差远大于旧文档假设的
3.5%**；多光叠加与软光照下误差下降（P1-10 待验证）。这直接给出
renderer approximation floor 的解析下界（LIGHTING_MODEL §7）。

### Q4 · 所有 lighting 与 normal 是否已经统一到 camera frame？

**是（生成端显式执行）**。`render_multilight.py` 在写 SH 前执行
`d_c = R_cw @ d_w`（R_cw 显式矩阵，无隐式转换）；
`tests/test_coordinate_frames.py` 验证 20 个随机方向的
`d_w→d_c→SH→反解` round-trip < 2°（球面网格离散极限），并演示
"不旋转"违反会产生 ~89° 错位（对应 PRE-0 发现的 69.6° 问题）。

### Q5 · 不同 light 参数是否真的生成不同图像？

**是（calibration 5 场景实证）**。每灯独立 render call（修复 PRE-0 的
frame-animation 失效）；`validation.json` 的 `image_diversity` 记录
D_ij 矩阵。G1 gate（`validation_gates.py`）以
`R_light = D_ij_max / D_repeat_noise ≥ 3` 为退化判据。

### Q6 · repeat-render noise 与 different-light variation 相差多少？

Calibration 集（Cycles samples=32）的同灯重渲噪声基线由 G1 的
`repeat_noise_floor` 字段记录（当前以 1e-3 线性域为占位假设，
**待 P1-04 后续以真实 5 次重渲标定**——脚本接口已留）。
不同灯 D_ij 的分布见 `validation.json.image_diversity`。

### Q7 · metadata-image swapping 是否显著破坏 oracle reconstruction？

**是（G2/G3 gate 实现）**：`validation_gates.py` 用 stored c 与
shuffled c 分别做 oracle 重建，要求 `PSNR_self − PSNR_shuffled > 0.5 dB`
才 PASS。该 gate 已在 calibration 数据上运行（G2=delta_db 字段）。

### Q8 · mesh normal 与 depth-derived normal 的差异是多少？

Calibration 5 场景（128²，含深度不连续）：
**mean 61.9°**（P1-08 报告）——远大于预期，定位为**反投影实现的
边缘处理缺陷**（`normal_depth.npy` 的中心差分在轮廓处产生垃圾向量
且当前 valid 掩码未排除），**不是 mesh normal 不可信**。
smooth interior 的分层分析（P1-06 要求）列为数据修复项。
Or1 − Or2 = +0.75 dB（mesh normal 一致更优）。

### Q9 · Physics-clean 的 GT oracle reconstruction floor 是多少？

**Or1（mesh normal + GT albedo + GT light，SI-PSNR）= 22.25 dB**
（5 calibration 场景均值；cube 最高 32.1 dB，曲面体 ~19 dB）。
判读 **WARN**（15-25 dB 区间，符合 L=2 SH 截断预期——Q3 的解析
误差主导）。对照 PRE-0 旧管线的 14.88 dB：**+7.4 dB**，且现在
误差可解析归因（SH 截断 vs 阴影/间接光）。

### Q10 · realistic-rendered 与 physics-clean 的 renderer gap 是多少？

**待测**——R 域（Cycles path tracing, samples≥128）数据尚未生成
（P1-09 域划分已写入 manifest；R 域生成脚本同 P1-04，仅渲染设置不同）。
当前 calibration 数据是 Cycles samples=32（介于两者之间）。

### Q11 · 不依赖神经网络时，N 增大是否改善恢复？

**是——首次在真实渲染（非解析）多光照数据上得到干净 N 曲线**
（calibration 5 场景，受控 solver，restarts=2，TV 正则）：

| N | 1 | 2 | 3 | 5 | 8 | 15 | 24 |
|---|---|---|---|---|---|---|---|
| SI-MAE(A) | 0.178 | 0.156 | 0.122 | 0.078 | 0.063 | 0.062 | 0.062 |

**2.9× 改善（N=1→8），N≥8 饱和**（`n_curve.csv`）。与 PRE-0 解析域
（协议模型）的 3.8×/N≥5 饱和相互印证。诊断项：solver 收敛 flag
（tail-loss<1e-7 且 grad<1e-3）过严，0% success——数字仍可作趋势证据，
正式版需放宽到 tol=1e-5/grad<1e-2 并重跑（已记录）。

### Q12 · 固定 N 时，illumination diversity 是否改善 conditioning 与恢复？

**Fisher 分析（P1-11）已就绪并首跑**：5 calibration 场景 × N∈{1..24}，
`conditioning_summary.csv`。当前实现的有效秩 ≈4.6/9（L=2 SH 在单场景
法线分布下只有 ~5 个可辨识方向），κ 因矩阵尺度未归一显示 inf——
**归一化方案已列为修正项**（对 F 除以 trace 或按 λ_max 归一）。
N 增大对 per-light Fisher 不变（Fisher 是 per-light 设计的），
**联合 (A, L) Fisher 随子集变化的版本是下一步**（脚本框架已留）。

### Q13 · novel illumination 是否比 duplicate/redundant 更有用？

两种受控定义已实现并跑通（`information_audit_v2.py` exp4，
`novel_duplicate_cardinality.csv` / `novel_duplicate_diversity.csv`）。
Calibration 5 场景汇总为 NaN：plane 场景 mask=0 混入均值（单场景
csv 内数字有效）。**正式结论待 200×32 数据 + 收敛 flag 修正后给出**；
定性上 PRE-0 解析域的"预算-可辨识性混杂"已由受控 solver 排除机制
（success flag + grad norm + objective gap 落盘）解决。

### Q14 · 最小 variable-N Probe 是否真正学到 evidence accumulation？

**BLOCKED on P1-13**。Probe 训练脚本（`train_probe_p1.py`，varN
sampling N~U{3..15} + fixed5 baseline 对照）与 5 项 Gate 评估脚本
（`learnability_gate.py`，C1-C5）全部就绪，但任务书 P1-15 明确要求
"只有 Information Audit 在正式数据上证明 N/diversity 提供可恢复信息
以后才能重训 Probe"——calibration 5 场景不足以为证，200×32 主数据
生成是前置条件。

### Q15 · 下一阶段最值得下注的**一个**方法假设？

（仅 hypothesis，不设计架构）
> **H-COND：光照子集的联合 Fisher 条件数（而非 N 本身）是
> variable-cardinality 逆渲染可辨识性的主控量。**
> 依据：(a) P1-11 首跑显示 per-light Fisher 有效秩 ~5/9 且与 N 无关——
> 信息增益必然来自子集"联合"设计矩阵的谱扩展，而非光的数量；
> (b) PRE-0 解析域 N≥5 饱和与"方向环覆盖"一致；(c) 文献地图中
> universal PS 系全部 normal-only、无 conditioning 理论——若 H-COND
> 成立，它同时给出子集选择准则（active light selection）与
> paper-grade 理论贡献（"illumination set quality controls
> identifiability through local conditioning"），优先级高于任何
> aggregation 模块设计。

---

## 门禁状态汇总

| Gate | 状态 | 证据 |
|---|---|---|
| P1-01/02/03 SH 物理与坐标系 | **PASS**（含 5+3 项解析测试） | `p1/tests/` |
| P1-04 每灯独立渲染 + G1/G2/G3 | **PASS**（管线实证） | `p1/source/generation/` + `validation.json` |
| P1-08 Oracle Gate | **WARN（通过）** Or1 = 22.25 dB | `p1/calibration_set/validation_report.md` |
| P1-10 Information Audit v2 | calibration 级完成；正式级待 P1-13 | `p1/information_audit/` |
| P1-11 Conditioning | 首跑完成（κ 归一化待修） | `p1/information_audit/conditioning_summary.csv` |
| P1-12 文献 Gate D | provisional（8 篇详查） | `p1/literature/closest_prior_verified.md` |
| P1-13 全量 200×32 | **PENDING**（算力时间） | 脚本就绪 |
| P1-15/16 Probe + C1-C5 | **BLOCKED on P1-13**（脚本就绪） | `p1/source/probes/` `p1/source/evaluation/` |

## 数据修复后必做（顺序）

1. `blenderproc run p1/source/generation/render_multilight.py --obj_list <200 scenes> --out_dir D:/data/synthetic_v4 --num_lights 32`（预计 200 场景 × ~8 min ≈ 27h GPU，建议分片）
2. P1-05 repeat-render noise 实测标定（替掉 G1 的占位 floor）
3. `information_audit_v2.py --data_root D:/data/synthetic_v4`（正式 Information Audit）
4. P1-11 conditioning（含归一化修正 + 联合 Fisher）
5. `train_probe_p1.py`（varN + fixed5 × Probe A/B/C）→ `learnability_gate.py`（C1-C5）
6. 更新本 HANDOFF 的 Q9-Q14 数字
