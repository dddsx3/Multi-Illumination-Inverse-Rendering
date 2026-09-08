# CI03（线性 oracle MC）· 1 页 memo（2026-09-08 · 卡 C12）

**判定：Gate B 素材通过。** CI03 线性 MC 18/18 checks 全绿（6 case 族 × 3 场景；
photometric N∈{1,3}×两档噪声 + 高条件数终止子场景（trials=10000）+ 随机块）。
config=`configs/ci03/ci03_formal.yaml`，摘要=`artifacts/frozen/ci03_formal_summary.json`。

## 核心数字（预注册 gate）

| 指标（最弱 5 模式 = 预测协方差特征基，mode-resolved） | 观测范围 | gate |
|---|---|---|
| joint ensemble 方差比 emp/pred（Prop 1 主判据） | **0.977–1.009** | median∈(0.95,1.05), IQR⊂(0.90,1.10) ✅ |
| coverage 68% / 95%（median across weak modes） | 0.676–0.693 / 0.947–0.955 | \|emp−nom\|<0.05 ✅ |
| fixed-δc 偏差闭式复现（SNR>10 才判定） | **0.960–1.013**（红队 1.04） | (0.9,1.2) ✅ |
| marginal 均值残差（joint sampling 的必要性） | <0.2 全过（红队 0.052） | <0.2 ✅ |
| 条件方差闭式 emp/pred（同模式向量二次型） | **0.975–1.026**（红队 1.0015） | (0.9,1.1) ✅ |
| 方向性 cond<marg（M 收缩映射） | 报告项不 gate（收缩≈1 的场景在 MC 噪声内） | 诊断 ✅ |

random_block 的 bias SNR=0.0–0.3（Λ=0.11 → bias 淹没在 MC 均值噪声地板下），
如实记 not-measurable 并豁免 bias gate（预注册 gate_bias_snr_min=10）——不是理论失败。

## 本轮三处实现层事故（如实记录，均已修复后整卡重跑）

1. **np.empty 未初始化行混入经验协方差**（最重）：per-case 扩样 trials=10000 时，
   系综循环误用全局 trials=4000 → E 的后 6000 行是未初始化内存 → photo_term_hi
   全指标假性 0.40（紧 IQR 0.403–0.405）。修复=循环用 trials_c；这是"降档决策
   基于实际运行路径"红线的姊妹坑：**扩样必须作用到循环本身**。
2. **条件方差比率用错了归一化**：首轮拿 marginal 预测当分母 → 比率 0.27–0.98 散布
   （恰 = M 收缩因子本身，是物理不是错误；红队 1.0015 是全像素口径）。修正 = 同一
   模式向量上的条件预测二次型 vᵀ(σ²ΔF⁻¹AᵀM²AΔF⁻¹)v。
3. **bias 可测性**：Λ 小的场景 bias 低于 MC 均值噪声地板，比率 4.5×/69× 是噪声；
   修复 = SNR 门（bias_snr=‖Fh·bias‖/sqrt(tr(C_cond)/trials) > 10 才判定）。

## 与红队 V1 的关系

红队 V1（全像素 canonical 口径，median 1.0045/IQR [0.984,1.025]）在 CI03 弱模式
白化口径下复现为 0.977–1.009——同一 Prop 1 恒等式，mode-resolved 读出精度相当。
条件方差收缩因子的场景依赖（0.27–0.98 散布）本身是 Fig.5 的素材（弱模式受 nuisance
伤害最大 → 收缩最强），已在 frozen 数据中留档。

## 结论与下一步

- C2（Cov=σ²ΔF⁻¹）与 Prop 1 Corollary 的有限样本恒等式在 mode-resolved 口径下全部
  验证；Gate B 主判据素材齐；
- 下一步 **C13**：非线性 validity envelope（首选 B 臂 = 解析非线性 SH+ReLU 管线，
  mask-flip 率/‖δc‖/SNR 三轴 → 理论误差 <10% 边界曲面）。
