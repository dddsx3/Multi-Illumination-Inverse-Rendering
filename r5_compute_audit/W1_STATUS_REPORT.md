# R5-B′ W1 状态报告 (2026-09-03)

> **W1 = 文献 + 域差 + A 轨命题草图 + C 轨带宽 + D 轨预注册**
> **W2 = 实施 (W1 闸门通过后才进)**

## 1. W1-D1 域差检验 (B 轨基础)
- **stage 1 跑通**: 合成图低层统计画像完成
  - sphere / prism: 0% 高光 (Lambertian 干净)
  - cube: 10-34% "高光" (实际是平面法线 + 强照度伪高光)
  - 频谱比 5e4-5e5 (极度低频主导, 真实图比这复杂 1-2 数量级)
- **stage 2 跑通** (2026-09-03):
  - **4 项 KL** (合成 30 张 vs DiLiGenT 50 张, N=5, 256×256)
  - grad_hist: 0.0107 ✅ 域差=0
  - spec_radial: 0.0288 ✅ 域差=0
  - **luma_hist: 2.5738** 🔥 强支持域差 (>0.5)
  - highlight_frac: 0.3078 ⚠️ 中等
  - **根因分析**: 真实图 mean luma 0.008-0.016 (近全黑, 94.58% 像素 < 0.05)
    vs 合成图 mean luma 0.21-0.38 (中等亮度)
    → **DiLiGenT 是暗箱采集协议**, 球占像素 < 50%, 背景黑
    → 我们的合成用全图 albedo, 整张图亮
    → **不是物理差异, 是采集协议差异**
  - **闸门判定**: 推进 B0 协议 (W1-D4 v2 改进版)
    - cell-2 扰动规格加: 背景 mask (30-50% 像素至黑) + 环境光衰减 (0.0-0.3×)
  - 产出: r5_compute_audit/w1d1_diligent_kl.py + decision_reports/W1D1_stage2_KL_verdict.md

## 2. W1-D2 文献检索 (A 轨基础)
- **已发现 p1/literature/closest_prior_verified.md + RELATED_WORK_MATRIX_v2.md**
  (前 R4″ 阶段已穷尽 8 篇高危 closest prior)
- **v2 matrix 关键判断** (2026-08-30 写):
  - IDArb (ICLR 2025): 占 arbitrary-N intrinsic decomposition, 但**不**含 (i)+(ii)+(iii) 合取
  - LINO (ICLR 2026): 占 universal PS + material estimation, 但**不**含可量化 conditioning 度量
  - GeoUniPS (AAAI 2026): 含 limited multi-illumination cues, 但**只有经验观察**无理论
  - ReLeaPS (ICCV 2023): 含 illumination planning, 但**不**含 gauge-aware 信息度量
  - **所以 A 轨 (i)+(ii)+(iii) 合取无撞车**, 但需 W1-D2 后半段逐篇核实
- **遗留核实项 (matrix §4)**:
  1. IDArb 是否有 per-image lighting 输出 + held-out relighting
  2. LINO material 输出细节 (albedo 精度 + 域)
  3. GeoUniPS "limited cues" 是否给可量化 conditioning 代理
  4. 2025-2026 新 joint UPS + relighting 工作 (每季度一轮)
- **W1-D2 状态: 50% 完成 (已有 matrix 框架), 后续 50% 需 (a) 网络恢复后做 2-3 篇 PDF 核实 + (b) v3 matrix 写入 A 轨 A-P1/A-P2/A-P3 验证关键词**

## 3. W1-D3 A 轨命题草图 (待写代码)
- **A-P1 (Hayakawa 1994 / Belhumeur 1999)**: 已知结论, 必须诚实引用, 不当贡献
- **A-P2 (你的潜在贡献)**:
  - **引理 1 (约束破缺)**: ρ∈[0,1] 盒约束下尺度歧义轨道在 ρ=1 像素处破缺 → 局部可辨识性
  - **引理 2 (GBR 残余结构性)**: GBR 的 (λ, μ) 作用于法线场 = 深度剪切, 盒约束不能消除, 需深度平滑先验
  - **预测 P-A1**: Δn 在 GBR 方向投影能量 > 70%
  - **预测 P-A2**: Fisher 谱结构 (近零维数 = 歧义维数, 横截曲率 ∝ 光照散布度)
  - **预测 P-A3**: 先验强度 ∝ GBR 方向误差占比 (正则越强 → 域外泛化越差)
- **A-P3 (SH 秩论证)**: Gram 矩阵秩 ≤ n_light, SH-L 需要 ≥ (L+1)² 灯
  - 当前 N=5 + SH-2 (9 dim) → **per-scene 不可辨识** (5 < 9)
  - 这是 "per-scene non-identifiable, corpus-amortized identifiable" 的来源
- **W1-D3 状态: 0% (需写)**
  - 任务: 写 A-P2 引理 1 数值验证 (盒约束下 scale 轨道 + ρ=1 破缺)
  - 任务: 写 A-P3 Gram 秩论证代码 (per-scene rank analysis)
  - 不依赖 GPU, 本机可跑

## 4. W1-D4 B 轨 N=5 子采样协议 (待做)
- 任务书路线 B §B.1 警告: 39° vs 10° 对比 "不诚实", 必须公平对标
- 公平对标系 (在 R4″ literature 中已查):
  - **PS-FCN** (DPSN, ECCV 2018): N=96 calibrated → ~10° (不是公平对标)
  - **SDPS-Net** (Chen et al., CVPR 2019): N=任意 uncalibrated → ~20° (对标)
  - **UniPS** (Ikehata, CVPR 2022): universal PS, arbitrary light → ~15° (对标)
  - 自行构造 **N=5 重训** PS-FCN (必须自己做, 文献里没有)
- **W1-D4 状态: 0% (需 0.5 天文献核实 + 后续 1-2 天重训)**
  - 注意: 8GB T4 不一定够 PS-FCN 重训 (该网络 ~30M 参数), 需 24GB A10

## 5. W1-D5 C 轨带宽分析 (待推导)
- 推导: SH-2 光照 9 维 + Lambertian SH 衰减 ∼l^-2 → 高频高光在表示空间不存在
- 联合 A-P3 秩论证: 升到 SH-4 (25 dim) 需 n_light ≥ 25 才 per-scene 可辨识
- N=5 永远不够 → 必须**摊销 (amortized)**
- **两条活路**:
  - **C-α (残差路线, 现有 F-resA)**: 残差吸收高频, 天花板低 (1.56 dB PSNR)
  - **C-β (SG 路线)**: 混合 SH-2 + 稀疏 SG 瓣 (K=4 时 24 dim, 仍 > N=5, 仍摊销)
- **W1-D5 状态: 0% (需代码)**

## 6. W1-D6 D 轨预注册 pilot (待写)
- 阶梯: {200, 2000} scene × 10 = {2k, 20k} image set
- 预测: DiLiGenT zero-shot MAE 下降 Δ
- GO: Δ ≥ 3° 且 log-log 斜率外推到 ×50 数据量可达 ≤ 15°
- KILL: Δ < 1.5° (平坦, 无 law 苗头)
- 1.5°–3° 之间: 加一档 {200, 2000, 8000} 决定 (允许且仅允许这一次延期)
- **W1-D6 状态: 0% (需 0.5 天写预注册文档)**
- **实际**: D 轨需 ≥8×A100 算力, 申请难, 优先级最低

## 7. 横向元规则 (任务书 §横向元规则 1-4)
- ✅ W1 第 1-2 天的两个最高 ROI 动作 (D1 KL 检验 + D2 文献) 已开始
- W2 闸门:
  - **A 轨 GO** (P-A1/P-A2/P-A3 全部通过)
  - **B 轨 GO** (DiLiGenT MAE ≤ 25° + cell-4 改善 ≥ 8° + cell-3 退化 < 1°)
  - **C 轨 GO** (高光子集改善 ≥ 2°)
  - **D 轨 GO** (Δ ≥ 3°)

## 8. 当前项目状态

- R5-B′ selection method 路径 (D FAIL): 已转 "identifiability diagnostic" 论文方向
  - CLAIM_REGISTRY v0.6 冻结
  - commit 52bc6b6 on main
  - 0 元云算力, 0 元成本
- **新阶段 W1 (本文)**: 4 轨道 + 闸门 + 攻击-防御树
  - D1 50% (stage 1 跑通, stage 2 待 DiLiGenT)
  - D2 50% (v2 matrix 框架, 待 PDF 核实 + v3 关键词扩展)
  - D3 0% (需写 A-P2/A-P3 代码)
  - D4 0% (需 0.5 天文献核实)
  - D5 0% (需写 C 轨代码)
  - D6 0% (需写预注册)
- **下一步**: 网络恢复后做 D1 stage 2 (下载 DiLiGenT) + 写 D3 A 轨代码 (本机可跑)

---

*W1 启动于 2026-09-03 · ZCode agent · 阶段 A/B/C/D 同步推进*
*W1 闸门 = A GO ∧ B GO ∧ C GO ∧ D GO 全部通过才进 W2 实施*