**Calibration--Confidence Continuum\
全新实验设计书与 TCI 论文执行计划**

版本 v1.0 · 2026-09-07 · 冻结理论后的执行版

**目标：把三轮红队已冻结的理论骨架转化为可审计、可复现、可投稿 IEEE Transactions on Computational Imaging 的完整研究工程。**

本设计书不再重新讨论顶层方向；所有新实验只能服务于已冻结主张、证伪风险或论文必要证据链。任何新"漂亮现象"若不改变 CLAIMS_REGISTRY，不得挤占主线。

仓库集成说明：当前对话尚未提供可读取的 GitHub 仓库 URL/授权，因此本文中的目录与模块是"目标架构"，不是对现有仓库的虚构描述。GitHub 连接后应执行一次 path-by-path migration map，把现有脚本映射到本架构；理论、Gate、实验矩阵和写作提纲不依赖该映射。

# 0. 执行总则：冻结什么、允许改变什么

最终红队已经放行理论冻结：Lemma 1（HCRB/Schur 连续谱与单调性）、Prop. 1（affine-Gaussian 有限样本协方差恒等式）、Prop. 2（gauge-lifting 精确谱响应）、λ⋆ 诊断、calibration-retention spectrum，以及"合成精确性 → Monte-Carlo → 受控真实 corruption"的验证阶梯。后续只允许修实现、修统计协议、修表述，不允许再换主线。

-   主问题：光照校准从完全已知到完全未知的连续变化，如何以逐模式（mode-resolved）的方式改变逆渲染/光度立体中的有效信息与可恢复性？

-   主因果链：calibration uncertainty → nuisance coupling → mode-specific information loss → gauge lifting / weak-mode transition → empirical reconstruction degradation。

-   新颖性权重：不放在 Schur/HCRB/GLS 本身；放在 inverse-rendering 实例化、gauge lifting 的解析响应、场景条件 λ⋆ 诊断、逐模式预测，以及受控真实数据的定量吻合。

-   强制禁令：不得用 trace/logdet 作为 continuum 主证据；不得把 retention spectrum 用于定义 λ⋆；不得按"第 j 小特征值"跨 λ 追踪模式；不得在 Λ=0 秩亏路径调用普通 solve；不得把 fixed-δc Monte-Carlo 当成 marginal CRB 验证。

# 1. 研究问题、假设与成功标准

## 1.1 核心研究问题

1\. RQ1：在白化观测空间中，校准协方差 Σ_c 与图像噪声 σ² 如何共同决定有效 calibration confidence Λ = σ²Σ_c⁻¹，并诱导从 uncalibrated 到 calibrated 的信息连续谱？

2\. RQ2：当经典 gauge 满足 Aa = Bc̄ 时，有限 calibration confidence 如何以可解析方式抬升该方向的信息，并在何种场景条件下越过本征弱模式地板？

3\. RQ3：理论 ΔF(Λ) 的逐模式方差预测是否能在 affine-Gaussian Monte-Carlo 中达到有限样本精确性，并在非线性渲染、mask 变化与真实数据 corruption 中保持可用的秩/尺度预测能力？

4\. RQ4：归一化 retention spectrum ρ_j(Λ) 能否作为 within-scene 的稳定读出，并在多灯/多图条件下通过连续性追踪避免 mode swapping？

5\. RQ5：在 DiLiGenT/OpenIllumination 等真实数据上，理论预测是否能解释"哪些模式先坏、何时坏、坏多少"，而不仅仅展示平均误差随 corruption 单调上升？

## 1.2 可证伪假设

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **ID**            **假设**                                                                                              **主实验**        **失败判据**
  ----------------- ----------------------------------------------------------------------------------------------------- ----------------- ----------------------------------------------------------
  H1                在 affine-Gaussian、mask 固定且模型正确时，joint MAP/GLS 的 marginal covariance 与 σ²ΔF(Λ)⁻¹ 一致。   CI01/CI03         逐模式方差比超出 MC 误差预算或系统偏离。

  H2                gauge 绝对信息满足精确谱闭式，小 λ 线性抬升、大 λ 饱和。                                              CI02              闭式与数值 Rayleigh 商不一致，或端点斜率/饱和值错误。

  H3                λ⋆ 的小 λ 预测 μ_floor/Σα_i² 在适用域内给出正确数量级。                                               CI02              跨场景长期偏差 \>1 decade，且无法由"离开线性区"解释。

  H4                retention spectrum 0≤ρ≤1；其模式仅用于 within-scene 解释。                                            CI01/CI02         出现稳定越界且非数值误差，或跨场景比较被迫依赖模式身份。

  H5                真实 corruption 的弱模式退化与 ΔF 连续谱预测存在显著秩相关/尺度相关。                                 CI04/CI05         相关性接近零，或只在平均 MAE 上单调而逐模式预测失败。
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1.3 投稿级成功标准

-   理论：所有正文命题均有独立手工证明 + 符号/数值 sanity test；正文不把成熟 HCRB、mixed-model、GUM 结果包装为新定理。

-   实现：每个 Fisher/Schur 计算至少两条独立路线交叉验证；CI 必须运行 V1--V6 对应的 known-answer tests。

-   统计：主结论使用 mode-resolved 指标、置信区间/重复种子、预注册 Gate；不能靠单一漂亮 seed。

-   真实数据：OpenIllumination 做 controlled corruption 定量验证；DiLiGenT 只承担 real sanity，不把未知模型失配归因于 calibration uncertainty。

-   可复现性：一条命令从 config 生成论文 Figure/Table；输出带 git commit、config hash、seed、dataset manifest。TCI 官方明确鼓励公开生成图表所需代码与数据。

# 2. 冻结数学模型与符号规范

## 2.1 统一白化模型

所有理论与代码从加权/白化形式开始，不再把同方差噪声当作唯一入口。对局部线性化后的观测：

y = A x + B δc + ε, ε \~ N(0, σ² I), δc \~ N(0, Σ_c).

若原始观测噪声为 Σ_y，则先以 W = Σ_y\^{-1/2} 白化 A、B、残差；后续公式在白化空间使用。定义

Λ = σ² Σ_c\^{-1}, M(Λ)=I−B(BᵀB+Λ)\^{-1}Bᵀ, ΔF(Λ)=AᵀM(Λ)A.

物理解释：continuum 横轴不是"裸 prior precision"，而是 calibration information relative to image-noise level。参数化时使用 δc = Σ_0\^{1/2} z, z\~N(0,τ\^{-1}I)，把 τ 作为主横轴；Λ 仅作为计算记号。

## 2.2 两套读出必须严格分离

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **度量**           **定义**                                      **用途**                            **限制**
  ------------------ --------------------------------------------- ----------------------------------- --------------------------------------------------------------
  绝对信息谱         eig(ΔF(τ)) 或指定方向 Rayleigh quotient       gauge lifting、μ_floor、λ⋆          可跨 τ 比同一物理参数化；跨场景只在单位一致时谨慎比较

  归一化 retention   R(τ)=F∞\^{-1/2}ΔF(τ)F∞\^{-1/2}, ρ_j∈\[0,1\]   within-scene 信息保留、模式可视化   禁止用于 λ⋆；跨场景只比较统计分布，不宣称 mode identity 可比
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. 每个 Lemma / Proposition 的严格证明计划

## Lemma 1：Schur/HCRB calibration continuum 与单调性

目标陈述：在 BᵀB+Λ ≻0 的可逆域，ΔF(Λ)=Aᵀ\[I−B(BᵀB+Λ)\^{-1}Bᵀ\]A；若 0≼Λ₁≼Λ₂，则 ΔF(Λ₁)≼ΔF(Λ₂)≼F∞=AᵀA。Λ→0 用 Moore--Penrose 极限定义，Λ→∞ 得 F∞。

### 证明步骤：

1\. 从 joint information/Hessian 写块矩阵 H=\[\[AᵀA,AᵀB\],\[BᵀA,BᵀB+Λ\]\]。

2\. 对 nuisance 块取 Schur complement 得 ΔF。

3\. 利用 Q₁≼Q₂ ⇒ Q₂\^{-1}≼Q₁\^{-1} 推出 M、ΔF 的 Loewner 单调。

4\. Λ=0 秩亏时用 B=USVᵀ 的 thin-SVD 展开，证明 M(0)=I−UUᵀ；禁止把 full-SVD 零奇异方向写成 0/0。

5\. 一般 W 情形先白化，或直接写 AᵀWA−AᵀWB(BᵀWB+Λ)\^{-1}BᵀWA。

### 审计与失败条件：

-   HCRB/Schur complement 是成熟结果，正文用 recall/instantiate，不用"novel theorem"。

-   单元测试：V1b、N3-fix、随机 Loewner tests、Λ=0 rank-deficient case。

## Proposition 1：affine-Gaussian 下 joint MAP / GLS 的有限样本 covariance identity

目标陈述：若 ΔF 可逆，profiled joint MAP 为 x̂=ΔF\^{-1}AᵀMy；在 joint ensemble δc\~N(0,Σ_c), ε\~N(0,σ²I) 下 E\[x̂\]=x 且 Cov(x̂)=σ²ΔF\^{-1}。

### 证明步骤：

1\. 由 MAP objective \|\|y−Ax−Bc\|\|²+cᵀΛc 消去 c，得到 profiled normal equation ΔF x̂=AᵀMy。

2\. 边缘化 δc 得 y\|x\~N(Ax, σ²I+BΣ_cBᵀ)。

3\. 用 Woodbury 证明 V\^{-1}=σ\^{-2}M，故 x̂ 是 GLS。

4\. 代入 GLS covariance (AᵀV\^{-1}A)\^{-1}=σ²ΔF\^{-1}，得到有限样本恒等式。

5\. 附加 Corollary：fixed-δc 时 bias=ΔF\^{-1}AᵀMBδc；conditional covariance=σ²ΔF\^{-1}AᵀM²AΔF\^{-1}。

### 审计与失败条件：

-   引用 HCRB + linear mixed models/Henderson/Harville；定位为"verify/recall in our setting"。

-   单元测试：V1 2000+ MC、V1-fix、不同 cond(ΔF) 的误差预算曲线。

## Proposition 2：gauge-lifting 精确谱响应

目标陈述：若 Aa=Bc̄，且白化 nuisance prior 为 Λ=λI，B=USVᵀ，α=Vᵀc̄，则 aᵀΔF(λ)a=Σ_i α_i² s_i² λ/(s_i²+λ)。因此小 λ 线性抬升，大 λ 饱和到 \|\|Aa\|\|²。

### 证明步骤：

1\. 由 ΔF a=AᵀB(BᵀB+λI)\^{-1}λc̄。

2\. 左乘 aᵀ 并用 Aa=Bc̄，将式子化为 c̄ᵀBᵀB(BᵀB+λI)\^{-1}λc̄。

3\. 代入 SVD 对角化得到逐奇异方向闭式。

4\. Taylor 展开：λ→0 时对 s_i\>0 得 λΣα_i²+O(λ²)；λ→∞ 时趋于 Σα_i²s_i²=\|\|Bc̄\|\|²。

5\. 各向异性 Λ 仅作为补充：保留矩阵式，不强求一维闭式。

### 审计与失败条件：

-   V2 25-decade grid 作为 known-answer；数值实现必须同时用闭式与直接 Rayleigh quotient。

-   正文称 Proposition，不称"new theorem"；新颖性来自 render-specific interpretation 与验证。

## Lemma 2：calibration-retention spectral bound

目标陈述：在 F∞ 正定的可识别子空间，R(Λ)=F∞\^{-1/2}ΔF(Λ)F∞\^{-1/2} 满足 0≼R≼I，故 0≤ρ_j≤1。

### 证明步骤：

1\. 由 Lemma 1 得 0≼ΔF≼F∞。

2\. 对 Loewner 不等式作 congruence transform F∞\^{-1/2}(·)F∞\^{-1/2}。

3\. 若 F∞ 秩亏，先限制到 range(F∞)，用 eigen/SVD basis 明确可识别子空间。

### 审计与失败条件：

-   严禁 F∞\^{-1/4} 等实现错误；用 generalized eigenvalue 与 whitening 两条路线交叉验证。

-   V3 谱界是回归测试。

## Definition/Diagnostic：scene-conditioned observability crossover λ⋆

不是普适定理。定义在绝对信息单位：gauge 信息曲线与场景本征弱模式地板 μ_floor 的交叉。小 λ 线性域预测 λ⋆≈μ_floor/Σα_i²。

### 证明步骤：

1\. 先给 μ_floor 的操作性定义：去除 gauge 子空间后，在 uncalibrated/near-uncalibrated 基线中的最小非 gauge Rayleigh/eigen floor。

2\. 证明小 λ 一阶近似后解线性方程得到预测器。

3\. 给出适用条件：λ⋆≪min\_{α_i≠0}s_i²；否则使用闭式数值求根而非线性近似。

4\. 报告 prediction/scan ratio 和 log10 error，不用单一绝对差。

### 审计与失败条件：

-   R(Λ) retention 不参与 λ⋆ 定义。

-   V3 当前单场景 ratio 0.74 只能作为先验 sanity，不得写成泛化结论。

# 4. CI01--CI05 最终实验矩阵（冻结规格）

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **ID**   **主题**                                **问题**                                       **数据**                                **横轴/因素**                                   **主输出**                                               **Gate**                                        **失败处置**
  -------- --------------------------------------- ---------------------------------------------- --------------------------------------- ----------------------------------------------- -------------------------------------------------------- ----------------------------------------------- -----------------------------------------------------------
  CI01     Algebra / invariance                    公式、尺度、参数化、秩亏实现是否正确           synthetic linear blocks                 τ: 0→∞；rank(B)；W；reparameterization          machine precision identities；ρ bounds；endpoint         所有核心误差 \<1e-10（MC 除外）；Λ=0 无 solve   任何代数 identity 失败：停止后续实验，先修 core

  CI02     Gauge lifting / λ⋆                      gauge 如何被 calibration confidence 抬升       controlled synthetic scenes             τ/λ；scene geometry；albedo distribution        absolute gauge info；μ_floor；λ⋆；retention heatmap      闭式 vs direct \<1e-8；λ⋆ log error 预注册      若 λ⋆ 失效：保留闭式，降级 crossover claim

  CI03     Estimator calibration / linearization   理论 covariance 在何时可达、何时因非线性失效   linear + nonlinear renderer MC          σ；Σ_c；SNR；mask-flip rate                     mode variance ratio；bias；coverage；mask flip           joint ensemble 95% CI 覆盖；线性域误差预算      若随 mask flip 系统恶化：建立 validity envelope，不硬解释

  CI04     Controlled real corruption              真实图像上理论能否预测 corruption 退化         OpenIllumination OLAT / calibrated GT   injected light direction/intensity covariance   predicted vs empirical mode degradation；rank corr       Spearman + affine-scale fit + bootstrap CI      若只有 MAE 单调而 mode 失败：真实定量主张不成立

  CI05     External sanity / robustness            主结论对材质、对象、错设是否稳健               DiLiGenT + selected DiLiGenT10²         object/material；Λ misspec；noise model         distribution of within-scene metrics；failure taxonomy   不设"必须赢 SOTA"；要求方向一致与边界透明       若模型失配主导：作为 limitation/failure case，不扩 claim
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 4.1 CI01 --- 代数正确性、白化、参数化不变性

-   输入：随机但可控的 A,B；q∈{1,3,9,36}；m/q 覆盖欠定、临界、过定；Σ_y 包含同方差与对角异方差；Σ_c 既有 full-rank 也有低秩物理参数映射。

-   双路线：Route A 直接边缘协方差 V=σ²I+BΣ_cBᵀ 后求 AᵀV\^{-1}A；Route B Schur/ΔF。两者逐元素相对误差作为第一主指标。

-   参数化不变性：φ-space prior → c=Jφ 映射，与 c-space induced low-rank prior 的 marginal information 必须一致。

-   尺度自检：(σ,Σ_c)→(kσ,k²Σ_c) 时 retention spectrum 不变；只改 σ 时 continuum 必须移动。

-   秩亏红线：Λ=0 只走 SVD/lstsq/pinv；测试 m_k\<q、重复列、退化法线；显式断言普通 solve 路径未被调用。

## 4.2 CI02 --- Gauge lifting、绝对信息与 λ⋆

-   构造至少 30 个 synthetic scenes：几何条件数、albedo spread、light count、B singular spectrum 分层采样；每个 scene 保存真实 gauge vector a 与 c̄。

-   λ/τ 网格：对每 scene 以 singular scales 自适应，覆盖 10\^{-6}·s_med² 到 10\^{6}·s_med²；另保留固定 log grid 便于跨场景统计。

-   主图不是 trace，而是：gauge absolute information、non-gauge floor μ_floor、闭式曲线、扫描 λ⋆、一阶预测 λ⋆\^lin。

-   同时生成 retention spectrum heatmap，但只做 within-scene 解释；多图条件使用连续性 mode tracking。

-   统计：报告 log10(λ⋆\^pred/λ⋆\^scan) 的 median/IQR/90% 区间，并按"预测点是否仍在小 λ 区"分层。

## 4.3 CI03 --- Monte-Carlo 紧性与线性化有效域

-   Linear oracle：每格至少 2,000 次 joint samples；高条件数场景增加到 10,000 或使用 batch sequential stopping，使弱模式方差比 CI 宽度 \<10%。

-   Nonlinear renderer：从同一 ground truth 生成 δc 与 ε，重新渲染后再估计；记录 mask/terminator 是否变化。

-   必须同时跑 marginal ensemble 与 fixed-δc diagnostic；后者只验证 closed-form bias 与 conditional covariance，不用于 CRB tightness claim。

-   coverage：对 tracked modes 计算 nominal 68%/95% interval 的 empirical coverage；若只看 variance ratio，容易漏掉 bias。

-   validity envelope：以 mask-flip rate、\|\|δc\|\|、SNR 为轴，找出理论误差从 \<10% 上升到不可接受的边界。

## 4.4 CI04 --- OpenIllumination 受控真实 corruption

-   优先 OLAT 子集：利用 illumination ground truth 与 segmentation masks；先选 6--10 个材质差异明显对象做 development，冻结后扩展到更大对象集。

-   corruption 类型至少三类：light intensity、direction/position 参数化扰动、联合扰动；每类由物理单位定义 Σ_c，不用 raw-coordinate λI。

-   对每个 corruption level 重复随机注入 ≥20 seeds；理论在 corruption 前由 GT/nominal state 计算，不允许用结果反调 Λ。

-   主检验：predicted weak-mode degradation vs empirical reconstruction-mode error；Spearman rank、bootstrap CI、允许一个全局 affine scale。

-   Σ_c_real 风险：以 Σ_c_eff=Σ_c_inject+Σ_c_real 的敏感性带报告；不要假装真实 calibration GT 零误差。

## 4.5 CI05 --- DiLiGenT sanity、外部稳健性与失败案例

-   DiLiGenT 的职责是确认真实非 Lambertian、shadow/specularity 条件下弱模式现象仍有解释力，不承担严格"理论精确吻合"。

-   可选扩展 DiLiGenT10²：利用 shape/material controlled variation 做跨对象分布统计，避免只挑 10 个经典对象讲故事。

-   与 Quéau et al. 类 inaccurate-lighting reconstruction 方法的关系：可以作为 reconstruction baseline，但论文问题不是"谁 MAE 最低"，而是"信息预测是否解释退化"。

-   Λ misspecification、Poisson-Gaussian noise、mask policy、number of lights 作为 robustness appendix；N5 式 4× misspec 当前只支持"单场景二阶效应"，不允许泛化为 robust。

# 5. 统计设计、预注册 Gate 与停止规则

## 5.1 统一统计原则

-   主分析单位是 scene × tracked mode × confidence level，而不是把所有像素/模式混成一个 trace。

-   所有随机实验至少 5 个 scene seeds；关键结论报告 median/IQR + bootstrap 95% CI。

-   Monte-Carlo 不是固定次数迷信：先跑 pilot 估计弱模式方差的不确定性，再按目标 CI 宽度扩样。

-   多重比较不对每个 λ 点逐点做显著性检验；优先拟合整条预测曲线、rank correlation、calibration slope。

-   所有阈值在看最终 test objects 前冻结；development/test 对象清单写入 dataset manifest。

## 5.2 Gate A--E（项目级）

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Gate**                        **通过标准**                                                                       **失败动作**
  ------------------------------- ---------------------------------------------------------------------------------- ------------------------------------------------------------------------------
  Gate A --- Algebra              CI01 全通过；双路线误差、谱界、参数化/尺度不变性均通过                             否则停止项目主实验；不允许"先跑真实数据再说"。

  Gate B --- Estimator            linear joint ensemble 与 σ²ΔF\^{-1} 在预注册误差预算内；fixed-δc 诊断吻合闭式      否则检查 sampling、M²、σ²、condition number；若理论仍失败，Prop.1 不入正文。

  Gate C --- Nonlinear validity   在低 mask-flip 区域理论预测保持可校准；明确 validity envelope                      若不存在有效域，则论文只能保留线性理论，不做真实预测主张。

  Gate D --- Real quantitative    OpenIllumination 上逐模式预测与实测退化有稳定 rank/scale agreement                 若失败，真实章降为 limitation；不能用"平均 MAE 单调"替代。

  Gate E --- Paper readiness      所有主 Figure/Table 可由 clean checkout 一键重建；CLAIMS_REGISTRY 每条有证据指针   否则不投稿。
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. 实验目录 / 代码架构（目标仓库结构）

原则：核心数学、数据适配、实验 orchestration、绘图、论文产物必须解耦。实验脚本不得复制 Fisher 公式；所有公式只存在于 src/calibinfo/information/。

> repo/
>
> ├─ pyproject.toml
>
> ├─ README.md
>
> ├─ CLAIMS_REGISTRY.yaml
>
> ├─ Makefile
>
> ├─ configs/
>
> │ ├─ ci01/\*.yaml
>
> │ ├─ ci02/\*.yaml
>
> │ ├─ ci03/\*.yaml
>
> │ ├─ ci04/\*.yaml
>
> │ └─ ci05/\*.yaml
>
> ├─ src/calibinfo/
>
> │ ├─ models/
>
> │ │ ├─ photometric.py
>
> │ │ ├─ linearize.py
>
> │ │ └─ noise.py
>
> │ ├─ information/
>
> │ │ ├─ whitening.py
>
> │ │ ├─ schur.py
>
> │ │ ├─ gauge.py
>
> │ │ ├─ retention.py
>
> │ │ └─ mode_tracking.py
>
> │ ├─ estimators/
>
> │ │ ├─ joint_map.py
>
> │ │ └─ gauss_newton.py
>
> │ ├─ datasets/
>
> │ │ ├─ synthetic.py
>
> │ │ ├─ diligent.py
>
> │ │ └─ openillumination.py
>
> │ ├─ metrics/
>
> │ │ ├─ mode_metrics.py
>
> │ │ ├─ calibration.py
>
> │ │ └─ mask_flip.py
>
> │ └─ io/manifest.py
>
> ├─ experiments/
>
> │ ├─ ci01_algebra.py
>
> │ ├─ ci02_gauge.py
>
> │ ├─ ci03_mc_validity.py
>
> │ ├─ ci04_real_corruption.py
>
> │ └─ ci05_sanity_robustness.py
>
> ├─ tests/
>
> │ ├─ unit/
>
> │ │ ├─ test_v1_covariance.py
>
> │ │ ├─ test_v2_gauge_closed_form.py
>
> │ │ ├─ test_v3_retention_bounds.py
>
> │ │ ├─ test_v4_parameterization.py
>
> │ │ ├─ test_v5_scale.py
>
> │ │ ├─ test_v6_mode_tracking.py
>
> │ │ └─ test_rank_deficient_lambda0.py
>
> │ └─ regression/
>
> ├─ scripts/
>
> │ ├─ run_ci.py
>
> │ ├─ make_figures.py
>
> │ ├─ make_tables.py
>
> │ └─ reproduce_paper.sh
>
> ├─ results/ \# gitignored raw runs
>
> ├─ artifacts/ \# frozen CSV/JSON summaries + manifests
>
> ├─ paper/
>
> │ ├─ figures/
>
> │ ├─ tables/
>
> │ ├─ supplement/
>
> │ └─ provenance/
>
> └─ legacy_redteam/ \# redteam2/3 scripts, read-only reference

## 6.1 核心 API 契约

  ---------------------------------------------------------------------------------------------------------------------------
  **API**                               **契约**
  ------------------------------------- -------------------------------------------------------------------------------------
  whiten_system(A,B,Sigma_y)            返回 Aw,Bw 与 whitening metadata；所有下游只接受白化后的系统。

  delta_f(A,B,Lambda, rank_policy)      唯一 Schur 实现；Lambda=0 自动 SVD/pinv；返回 ΔF、M、rank diagnostics。

  delta_f_marginal(A,B,Sigma_c,sigma)   独立 Route A；只用于交叉验证/测试，不复用 schur.py。

  gauge_response(B,cbar,lambda)         闭式 Prop.2；返回 exact、small-λ slope、saturation。

  retention_spectrum(DeltaF,Finf)       限制到 identifiable subspace；返回 ρ、basis、condition diagnostics。

  track_modes(prev_vecs,curr_vecs)      基于 \|V_prevᵀV_curr\| 的 assignment + degenerate-subspace handling。

  joint_map(\...)                       返回 xhat, chat, diagnostics；不得在 estimator 内偷偷重定义 Λ。

  run_manifest(\...)                    写 git SHA、config hash、dataset version、seed、hostname/runtime、code dirty flag。
  ---------------------------------------------------------------------------------------------------------------------------

## 6.2 CI / 测试红线

-   每个 PR 必跑 unit tests；main 分支额外跑 CPU synthetic regression。

-   任何新增 Fisher/Schur 代码必须有 known-answer + independent-route test；"图看起来合理"不算验证。

-   数值阈值按 dtype 与 condition number 分层；禁止所有测试统一 atol=1e-5。

-   模式追踪测试必须复现 V6：N=1 子空间 0° 旋转、N\>1 可大旋转；确保排序索引不会被误用。

-   论文 Figure/Table 只读取 artifacts/frozen/ 下的机器可读摘要，不直接读临时 notebook。

# 7. 具体 Figure / Table 设计

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **编号**       **标题**                                 **目的**                                                                     **版式**                                                                                      **落点**
  -------------- ---------------------------------------- ---------------------------------------------------------------------------- --------------------------------------------------------------------------------------------- ---------------------------
  Fig. 1         Problem & continuum overview             一张图讲清 calibrated↔uncalibrated、Σ_c/τ、nuisance coupling、weak modes。   概念图 + 2D information ellipse；不放实验数字。                                               Introduction

  Fig. 2         Theory anatomy                           Schur elimination、absolute information 与 retention 两套读出。              左：block information；中：ΔF；右：R∈\[0,1\]。                                                Method

  Fig. 3         Gauge lifting law                        Prop.2 的核心理论图。                                                        x=log λ；y=absolute gauge info；exact curve、small-λ tangent、saturation、μ_floor、λ⋆。       Theory

  Fig. 4         Mode-resolved continuum                  展示 trace 为什么骗人以及 mode tracking 为什么必要。                         上：trace 近乎不变；中：tracked ρ heatmap；下：N=1 vs N=3 subspace angle。                    Theory/CI02

  Fig. 5         Finite-sample calibration                Gate B 主图。                                                                predicted vs empirical variance scatter；mode-wise ratio violin；coverage；fixed-δc inset。   CI03

  Fig. 6         Linearization validity envelope          把"何时理论会坏"显式化。                                                     mask-flip rate × corruption strength 热图；颜色=prediction error。                            CI03

  Fig. 7         OpenIllumination controlled corruption   真实定量主图。                                                               每对象 predicted vs empirical degradation；rank correlation；代表性图像。                     CI04

  Fig. 8         Real sanity & failure taxonomy           DiLiGenT/10² 上展示材质/阴影/高光边界。                                      对象×材质小 multiples；成功/失败例并列。                                                      CI05

  Fig. 9         Ablation / robustness                    白化、raw λI、Λ misspec、heteroscedastic noise 的影响。                      box/violin + calibration slope。                                                              Appendix or main if space
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 7.1 每张主图的"不可替代信息"

-   Fig.3 必须让审稿人一眼看到：校准不是二元开关，而是对 gauge information 的可解析连续抬升；λ⋆ 是 scene-conditioned，而非万能阈值。

-   Fig.4 必须同时出现 trace 与弱模式，直观证明为什么传统 scalar summary 会错过伤害；并用 mode rotation 解释连续追踪协议。

-   Fig.5 必须证明"理论不是只画 Fisher"：它对可达 estimator 的 sampling covariance 有有限样本对应。

-   Fig.7 是论文能否从漂亮理论进入 TCI 级完整工作的关键：真实 corruption 必须和理论预测定量对齐。

  ---------------------------------------------------------------------------------------------------------------------------------------
  **编号**                **名称**                        **内容**
  ----------------------- ------------------------------- -------------------------------------------------------------------------------
  Table I                 Related-work positioning        Calibrated / inaccurate / uncalibrated PS；HCRB/mixed models；GUM；本文定位。

  Table II                Notation & claims               每个 Lemma/Prop/Diagnostic 的假设、结论、新颖性权重、对应实验。

  Table III               CI protocol                     CI01--CI05 输入、横轴、指标、Gate、失败动作。

  Table IV                Synthetic validation            closed-form errors、MC ratios、coverage、λ⋆ prediction stats。

  Table V                 OpenIllumination quantitative   对象/扰动类型的 Spearman、slope、CI、mask-flip。

  Table VI                Robustness & ablations          parameterization、noise、Λ misspec、light count、mask policy。

  Table VII               Claims registry                 允许表述 / 禁止表述 / evidence artifact / commit SHA（supplement）。
  ---------------------------------------------------------------------------------------------------------------------------------------

# 8. 12 周逐周执行计划

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **周**         **主题**                       **任务**                                                                                                  **交付物**                                             **周末阻断条件**
  -------------- ------------------------------ --------------------------------------------------------------------------------------------------------- ------------------------------------------------------ -----------------------------------------------
  W1             冻结规格与仓库重构             建立 CLAIMS_REGISTRY.yaml；迁移 redteam V1--V6 为 tests；实现 run manifest；锁定符号与 config schema。    Gate A smoke test；目录重构 PR。                       任何核心 identity 未变成 test。

  W2             CI01 完成                      weighted/whitened ΔF；marginal-vs-Schur 双路线；rank-deficient Λ=0；parameterization/scale invariance。   CI01 report + Table IV 前半。                          误差 \>1e-10 或存在 solve 静默路径。

  W3             理论证明定稿                   完成 Lemma1、Prop1、Lemma2 手稿；补 HCRB/mixed-model/GUM 引用；写 supplement proof skeleton。             Theory note v1 + proof tests。                         证明依赖未声明可逆性/秩假设。

  W4             CI02 gauge                     30+ scenes；Prop2 exact curve；μ_floor；λ⋆ exact root 与 linear predictor；scene stratification。         Fig.3 draft + λ⋆ stats。                               λ⋆ 误差无可解释结构。

  W5             Retention + mode tracking      实现 generalized eig/whitening 双路线；assignment/subspace tracking；复现 V6；生成 Fig.4。                Fig.4 frozen candidate。                               仍按 eigenvalue index 追踪。

  W6             CI03 linear MC                 joint ensemble sequential MC；mode variance ratio、coverage；fixed-δc diagnostic。                        Fig.5 linear panel；Gate B。                           弱模式 MC 误差未量化。

  W7             CI03 nonlinear validity        接入真实 renderer/Jacobian；corruption 扫描；mask-flip；validity envelope。                               Fig.6 + Gate C。                                       不记录 mask flip 或把非线性失配误判理论错误。

  W8             OpenIllumination development   数据适配；选 development objects；定义物理 Σ_c；小规模 corruption pilot。                                 CI04 protocol freeze。                                 根据 pilot 结果反调 test 阈值。

  W9             OpenIllumination final         冻结 test objects；≥20 corruption seeds；predicted-vs-empirical mode analysis；bootstrap。                Fig.7 + Table V + Gate D。                             只有平均误差单调，无 mode agreement。

  W10            DiLiGenT / robustness          real sanity；DiLiGenT10² 可选；Λ misspec、heteroscedastic noise、light count、mask policy。               Fig.8/9 + Table VI。                                   把模型失配包装成 calibration claim。

  W11            论文组装                       冻结所有 Figure/Table；Introduction/Related Work/Method/Theory/Experiments 初稿；supplement proofs。      完整 manuscript v0.8。                                 图表仍由 notebook 手工导出。

  W12            复现与内部审稿                 clean checkout 一键复现；第三方机器 smoke；逐条 CLAIMS_REGISTRY 审计；TCI 格式与叙事压缩。                submission candidate v1.0 + reproducibility bundle。   任一主 claim 无 artifact/commit 指针。
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 8.1 每周固定节奏

-   周一：冻结本周 config 与 acceptance criteria；不在周中偷偷改 Gate。

-   周二--周三：实现与小规模 pilot；每个新数学函数先 test 后跑大实验。

-   周四：全量 run + artifact 汇总；禁止手工复制数字到论文。

-   周五：Figure/Table + 1 页实验 memo（结果、异常、claim 是否变化）。

-   周末：只有 Gate 通过才进入下一周；失败则执行预注册降级路径，不回到顶层重写故事。

# 9. IEEE TCI 论文逐节写作提纲

TCI 的官方 scope 明确覆盖 computational imaging 的理论、model-based inversion 与 advanced mathematical techniques，并鼓励公开生成论文图表所需的代码与数据。因此文章应写成"计算成像中的不确定校准如何改变可恢复信息"，而不是纯统计推导或纯 photometric-stereo benchmark。

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **章节**                                               **写什么**                                                                                                                                                                                             **写作约束**
  ------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------
  Title                                                  首选：From Calibrated to Uncalibrated Lighting: A Mode-Resolved Information Continuum for Photometric Inverse Problems                                                                                 避免标题承诺"new CRB"；突出 continuum + mode-resolved + lighting calibration。

  Abstract                                               四句结构：问题缺口 → continuum/解析 gauge result → finite-sample + controlled-real validation → 结论/用途。                                                                                            必须出现"mode-resolved"；不写"novel Fisher information"。

  I. Introduction                                        1\) calibration 从不是完美二元变量；2) scalar reconstruction error 看不到弱模式；3) 本文问题；4) 三条贡献；5) 结果预览。                                                                               Fig.1 放这里；贡献只保留 3 条。

  II\. Related Work                                      A. calibrated/inaccurate/uncalibrated photometric stereo；B. hybrid estimation + mixed linear models；C. measurement uncertainty propagation/GUM；D. inverse-rendering information/identifiability。   明确 Quéau 等做 robust reconstruction；本文做 uncertainty→information prediction。

  III\. Problem Formulation                              成像模型、局部线性化、白化、Σ_c 物理参数化、τ continuum、mask 固定假设。                                                                                                                               在这里一次性定义 W/σ²/Σ_c/Λ，避免后文尺度混乱。

  IV\. Calibration-Confidence Information Continuum      Lemma1；Schur form；端点；单调性；retention spectrum 定义与界。                                                                                                                                        把成熟理论标为 recall/instantiate；Fig.2。

  V. Gauge Lifting and Scene-Conditioned Observability   Prop2；small-λ slope；saturation；μ_floor；λ⋆ exact/linear predictor；N=1 mode-basis remark。                                                                                                          Fig.3；λ⋆ 只在 absolute information。

  VI\. Estimator Calibration and Experimental Protocol   Prop1/GLS identity；marginal vs fixed-δc；mode tracking；Gate B；mask-flip validity。                                                                                                                  把理论"可达性"与实验 protocol 绑在一起。

  VII\. Synthetic Experiments                            CI01/CI02/CI03：algebra、gauge、retention、MC、nonlinear envelope。                                                                                                                                    Fig.4--6，Table IV。

  VIII\. Controlled Real-Data Validation                 OpenIllumination corruption；Σ_c_real sensitivity；predicted vs empirical mode degradation。                                                                                                           Fig.7，Table V；这是主真实证据。

  IX\. External Sanity, Robustness, and Limitations      DiLiGenT/10²；non-Lambertian failure；Λ misspec；heteroscedastic noise；near-field/model mismatch。                                                                                                    Fig.8/9，Table VI；主动划边界。

  X. Discussion                                          何时 calibration 值得继续提高；为什么 trace 会误导；如何把框架迁移到其他 inverse problems。                                                                                                            只讨论由证据支持的外推。

  XI\. Conclusion                                        重述 continuum、mode-specific prediction、gauge lifting、real validation。                                                                                                                             不新增 claim。

  Supplement                                             完整证明；更多 scene seeds；mode-tracking algorithm；dataset manifests；extra robustness；CLAIMS_REGISTRY。                                                                                            主文保持紧凑，审计材料放 supplement。
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 9.1 最终三条 contribution 建议

1\. A calibration-confidence continuum for photometric inverse problems that maps physically parameterized lighting uncertainty to mode-resolved effective information, with a normalized retention spectrum for within-scene interpretation.

2\. An exact gauge-lifting spectral response and a scene-conditioned observability diagnostic that predict how finite calibration confidence lifts otherwise ambiguous/weak directions.

3\. A reproducible validation protocol connecting the information prediction to finite-sample estimator covariance and to controlled real-data corruption, including explicit validity boundaries from mask changes and model mismatch.

注意：第一条不能写成"we introduce a new information measure"；第二条不能写成"universal observability theorem"；第三条不能写成"CRB predicts all real reconstruction errors"。

# 10. CLAIMS_REGISTRY 与证据链

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Claim**      **允许表述**                                                                                        **证据**           **权重**            **禁止表述**
  -------------- --------------------------------------------------------------------------------------------------- ------------------ ------------------- -------------------------------
  C1             ΔF 是 hybrid/marginal information 的 Schur form；随 calibration confidence 单调。                   Lemma1 + CI01      成熟理论实例化      "new information geometry"

  C2             joint ensemble 下 affine-Gaussian covariance = σ²ΔF⁻¹。                                             Prop1 + CI03       精确性/一致性资产   "new estimator"

  C3             gauge absolute information obeys exact spectral response。                                          Prop2 + CI02       核心解析资产        "universal new gauge theorem"

  C4             λ⋆ 可由 scene quantities 预测，在小 λ 域用 μ_floor/Σα² 近似。                                       CI02 multi-scene   诊断资产            "universal threshold"

  C5             ρ∈\[0,1\]，可读作 within-scene information retention。                                              Lemma2 + CI01      解释工具            跨场景逐 mode 直接比较

  C6             mode-resolved prediction explains controlled real corruption better than scalar trace summaries。   CI04               核心实证资产        只凭 MAE 单调即宣称成功
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. 可复现性、数据治理与审计

-   每个 run 生成 manifest.json：git SHA、dirty flag、Python/NumPy/PyTorch 版本、config hash、seed、dataset checksum、机器信息、开始/结束时间。

-   results/raw/ 不进 Git；artifacts/frozen/ 只保存足以重画论文图表的 CSV/JSON 与 manifest。

-   每张 Figure/Table 有唯一 recipe，例如 \`python scripts/make_figures.py \--figure 3 \--artifact \...\`。

-   数据 split 先写 manifest 再跑实验；OpenIllumination development/test 对象严格分离。

-   所有随机性显式 seed；GPU 非确定算子若不可避免，记录 deterministic flag 与重复方差。

-   最终发布仓库保留 legacy_redteam/ 作为审计历史，但 README 明确其不是主实验入口。

# 12. 风险登记册与降级路径

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **风险**          **内容**                                             **等级**          **处置**
  ----------------- ---------------------------------------------------- ----------------- -----------------------------------------------------------------------------------------
  R-A               OpenIllumination corruption 与真实未知校准误差叠加   高                Σ_c_real sensitivity band；只 claim trend/rank if scale uncertain。

  R-B               mask/terminator flip 破坏线性化                      高                CI03 validity envelope；主结论限制在低 flip 区。

  R-C               多图 mode rotation 导致错误匹配                      高                continuity/subspace tracking；V6 regression test。

  R-D               真实材质/BRDF mismatch 主导                          中高              DiLiGenT 仅 sanity；失败例显式归因 model mismatch。

  R-E               Λ/Σ_c 参数化不物理                                   高                whitened physical coordinates；V4/V5 tests。

  R-F               高条件数导致 MC 假失配                               中                sequential MC + condition-aware CI；报告 uncertainty budget。

  R-G               审稿人认为理论均为成熟统计                           高                主动引用 HCRB/mixed models/GUM；新颖性集中 render-specific gauge/mode/real validation。

  R-H               仓库遗留脚本复制公式导致漂移                         高                core API 单源；legacy scripts read-only；CI known-answer。
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. 投稿前最终 Checklist

-   □ Lemma1/Prop1/Prop2/Lemma2 的假设、可逆性、秩条件在正文首次出现时完整写出。

-   □ λ⋆ 全文只出现在 absolute information 语境；retention 语境没有"交叉阈值"措辞。

-   □ 所有 continuum 主图都是 mode-resolved；trace/logdet 只作为反例/辅助。

-   □ fixed-δc 与 marginal ensemble 的公式、图例、代码路径完全分开。

-   □ OpenIllumination test objects 在最终分析前冻结；无事后删对象。

-   □ DiLiGenT 不被用来证明 calibration-only 因果。

-   □ Figure 3/5/7 分别回答"理论曲线是什么 / estimator 是否达到 / 真实数据是否跟随"。

-   □ 每个主 claim 在 CLAIMS_REGISTRY 中有 artifact path + commit SHA。

-   □ clean clone 可在 README 指令下生成所有主表图；TCI reproducibility 要求被实际满足。

-   □ Related Work 主动覆盖 HCRB、linear mixed models、GUM、inaccurate/semi/uncalibrated photometric stereo。

-   □ 摘要和贡献没有"new CRB / new information geometry / universal threshold"等过度措辞。

-   □ Supplement 包含完整 proofs、额外 seeds、mode tracking、robustness、failure cases。

# 14. 文献与数据锚点（执行版，不是最终 bibliography）

-   Noam, Y. & Messer, H. (2008), The hybrid Cramér--Rao bound and the generalized Gaussian linear estimation problem. 用于 HCRB 谱系与 asymptotic efficiency 定位。

-   Harville, D. A. (1977), Maximum Likelihood Approaches to Variance Component Estimation and to Related Problems. 用于 mixed linear models / fixed-random effects 谱系。

-   JCGM 100:2008 / JCGM 101:2008 (GUM and Monte Carlo propagation). 用于 measurement-uncertainty propagation 谱系。

-   Shi et al. (CVPR 2016), DiLiGenT. calibrated directional lighting + ground-truth normals；本文 real sanity。

-   Liu et al. (NeurIPS 2023), OpenIllumination. camera parameters + illumination ground truth + masks；本文 controlled corruption 主数据。

-   Quéau et al. (CVPR 2017), Photometric Stereo Under Inaccurate Lighting. 用于 reconstruction-under-inaccurate-lighting 邻接与切割。

-   IEEE Transactions on Computational Imaging official scope/reproducibility guidance. 用于投稿定位与发布工程要求。

# 15. 当前唯一未闭环项：与现有 GitHub 仓库做 1:1 映射

一旦仓库可读，下一步不是改本设计书的理论/实验逻辑，而是做"迁移矩阵"：现有路径 → 目标模块 → 保留/重写/删除 → 对应 test → 对应 Figure/Table。建议输出 REPO_MIGRATION.md，并把所有现有 exp\*.py、redteam\*.py、notebook、dataset loader、renderer/Jacobian 实现逐项归档。

  ---------------------------------------------------------------------------------------------------------------------------------
  **现有资产类别**         **目标位置**                    **动作**                                      **必须绑定**
  ------------------------ ------------------------------- --------------------------------------------- --------------------------
  现有 Fisher/Schur 代码   src/calibinfo/information/      保留正确实现，删除重复公式                    V1b/V4/V5/rank-def tests

  redteam3_exp.py          legacy_redteam/ + tests/unit/   脚本冻结；断言拆成正式 tests                  V1--V6

  旧 exp14 / T0--T9        experiments/ci0x + configs/     按 claim 重新命名，不保留历史编号作为主入口   CI01--CI05

  renderer/Jacobian        src/calibinfo/models/           保留，补 whitening/mask diagnostics           CI03

  数据加载器               src/calibinfo/datasets/         统一 manifest 与 split                        CI04/CI05

  绘图 notebook            scripts/make_figures.py         只保留探索 notebook；论文图脚本化             Fig.1--9
  ---------------------------------------------------------------------------------------------------------------------------------

# 结论

这份计划把项目从"继续寻找理论亮点"切换为"证据链工程"：理论已冻结，真正决定论文质量的是 CI01--CI05 是否按预注册 Gate 完成、Fig.3/5/7 是否形成解析---统计---真实数据三段闭环，以及仓库是否做到一键复现。只要 Gate D 能通过，路线具备完整 TCI 论文形态；若 Gate D 失败，也有明确降级路径，不需要再回到已经关闭的顶层理论争论。
