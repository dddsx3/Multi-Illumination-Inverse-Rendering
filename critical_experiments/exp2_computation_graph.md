# 实验 2 · SH-9 联合 Fisher + 稀疏 Schur 补 · 计算图与关键实现逻辑

> 状态:设计冻结(实现前预注册) · 2026-09-05 · 主智能体关键实验指令之实验 2
> 纪律:本文件先于代码结果写定所有公式与判据,代码只实现不更改。

## 1. 前向模型(必须与仓库 physics_renderer.py 逐符号一致,不自造)

```
参数: z ∈ R^P (深度图, P=H×W)  ρ ∈ R^P (灰度 albedo)  C ∈ R^{N×9} (每光 SH-2 系数)

z --Sobel(Sx,Sy, same-pad)--> (gx, gy)                    [physics_renderer.py:63-64]
n  = normalize([-gx, -gy, 1])                              [physics_renderer.py:76-81]
Y_p = SH2(n_p) ∈ R^9   (C0=0.282095, C1=0.488603, C2=[1.092548,1.092548,0.315392,1.092548,0.546274])
s_kp = ReLU(Y_p · C_k)                                    [physics_renderer.py:199-202]
h_kp = 1[Y_p·C_k > 0]                                     (ReLU 指示)
I_kp = ρ_p · s_kp                                          [physics_renderer.py:345]
```

## 2. 解析 Jacobian(列序 [z(P) | ρ(P) | C(N×9)])

### ∂I_kp/∂ρ_q = s_kp·δ_pq  → D_ρ = diag(s_k) (P×P, 每光一个对角阵)

### ∂I_kp/∂C_k[j] = ρ_p·h_kp·Y_p[j]  → J_C,k = diag(ρ·h_k)·Y (P×9)
(与 gauge_fisher_v2.fisher_blocks 的 B_k 行约定一致:s = ReLU, h = 指示, Y = SH2(法线))

### ∂I_kp/∂z_q —— 链式: I→s→n→(gx,gy)→z。每行 9 个非零(Sobel 3×3)。
```
∂n_p/∂z_q = (1/|v_p|)·[ P_⊥(n_p) · ∂v/∂z_q ]   v_p=(-gx,-gy,1), P_⊥ = I − n nᵀ
∂gx_p/∂z_q = Sx[p,q]  (Sobel 核,稀疏 9 点)
∂s_kp/∂z_q = h_kp · C_kᵀ · ∂Y_p/∂n_p · ∂n_p/∂z_q
            其中 ∂Y_p/∂n_p (9×3) 是 SH2 对法线的解析导数:
            Y=[C0, C1·ny, C1·nz, C1·nx, C2₀·nx·ny, C2₁·ny·nz, C2₂·(3nz²−1), C2₃·nx·nz, C2₄·(nx²−ny²)]
            ∂Y/∂nx=[0,0,0,C1, C2₀·ny, 0, 0, C2₃·nz, C2₄·2nx]
            ∂Y/∂ny=[0,C1,0,0, C2₀·nx, C2₁·nz, 0, 0, −C2₄·2ny]
            ∂Y/∂nz=[0,0,C1,0, 0, C2₁·ny, C2₂·6nz, C2₃·nx, 0]
∂I_kp/∂z_q = ρ_p · ∂s_kp/∂z_q = ρ_p·h_kp·(C_kᵀ ∂Y/∂n)(3×1)·(1/|v_p|)·P_⊥(n_p)(3×3)·(−Sx/Sy 行)
```
即每像素行 ∂I_kp/∂z 只在 Sobel 3×3 窗口的 9 个深度像素上非零 → J_z 稀疏(NP × P, 每行 9 非零)。

## 3. Fisher 组装与分块

```
F = J^T J,  J = [J_z | J_ρ | J_C]   (NP × (2P + 9N))
F_zz = Σ_k J_z,kᵀ J_z,k    (P×P, 稀疏: Sobel 双线性 → 每行 ≤ 41 非零)
F_ρρ = Σ_k diag(s_k²)        (P×P, 对角)
F_ρz = Σ_k diag(s_k)·J_z,k   (P×P, 稀疏, 与 F_zz 同稀疏模式)
A := F_zz + F_ρρ + 2·F_ρz  —— 不做: 分块更稳,见下
实际分块(设计文档 §3 口径):
  A = [[F_zz, F_zρ],[F_ρz, F_ρρ]]  ((2P)×(2P) 稀疏对称)
  B = [[F_zC],[F_ρC]]              ((2P)×(9N))
  S = F_CC − Bᵀ A⁻¹ B              (9N×9N 稠密小矩阵)
```

**深度平移奇异性(设计文档 §5)**:正交投影下 z→z+c 不变 (gx,gy) → n 不变 → I 不变 → J_z·**1** = 0 →
F_zz·**1** = 0 精确成立。处理:钉住一个像素的深度(删 A 的对应行列)或伪逆。**选伪逆路径**(不改变
问题规模,且与谱分析一致);splu 对奇异 A 会失败,故对 A 加微小 Tikhonov 后求解并验证
‖A·**1**‖ 恒等式仍成立(数值核验)。

## 4. 求解路径(设计文档 §3)

1. A 用 scipy.sparse.linalg.splu(factorization);9 非零/行 × 2P 规模 → 64×64(P=4096, A 是 8192²)先跑通;
2. 256×256(P=65536, A 是 131072²)用 splu 的 solve 进行 Bᵀ A⁻¹ B 的双三角求解;
3. S 做稠密 eigh(9N×9N, N=5 → 45×45, 小)。

## 5. 判据(预注册)

- **1 个精确零**:全局尺度 gauge (δρ, δC)=(ρ, −C·λ⁻¹…):解析方向
  δθ = (0_z, ρ, −C)?? ——注意本模型 I=ρ·s(C), s 齐次于 C,故 (ρ,C)→(λρ, C/λ) 不变 →
  精确零方向 = (0, ρ, −C 逐光线性缩放):验证 S 在该方向 Rayleigh 商 ≤ 机器精度;
- **3 个近零(GBR)**:仅当 N≥2 且深度可变时出现(实验 3 用解析 GBR 生成元验证);
- **N×k 个近零(SH 病态)**:G_Y = Σ ρ² Y(n)Y(n)ᵀ 的秩亏损(法线受限场景);
- 运行时间逐级报告(64×64 与 256×256 的组装+分解+Schur 秒数);
- 谱输出:S 的前 10 特征值(log 尺度),相对 λ_max。

## 6. 数据接口(多模态 agent)

- `exp2_joint_fisher_spectrum.json`:每 (scene, res) 的 {runtime_s, eig_10, rayleigh_scale_gauge, …}
- 待绘图:谱 log 图(X=特征值序号,Y=log10(λ/λ_max)),标注 1 精确零/3 GBR/SH 病态预测带。

*设计冻结 · 实现前落盘 · 代码必须逐条对上本文件,任何偏差在代码注释中标注。*
