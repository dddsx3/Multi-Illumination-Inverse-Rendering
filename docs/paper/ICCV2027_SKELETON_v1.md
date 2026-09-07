# ICCV 2027 论文骨架 v1 · 卡I(2026-09-07)

> 依据:任务书 v3.2 卡 I(四支柱章节映射 + 证据指针表)。
> 状态:骨架 v1(S 腿已闭环; 支柱③重定义裁决权归主智能体; 卡H GPU 臂待排)。

## 0. 一句话定位(本节唯一主张)

**在多光照逆渲染中, N 曲线平坦不是数据缺信息, 而是网络未利用信息; 我们用 Fisher/CRB
工具链证明信息在数据里、网络是待诊断对象, 并如实报告判别诊断在真实数据上的失败模式与
可用的规范处方。**

- 观察现象:N 曲线平坦(14.875°→14.887°, EX-01 冻结)
- 理论带宽:E_min 51.9–78.3% 降幅可兑现(exp7)
- 与现象矛盾的解析:信息在数据、不在网络用法(exp6) → "诊断工具"叙事成立

## 1. 四支柱章节映射

| 论文节 | 支柱 | 本节唯一主张 | 证据指针 | 状态 |
|---|---|---|---|---|
| Abstract | 综述 | 信息存在(Fisher/CRB 12/12 有效性)+ 网络未用(0.010°)+ 诊断判别失败模式(0/10)+ 处方(控制子空间 0.765) | 四项各 1 句, 措辞包 §3 | ✅ |
| §1 Intro | 现象 | N 曲线平坦是"网络侧现象", 需诊断工具而非更大模型 | EX-01; exp6/7 同图 | ✅ |
| §2 Related Work | 边界 | 8 篇最近工作 + novelty 风险(可变基数证据累积) | `p1/literature/RELATED_WORK_MATRIX_v3.md` | ✅ |
| §3 Method | 支柱① 工具链 | SH-9 联合 Fisher + 稀疏 Schur; GBR 三类方向解析分离; CRB 有效性(12/12) | exp2/3/4 计算图+json; 措辞包 §2 | ✅ |
| §4 Method(诊断定义) | 支柱③(现:负结果) | 迹推前诊断定义 + 预注册评估协议; 判定不成立及 oracle 审计 | exp8r_preregistration_v3; exp8r_verdict_v3; exp8S | ✅(裁决待) |
| §5 Diagnostics(对象) | 支柱② 网络侧 | 复制/打乱/敏感度三前向: 网络对光照多样性近零响应 | exp6; EX-01/EX-04/EX-05 验收 | ✅ |
| §6 Experiments(真实数据) | 支柱③真实侧 | DiLiGenT 10 物体: 0/10 显著正/6/10 显著负; 异质性先于合并 | exp8r_verdict_v3; 73 备用 json | ✅ 负结果 |
| §7 Prescription | 支柱④ 处方 | 控制子空间/谱先验(Gauge/GBR-informed): 124/124 领先; 谱分层与先验连续谱 | exp13b; exp14; exp9c; exp10; exp13 | ✅ |
| §8 Limitations | 全集 | SH-2 截断 GBR 非精确; 深度通道全病态; 支柱③未成立; 卡H 单模型证据 | 措辞包 §3; FLAG 文件 | ✅ |
| §9 Conclusion | 收束 | 诊断必要(②)+ 工具可靠(①)+ 判别需重定义(③)+ 处方可行(④) | 同上 | 待支柱③裁决 |

## 2. 证据指针表(每格 = 卡/实验 → 文件)

| 卡/实验 | 命题 | 证据文件(相对路径) | 判定 |
|---|---|---|---|
| exp1(P-A2) | 2.59/0.37 口径错位 | `exp1_pa2_*` | 解除 |
| exp2 | SH-9 Fisher+稀疏 Schur | `exp2_*` | ✅ |
| exp3 | GBR 三类方向分离 | `exp3_direction_separation.*` | ✅ |
| exp4 | CRB 有效性(金标准) | `exp4_crb_validity.*` | ✅ 12/12 |
| exp5/5b | 散布度负结果族 + 低秩机制 | `exp5_*` | 负结果(机制级) |
| exp6 | 网络三前向测试 | `exp6_*` | ✅ |
| exp7 | CRB-vs-N 同图 | `exp7_*` | ✅ |
| exp8R v3(卡S) | DiLiGenT 判别力(预注册) | `exp8r_verdict_v3.*` | **不成立(0/10 正 6/10 负)** |
| exp8S(卡S') | oracle 审计(三变体) | `exp8s_oracle_schur_audit.*` | 假说全排除 |
| exp9c(卡N) | 谱分层闭环 | json+图 | ✅ |
| exp10(卡O链条) | slope 收口 | `exp10_slope_closure.*` | ✅ |
| exp11c(卡O) | 反向切法 κ_w | `exp11c_reverse_cut.*` | ✅ |
| exp12v2(卡L) | 联合 GN 判别 | `exp12v2_*` | 真负结果(病态) |
| exp12v3(卡M预备) | 匹配判别 | `exp12v3_*` | ✅ |
| exp13b(卡M) | 控制子空间双对照 | `exp13b_control_subspaces.*` | ✅ 124/124 |
| exp13 | gauge 误差占比 | `exp13_gauge_error_fraction.*` | ✅ |
| exp14 | 先验连续谱 | `exp14_prior_continuum.*` | ✅ |
| 卡T/U/V/R | (见 5f6e4de→1037da7 提交) | git log | 已闭合 |
| 卡H seed2024 | 网络侧 seed 复现臂(GPU ~8h) | 命令见 CLOSURE_20260906 §未完成 | **待排** |
| 卡W | 措辞包 v0.4 + 方法论小节 | `docs/论文措辞包_v0.4.md` | ✅ |
| 卡I | 本骨架 v1 | 本文件 | ✅v1 |

## 3. gate-SKELETON-FULL 检查单

- [x] 卡L(exp12v2 真负结果) / 卡M(exp13b 124/124) / 卡N(exp9c 谱闭环) / 卡O(exp11c)
- [x] 卡S(exp8R v3 判定 0/10→不成立)+ exp8S 审计
- [x] 卡W(措辞包 v0.4 + 方法论清单小节, 本包 §1–§3 落地)
- [x] 卡I(骨架 v1:四支柱映射 + 证据指针表)
- [ ] **卡H seed2024**(唯一 GPU 臂, ~8h 本机夜跑; 云实例已暂停, 待本机或新实例排期)
- [ ] **支柱③重定义裁决**(主智能体; 裁决卡见 `exp8r_verdict_v3.md` §4; 措辞条目 5 跟随更新)

> 未裁决/未跑两格是 gate 开闸的前置; 其余 S 腿证据已全部在案且可复现。