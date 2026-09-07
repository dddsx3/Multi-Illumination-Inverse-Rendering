# CI01 Gate A · 1 页 memo（2026-09-07 · 卡 C07）

**判定：Gate A 通过。** CI01 正式 run 100/100 checks 全绿（config=`configs/ci01/ci01_formal.yaml`，
5 seeds × 20 case 变体，摘要=`artifacts/frozen/ci01_formal_summary.json`，manifest 同放）。

## 核心数字（预注册 gate = 1e-10）

| 指标 | 观测最差 | gate |
|---|---|---|
| 双路线（Schur vs marginal×σ²）逐元素相对误差 | **2.21e-12** | <1e-10 ✅ |
| 参数化不变性（φ-profiled vs c-marginal，V4 恒等式） | **2.23e-12** | <1e-10 ✅ |
| 尺度不变 (σ,Σc)→(kσ,k²Σc) retention 谱 | **1.72e-15** | <1e-10 ✅ |
| Λ=0 秩亏 M(0) 幂等性 | **8.63e-16** | <1e-10 ✅ |
| retention 谱界 0≤ρ≤1 | 100/100 ok | ✅ |
| σ 单独改变 → 移动（random 块非空洞性） | min 5.40e-02 | >1e-6 ✅ |
| σ 单独改变 → 移动（photometric V5 同构） | min **0.799** | >0.1 ✅ |

覆盖：q∈{1,3,9,36}；m/q 三区（欠定/临界/过定）；Σ_y 同方差+对角异方差；Σ_c full-rank+低秩 Jφ；
solve 禁令静态盯防（`test_information_modules.py::test_solve_never_called`）。

## 首轮正式 run 暴露的两个 harness 缺陷（如实记录，core 无恙）

1. **未归一 J 注入人工病态**：随机 J (q,r) 使 cond(JᵀJ)≈cond(J)² 进入 B_phi=B@J，
   identity 误差被顶到 4.46e-09（κ·eps 量级）——非代数错误。修法：J 取正交列（QR），
   物理上等价（任意 r 维子空间映射都是合法参数映射）。修后最差 2.2e-12。
2. **λ 锚定在退化带**：m<q 时 BᵀB 有近零奇异值，median 落在零带 → σ-shift 检查空洞
   （8.5e-16）。修法：工作带纪律（红队 handoff §2.2 谱截断）——只对 >1e-3·λ_max 的
   奇异值取 median。修后 random 块 min shift 5.4e-02。

两处修复均属 harness 构造缺陷（判据未动、gate 未动）；修后整卡重跑（Gate A 失败动作执行正确）。

## 结论与下一步

- C1（ΔF=Schur 单调）与 C5（ρ∈[0,1] within-scene）的代数/白化/参数化/尺度资产全部
  machine-precision 落地；V1–V6 红队判定作为单元测试持续回归（48 tests 全绿）。
- **Gate A 通过 → 进 P2（C09 scene 工厂 → C10 CI02 gauge/λ⋆ → C11 retention+tracking）**，
  不碰真实数据（宪法 Gate A 失败动作的反面即本次路径）。
