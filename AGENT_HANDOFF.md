# AGENT_HANDOFF · 交接文档（calibration-continuum 版 · 2026-09-07 重写）

> **重写**：2026-09-07 · 卡 C03（主控计划书 §4）。旧版（08-31，GA-ISI 主线）的 §7 待办
> （R4' 修正、P1-13 全量 200×32、Probe 训练）**全部作废**；§5 的 21 条踩坑清单仍然有效
> （原样保留在文末，渲染/SH/评估类步骤适用）。

---

## 1. 30 秒现状

**主线（唯一）**：校准置信度连续谱 → 逐模式信息预测 → 受控真实数据验证
（**CI01–CI05 → IEEE TCI 投稿**）。理论层已冻结（三轮红队 R1–R4 修订入版，顶层讨论永久关闭）。

- **宪法**（不再讨论顶层）：`docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md`
- **执行条例**（逐卡 C01–C23）：`docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md`
- **状态唯一记录**：仓库根 `STATE.md`（进度只认 STATE.md，不认对话记忆）
- **主张登记**：仓库根 `CLAIMS_REGISTRY.yaml`（C1–C6 + 红队 N/V 增补 + 实现红线）
- **迁移矩阵**：`docs/REPO_MIGRATION.md`（迁移期间只按该表行动）

## 2. 交接指令（固定开场）

```
读 STATE.md + REPO_MIGRATION.md 锚点列 + 主控计划书 §4 当前卡，继续执行。
```

开工先读 STATE.md 与未闭合卡；收工必须更新 STATE.md 并 commit+push。

## 3. 环境

| 项 | 值 |
|---|---|
| 本仓库 | `D:\MIR_Archive_20260829\Multi-Illumination-Inverse-Rendering`（GitHub 权威存档，push=双份归档之一） |
| 包安装 | `pip install -e .`（calibinfo，Python ≥3.10；本机 3.14 + numpy 2.4.1 + scipy 1.17.0 实测绿） |
| 测试 | `python -m pytest`（testpaths=tests/unit+regression；红队 V 系列 23 项 + 迁移绑定测试） |
| git push 代理 | `https_proxy=http://127.0.0.1:63770 git push origin main`（**会轮换**：失败先查注册表 `HKCU\...\Internet Settings\ProxyServer`，再试旧端口 57877/58240） |
| 数据 | DiLiGenT `D:\data\DiLiGenT\pmsData`（936MB，不入 git，loader=src/calibinfo/datasets/diligent.py）；OpenIllumination 未下载（C14，HF `OpenIllumination/OpenIllumination`，先列文件实测体积） |
| 本机 | 32 核 / 16GB（大 P>2000 需子采样+内存预算）/ RTX 5070Ti 12GB（仅 CI03 增强臂用） |
| 机器分工 | 本机 = 全部开发+CI01–CI03（CPU 秒级）；云 Linux/A10 已从主线移除（云跑经验保留在 legacy 脚本） |

## 4. 纪律红线（每卡检查，违例=返工）

1. **宪法 §0 五禁令**：trace/logdet 不作 continuum 主证据；retention 谱不定 λ⋆；
   禁"第 j 小特征值"索引追踪模式（必须 track_modes 连续性）；Λ=0 秩亏路径禁 solve；
   fixed-δc MC 不当 marginal CRB 验证（其条件方差用 σ²ΔF⁻¹AᵀM²AΔF⁻¹）。
2. **红线 #8（已知答案单测）**：任何 Fisher/Schur/边缘化/推前实现，入库前过随机小规模
   稠密对照（rel<1e-10）+ 双路线交叉验证。模板：`tests/unit/test_diagnostics_dense_contrast.py`。
3. **红线 #9（声称=实现）**：docstring 声称被边缘化的每个参数必须有对应代码项。
4. **红线 #11（GT 显式声明）**：GT 只进评分/oracle 臂；`assert_data_driven_init`（estimators.gauss_newton）断言拦截。
5. **卡门禁三问**（主控计划书 §6.1）：服务哪条 claim？判据跑前写死了吗？失败降级路径是什么？
6. **停车场纪律**：新想法只记 `PARKING_LOT.md` 一行，当天不展开。
7. **公式单源化**：Fisher/Schur/ΔF 只存在于 `src/calibinfo/information/`；
   `legacy_redteam/` 与 `critical_experiments/` 只读，禁止 import。

## 5. 卡索引（详见主控计划书 §4）

| 阶段 | 卡 | 状态见 STATE.md |
|---|---|---|
| P0 | C01 地基 / C02 V1–V6 测试 / C03 迁移 / C04 manifest / C05 Gate A 冒烟 | C01/C02 done |
| P1 | C06 双路线 ΔF / C07 CI01 / C08 理论定稿 | — |
| P2 | C09 scene 工厂 / C10 CI02 / C11 retention+tracking | — |
| P2b | C12 CI03 线性 / C13 CI03 非线性 | — |
| P3 | C14 OpenIllumination / C15 Σ_c 生成器 / C16 CI04 / C17 DiLiGenT | — |
| P3b/P4 | C18–C19 / C20–C23 | — |

条件卡：OPT-1 VarPro（估计器单解>数秒成瓶颈才启用）；卡 R2 复活（CI05 需合成侧判别力失败案例）。

## 6. 资产速查

- 红队数值资产：`legacy_redteam/redteam{,2,3}_exp.py`（E1–E6/N1–N5/V1–V6，种子 20260907，
  纯 CPU 5–71s）+ 三轮报告 `D:\红队报告*_20260907*.md`；
- 估计器（已迁移）：`src/calibinfo/estimators/`（joint_map 与 legacy exp8R v3 joint_trf
  **对账 x 逐位一致 max|Δ|=0.0**）；诊断参考实现（exp8S 变体 A 代数）= estimators/diagnostics.py；
- 数据 loader（已迁移）：`src/calibinfo/datasets/diligent.py`（与 legacy load_object 逐位一致）；
- 信息层（已迁移）：`src/calibinfo/information/`（whitening/schur/gauge/retention/mode_tracking）；
- 旧四支柱历史结论：`critical_experiments/`（原位只读，禁引清单见 `docs/archive/legacy_iccv/论文数字口径说明_v0.5.md`
  与 CLOSURE_20260906.md 增补指针）；
- 长期记忆：`~/.clawsgo/memory/inverse-rendering-sci-plan.md`（每里程碑追加）。

## 7. Git 锚点（新宪法起点）

```
6e28ad8 卡C02 红队V1-V6测试迁移(23绿,legacy逐位对照) + exp9d查证闭阻塞
c23cedb 卡C01 calibinfo包骨架+CLAIMS_REGISTRY+STATE/PARKING_LOT
14b10c9 宪法+主控计划书+REPO_MIGRATION 入库
29c742a (旧主线终点) 窗口终局交接
```

## 8. 踩坑清单 21 条（原样继承，AGNET_HANDOFF 08-31 版 §5）

**BlenderProc / 生成器**
1. `blenderproc run` 的脚本**第一有效行必须是 `import blenderproc`**（docstring 都不能在前面）；主 python 里 import 它会直接 RuntimeError。
2. `bproc.renderer.enable_depth_output()` 等 enable 是**进程级一次性**——多场景复用同进程会 "can not be called twice"。**每场景独立进程跑**（bash for 循环）。
3. **BlenderProc normals AOV 已是相机系**——不要再做 world→cam 旋转（二次旋转曾把 oracle 打到 15 dB，修复后 26.7+）。
4. **SUN 能量语义**：strength=100 会全图饱和（线性域裁到 1.0，形状信息全毁）。用 `--light_energy 3.0`（I_eff=S/π≈0.95）。渲染后必查饱和占比=0。
5. **P 域必须纯 Diffuse BSDF**（默认 Principled 有 specular=0.5，colors 通道混高光）；且 `materials.clear()` 要无条件执行（OBJ 无材质时 if 判断会跳过 append）。
6. BlenderProc frame animation（`Light.set_location(..., frame=k)`）**不可信**——synthetic_v3 五图同图灾难根源；**每灯独立 render call**。
7. Blender 4.2 没有 `bpy.ops.export_scene.obj`（用脚本自带 write_obj）。
8. 空掩码场景（如水平 plane 在 30° 俯角下不可见）要 raise 干净跳过。
9. Windows git-bash **不支持 process substitution** `<(echo ...)` 当 obj_list——用临时文件；且别把 .obj 本身当 list 传（每行 OBJ 内容会变成"场景名"建垃圾目录）。
10. 渲染输出 dtype 自动检测：uint8→sRGB 反变换；float→已是线性。

**SH / 物理**
11. 卷积系数用 `A_L=[π,2π/3,π/4]`（`sh.py`），**不是** K_L 旧值；E_L2 解析式 = 0.25+0.5μ+0.3125·P₂(μ)。
12. 全 (0,0,1) 法线初始化是**鞍点**（SH x/y 项梯度恒 0）——联合优化必须随机小扰动 + c 小噪声破对称（exp2 教训）。
13. 存在全局旋转 gauge（法线+光照同旋转图像不变）——法线角误差必须 Kabsch 对齐后再报告；albedo/light 尺度 gauge → SI-MAE。

**训练 / 求解器**
14. 训练渲染用 `use_edge_aware=False`（与数据定义一致；True 是旧 config 遗留冲突）。
15. GPU 双进程并行会互相拖慢数倍（小 kernel 串行化）——**串行跑大任务**。
16. 后台跑长任务：stdout 会缓冲看不到进度，用 `python -u` 或查 checkpoint/csv 落盘时间戳；`blenderproc run -u` 不认 -u（会传给脚本）。
17. 系统内存可能被残留 python 进程吃满（出现过 2.2MB 都分配失败的 OOM）——跑大任务前 `Get-Process python` 清点并清理。
18. `SceneBatcher._CACHE` 类内缓存跨实例共享——注意多 loader 场景的内存。

**评估 / 统计**
19. pooled 回归跨场景比较必须先按场景 z-score 归一（场景尺度效应会淹没子集效应）——R4 G2 的教训；且别只用最弱指标（logdet）下结论。
20. solver 收敛判据（tail-loss<1e-7 & grad<1e-3）过严会 0% success——对比实验前先标定判据；对比只用收敛 trials（P1-10 纪律）。
21. 评估代码里 `[B,K,H,W]` vs `[B,K,1,H,W]` 广播是复发 bug 源（recon PSNR 曾错 20dB）——shape 断言先写。

---

*重写执行 agent · 2026-09-07 · 卡 C03 · 交接指令 = §2*
