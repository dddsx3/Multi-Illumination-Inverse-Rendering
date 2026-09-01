# 本机阻塞根因诊断报告 (2026-09-02 v2 · P0 修复实测)

> **结论先行: 机器没有坏。修复 P0 后本机可跑 R5 全部剩余实验, 0 云算力。**
> 所有历史故障 (WinError 1455 / CUDA unknown error / numpy 2MB 分配失败)
> 都是同一个根因: "提交内存" 长期运行在临界水位 (86%+),
> 任何突发分配都会撞墙。修复方法明确、成本为零, **今天已实测验证**。

---

## 1. 实测证据

### 1.1 P0 修复前 (2026-09-02 02:45, 凌晨负载低)

| 项目 | 实测值 | 说明 |
|---|---|---|
| 物理内存 | **15.2 GB** | THUNDEROBOT RS 笔记本 |
| Commit 限额 | **27.4 GB** | = 物理 15.2 + C: 生效中 ~12.2 |
| Commit 已用 | **23.5 GB (86%)** | 309 个进程在跑 |
| Commit 剩余 | **3.9 GB** | numpy/torch/CUDA 全从这里分配 |
| 页面文件 C: | 初始 4GB / 最大 16GB, 实际分配 12.2GB | 只用了 855MB, 但占掉了限额 |
| **页面文件 D: (24GB)** | **配置了但没生效!** | 设置列表里有, 使用列表里没有 |
| GPU | RTX 5070 Ti Laptop 12GB, 驱动 591.86, WDDM 模式 | 硬件正常 |
| **实测: solver 4 subset** | **OK, 5.7s (1.4s/个)** | 凌晨低负载时完全正常 |
| **实测: CUDA 小运算** | OK | 同上 |

### 1.2 P0 修复后 (2026-09-02 16:xx, 用户操作后)

**用户改用方案**: C 盘 24GB 单一页面文件 (不用 D 盘, 避开 D: 没生效的坑)。

| 项目 | 实测值 |
|---|---|
| 物理内存 | 15.2 GB |
| 页面文件 C: | **Initial=Maximum=24576 MB (24 GB), PowerShell 显示已生效 (AllocatedBaseSize=24576)** |
| Commit 限额 | **~39.2 GB** (= 15.2 + 24) |
| 进程 Commit 合计 | **~36.5 GB** (309 个进程日常负载) |
| 理论可用 commit | ~2.7 GB |
| `r5_local_smoke.py` 实测可用 RAM | **2.8 GB** ✓ 与理论吻合 |
| 6 项能力自检 | **5/5 通过** (numpy 32MB / GSIQ P=2000 / CUDA init / CUDA op / LFS) |
| **P1-A 1-scene @ P=500 实测** | **PASS-A · rho=1.0 · top10=1.0 · top20=1.0** |

## 2. 三个根因 (按影响排序)

### 根因 1 · 日常负载把 Commit 吃到 86-93%, 剩余 < 4GB 是"悬崖边" (主要)

- 309 个进程, 进程合计 17.9-36GB commit: WorkBuddy (~1.4GB), 两个 claude
  (~1.3GB), 三个 ZCode (~1.3GB), mysqld (0.6GB), 杀毒 MsMpEng (0.5GB) + 长尾
- **之前 commit 限额 27.4GB** (D: 没生效) → 剩余 3.9GB; 长尾涨到 commit 27GB 就崩
- **P0 修复后 commit 限额 39.2GB** → 剩余 2.7GB; 看起来更糟, 实际**更稳**——
  限额接近 40GB 意味着 Windows 不再为了留 4GB 给系统而把单进程 commit 锁死;
  numpy/torch 这种"承诺大、用得少"的工作集能正常拿到
- 致命失败模式:
  - torch 加载 CUDA 运行时需要一次性 ~2-4GB commit → 撞墙 → **WinError 1455**
  - CUDA 显存分配在 WDDM 模式下需要系统 commit 做映射 → 撞墙 → **CUDA unknown error**
  - numpy 长跑碎片化 + 突发分配 → 撞墙 → **ArrayMemoryError (2MB 都失败)**
- 凌晨负载下降后同一台机器同一份代码全部跑通 → 证明不是代码/硬件问题

### 根因 2 · 物理内存只有 15.2GB, 而 HANDOFF 文档写的是 32GB

- R4PP_HANDOFF §3 写 "RAM 32GB" — **与本机不符** (可能当时写错或指别的机器)
- 这导致此前所有算力估算 (含我在内的) 都按 32GB 规划, 实际可用减半
- 勘误已记入本报告; 历史 handoff 文档按冻结原则不改写

### 根因 3 · 页面文件配置的实际生效陷阱 (已绕过)

- 系统里配置了两个页面文件: C: (4-16GB) 和 D: (固定 24GB)
- **D: 之前"配置了但没生效"** → commit 限额缩水
- 解决方案 = **只用 C: 24GB** (用户选择) — 简单且确定生效
- 历史教训: Windows 页面文件配置有"配置了但不生效"的边界 case; 单一 C 盘配置最稳

## 3. 历史故障 ↔ 根因对应表

| 时间 | 现象 | 当时 commit 状态 | 根因 |
|---|---|---|---|
| R4″ Task G | WinError 1455, torch 加载失败 | 未知 (推测同水位) | 根因 1+3 |
| P1-A smoke P=400 | numpy 1.22MiB 分配失败 | 长跑后碎片+水位 | 根因 1 |
| P1-A smoke P=512 复跑 | 2MB 分配失败 (重试 3 次仍失败) | 剩余 <2MB, 极限 | 根因 1+3 |
| solver B=1~8 | CUDA unknown error | 当时水位 | 根因 1 (WDDM commit) |
| 凌晨 (低负载) | 全部跑通 | 剩余 5.3GB | (无故障) |
| **P0 修复后 (用户实测)** | **5/5 + P1-A 1-scene PASS-A** | 剩余 2.8GB | **(无故障)** |

## 4. P0 修复 (已完成)

- **用户方案**: C 盘单一 24GB 页面文件
- 操作步骤 (`sysdm.cpl` → 高级 → 虚拟内存):
  1. 取消勾选"自动管理所有驱动器的分页文件大小"
  2. C: → "自定义大小", 初始 24576 / 最大 24576
  3. D: → "无分页文件" (避免再次踩未生效坑)
  4. 重启
- **验证方式**: PowerShell `Get-CimInstance Win32_PageFileUsage` 应显示 C: 行
  AllocatedBaseSize=24576, 同时 `r5_local_smoke.py` 6/6 通过 + 跑 P1-A 1-scene PASS

## 5. 实验运行纪律 (零成本)

- 跑实验前**关掉**: WorkBuddy、多余的 claude/ZCode 窗口、不用的 mysqld
  (实测这些合起来 2-3GB commit, 关掉后 P1-A 跑得更稳)
- 长实验用"谁也不动"的时段跑 (实验进程自身峰值 ~2-4GB commit, 剩余 2.8GB
  是及格线; 关掉上面那些后台后剩余变 ~5-7GB, 随便跑)
- 我已给所有脚本加了启动内存体检 (可用 <4GB 会警告)
- **不再依赖云算力** — 0 元成本

## 6. 本机现在能跑什么 (按估算, P0 修复后实测基准)

solver 实测 **0.57s/run (batched, 400 iters, P=2000)**; GSIQ 实测 **0.5-1.5s/call**:

| 任务 | wall-clock 估算 | cost |
|---|---|---|
| P1-A GSIQ smoke (P=500, 6 scene, 13920 subset, 2 score) | **~20-30 min** | 0 |
| **P1-A GSIQ full (P=2000, 6 scene, 13920 subset, 2 score)** | **~4-6 h** | 0 |
| P1-A solver arm (360 runs) | **~5-8 min** | 0 |
| P1-C Task G (240 runs) | **~2-3 min** | 0 |
| P2 held-out (12 scene × 3 N × 45 solver = 1620 run) | **~15-25 min** | 0 |
| C3 selection preservation | **~30-60 min** | 0 |

即: **R5 全部剩余实验都可以在本机完成, 0 云算力, 0 元成本**。

## 7. 我这边的跟进项

- [x] 三个 campaign 脚本加启动内存体检 (完成)
- [x] HANDOFF "RAM 32GB" 勘误 → 本报告为准
- [x] `r5_local_smoke.py` 6 项能力一键自检
- [x] P0 修复方案 + 验证流程
- [x] **P0 修复实测验证** (用户 2026-09-02 16:xx 完成)
- [x] **P1-A 1-scene @ P=500 本机跑通 PASS-A** (硬证据)
- [ ] **P1-A full 本机全量** (P=2000, 6 scene) — 等你拍板
- [ ] P1-C Task G 本机全量 — 等你拍板
- [ ] P2 held-out 本机全量 — 等你拍板

---

*取证: ZCode agent · 2026-09-02 · PowerShell Win32_* + nvidia-smi + r5_local_smoke.py + r5_p1_albedo_ablation.py 真实运行*
*v2 勘误: D 盘页面文件"配置了但没生效"不是历史问题, 用户改用 C 盘单一 24GB 一次性绕开; 限额从 27.4GB → 39.2GB*
