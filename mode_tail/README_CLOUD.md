# mode_tail · 云端认证表运行包（P-LOWRANK-FULLRES v1）

全分辨率 / 全 142 灯的标定预算认证间隙表（原理与协议见
mode-resolved-calibration 仓库 `configs/lowrank_fullres.yaml` 内嵌说明）。

## 内容

- `app/src/calibinfo/` — 分析库（vendored，与 mode-resolved-calibration
  分支 `mode-tail-design` 的对应文件逐字一致）
- `app/experiments/lowrank_fullres.py` — 驱动（v1.1：逐对象 checkpoint
  断点续传 + 多进程按对象并行 + 顺序无关汇总）
- `app/experiments/openillumination_validation.py` — nominal 资产依赖
- `data/OpenIllumination/OLAT/...` — 11 对象 × 142 灯的
  com_masked_thumbnail/A1.png（17 MB，全部所需数据层）
- `data/OpenIllumination/light_pos.npy` — GT 灯位
- `run_cloud.py` — 启动器（线程数预设 / 依赖自检 / 数据路径改写 / 续传）

## 运行（默认 32 核 / 64 GB 机器）

```bash
cd mode_tail
python run_cloud.py                 # 自动 workers = max(1, cores//6)，OMP = cores//workers
# 或显式：
python run_cloud.py --workers 6 --objects obj_03_pumpkin,obj_04_dolphin
```

- **断点续传**：中断/崩溃后原命令重跑——`results/checkpoints_fullres/
  partial_<obj>.json` 存在的对象直接跳过。
- **预期耗时**：32 核、6 workers × 5 BLAS 线程：约 1.5–2.5 h（本机
  32 核串行实测 ~4.8 h；P 逐对象 6k–35k 不等）。
- **产出**：`results/lowrank_fullres.json`（55 行以下行表 + per-k 汇总
  + display + manifest）与 `results/checkpoints_fullres/`。

## 完成后

把 `results/lowrank_fullres.json` 与 `results/checkpoints_fullres/`
提交回本分支（或原样带回主仓库 `mode-tail-design` 的
`results/certification/`），由主仓库的
`tests/test_lowrank_identity.py` 等做一致性验收。

## 纪律

- 本包所有文本 LF（pack 时强制）；数据为只读快照（17 MB，与
  `data/manifests` 校验链同源）；
- 协议结果无关：无符号判据，全部行如实报告。
