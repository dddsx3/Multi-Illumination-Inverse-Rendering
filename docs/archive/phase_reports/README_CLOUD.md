# Phase 2 云端训练交付文档

**版本**：v1.0（2026-08-25）　**交付人：本地执行者　~~运行方：云端算力~~**

> **状态更新（2026-08-25 深夜）：云端计划已取消，七臂全部转本地串行执行。**
> 本文档保留作为七臂实验矩阵与回传清单口径的原始定义。本地执行入口：
> `run_phase2_all.py`（同一编排脚本，断点续跑语义不变，数据根默认
> `D:/data/synthetic_v3`）。已完成臂：R0 对照、F-N5-gray（评估
> `eval_output/p2_r0_v3gray_test`、`p2_t22_f_n5gray_test`）；剩余五臂由
> 编排器串行补齐。RGB 双链路配套改动见 trainer.py `_resolve_recon_target`
> 与 data_loader.py 的编码域 BT.709 luma 生成（与灰度 PNG 推导逐位同源）。

---

## 一、云端的具体任务是什么

在云端机器上**顺序执行 7 个全量训练 run**（每个 100 epoch，BF16，约 1.6–1.8 小时），
并在每个 run 结束后自动跑一次冻结 test 集 13 项指标评估。全部完成后把「回传清单」
（见第五节）打包传回。总 GPU 时间预算 **11–13 小时**；脚本支持断点续跑——
云端被抢占/kill 后**重新执行同一条启动命令即可从中断处继续**。

### Run 清单（串行执行，顺序即依赖顺序）

| # | run_id | 架构 | 模态 | 特殊参数 | 用途 |
|---|---|---|---|---|---|
| 1 | p2_r0_gray_R0BASE | 原 U-Net | gray | 无 | 共享对照基线臂 |
| 2 | p2_t22_f_n5gray | FusionUNet | gray | 无 | 核心创新主交付 |
| 3 | p2_t22_f_n5rgb | FusionUNet | **rgb** | 无 | 双模态链路 |
| 4 | p2_t23_f_physcon | FusionUNet | gray | --sh_constraint softplus | 物理约束消融 |
| 5 | p2_t25_f_resA | FusionUNet | gray | --residual_off | 残差关闭消融 |
| 6 | p2_t25_f_resC | FusionUNet | gray | --res_hidden 32 | 残差容量消融 |
| 7 | p2_t25_f_albOff | FusionUNet | gray | --no_per_light_albedo | 逐光照反照率消融 |

## 二、环境要求

- GPU：≥12GB 显存（实测单实例峰值 ~8GB @bs8/BF16）；NVIDIA + CUDA 12.x
- Python ≥3.10；依赖见 `code/requirements.txt`（torch/torchvision/numpy/Pillow/tqdm）
- 操作系统不限（脚本纯 Python 跨平台）；磁盘 ≥30GB（数据集解压 ~2GB + 7 run × ~1.7GB checkpoint）
- **不需要 BlenderProc**（渲染已完成，只训练）

## 三、包结构与放置

```
phase2_cloud_package.zip
├── README_CLOUD.md          ← 本文档
├── code/                    ← 仓库快照（含全部源码与划分清单）
│   ├── main.py / trainer.py / fusion_unet.py / evaluate_model.py ...
│   ├── splits/synthetic_v3.json   （冻结划分：train 456 / val 50 / test 127）
│   ├── run_phase2_all.py    ← ★ 一键编排入口
│   ├── eval_n_curve.py      ← N 敏感性曲线评估
│   └── requirements.txt
└── data/
    └── synthetic_v3/         ← 626 个场景目录（每场景 15 个文件）
```

解压后保持该结构；`data/synthetic_v3` 与 `code/` 的相对位置即默认约定
（编排器按 `../data/synthetic_v3` 相对定位；如放别处，用环境变量 `P2_DATA_ROOT` 覆盖）。

## 四、启动命令（三选一，按云端 OS）

```bash
# Linux/macOS
cd phase2_cloud_package/code
pip install -r requirements.txt
python -u run_phase2_all.py                # 全部 arm 串行执行

# Windows PowerShell
cd phase2_cloud_package\code
python -u run_phase2_all.py
```

**断点续训语义**：编排器维护 `progress.json`；每个 arm 内部按 10 epoch 一段切分，
每段结束扫描 `checkpoints/{run_id}/checkpoint_epoch_NNNN.pth` 取最大值作为续训起点，
通过 `--resume --checkpoint <latest>` 热重启（优化器/调度器/阶段状态全恢复）。
云端任意时刻被 kill：重新执行同一命令即可，已完成的 epoch 与已完成的 arm 均不重跑。

常用变体：
- `python run_phase2_all.py --status` —— 只看各 arm 进度
- `python run_phase2_all.py --only p2_r0_gray_R0BASE,p2_t22_f_n5gray` —— 只跑指定 arm

## 五、需要回传的产物（验收证据）

**必须回传（缺一无法验收）：**

1. `checkpoints/{每个run_id}/best_model.pth` —— 是否交付模型？**是**，每 run 一个
   best（~96MB×7）；`latest_model.pth` 可选（用于续训兜底）；
   中间的 checkpoint_epoch_NNNN.pth 不需要回传；
2. `eval_output/{每个run_id}/eval_summary.json + per_scene_metrics.csv` ——
   13 项指标的聚合与逐场景原始数据（审计将抽查溯源）；
3. `_train_{run_id}.txt` 训练日志全程 + `logs/{run_id}/` TensorBoard 目录；
4. `progress.json`（编排器进度账本）。

**建议附带回传（提升验收质量）：**

5. `eval_output/n_curve_raw.json`（若跑了 N 曲线扩展评估）；
6. 云端 `nvidia-smi` 快照与 `pip freeze` 输出（环境复现凭据）。

## 六、验收标准对照（云端侧只需保证 1–4 可回传）

- G2.2：F-N5-gray 与 R0 的 PSNR 对比（不降即放行）、置换测试已在本地通过；
- G2.3：F-physcon vs F-N5-gray（softplus vs clamp 单变量）；
- G2.5：五维消融矩阵完整性（所有 arm 的 json/csv 缺一不可）；
- G2.6b：N 子集曲线原始数据（子集索引入库）。

## 七、故障排查

| 症状 | 处理 |
|---|---|
| CUDA OOM | 确认无其他进程占卡；仍 OOM 则 batch_size 降 4 并记录（所有后续 arm 统一改） |
| 页面文件太小 | 云端加大 swap/pagefile 或增加物理内存（本机教训，见 INC-0004 关联） |
| dataloader worker 崩溃 | --num_workers 0 重试 |
| 中断后重跑重复训练 | 检查 checkpoints/{run_id} 是否有 checkpoint_epoch_*.pth；有则编排器自动 resume |

## 八、已知限制（如实声明）

- v3 每场景仅 5 光 => N 敏感性曲线只能测 N∈{1..5}；任务书 {7,10} 需未来渲染
  更多光照后补测（已在设计文档声明）；
- pad+mask 变长批处理未实现（按 N 分桶规避）；
- F-physcon 的 softplus SH0 若出现学习停滞（SH0 长期不动），按风险登记 R2 处理。