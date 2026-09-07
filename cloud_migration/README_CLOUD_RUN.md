# exp8R v3 云实例运行包(2026-09-06)

## 背景
本机(RTX 5070 Ti laptop, 32核)实测单物体 3.5-4h(trf 数值差分固有成本),
10 物体全量 ~35-45h,判定为对本机不可用的阻塞 → 搬云(64核/128GB, 预计 3-4h)。

## 运行方法(云 Linux 实例)

```bash
# 1. 环境(Python 3.10+)
pip install numpy scipy pillow

# 2. 数据布局(把 DiLiGenT 解压/放置到):
#    /data/DiLiGenT/pmsData/   (10 个 PNG 目录, 每个含 001-096.png + 光方向/强度 txt + Normal_gt.mat + mask.png)

# 3. 单进程试跑(先验证 1 物体端到端):
python exp8r_diligent_discrimination_v3.py --only ballPNG

# 4. 并行全量(10 物体天然独立, 一物体一进程):
# 脚本已支持 --only <object> 参数; 用 GNU parallel 或 bash 后台:
for obj in ball bear buddha cat cow goblet harvest pot1 pot2 reading; do
  python exp8r_diligent_discrimination_v3.py --only ${obj}PNG &
done
wait
# 5. 合并:
python exp8r_merge.py
```

## 文件清单
- exp8r_diligent_discrimination_v3.py — 主脚本(已加 --only 支持)
- exp8r_merge.py — 逐物体 json 合并 + 判定(待生成)
- exp8r_preregistration_v3.md — 协议(判据冻结, 禁改)

## 关键纪律(迁移不改变)
- 种子 SEED=20260906 确定性(结果与本机部分跑一致)
- 预注册 v3 判据: N=3 层 50 次 ≥6/10 物体 LAE 显著正(p<0.05)
- 像素下采样(每8像素取1, 固定种子)= 数值预算通用预处理, 已在脚本内

## 进度查看(运行期间, 另开终端)

```bash
watch -n 15 bash show_progress.sh        # 仓库根; 或 cloud_migration/ 下同名脚本
```

- 每物体 DONE/RUN/PENDING + 已完成子集计数(总 110 子集/物体: N=3×50 + otherN×60);
- 每物体自估 ETA + 整体墙钟 ETA(取运行物体最大值); 明细在 `logs/<物体>.log` 的 `[P]` 行。

## 线程钉扎(入口脚本自动设置)

10 物体并行时若 BLAS 默认全核, 会 10×64 线程超额订阅拖慢; 脚本已按
`NCPU/PARALLEL` 为每物体导出 OMP/OpenBLAS/MKL 等线程数(64 核 10 物体 → 每物体 6 线程)。
