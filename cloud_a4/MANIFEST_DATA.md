# cloud_a4 · 数据上传规格（仅当 run_all.sh 报 DATASET NOT FOUND 时需要）

run_all.sh 会自动探测以下路径的数据；全部找不到时需要手动上传。

## 需要的文件（精确清单，来自冻结的 loader `datasets/openillumination.py`）

```
data/OpenIllumination/
  light_pos.npy                              # (142,3) float64 — GT 灯位（全局共享）
  OLAT/obj_03_pumpkin/Lights/000..141/com_masked_thumbnail/A1.png   # 142 张/对象
  OLAT/obj_03_pumpkin/output/com_masks/A1.png                       # 1 张（交叉校验备用）
  ... (以下 10 对象同结构)
data/OpenIllumination_meta/
  light_pos.npy                              # 同上（loader 的 fallback 路径）
  dev_selection.json                         # 开发清单（现有实验脚本读取）
  data_olat.json                             # 官方 OLAT manifest（审计留档用）
```

11 个对象：obj_03_pumpkin, obj_04_dolphin, obj_07_pumpkin2, obj_09_ball,
obj_10_pumpkin3, obj_11_pine, obj_13_mushroom, obj_16_friends_cup,
obj_17_pumpkin5, obj_18_fabric_hat, obj_19_cylinder

**总量：约 19–24 MB（1,573 个文件）。**

## 探测顺序

run_all.sh 依次探测（找到即用）：
1. `$MRC_DATA_ROOT`（环境变量）
2. `./data/OpenIllumination`（包目录内）
3. `$HOME/data/OpenIllumination`
4. `/data/OpenIllumination`

## 打包好的数据

`openillumination_data_bundle.zip`（作者提供，17.3 MB）解压后即为上述精确结构：
```zsh
unzip -o openillumination_data_bundle.zip -d data_extract
mv data_extract/data/OpenIllumination data/
mv data_extract/data/OpenIllumination_meta data/
```
