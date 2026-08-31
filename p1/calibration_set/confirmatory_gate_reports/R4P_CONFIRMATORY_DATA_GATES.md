# R4′-C 确认集数据 Gate 汇总

- scene 数：18（canary 1 + 批量 17；seed 20260901；horizon 18 mesh 中 7 mesh 渲染时空遮罩被剔除，见日志）
- 渲染协议：SUN 远场 / 纯 Diffuse / 128² / 32 samples / light_energy 3.0
- INC-001 帧级校验已嵌入；本批无坏帧逃出

## Validation Gates (G1/G2/G3)

| scene | G1 ratio | G2 Δ(dB) | G3 Δ(dB) | PASS |
|---|---|---|---|---|
| conf_cone_r04_d12 | 484.4 | 10.26 | 10.26 | ✓ |
| conf_cube_axis | 469.5 | 10.13 | 10.13 | ✓ |
| conf_cube_plus_cone | 434.5 | 9.66 | 9.66 | ✓ |
| conf_cyl_plus_sphere | 456.4 | 10.12 | 10.12 | ✓ |
| conf_cylinder_r03_d12 | 467.4 | 10.24 | 10.24 | ✓ |
| conf_cylinder_r06_d06 | 488.7 | 10.56 | 10.56 | ✓ |
| conf_egg | 456.5 | 10.62 | 10.62 | ✓ |
| conf_ellipsoid_x13z07 | 400.2 | 10.52 | 10.52 | ✓ |
| conf_ellipsoid_z06 | 433.3 | 10.91 | 10.91 | ✓ |
| conf_hemisphere_sq | 447.4 | 10.88 | 10.88 | ✓ |
| conf_icosphere_sub3 | 460.9 | 10.82 | 10.82 | ✓ |
| conf_prism8 | 483.1 | 10.42 | 10.42 | ✓ |
| conf_snowman | 446.0 | 10.11 | 10.11 | ✓ |
| conf_sphere_on_cube | 433.9 | 9.06 | 9.06 | ✓ |
| conf_sphere_r05 | 461.5 | 10.83 | 10.83 | ✓ |
| conf_torus_R05_r02 | 438.3 | 10.89 | 10.89 | ✓ |
| conf_torus_R06_r035 | 441.6 | 10.87 | 10.87 | ✓ |
| conf_two_spheres_row | 405.0 | 9.92 | 9.92 | ✓ |

**18/18 场景全部 3 个 Gate PASS**

## Oracle Gate（mesh normal + GT albedo + GT light）

| scene | Or1 SI-PSNR (dB) | Or2 SI-PSNR (dB) |
|---|---|---|
| conf_cone_r04_d12 | 28.47 | 13.13 |
| conf_cube_axis | 31.41 | 15.59 |
| conf_cube_plus_cone | 28.22 | 13.78 |
| conf_cyl_plus_sphere | 26.17 | 13.30 |
| conf_cylinder_r03_d12 | 28.08 | 13.41 |
| conf_cylinder_r06_d06 | 28.34 | 13.85 |
| conf_egg | 26.72 | 13.55 |
| conf_ellipsoid_x13z07 | 26.24 | 13.07 |
| conf_ellipsoid_z06 | 26.70 | 14.03 |
| conf_hemisphere_sq | 26.69 | 13.95 |
| conf_icosphere_sub3 | 26.76 | 13.88 |
| conf_prism8 | 30.41 | 14.23 |
| conf_snowman | 25.55 | 13.47 |
| conf_sphere_on_cube | 26.21 | 13.65 |
| conf_sphere_r05 | 26.73 | 13.89 |
| conf_torus_R05_r02 | 26.73 | 13.94 |
| conf_torus_R06_r035 | 26.71 | 13.93 |
| conf_two_spheres_row | 25.06 | 12.79 |

**汇总**
- Or1 SI-PSNR: mean=27.29 dB, min=25.06, max=31.41
- 阈值 (25.0 dB): 18/18 场景 Or1 PASS

## 结论

- 物理协议零改动：Or1 全员 > 阈值即 P 域 oracle 成立；
- 确认集与 Discovery (4 scene) 在同协议下表现一致；
- 数据可进入 R4′ 确认性 Gate（E2/G2/E3）。