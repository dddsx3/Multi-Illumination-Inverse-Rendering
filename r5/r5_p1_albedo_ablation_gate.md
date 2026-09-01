# R5-P1-A Gate · smoke (RTX 5070 Ti)

Scenes: ['conf_sphere_r05', 'conf_cube_axis', 'conf_prism8', 'conf_egg', 'conf_cylinder_r06_d06', 'conf_ellipsoid_z06']

NS: [3, 5]

Per-(scene, N) ranking diagnostics:

| scene | N | rho | top10 | top20 | O_mean | A_mean | n_deficient_O | n_deficient_A | n_at_cutoff_O_med | n_at_cutoff_A_med |
|---|---|---|---|---|---|---|---|---|---|---|
| conf_sphere_r05 | 3 | 1.0000 | 1.000 | 0.990 | -5.932 | -5.931 | 11 | 11 | 0 | 0 |
| conf_sphere_r05 | 5 | 1.0000 | 1.000 | 1.000 | -6.137 | -6.137 | 8 | 8 | 0 | 0 |
| conf_cube_axis | 3 | 1.0000 | 1.000 | 1.000 | -5.749 | -5.749 | 500 | 500 | 0 | 0 |
| conf_cube_axis | 5 | 1.0000 | 1.000 | 1.000 | -6.078 | -6.078 | 500 | 500 | 0 | 0 |
| conf_prism8 | 3 | 0.9970 | 1.000 | 1.000 | -5.823 | -5.823 | 500 | 500 | 1 | 1 |
| conf_prism8 | 5 | 1.0000 | 1.000 | 1.000 | -6.163 | -6.163 | 500 | 500 | 1 | 1 |
| conf_egg | 3 | 0.9999 | 1.000 | 0.980 | -5.925 | -5.925 | 10 | 10 | 1 | 1 |
| conf_egg | 5 | 1.0000 | 1.000 | 1.000 | -6.160 | -6.160 | 13 | 13 | 0 | 0 |
| conf_cylinder_r06_d06 | 3 | 1.0000 | 1.000 | 1.000 | -5.790 | -5.790 | 500 | 500 | 1 | 1 |
| conf_cylinder_r06_d06 | 5 | 1.0000 | 1.000 | 1.000 | -6.030 | -6.030 | 500 | 500 | 1 | 1 |
| conf_ellipsoid_z06 | 3 | 1.0000 | 1.000 | 0.990 | -5.972 | -5.972 | 12 | 12 | 0 | 0 |
| conf_ellipsoid_z06 | 5 | 1.0000 | 1.000 | 1.000 | -6.110 | -6.110 | 6 | 6 | 0 | 0 |

**median rho** = 1.0000

**median top10 overlap** = 1.000

**median top20 overlap** = 1.000

## Gate verdict: PASS-A

Next step: freeze a=1; P1-B normal/light proxy audit

## Boundary outliers (|ΔI| > 1e-03)

Total: 50 subsets (0.833% of all subsets)

| scene | N | subset | I_O | I_A | diff | status_O | status_A | d_extra_O | d_extra_A | n_at_cutoff_O | n_at_cutoff_A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conf_sphere_r05 | 3 | 0,4,12 | -5.8806 | -5.8795 | -1.0375e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_sphere_r05 | 3 | 0,6,14 | -5.8974 | -5.8963 | -1.1393e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_sphere_r05 | 3 | 0,8,14 | -5.9265 | -5.9254 | -1.0815e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_sphere_r05 | 3 | 0,12,14 | -5.8451 | -5.8441 | -1.0009e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_sphere_r05 | 3 | 0,14,19 | -5.9184 | -5.9174 | -1.0063e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_sphere_r05 | 3 | 1,2,12 | -7.0816 | -7.0806 | -1.0519e-03 | full | full | 0 | 0 | 6 | 6 |
| conf_sphere_r05 | 3 | 1,2,14 | -6.5887 | -6.5873 | -1.3744e-03 | deficient | deficient | 1 | 1 | 3 | 3 |
| conf_sphere_r05 | 5 | 4,7,12,14,17 | -7.5090 | -7.5078 | -1.2149e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_sphere_r05 | 5 | 1,12,16,25,30 | -6.3549 | -6.3537 | -1.1541e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_sphere_r05 | 5 | 1,12,19,25,30 | -7.4496 | -7.4482 | -1.3605e-03 | full | full | 0 | 0 | 11 | 11 |
| conf_sphere_r05 | 5 | 14,16,22,24,29 | -6.4631 | -6.4621 | -1.0461e-03 | full | full | 0 | 0 | 2 | 2 |
| conf_sphere_r05 | 5 | 9,14,19,20,25 | -7.2101 | -7.2088 | -1.2373e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_sphere_r05 | 5 | 6,9,11,12,22 | -7.0496 | -7.0480 | -1.6770e-03 | full | full | 0 | 0 | 4 | 4 |
| conf_sphere_r05 | 5 | 1,12,19,22,27 | -7.5467 | -7.5452 | -1.4809e-03 | full | full | 0 | 0 | 12 | 12 |
| conf_prism8 | 3 | 0,25,29 | -5.8243 | -5.7817 | -4.2550e-02 | deficient | deficient | 2 | 3 | 1 | 0 |
| conf_egg | 3 | 1,2,7 | -7.0057 | -7.0042 | -1.4467e-03 | deficient | deficient | 1 | 1 | 4 | 4 |
| conf_egg | 3 | 1,2,26 | -6.2830 | -6.2816 | -1.4190e-03 | deficient | deficient | 1 | 1 | 1 | 1 |
| conf_egg | 5 | 1,13,18,20,28 | -6.0980 | -6.0967 | -1.2773e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 4,15,18,28,31 | -6.5567 | -6.5556 | -1.1390e-03 | full | full | 0 | 0 | 2 | 2 |
| conf_egg | 5 | 1,7,18,22,26 | -6.3204 | -6.3192 | -1.1951e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 1,5,9,17,20 | -6.0195 | -6.0184 | -1.1685e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 4,6,18,25,30 | -6.3304 | -6.3291 | -1.2636e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 6,7,26,27,28 | -6.4563 | -6.4544 | -1.8481e-03 | deficient | deficient | 1 | 1 | 1 | 1 |
| conf_egg | 5 | 1,7,19,20,23 | -6.8092 | -6.8103 | +1.0967e-03 | deficient | deficient | 1 | 1 | 5 | 5 |
| conf_egg | 5 | 12,17,20,25,26 | -6.5058 | -6.5043 | -1.5145e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 7,9,12,13,20 | -6.0267 | -6.0254 | -1.3275e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 2,15,20,23,31 | -6.8437 | -6.8449 | +1.1989e-03 | deficient | deficient | 1 | 1 | 5 | 5 |
| conf_egg | 5 | 1,7,17,23,26 | -6.4640 | -6.4629 | -1.0440e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 1,6,23,26,30 | -6.4196 | -6.4180 | -1.5516e-03 | deficient | deficient | 1 | 1 | 1 | 1 |
| conf_egg | 5 | 12,23,26,28,29 | -6.4474 | -6.4464 | -1.0050e-03 | full | full | 0 | 0 | 2 | 2 |
| conf_egg | 5 | 1,7,18,21,27 | -6.1491 | -6.1481 | -1.0068e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 1,9,11,25,29 | -6.4430 | -6.4420 | -1.0101e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 7,17,18,30,31 | -6.5058 | -6.5046 | -1.1918e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 4,7,26,30,31 | -6.5030 | -6.5019 | -1.1447e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 7,11,24,25,28 | -6.5792 | -6.5780 | -1.1752e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_egg | 5 | 1,5,20,23,25 | -6.1272 | -6.1261 | -1.0791e-03 | full | full | 0 | 0 | 0 | 0 |
| conf_egg | 5 | 15,17,20,28,29 | -6.6174 | -6.6162 | -1.2254e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 1,9,20,26,31 | -6.4870 | -6.4859 | -1.1575e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_egg | 5 | 7,12,13,18,31 | -6.1223 | -6.1213 | -1.0470e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 4,7,12,24,30 | -6.4472 | -6.4482 | +1.0523e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 3,19,25,27,28 | -6.3498 | -6.3510 | +1.1808e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 3,12,24,28,30 | -6.2849 | -6.2861 | +1.1525e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 4,12,22,24,30 | -6.8195 | -6.8207 | +1.1272e-03 | full | full | 0 | 0 | 4 | 4 |
| conf_ellipsoid_z06 | 5 | 17,20,24,28,29 | -6.3923 | -6.3934 | +1.1125e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_ellipsoid_z06 | 5 | 6,14,17,30,31 | -6.4977 | -6.4965 | -1.1672e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 3,4,20,24,28 | -6.2234 | -6.2245 | +1.0231e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 4,11,12,27,30 | -6.9289 | -6.9300 | +1.0777e-03 | full | full | 0 | 0 | 5 | 5 |
| conf_ellipsoid_z06 | 5 | 3,6,24,25,27 | -6.7002 | -6.7013 | +1.1093e-03 | deficient | deficient | 1 | 1 | 4 | 4 |
| conf_ellipsoid_z06 | 5 | 16,17,23,29,31 | -6.1454 | -6.1443 | -1.0336e-03 | full | full | 0 | 0 | 1 | 1 |
| conf_ellipsoid_z06 | 5 | 6,22,25,27,29 | -6.6394 | -6.6404 | +1.0574e-03 | full | full | 0 | 0 | 3 | 3 |

→ Outliers arise from spec_cutoff=1e-8 boundary granularity
  (per-pixel albedo modulation shifts the smallest positive eigenvalue
  across the cutoff, see `r5/r5_p1_a_boundary_diagnostic.md`)

Gate criteria (R5-P1-A task book):
- PASS-A:        median(rho) >= 0.95  AND  median(top10 overlap) >= 0.80
- CONDITIONAL:   0.80 < median(rho) < 0.95
- FAIL-A:        median(rho) <= 0.80
