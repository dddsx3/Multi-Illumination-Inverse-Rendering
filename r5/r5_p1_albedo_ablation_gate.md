# R5-P1-A Gate · smoke (RTX 5070 Ti)

Scenes: ['conf_sphere_r05']

NS: [3, 5]

Per-(scene, N) ranking diagnostics:

| scene | N | rho | top10 | top20 | O_mean | A_mean | n_deficient_O | n_deficient_A | n_at_cutoff_O_med | n_at_cutoff_A_med |
|---|---|---|---|---|---|---|---|---|---|---|
| conf_sphere_r05 | 3 | 1.0000 | 0.998 | 0.999 | -6.779 | -6.779 | 1020 | 1018 | 3 | 3 |
| conf_sphere_r05 | 5 | 1.0000 | 1.000 | 1.000 | -6.589 | -6.589 | 8 | 9 | 0 | 0 |

**median rho** = 1.0000

**median top10 overlap** = 0.999

**median top20 overlap** = 0.999

## Gate verdict: PASS-A

Next step: freeze a=1; P1-B normal/light proxy audit

## Boundary outliers (|ΔI| > 1e-03)

Total: 103 subsets (1.996% of all subsets)

| scene | N | subset | I_O | I_A | diff | status_O | status_A | d_extra_O | d_extra_A | n_at_cutoff_O | n_at_cutoff_A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conf_sphere_r05 | 3 | 1,2,17 | -7.3803 | -7.3573 | -2.2991e-02 | deficient | deficient | 2 | 3 | 10 | 9 |
| conf_sphere_r05 | 3 | 1,4,7 | -8.0910 | -8.0899 | -1.0832e-03 | full | full | 0 | 0 | 19 | 19 |
| conf_sphere_r05 | 3 | 1,7,12 | -8.0010 | -8.0000 | -1.0240e-03 | deficient | deficient | 1 | 1 | 21 | 21 |
| conf_sphere_r05 | 3 | 1,7,17 | -7.8356 | -7.8345 | -1.1117e-03 | deficient | deficient | 1 | 1 | 18 | 18 |
| conf_sphere_r05 | 3 | 1,10,17 | -6.9520 | -6.9287 | -2.3249e-02 | full | deficient | 0 | 1 | 5 | 4 |
| conf_sphere_r05 | 3 | 1,12,28 | -7.7261 | -7.7482 | +2.2067e-02 | deficient | full | 1 | 0 | 21 | 22 |
| conf_sphere_r05 | 3 | 1,12,30 | -8.0295 | -8.0524 | +2.2879e-02 | deficient | deficient | 2 | 1 | 19 | 20 |
| conf_sphere_r05 | 3 | 1,17,22 | -8.1463 | -8.1205 | -2.5762e-02 | deficient | deficient | 3 | 4 | 19 | 18 |
| conf_sphere_r05 | 3 | 1,17,27 | -8.0893 | -8.0683 | -2.1027e-02 | deficient | deficient | 2 | 3 | 21 | 20 |
| conf_sphere_r05 | 3 | 1,19,25 | -7.6705 | -7.6493 | -2.1213e-02 | deficient | deficient | 1 | 2 | 15 | 14 |
| conf_sphere_r05 | 3 | 2,4,12 | -7.5756 | -7.5988 | +2.3225e-02 | deficient | full | 1 | 0 | 9 | 10 |
| conf_sphere_r05 | 3 | 2,12,23 | -7.3434 | -7.3659 | +2.2528e-02 | deficient | full | 1 | 0 | 5 | 6 |
| conf_sphere_r05 | 3 | 2,14,17 | -7.1459 | -7.1225 | -2.3388e-02 | deficient | deficient | 3 | 4 | 4 | 3 |
| conf_sphere_r05 | 3 | 4,6,11 | -7.5056 | -7.4832 | -2.2430e-02 | deficient | deficient | 1 | 2 | 13 | 12 |
| conf_sphere_r05 | 3 | 4,6,19 | -7.7068 | -7.6842 | -2.2554e-02 | deficient | deficient | 1 | 2 | 13 | 12 |
| conf_sphere_r05 | 3 | 4,6,24 | -7.1599 | -7.1366 | -2.3267e-02 | deficient | deficient | 1 | 2 | 7 | 6 |
| conf_sphere_r05 | 3 | 4,6,29 | -6.8663 | -6.8427 | -2.3515e-02 | full | deficient | 0 | 1 | 2 | 1 |
| conf_sphere_r05 | 3 | 4,7,9 | -8.0483 | -8.0472 | -1.1625e-03 | full | full | 0 | 0 | 17 | 17 |
| conf_sphere_r05 | 3 | 4,7,12 | -8.2868 | -8.2856 | -1.2044e-03 | deficient | deficient | 3 | 3 | 23 | 23 |
| conf_sphere_r05 | 3 | 4,7,17 | -8.1936 | -8.1923 | -1.3662e-03 | deficient | deficient | 1 | 1 | 20 | 20 |
| conf_sphere_r05 | 3 | 4,7,22 | -8.0398 | -8.0387 | -1.1646e-03 | deficient | deficient | 1 | 1 | 18 | 18 |
| conf_sphere_r05 | 3 | 4,9,17 | -8.2045 | -8.2035 | -1.0444e-03 | deficient | deficient | 2 | 2 | 9 | 9 |
| conf_sphere_r05 | 3 | 4,9,30 | -8.2341 | -8.2552 | +2.1164e-02 | deficient | deficient | 3 | 2 | 23 | 24 |
| conf_sphere_r05 | 3 | 4,12,15 | -7.7906 | -7.8128 | +2.2228e-02 | deficient | full | 1 | 0 | 12 | 13 |
| conf_sphere_r05 | 3 | 4,12,17 | -8.2058 | -8.2043 | -1.4997e-03 | deficient | deficient | 1 | 1 | 14 | 14 |
| conf_sphere_r05 | 3 | 4,17,20 | -8.1881 | -8.2095 | +2.1391e-02 | deficient | deficient | 4 | 3 | 23 | 24 |
| conf_sphere_r05 | 3 | 4,17,22 | -8.2319 | -8.2551 | +2.3260e-02 | deficient | deficient | 4 | 3 | 20 | 21 |
| conf_sphere_r05 | 3 | 4,17,25 | -8.3154 | -8.3376 | +2.2211e-02 | deficient | deficient | 3 | 2 | 23 | 24 |
| conf_sphere_r05 | 3 | 4,17,30 | -8.2026 | -8.2239 | +2.1313e-02 | deficient | deficient | 4 | 3 | 22 | 23 |
| conf_sphere_r05 | 3 | 4,17,31 | -7.0926 | -7.1154 | +2.2737e-02 | deficient | full | 1 | 0 | 3 | 4 |
| conf_sphere_r05 | 3 | 6,7,12 | -7.4425 | -7.4413 | -1.2152e-03 | deficient | deficient | 1 | 1 | 15 | 15 |
| conf_sphere_r05 | 3 | 6,7,14 | -7.4058 | -7.4047 | -1.0892e-03 | deficient | deficient | 1 | 1 | 13 | 13 |
| conf_sphere_r05 | 3 | 6,7,17 | -7.5627 | -7.5610 | -1.6145e-03 | deficient | deficient | 2 | 2 | 18 | 18 |
| conf_sphere_r05 | 3 | 6,7,22 | -7.4058 | -7.4281 | +2.2253e-02 | deficient | deficient | 2 | 1 | 13 | 14 |
| conf_sphere_r05 | 3 | 6,9,12 | -7.9176 | -7.9164 | -1.2511e-03 | deficient | deficient | 3 | 3 | 21 | 21 |
| conf_sphere_r05 | 3 | 6,9,17 | -7.9345 | -7.9331 | -1.4118e-03 | deficient | deficient | 3 | 3 | 17 | 17 |
| conf_sphere_r05 | 3 | 6,11,12 | -7.3276 | -7.3050 | -2.2688e-02 | deficient | deficient | 2 | 3 | 10 | 9 |
| conf_sphere_r05 | 3 | 6,11,14 | -7.5919 | -7.6140 | +2.2161e-02 | deficient | deficient | 2 | 1 | 15 | 16 |
| conf_sphere_r05 | 3 | 6,11,17 | -7.4772 | -7.4542 | -2.2965e-02 | deficient | deficient | 1 | 2 | 12 | 11 |
| conf_sphere_r05 | 3 | 6,11,20 | -7.1331 | -7.1096 | -2.3442e-02 | full | deficient | 0 | 1 | 9 | 8 |
| conf_sphere_r05 | 3 | 6,11,28 | -7.0114 | -6.9884 | -2.3017e-02 | full | deficient | 0 | 1 | 6 | 5 |
| conf_sphere_r05 | 3 | 6,12,19 | -7.5358 | -7.5121 | -2.3711e-02 | deficient | deficient | 2 | 3 | 16 | 15 |
| conf_sphere_r05 | 3 | 6,12,27 | -7.6135 | -7.5898 | -2.3677e-02 | deficient | deficient | 2 | 3 | 18 | 17 |
| conf_sphere_r05 | 3 | 6,12,29 | -6.8781 | -6.8544 | -2.3670e-02 | full | deficient | 0 | 1 | 3 | 2 |
| conf_sphere_r05 | 3 | 6,14,17 | -7.9001 | -7.8990 | -1.1804e-03 | deficient | deficient | 3 | 3 | 18 | 18 |
| conf_sphere_r05 | 3 | 6,14,27 | -7.7011 | -7.7249 | +2.3792e-02 | deficient | deficient | 2 | 1 | 16 | 17 |
| conf_sphere_r05 | 3 | 6,17,19 | -7.7024 | -7.6786 | -2.3852e-02 | deficient | deficient | 3 | 4 | 16 | 15 |
| conf_sphere_r05 | 3 | 6,17,20 | -7.5871 | -7.5625 | -2.4640e-02 | deficient | deficient | 3 | 4 | 20 | 19 |
| conf_sphere_r05 | 3 | 6,17,22 | -7.7925 | -7.7915 | -1.0162e-03 | deficient | deficient | 4 | 4 | 17 | 17 |
| conf_sphere_r05 | 3 | 6,17,25 | -7.7450 | -7.7204 | -2.4648e-02 | deficient | deficient | 3 | 4 | 20 | 18 |
| conf_sphere_r05 | 3 | 6,17,27 | -7.7464 | -7.7222 | -2.4158e-02 | deficient | deficient | 4 | 5 | 16 | 15 |
| conf_sphere_r05 | 3 | 6,17,30 | -7.6801 | -7.6789 | -1.1261e-03 | deficient | deficient | 5 | 5 | 17 | 17 |
| conf_sphere_r05 | 3 | 6,19,25 | -7.4409 | -7.4171 | -2.3850e-02 | deficient | deficient | 2 | 3 | 12 | 11 |
| conf_sphere_r05 | 3 | 6,20,25 | -7.4443 | -7.4432 | -1.1033e-03 | deficient | deficient | 3 | 3 | 17 | 17 |
| conf_sphere_r05 | 3 | 6,27,28 | -7.2146 | -7.1908 | -2.3796e-02 | deficient | deficient | 1 | 2 | 6 | 5 |
| conf_sphere_r05 | 3 | 7,9,12 | -8.0112 | -8.0100 | -1.2371e-03 | full | full | 0 | 0 | 19 | 19 |
| conf_sphere_r05 | 3 | 7,9,17 | -7.8678 | -7.8666 | -1.2308e-03 | deficient | deficient | 1 | 1 | 16 | 16 |
| conf_sphere_r05 | 3 | 7,9,20 | -7.8851 | -7.8837 | -1.3748e-03 | deficient | deficient | 2 | 2 | 17 | 17 |
| conf_sphere_r05 | 3 | 7,10,15 | -7.2975 | -7.3204 | +2.2854e-02 | deficient | deficient | 2 | 1 | 11 | 12 |
| conf_sphere_r05 | 3 | 7,12,17 | -8.0554 | -8.0539 | -1.5491e-03 | deficient | deficient | 2 | 2 | 21 | 21 |
| conf_sphere_r05 | 3 | 7,12,20 | -7.9906 | -7.9892 | -1.4293e-03 | deficient | deficient | 2 | 2 | 21 | 21 |
| conf_sphere_r05 | 3 | 7,12,22 | -7.7435 | -7.7423 | -1.1419e-03 | deficient | deficient | 2 | 2 | 16 | 16 |
| conf_sphere_r05 | 3 | 7,14,17 | -7.6793 | -7.6778 | -1.5002e-03 | deficient | deficient | 3 | 3 | 14 | 14 |
| conf_sphere_r05 | 3 | 7,17,20 | -7.8932 | -7.8909 | -2.3077e-03 | deficient | deficient | 3 | 3 | 20 | 20 |
| conf_sphere_r05 | 3 | 7,17,22 | -7.7733 | -7.7718 | -1.4097e-03 | deficient | deficient | 3 | 3 | 17 | 17 |
| conf_sphere_r05 | 3 | 7,20,25 | -7.8006 | -7.7995 | -1.0970e-03 | deficient | deficient | 3 | 3 | 24 | 24 |
| conf_sphere_r05 | 3 | 9,10,31 | -6.8603 | -6.8836 | +2.3281e-02 | deficient | full | 1 | 0 | 4 | 4 |
| conf_sphere_r05 | 3 | 9,12,17 | -8.1245 | -8.1234 | -1.0490e-03 | deficient | deficient | 2 | 2 | 16 | 16 |
| conf_sphere_r05 | 3 | 9,15,17 | -7.4118 | -7.3872 | -2.4570e-02 | deficient | deficient | 1 | 2 | 8 | 7 |
| conf_sphere_r05 | 3 | 9,17,23 | -7.1709 | -7.1469 | -2.4070e-02 | deficient | deficient | 1 | 2 | 4 | 3 |
| conf_sphere_r05 | 3 | 9,17,25 | -8.1136 | -8.0878 | -2.5811e-02 | deficient | deficient | 3 | 4 | 22 | 21 |
| conf_sphere_r05 | 3 | 9,17,31 | -7.0321 | -7.0310 | -1.0928e-03 | deficient | deficient | 1 | 1 | 2 | 2 |
| conf_sphere_r05 | 3 | 9,18,22 | -6.7706 | -6.7467 | -2.3912e-02 | full | deficient | 0 | 1 | 6 | 4 |
| conf_sphere_r05 | 3 | 9,25,27 | -7.9703 | -7.9919 | +2.1590e-02 | deficient | deficient | 4 | 3 | 21 | 22 |
| conf_sphere_r05 | 3 | 11,14,22 | -7.5130 | -7.5358 | +2.2776e-02 | deficient | deficient | 2 | 1 | 13 | 14 |
| conf_sphere_r05 | 3 | 11,20,30 | -7.0744 | -7.0975 | +2.3053e-02 | deficient | full | 1 | 0 | 6 | 7 |
| conf_sphere_r05 | 3 | 12,17,19 | -7.5106 | -7.5093 | -1.2621e-03 | deficient | deficient | 2 | 2 | 13 | 13 |
| conf_sphere_r05 | 3 | 12,17,20 | -7.9582 | -7.9568 | -1.3912e-03 | deficient | deficient | 4 | 4 | 19 | 18 |
| conf_sphere_r05 | 3 | 12,17,25 | -8.0700 | -8.0688 | -1.1910e-03 | deficient | deficient | 3 | 3 | 20 | 20 |
| conf_sphere_r05 | 3 | 12,17,30 | -7.9727 | -7.9716 | -1.1212e-03 | deficient | deficient | 3 | 3 | 21 | 21 |
| conf_sphere_r05 | 3 | 12,20,25 | -7.9111 | -7.9360 | +2.4910e-02 | deficient | deficient | 5 | 4 | 25 | 26 |
| conf_sphere_r05 | 3 | 12,20,30 | -7.7910 | -7.8149 | +2.3877e-02 | deficient | deficient | 4 | 3 | 23 | 24 |
| conf_sphere_r05 | 3 | 12,22,30 | -7.8296 | -7.8550 | +2.5426e-02 | deficient | deficient | 3 | 2 | 21 | 23 |
| conf_sphere_r05 | 3 | 14,17,22 | -7.9015 | -7.9003 | -1.2289e-03 | deficient | deficient | 5 | 5 | 16 | 16 |
| conf_sphere_r05 | 3 | 14,17,28 | -7.5509 | -7.5276 | -2.3341e-02 | deficient | deficient | 5 | 6 | 12 | 11 |
| conf_sphere_r05 | 3 | 15,17,20 | -7.6207 | -7.6197 | -1.0234e-03 | deficient | deficient | 2 | 2 | 10 | 10 |
| conf_sphere_r05 | 3 | 15,17,22 | -7.4085 | -7.4074 | -1.0693e-03 | deficient | deficient | 1 | 1 | 8 | 7 |
| conf_sphere_r05 | 3 | 15,18,20 | -6.9722 | -6.9712 | -1.0334e-03 | deficient | deficient | 1 | 1 | 6 | 6 |
| conf_sphere_r05 | 3 | 15,20,26 | -6.8665 | -6.8655 | -1.0834e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_sphere_r05 | 3 | 17,18,20 | -6.8826 | -6.8813 | -1.3178e-03 | full | full | 0 | 0 | 6 | 6 |
| conf_sphere_r05 | 3 | 17,18,25 | -6.8390 | -6.8379 | -1.1270e-03 | deficient | deficient | 1 | 1 | 5 | 5 |
| conf_sphere_r05 | 3 | 17,18,28 | -6.9007 | -6.8996 | -1.0169e-03 | deficient | deficient | 1 | 1 | 5 | 5 |
| conf_sphere_r05 | 3 | 17,19,20 | -7.3899 | -7.3887 | -1.1626e-03 | deficient | deficient | 3 | 3 | 12 | 12 |
| conf_sphere_r05 | 3 | 17,20,25 | -7.8632 | -7.8620 | -1.2046e-03 | deficient | deficient | 6 | 6 | 24 | 24 |
| conf_sphere_r05 | 3 | 17,20,26 | -6.8001 | -6.7989 | -1.1205e-03 | full | full | 0 | 0 | 2 | 2 |
| conf_sphere_r05 | 3 | 17,20,28 | -7.7650 | -7.7639 | -1.1475e-03 | deficient | deficient | 5 | 5 | 18 | 18 |
| conf_sphere_r05 | 3 | 17,22,25 | -7.8373 | -7.8363 | -1.0139e-03 | deficient | deficient | 5 | 5 | 16 | 16 |
| conf_sphere_r05 | 3 | 17,25,28 | -7.7081 | -7.7304 | +2.2285e-02 | deficient | deficient | 5 | 4 | 17 | 18 |
| conf_sphere_r05 | 3 | 17,25,30 | -7.7962 | -7.7950 | -1.2580e-03 | deficient | deficient | 4 | 4 | 16 | 16 |
| conf_sphere_r05 | 3 | 18,20,25 | -6.8965 | -6.9187 | +2.2157e-02 | deficient | full | 1 | 0 | 6 | 6 |
| conf_sphere_r05 | 3 | 18,20,30 | -6.8847 | -6.8836 | -1.0509e-03 | full | full | 0 | 0 | 7 | 7 |
| conf_sphere_r05 | 3 | 20,26,28 | -6.8971 | -6.8960 | -1.0990e-03 | full | full | 0 | 0 | 3 | 3 |
| conf_sphere_r05 | 5 | 10,15,17,18,22 | -6.9011 | -6.8772 | -2.3810e-02 | full | deficient | 0 | 1 | 6 | 5 |

→ Outliers arise from spec_cutoff=1e-8 boundary granularity
  (per-pixel albedo modulation shifts the smallest positive eigenvalue
  across the cutoff, see `r5/r5_p1_a_boundary_diagnostic.md`)

Gate criteria (R5-P1-A task book):
- PASS-A:        median(rho) >= 0.95  AND  median(top10 overlap) >= 0.80
- CONDITIONAL:   0.80 < median(rho) < 0.95
- FAIL-A:        median(rho) <= 0.80
