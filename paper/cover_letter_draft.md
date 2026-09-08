# Cover Letter（草稿 · 卡 F6）· IEEE Transactions on Computational Imaging

Dear Editor-in-Chief,

We submit our manuscript "From Calibrated to Uncalibrated Lighting: A Mode-Resolved
Information Continuum for Photometric Inverse Problems" for consideration in IEEE
Transactions on Computational Imaging.

**Problem.** Lighting calibration is never a binary variable. Between the calibrated
idealization and the uncalibrated idealization lies the regime where every real capture
system actually operates — and where the effective, *mode-resolved* information available
about a scene changes by orders of magnitude while any scalar summary (e.g., total Fisher
trace) changes by only a percent.

**Contributions** (worded per our claims registry; each statement is tied to an evidence
artifact and commit):

1. A calibration-confidence continuum for photometric inverse problems that maps physically
   parameterized lighting uncertainty to mode-resolved effective information, with a
   normalized retention spectrum for within-scene interpretation (verified to machine
   precision: 100/100 identity checks, dual-route cross-validation ≤2.2e-12).
2. An exact gauge-lifting spectral response and a scene-conditioned observability
   diagnostic λ⋆ with a closed-form predictor — validated across 54 stratified scenes
   (median log₁₀ prediction error 0.00000) — showing that gauge directions are lifted
   linearly in calibration confidence toward a scene-dependent observability floor.
3. A reproducible validation protocol connecting these predictions to finite-sample
   estimator covariance (variance ratios 0.977–1.009; 68/95% coverage within ±0.013) and
   to controlled corruption on OpenIllumination real data (test set of 11 frozen objects:
   Spearman 0.728, 95% CI [0.668, 0.783]), with explicit validity boundaries from mask
   changes (linearization valid below 0.8–3.0% mask-flip rate per scene family).

**Relation to prior art.** The photometric-stereo error-analysis lineage (Jiang & Bunke
1991; Kobayashi et al. 2011; Klaudiny & Hilton 2014; Chen et al. 2022; and the
inaccurate/semi-calibrated lines of Quéau et al. 2017 and Cho et al. TPAMI 2020) analyzes
how *given* calibration errors propagate to reconstruction. Our contribution is
complementary: a *continuous* confidence model with nuisance-marginalized, mode-resolved
information quantities and their validation against controlled real-data corruption. We
position the work explicitly with respect to Gupta et al. (TCI 2024, "Differentiable
Uncalibrated Imaging"), which addresses joint calibration/reconstruction; we instead
diagnose how calibration confidence changes the available information and its weak modes.

**Reproducibility.** A clean checkout reproduces every figure and table with one command
(`bash scripts/reproduce_paper.sh`); each figure carries provenance (commit SHA + artifact
hash); a clean-room re-run reproduces CI01/CI03 bit-exactly and Fig.3/4/7 byte-identically.
The repository, claims registry, and frozen artifacts are archived and referenced from the
manuscript.

The manuscript is 13 pages in the IEEE two-column format, with proofs, robustness
ablations, and the full validation ledger in the supplementary material. This work has not
been published elsewhere and is not under review at another venue.

Sincerely,
The Authors
