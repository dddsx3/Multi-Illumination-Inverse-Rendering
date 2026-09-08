# From Calibrated to Uncalibrated Lighting: A Mode-Resolved Information Continuum for Photometric Inverse Problems

**IEEE Transactions on Computational Imaging · manuscript v0.8（卡 C21 组装稿）**
Provenance：全部实验数字取自 `artifacts/frozen/`（每图/表 = make_figures.py --figure N 一键重建；
逐图 git SHA + artifact hash 见 `paper/provenance/`）。

---

## Abstract

Lighting calibration in photometric inverse problems is never a perfect binary variable, yet
classical analyses treat it as either known or unknown. We ask a quantitative version of this
question: how does the *level* of calibration confidence change the effective, mode-resolved
information available about a scene, as it varies continuously from calibrated to uncalibrated?
We instantiate the hybrid Cramér–Rao bound in the whitened observation space of multi-light
imaging and obtain (i) a **calibration-confidence continuum** — a Schur-complement effective
information ΔF(Λ) that is provably monotone in the relative calibration information Λ = σ²Σ_c⁻¹;
(ii) an **exact gauge-lifting spectral response** (Proposition 2) showing that otherwise
unobservable gauge directions are lifted linearly in calibration confidence before saturating,
together with a scene-conditioned observability diagnostic λ⋆ that is predictable from scene
quantities alone (a scene-conditioned model-internal diagnostic; 54/54 stratified scenes
in-band); and
(iii) a **mode-resolved validation protocol** connecting these predictions to finite-sample
estimator covariance (variance ratio 0.98–1.01, nominal coverage within ±0.01), to controlled
real-data corruption on OpenIllumination (pooled Spearman 0.728, object-cluster 95% CI
[0.705, 0.754], 11 held-out objects; descriptive — the load-bearing evidence is within-cell
mode ranking, R_A = 0.90), and to explicit validity boundaries from mask changes (linearization valid for
mask-flip rates up to 0.8–3.0% per scene family). Mode-resolved predictions detect weak-mode
damage that scalar trace summaries miss by three orders of magnitude; we report both the
successes and the quantitative boundaries of the approach.

## I. Introduction

1) **Calibration is never binary.** Every photometric capture pipeline carries residual
uncertainty in light direction and intensity — from motion, temperature, or synchronization.
Classical photometric stereo assumes calibrated point lights; uncalibrated methods absorb the
uncertainty into gauge freedoms. Neither extreme describes practice.

2) **Scalar reconstruction error cannot see weak modes.** In our setting the ratio of
calibrated to uncalibrated total (trace) information is **1.01**, while the weakest-mode
condition number spans 10⁻⁷: nearly all damage from unknown lighting concentrates in a few
weakly-informed directions that any trace-level summary averages away (Fig.4, trace panel).
Predicting reconstruction quality therefore requires mode-resolved readouts, not scalars; throughout, trace-type summaries appear only as counterexamples.

3) **This paper.** We treat the calibration level as a continuous random variable δc ~ N(0, Σ_c)
— with Σ_c in physical units — and characterize, per mode, how the resulting hybrid estimation
problem transfers information from nuisance to parameters.

4) **Contributions** (exactly three, per the frozen claims registry):
   - C1–C5: A calibration-confidence continuum for photometric inverse problems that maps
     physically parameterized lighting uncertainty to mode-resolved effective information,
     with a normalized retention spectrum for within-scene interpretation (Lemma 1 + Lemma 2);
   - C3–C4: An exact gauge-lifting spectral response and a scene-conditioned observability
     diagnostic λ⋆ with a closed-form predictor valid in the small-λ regime (Proposition 2 +
     Definition; CI02: 54/54 scenes in the linear band);
   - C2/C6: A reproducible validation protocol connecting information prediction to
     finite-sample estimator covariance (CI03) and to controlled real-data corruption
     (CI04), including explicit validity boundaries from mask changes and model mismatch
     (CI03-nl, CI05).

5) **Results preview.** Fig.3 shows the analytic gauge-lifting law; Fig.5 the finite-sample
estimator calibration; Fig.7 the controlled real-data validation; Fig.6 the validity envelope.

## II. Related Work

**A. Calibrated / inaccurate / uncalibrated photometric stereo.** Calibrated PS [Shi et al.
2016, DiLiGenT] fixes light directions and intensities from hardware. Uncalibrated PS absorbs
them into gauge freedoms (the classic ρ-scale and light-baseline ambiguities; for modern
treatments see [Quéau et al. 2017]). A middle line studies *inaccurate* lighting: Quéau et al.
jointly refine light intensity and direction under a variational robustness model. **These works
reconstruct; we characterize information.** Our λ⋆ diagnostic answers, before any reconstruction,
which inaccuracies are survivable for a given scene.

**B. Hybrid estimation and mixed linear models.** The hybrid Cramér–Rao bound [Noam & Messer
2008] gives tight bounds when deterministic parameters coexist with random nuisances; the
linear mixed-model literature (Henderson's mixed model equations; Harville 1977) provides the
estimator-side counterpart. Lemma 1 and Proposition 1 recall and instantiate these results in
the whitened photometric setting — we claim no new information geometry (Table II).

**C. Measurement-uncertainty propagation.** Metrology's GUM framework (JCGM 100; GUM
Supplement 1 Monte Carlo propagation) formalizes how calibration uncertainty propagates to
measurands. We are not aware of prior work turning this into a per-mode information prediction
for photometric stereo; our Related Work makes this lineage explicit.

**D. Inverse-rendering information and identifiability.** Recent inverse-rendering analyses
derive identifiability conditions and degeneracy structure; our continuum complements them by
grading identifiability by calibration confidence.

## III. Problem Formulation

Locally linearized multi-light observation (whitened form; all later formulas in this space):

  y = A x + B δc + ε,   ε ~ N(0, σ²I),   δc ~ N(0, Σ_c),   Λ = σ²Σ_c⁻¹,

with A ∈ R^{m×n} the albedo (per-pixel) design, B ∈ R^{m×q} the nuisance (lighting) design
from the SH-9 basis and mask, Σ_c the physical-unit prior (log-intensity variance and angular
variance in radians²; never a raw-coordinate λI — the whitened/physical parameterization is
essential: same nominal λ in raw vs whitened coordinates changes weak-mode predictions by ~3–10%,
Fig.9 panel 1). Parameterization: δc = Σ₀^{1/2} z with τ = z-precision as the user-facing axis.
The two readouts are **strictly separated** (Table I of the supplement): absolute information
spectrum eig(ΔF(τ)) for cross-τ gauge analysis and λ⋆; normalized retention
R(τ) = F∞^{-1/2}ΔF(τ)F∞^{-1/2} ∈ [0,1] only for within-scene interpretation. Mask fixed
(terminator flips treated explicitly in §VI).

## IV. Calibration-Confidence Information Continuum

**Lemma 1 (recall + instantiate).** ΔF(Λ) = Aᵀ[I − B(BᵀB+Λ)⁻¹Bᵀ]A is the Schur complement of
the hybrid information matrix; 0 ≼ Λ₁ ≼ Λ₂ ⇒ ΔF(Λ₁) ≼ ΔF(Λ₂) ≼ F∞ = AᵀA; at Λ = 0 the
Moore–Penrose limit via thin-SVD applies (implementation: lstsq/SVD path only; `solve` is
forbidden and statically asserted). All identities verified on 100 stratified random systems:
dual-route element-wise error ≤ 2.2e-12 (gate 1e-10), parameterization invariance ≤ 2.2e-12,
scale invariance (σ,Σ_c) → (kσ, k²Σ_c) exact to 1.7e-15, retention bounds 100/100 (CI01,
Fig.1). **Lemma 2 (bounds).** R(Λ) ∈ [0, I] on the identifiable subspace range(F∞), via
congruence transform; the positive square root is constructed by eigendecomposition (the
F∞^{-1/4} trap that produces a spurious 0.716 upper bound is caught by the dual-route
generalized-eigenvalue cross-check, max deviation 1.6e-15 over 54 scenes).

## V. Gauge Lifting and Scene-Conditioned Observability

**Proposition 2 (exact spectral response).** If Aa = Bc̄ (gauge identity) and Λ = λI with
B = USVᵀ, α = Vᵀc̄, then aᵀΔF(λ)a = Σᵢ αᵢ²sᵢ²λ/(sᵢ²+λ): linear lifting λΣαᵢ² as λ→0,
saturation to ‖Aa‖² as λ→∞. Verified against direct Rayleigh quotients on 54 stratified
scenes × 25-decade grids: well-conditioned region p50 3.1e-09 (gate 1e-8); the deep-uncalibrated
end is governed by a documented cancellation floor C·ε·sat/|direct| (Fig.3). In the theory
chapter the general parallel-sum form is written as S:Λ = S(S+Λ)⁺Λ (Anderson & Duffin 1969),
valid for positive-semidefinite, possibly rank-deficient operands; the subtractive form
S − S(S+Λ)⁻¹S is used only where Λ ≻ 0 is explicitly stated. λ⋆ retains its role as a
model-internal consistency diagnostic (closed-form root and linear predictor derive from the
same analytic curve); it is not reported as an externally validated prediction accuracy.
**Definition
(diagnostic λ⋆).** The gauge curve crosses the scene's intrinsic weak-mode floor μ_floor at
λ⋆, predictable in the linear band by λ⋆ ≈ μ_floor/Σαᵢ²; applicability requires
λ⋆ ≪ min_{αᵢ≠0} sᵢ². Across 54 scenes the closed-form bisection root and the linear predictor
agree to within grid resolution across all 54 in-band scenes (worst λ⋆/min sᵢ² = 0.99); as both the root and the predictor derive from the same analytic curve, this agreement is a model-internal consistency diagnostic rather than an external prediction-accuracy claim.
λ⋆ is defined only in absolute information units; in the retention metric the gauge and
weakest modes provably never cross (deficit-coefficient ordering 52.4 < 65.5).

## VI. Estimator Calibration and Experimental Protocol

**Proposition 1 (finite-sample identity).** For the profiled joint MAP/GLS estimator
x̂ = ΔF(Λ)⁻¹AᵀM(Λ)y, under joint sampling of δc and ε: E[x̂] = x, Cov(x̂) = σ²ΔF(Λ)⁻¹.
This identity holds under four explicit conditions: (1) the affine-Gaussian model is
correctly specified; (2) A, B, Σ_c and σ² are the true and known quantities; (3) δc ~
N(0,Σ_c) and ε ~ N(0,σ²I) are independently re-drawn per trial; (4) the matched
GLS/joint-MAP estimator is used. It is an exact sampling covariance under the stated
joint ensemble; it is not a universal bound under model mismatch, fixed corruption,
nonlinear mask changes, or unmodeled real calibration error.
The fixed-δc control exhibits closed-form bias ΔF⁻¹AᵀMBδ̄ (reproduction 0.96–1.01) and a
systematically *lower* conditional variance σ²ΔF⁻¹AᵀM²AΔF⁻¹ (0.98–1.03) — fixed-δc ensembles
underestimate uncertainty and are never used for tightness claims. **CI03 (Gate B)**: mode-
resolved variance ratios 0.977–1.009 and 68%/95% coverage within ±0.013 across 18 scene-cases
incl. a 10k-trial high-conditioning stratum (Fig.5). **Mode tracking**: cross-λ identities are
carried by |VᵀW| assignment (Hungarian + degeneracy flags), never by eigenvalue index; the
N=1 rotation is 0.00° and N=3 reaches 25.9° in our scene families (Fig.4). **Validity
envelope (R-B)**: with analytic SH+ReLU re-rendering, prediction error stays <10% for
mask-flip rates up to 0.8% (bumpy) / 3.0% (sphere) / 2.15% (terminator-heavy), rising
monotonically to ~60% at 20% flips (Fig.6) — real-data claims are restricted to the low-flip
region.

## VII. Synthetic Experiments (CI01–CI03)

Table IV (synthetic validation): closed-form errors ≤ 2.2e-12; MC variance ratios 0.977–1.009;
coverage 68% within ±0.013; λ⋆ consistency: 54/54 in-band scenes (model-internal diagnostic);
retention dual-route 1.6e-15. Ablations (Fig.9, Table VI): whitened vs raw λI changes weak-mode
readouts by median 0.034 (IQR 0.025–0.049) — a first-order effect, consistent with the
parameterization-invariance analysis; Λ 4× misspecification moves the weak-mode trace by
−74.8% (single-scene second-order effect, not generalized); light count 1→3 lifts the weakest
mode by 10^2.8; heteroscedastic-vs-homoscedastic whitening perturbs retention by median 0.008.

## VIII. Controlled Real-Data Validation (CI04)

Protocol (estimator-side corruption; pre-registered): the OpenIllumination light dome GT
(positions; 142 OLAT lights) is the truth; δc ~ N(0, Σ_c) in physical units is injected into the
*estimator's assumed* calibration; the nominal Lambertian calibration (ρ̂, n̂; 48-light analysis
subset, 1200-pixel subsample) defines the linearization; prediction = weak-5-mode degradation
λ_j(F∞)/λ_j(ΔF(Σ_c)) computed **before** any corruption; empirical = the same whitened
per-pixel GLS re-run with corrupted calibration, scale-gauge aligned, normalized by a
residual-bootstrap control arm. Test set: 11 objects frozen before the run (one excluded for a
dataset-side missing layer, documented). **Result (Gate D)**: pooled Spearman 0.728, object-cluster bootstrap 95% CI [0.705, 0.754] —
descriptive, partly driven by the shared corruption-level axis — with every per-object
coefficient positive in [0.72, 0.80] (Fig.7, Table V). The pre-registered confirmatory
re-analysis (CI04-R) separates the two load-bearing questions. *Within-cell mode ranking*
(承重墙): R_A = 0.90, object-cluster 95% CI [0.90, 0.95], 66/66 cells and 11/11 objects
positive — the mode-resolved retention scores consistently rank the relative degradation of
tracked modes within the same object and corruption level. *Fixed-level severity* (co-primary,
fair comparison against scalar baselines): the mode-resolved score matches but does not exceed
the best scalar baseline (Δ = +0.00, cluster 95% CI [−0.33, 0.00] versus E-min). This
degeneracy is structural: the tracked modes are, by construction, the bottom eigenvectors of
the same retention operator, so the E-optimality worst-mode score coincides exactly with the
mode-resolved score (max deviation 1.8e-15 over all 66 cells). Against the remaining scalar
baselines the stratified medians are 0.536 (mode-resolved) versus 0.400 (trace) and 0.418
(log-determinant) — a descriptive ordering without a pre-registered interval, reported without
significance claims. One boundary on reading identity tracking: the empirically worst mode is
the bottom tracked mode in 60 of 66 cells, and only 4 of 11 objects change their worst-mode
identity across corruption levels — the ranking evidence, rather than rich identity dynamics,
carries the identify-and-track wording. The locked Branch-B wording: scalar worst-mode
criteria such as E-optimality can summarize degradation severity nearly as well as the
mode-resolved score; the additional value of the mode-resolved spectrum is not a universally
better scalar predictor, but the ability to identify, track, and interpret which directions
become vulnerable as calibration confidence changes. Magnitude-level agreement is limited by
Σ_c_real and
model mismatch (~10², the empirical form of risks R-A/R-D): absolute magnitudes remain
descriptive, and the claim is never reduced to average-MAE monotonicity.

## IX. External Sanity, Robustness, and Limitations (CI05)

DiLiGenT (10 objects, geometry-known calibration side only): weak-mode structure exists in
all objects (floor/median ∈ [2.7e-3, 7.1e-3]); λ⋆ finite everywhere; the failure taxonomy places all objects in the band where the calibration-associated term is substantially larger than the residual mismatch term under this diagnostic decomposition (56–1131× across objects; median 273×); this comparison is descriptive and must not be read as a causal decomposition of real reconstruction error. Trace-like aggregate summaries can dilute localized information loss; E-optimality summarizes worst-mode severity, whereas the mode-resolved spectrum is what identifies, tracks, and interprets vulnerable directions individually. We do not claim mode-resolved superiority for scalar severity prediction, and we do not attribute natural non-Lambertian residuals to calibration uncertainty (constitution-level
separation; Fig.8). **Limitations** (explicit): (i) magnitude-level real-data agreement is
bounded by Σ_c_real and BRDF/shadow mismatch; (ii) the Λ-misspecification sensitivity is a
single-scene second-order observation; (iii) DiLiGenT serves as sanity only; (iv) mask/terminator
flips bound all continuum claims (Fig.6); (v) near-field structure mismatch is outside the
causal chain by design.

## X. Discussion

When is more calibration worth acquiring? The continuum gives a per-mode answer: directions
whose lifted information remains below μ_floor are unrecoverable regardless of effort below
λ⋆; above λ⋆, calibration investment transfers linearly until saturation. Trace-like
aggregate summaries can strongly dilute localized information loss: the
calibrated/uncalibrated trace ratio is 1.01 while weak modes span 10⁻⁷ in conditioning.
This is a statement about trace-type dilution, not about scalar summaries in general — in
the controlled real-data experiment, E-optimality and log-det baselines reach pooled
Spearman 0.87/0.86, and the E-optimality worst-mode score coincides exactly with the
mode-resolved one (structurally: the tracked modes are the bottom eigenvectors of the same
retention operator). What scalar summaries cannot provide is the identity, tracking, and
interpretation of vulnerable directions — the contribution the mode-resolved spectrum
retains under Branch B. The framework — whitened
hybrid information + gauge spectral response + continuity tracking — transfers to any inverse
problem with block-random calibration (camera color response, projector geometry, spectral
basis calibration).

## XI. Conclusion

A mode-resolved calibration-confidence continuum, an exact gauge-lifting law with a predictable
scene-conditioned crossover, and a validation protocol reaching controlled real data with
explicit validity boundaries. All figures/tables regenerate from frozen artifacts with one
command per figure; the claims registry ties each statement to its evidence artifact and commit.

## References (anchor list; full bibliography in supplement)

[1] Noam & Messer, *The hybrid Cramér–Rao bound and the generalized Gaussian linear estimation
problem*, IEEE SAM/TSP 2008.
[2] Harville, *Maximum likelihood approaches to variance component estimation*, 1977
(Henderson's mixed model equations lineage).
[3] JCGM 100:2008 (GUM); JCGM 101:2008 (Monte Carlo propagation).
[4] Shi et al., *DiLiGenT: A photometric stereo dataset*, CVPR 2016.
[5] Liu et al., *OpenIllumination*, NeurIPS 2023 (CC BY 4.0).
[6] Quéau et al., *Photometric stereo under inaccurate lighting*, CVPR 2017.
[7] IEEE TCI scope & reproducibility guidance.

---

*Provenance: frozen summaries `artifacts/frozen/ci0{1,2,3,3nl,4,5}*_summary.json` + manifests;
figure recipes `scripts/make_figures.py --figure 1..9`; claims registry `CLAIMS_REGISTRY.yaml`
(C1–C6 with artifact + commit pointers).*
