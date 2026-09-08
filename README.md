# calibinfo · Calibration-Confidence Continuum for Photometric Inverse Problems

> **From Calibrated to Uncalibrated Lighting: A Mode-Resolved Information Continuum
> for Photometric Inverse Problems** — IEEE Transactions on Computational Imaging
> (submission candidate v1.0, 2026-09-08).

Given the same scene captured under many lights, how much usable information about the scene
survives as lighting calibration degrades continuously from perfect to unknown — *per mode*,
not as a single scalar? This repository answers that question with frozen theory, a
pre-registered validation protocol, and a fully reproducible evidence chain.

**One-line story.** In whitened observation space, calibration uncertainty acts as a Gaussian
nuisance; marginalizing it yields a Schur-complement information continuum ΔF(Λ) that is
provably monotone in the relative calibration confidence Λ = σ²Σ_c⁻¹. Gauge directions —
exactly unobservable when calibration is unknown — are lifted *linearly* in confidence
(Prop. 2) until they cross the scene's intrinsic weak-mode floor at a predictable crossover
λ⋆ ≈ μ_floor/Σαᵢ². Scalar summaries are structurally blind to this: the
calibrated/uncalibrated trace ratio is 1.01 while weak-mode conditioning spans 10⁻⁷.

## Repository layout

```
├── CLAIMS_REGISTRY.yaml      # every claim C1–C6 ↔ allowed/forbidden wording ↔ evidence artifact + commit
├── STATE.md                  # progress ledger (C01–C23 execution cards + F1–F6 closure stage) — single source of truth
├── PARKING_LOT.md            # idea parking lot (kept out of the main line by rule)
├── src/calibinfo/            # the library (single source of all Fisher/Schur math)
│   ├── information/          #   whitening · schur (delta_f) · gauge (Prop.2) · retention · mode_tracking
│   ├── estimators/           #   joint_map (trf+sparse J) · gauss_newton (FD gate, GT-init assert) · diagnostics (variant-A reference)
│   ├── datasets/             #   synthetic scene factory · DiLiGenT · OpenIllumination loaders (+ sha256 manifests)
│   ├── models/               #   physical-unit corruption generator (intensity/direction/joint)
│   └── io/                   #   run manifests (git SHA, config hash, seed, checksums)
├── experiments/              # ci01_algebra … ci05_sanity (+ ci03nl envelope, ci05_ablation) — each run() is pre-registered
├── configs/                  # frozen YAML per experiment (pilot + formal)
├── tests/unit/               # 54 tests: red-team V1–V6 regressions, dense-contrast (red-line #8), loader parity, scene factory
├── scripts/                  # run_ci.py (orchestrator) · make_figures.py --figure N · reproduce_paper.sh
├── artifacts/frozen/         # machine-readable summaries + manifests (the ONLY inputs to figures/tables)
├── results/raw/              # gitignored raw runs
├── paper/                    # manuscript v0.8 · theory note · figures (per-figure provenance) · cover letter
├── docs/                     # constitution (experiment design v1.0) · master plan · migration matrix · per-card memos · novelty matrix
│   └── archive/              # 45 pre-2026-09 legacy docs (read-only, indexed by docs/archive/README.md)
├── critical_experiments/     # legacy project stage exp1–exp14 (read-only; honest negatives preserved)
└── legacy_redteam/           # red-team attack scripts E1–E6/N1–N5/V1–V6 (read-only; verdicts became unit tests)
```

## Quick start

```bash
pip install -e .
python -m pytest                    # 54 tests (red-team regressions + dense-contrast known-answer tests)
bash scripts/reproduce_paper.sh    # full chain: tests → all formal runs → Figures 1–9 (with provenance)
```

Requirements: Python ≥ 3.10, numpy, scipy, pyyaml, matplotlib, pillow.
Real-data stages (CI04/CI05) additionally need `D:/data/OpenIllumination` and
`D:/data/DiLiGenT/pmsData`; data contracts are frozen in `artifacts/frozen/*_manifest.json`
(checksums + downsampling verdicts). Synthetic stages (CI01–CI03) run on CPU in minutes.

## Key results (all numbers from frozen artifacts; regenerate with one command)

| Stage | Result | Gate |
|---|---|---|
| CI01 algebra/invariance | 100/100 identity checks; dual-route ≤ 2.2e-12; Λ=0 rank-deficient path solve-free | A ✅ |
| CI02 gauge lifting & λ⋆ | closed form vs direct p50 3.1e-9; λ⋆ predictor median \|log₁₀\| = 0.00000 (54 scenes) | ✅ |
| CI03 estimator calibration | weak-5-mode variance ratios 0.977–1.009; 68/95% coverage within ±0.013 | B ✅ |
| CI03-nl validity envelope | theory error <10% for mask-flip ≤ 0.8–3.0% (scene-family stratified) | C ✅ |
| CI04 controlled real corruption | OpenIllumination, 11 frozen test objects, 20 seeds: **Spearman 0.728 [0.668, 0.783]** | D ✅ |
| CI05 sanity & ablation | DiLiGenT 10/10 weak-mode structure; 4 within-scene effect sizes | ✅ |

## Method integrity (the part we are most proud of)

- **Known-answer tests before runs** (red-line #8): every Fisher/Schur implementation ships with
  a randomized dense-contrast test (rel < 1e-10) and independent-route cross-validation.
- **Determinism**: fixed seeds everywhere; a fresh clone reproduces CI01/CI03 bit-exactly and
  Fig.3/4/7 byte-identically (see `docs/memo_F3_crossmachine_20260909.md`).
- **Claims governance**: `CLAIMS_REGISTRY.yaml` binds each manuscript statement to allowed and
  forbidden wording, with artifact + commit pointers — no claim without evidence.
- **Provenance**: every figure regenerates via `python scripts/make_figures.py --figure N`,
  writing its git SHA + artifact hash to `paper/provenance/`.
- **Honest negatives preserved**: superseded/void verdicts from earlier project stages remain
  in `critical_experiments/` (read-only) and `docs/archive/` — the ledger includes what did not work.

## Documentation map

| Document | Role |
|---|---|
| `docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md` | Constitution: frozen theory (Lemma 1/Prop 1/Prop 2/Lemma 2/λ⋆), experiment matrix CI01–CI05, gates A–E, claims registry, writing constraints |
| `docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md` | Master plan: cards C01–C23, red lines, stop rules |
| `docs/任务布置_投稿闭口阶段_20260908_卡F1–F6.md` | Closure stage: novelty audit, claim lock, cross-machine repro, submission |
| `docs/novelty_comparison_matrix.md` | 8 nearest-neighbor works (1991–2024) vs this paper, with DOIs |
| `docs/memo_*.md` | One-page memos per experiment card (results, incidents, dispositions) |
| `docs/REPO_MIGRATION.md` | Asset migration ledger from the legacy codebase |
| `docs/repo_reorganization_manifest_20260909.md` | Where every legacy file went (2026-09-09 reorganization) |
| `docs/archive/` | 45 legacy docs (old project line), indexed and read-only |

## Citation & license

Code: MIT (see `LICENSE`). Datasets: DiLiGenT (Shi et al., CVPR 2016) and OpenIllumination
(Liu et al., NeurIPS 2023, CC BY 4.0) — follow their respective terms. Paper preprint link to
be added upon TCI acceptance.
