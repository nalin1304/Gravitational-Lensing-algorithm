# Tier-1 Topic and Problem Definition (IEEE TCI Target)

Last updated: 2026-02-26

## 1. Proposed Journal Topic

**Title direction**:  
Physics-Constrained and Uncertainty-Calibrated Computational Imaging for
Strong Gravitational Lensing Under Real-Data Domain Shift

**Why this topic is Tier-1 suitable**:
1. It is a computational imaging inverse problem with direct physical constraints.
2. It addresses both reconstruction quality and scientific reliability.
3. It explicitly treats reproducibility and uncertainty calibration as first-class outputs.

## 2. Precise Problem Statement

Given observed lensing image data `I`, infer convergence map `kappa` and lens
parameters `phi` such that:
1. The forward model remains consistent with thin-lens physics.
2. Inference uncertainty is calibrated (not only low average error).
3. Performance is robust when moving from synthetic training distributions to
   real observational distributions (SLACS/HST-like data).

This targets a known gap: many ML lensing pipelines report point accuracy but
do not quantify modeling/systematic reliability sufficiently for cosmographic use.

## 3. Research Questions and Testable Hypotheses

1. `H1` Physics constraints improve reconstruction:
   adding equation-level losses reduces RMSE and gradient inconsistency vs
   data-only models.
2. `H2` Uncertainty is meaningfully calibrated:
   empirical coverage and calibration diagnostics remain within pre-registered bounds.
3. `H3` Real-data transfer remains stable:
   metrics on SLACS-like systems remain inside predefined confidence intervals.
4. `H4` Methodological reproducibility is executable:
   full validation gates can be run from a clean checkout with deterministic outputs.

## 4. Scientific Rigor Requirements for Submission

1. **Pre-registered acceptance criteria**:
   thresholds must be set before final benchmark runs (not tuned post hoc).
2. **Uncertainty diagnostics beyond mean error**:
   include coverage/calibration summaries, not RMSE alone.
3. **Ablation evidence with effect sizes**:
   report metric deltas and confidence intervals for each removed component.
4. **Transparent baseline positioning**:
   separate physically interpretable baselines (e.g., parametric NFW/SIE) from
   learned baselines and avoid unsupported SOTA claims.
5. **Executable reproducibility artifact**:
   scripts and gate reports archived with commit hash.

## 5. Immediate Repository Actions (Implemented)

1. Added executable publication gate: `scripts/publication_gate.py`.
2. Added known-system validation with local path safety:
   `scripts/validate_known_systems.py`.
3. Added statistical rigor report generator:
   `scripts/statistical_rigor_report.py`.
4. Added CI/path consistency fixes and test guards for script-level reproducibility.

## 6. Remaining High-Impact Scientific Gaps

1. Raw mass recovery remains weaker than calibrated recovery on some systems.
2. Current benchmark scripts include proxy simulation pathways; manuscript text
   must clearly distinguish proxy vs trained-model evidence.
3. Final manuscript should include explicit confidence intervals and hypothesis
   outcomes table (accept/reject per hypothesis).

## 7. Web-Sourced Standards and Evidence Anchors

1. IEEE SPS TCI information for authors and journal scope:  
   [IEEE TCI publication resources](https://signalprocessingsociety.org/publications-resources/ieee-transactions-computational-imaging)
2. IEEE guidance on reproducibility/artifact support:  
   [IEEE SPS information for authors](https://signalprocessingsociety.org/publications-resources/information-authors)
3. IEEE guidance on sharing code/data and posting policy:  
   [IEEE author posting policy](https://www.ieee.org/publications/subscriptions/rights/author-posting-policy.html)
4. Strong-lens benchmark context (challenge realism and metrics):  
   [Strong Gravitational Lens Finding Challenge (A&A 625, A119)](https://arxiv.org/abs/1802.03609)
5. Systematics pressure in lens cosmography:  
   [Kochanek 2020, H0LiCOW/TDCOSMO systematics discussion](https://arxiv.org/abs/1911.05083)
6. Calibration diagnostics reference for predictive confidence:
   [On Calibration of Modern Neural Networks (ICML 2017)](https://arxiv.org/abs/1706.04599)
