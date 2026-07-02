# GCS-MaOEA — Hardening & Originality Plan (anti-desk-reject)

**Target venue:** IEEE Access
**Goal:** raise GCS-MaOEA from "three 2024 ideas combined" to a single, defensible,
original mechanism with isolated, measurable contributions and a bulletproof
empirical/theoretical package.
**Author stance:** senior researcher (25y) — every claim must be falsifiable,
every contribution isolable by ablation, every comparison fair.

This file is self-contained and resumable. Update the **Progress Tracker** at the
bottom. en-US, UTF-8 no BOM. Does not change the existing `PLAN.md` (research log);
this is the improvement/hardening track.

---

## 0. The core problem to fix

A critical reviewer's verdict (recorded): originality is **incremental** — ARV ≈
AR-MOEA/RVEA*/A-NSGA-III/NRV-MOEA; DVA ≈ LERD + a scalar term; multi-archive ≈
CMOEA-CD minus constraints. The "coupling" narrative is the only real novelty but
the code couples weakly (`f1 = con + 0.5·dir`). Plus: DVA never activates in the
proposed roster (`n_var<100`), baselines are stubs, no theory.

**Reframing (the originality lever).** Stop presenting three mechanisms. Present
**one**: a single online-learned **reference geometry** `G = (H, W_repaired)`
(normal hyperplane + repaired vectors) that simultaneously (i) adapts the
decomposition, (ii) **partitions decision variables by their directional
influence on `G`**, and (iii) drives **direction-targeted reproduction**. The
novelty claim becomes: *a shared latent geometry that couples search in objective
AND decision space* — which none of the cited works do jointly. Working name for
the mechanism: **Geometry-Coupled Search (GCS)**.

---

## 1. Defensible novelty claim (one sentence, falsifiable)

> "GCS-MaOEA introduces a single online-learned reference geometry that jointly
> (a) repairs the decomposition to the true front shape and (b) assigns disjoint
> decision-variable groups to individual reference directions, enabling
> direction-targeted reproduction; we show this objective↔variable geometric
> coupling outperforms decoupled adaptive-reference and decision-variable methods
> specifically on irregular and large-scale many-objective problems."

Falsifiable parts: "jointly", "direction-targeted reproduction", "specifically on
irregular and large-scale" — each must be supported by an ablation or a targeted
experiment, or the claim is dropped.

---

## The Plan — 10 phases (each shippable, leaves the repo green)

Validation gate every phase: `python -m pytest tests/ -q` for touched tests +
smoke run of GCS-MaOEA on one small problem. Do not commit unless asked.

### Phase 1 — Positioning & reviewer threat model (no code)
- **Deliverable:** `algorithms/gcs_maoea/POSITIONING.md` with (a) a delta table
  "what is new vs AR-MOEA, RVEA*, A-NSGA-III, NRV-MOEA(2024), LERD(2024),
  CMOEA-CD, LMOEA/LSMOF"; (b) the single novelty claim (Section 1); (c) a
  reviewer threat model: list the 8 most likely objections and the planned
  rebuttal/experiment for each.
- **Acceptance:** every prior-art method has an explicit one-line "they do X, we
  do Y" differentiator; no claim without a planned evidence source.

### Phase 2 — Elevate the coupling into a real mechanism (the originality core)
- **Action:** redesign DVA from a soft scalar term into **direction-partitioned
  variable analysis**: assign each (or a cluster of) repaired reference direction
  a *disjoint* variable group via the directional-influence score; store the
  assignment `var_group[d]`. Then add **direction-targeted reproduction**: when
  generating offspring aimed at direction `d`, perturb mainly `var_group[d]`.
  Couple ARV→DVA→reproduction through the shared `G`.
- **Files:** `dva.py` (partitioned scoring + assignment), `gcs_maoea.py`
  (`_infill` uses direction-targeted variation), small new `coupling.py` if needed.
- **Acceptance:** ablation hook exists to disable the partitioning (fall back to
  the old scalar term) so the coupling's effect is measurable; smoke run shows
  non-empty per-direction groups on a large-scale problem.

### Phase 3 — ARV rigor & differentiation
- **Action:** formalize the repair operator `R(w; H, z)`: prove/establish
  (i) extreme-point preservation, (ii) idempotence on the hyperplane
  (projection property `R∘R = R`), (iii) a bound on angular distortion vs the
  uniform set. Make it parameter-light (remove any incidental magic constants).
  Add a crisp contrast to AR-MOEA reference adaptation and NSGA-III normalization.
- **Files:** `reference_repair.py` (docstrings with the properties + asserts in a
  test), `POSITIONING.md` (the contrast).
- **Acceptance:** `tests/test_gcs_maoea_arv_properties.py` empirically verifies
  idempotence and extreme-point preservation within tolerance.

### Phase 4 — Experimental design aligned to contributions
- **Action:** define the roster so each contribution is *tested*:
  - **Regular (sanity):** DTLZ1–4, M∈{5,8,10,15}.
  - **Irregular (tests ARV):** MaF1–15, DTLZ5/6/7, WFG1/2/3, M∈{5,8,10,15}.
  - **Large-scale (tests DVA — currently MISSING):** LSMOP1–9, M∈{5,8,10},
    `n_var ∈ {100, 500, 1000}` so `DVAManager.enabled` is true.
  - Map each problem block → which contribution it validates.
- **Files:** `tests/gcs_maoea_benchmark.py` (add the LSMOP/large-scale block),
  `POSITIONING.md` (the problem→contribution matrix).
- **Acceptance:** a dry-run lists the full design and confirms DVA activates on
  the large-scale block (`n_var≥100`).

### Phase 5 — Real, fair baselines
- **Action:** port the **real** NRV-MOEA (Hua 2024) and LERD (He 2024) from the
  PlatEMO MATLAB into `algorithms/nrv_moea/`, `algorithms/lerd/` (the PLAN flags
  the current ones as stubs); add AR-MOEA and RVEA if absent. Verify fidelity the
  same way we audited the PlatEMO problems (defaults, operators, structural
  parity).
- **Acceptance:** each baseline runs end-to-end on a small problem and matches its
  source's default configuration; a short note records any deviation.

### Phase 6 — Ablation & sensitivity (isolate every claim)
- **Action:** implement switches and run: (A1) −ARV (fixed uniform vectors),
  (A2) −DVA, (A3) −direction-partitioning (scalar coupling only — isolates the
  *new* coupling), (A4) −dynamic allocation (fixed thirds). Plus hyperparameter
  sensitivity (warmup, dva_*, de_cr/f) and empirical per-generation cost.
- **Acceptance:** an ablation table where A3 (no coupling) is *worse than full*
  on irregular+large-scale — i.e., the coupling earns its place; otherwise the
  claim is revised honestly.

### Phase 7 — Statistical rigor
- **Action:** 30 independent runs per (alg, problem, M); report mean±std,
  Wilcoxon rank-sum (α=0.05) **with Holm-Bonferroni**, Friedman + Nemenyi
  critical-difference diagrams, effect sizes (A12/Cliff's delta), win/tie/loss,
  average ranks. Reuse the project's `core/analysis/stat_tests.py`.
- **Acceptance:** all comparison tables carry significance markers + corrected
  p-values; CD diagram generated.

### Phase 8 — Theory & complexity
- **Action:** per-generation complexity (show same asymptotic as MOEA/D-DE plus
  the `O(N²M)` hyperplane/repair term); the ARV lemmas from Phase 3; a coverage
  argument linking repaired vectors to front coverage; honest assumptions.
- **Acceptance:** a self-contained `THEORY.md` section ready to drop into the
  methodology; every lemma has a proof sketch or an empirical check.

### Phase 9 — Reproducibility package
- **Action:** fixed seed sequence (1..30), versioned configs, raw result CSVs,
  per-run SHA-256 execution manifests (reuse `core/execution/reproducibility.py`),
  one-command rerun script, environment pin. Code + scripts released with the paper.
- **Acceptance:** a clean checkout reproduces a representative cell
  (alg×problem×M) within run-to-run statistical tolerance.

### Phase 10 — Manuscript anti-desk-reject pass
- **Action:** assemble/verify against an IEEE Access checklist: scope fit,
  polished en-US, clear structure, the **explicit novelty table** (Phase 1),
  contribution→experiment traceability, a candid **limitations & threats-to-
  validity** section, no overselling. Map every reviewer objection (Phase 1
  threat model) to where it is answered in the text.
- **Acceptance:** a reviewer-response pre-mortem: for each of the 8 anticipated
  objections, point to the figure/table/section that defuses it.

---

## Anti-desk-reject mapping (why each risk is closed)

| Reviewer risk | Closed by |
|---|---|
| "Incremental combination" | Phase 2 (one coupled mechanism) + Phase 1 (delta table) |
| "Coupling is cosmetic" | Phase 2 (direction-partitioned) + Phase 6 A3 ablation |
| "DVA contribution untested" | Phase 4 (LSMOP large-scale where DVA activates) |
| "Unfair/stub baselines" | Phase 5 (real NRV-MOEA/LERD, fidelity-checked) |
| "Tuned to win" | Phase 6 (sensitivity) + Phase 9 (fixed seeds/configs) |
| "No statistical rigor" | Phase 7 (corrected tests, CD diagrams, effect sizes) |
| "No theory" | Phase 8 (complexity + lemmas) |
| "Not reproducible" | Phase 9 (seeds, manifests, code release) |

## Honesty guardrails (non-negotiable)
- If Phase 6 A3 shows the coupling does **not** help, **revise the claim** — do not
  bury the ablation. A defensible smaller claim beats an oversold rejected one.
- Report where GCS-MaOEA **loses** (e.g., regular fronts where ARV adds overhead);
  reviewers trust papers that show their own limits.
- No metric cherry-picking: IGD + HV + DeltaP reported for all cells.

---

## Progress Tracker (update when resuming)

| Phase | Status | Notes |
|---|---|---|
| 0 — Critical assessment | DONE | Verdict: incremental originality; desk-reject risk on novelty, not automatic at Access; coupling is the lever. |
| 1 — Positioning & threat model | TODO | start here next |
| 2 — Coupling → real mechanism | TODO | the originality core |
| 3 — ARV rigor & differentiation | TODO | |
| 4 — Experimental design (add LSMOP) | TODO | |
| 5 — Real baselines (NRV-MOEA, LERD) | TODO | |
| 6 — Ablation & sensitivity | TODO | |
| 7 — Statistical rigor | TODO | |
| 8 — Theory & complexity | TODO | |
| 9 — Reproducibility package | TODO | |
| 10 — Manuscript anti-desk-reject pass | TODO | |

### Resume protocol
1. Open this file; find the first `TODO` in the tracker — that is the next phase.
2. Re-read that phase's Action/Acceptance.
3. Execute, run the validation gate, update the row to DONE with a one-line note.
4. Do not commit unless asked.
