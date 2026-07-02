# GCS-MaOEA Research & Implementation Plan

**Algorithm:** GCS-MaOEA — Adaptive Reference-vector Repair via Decision Variable Analysis for Many-Objective Optimization  
**Working package:** `algorithms/gcs_maoea/`  
**Target venue:** IEEE Access  
**Focus:** Unconstrained many-objective optimization (MAOPs), emphasis on scalability in the number of objectives.

---

## 1. Research Motivation

State-of-the-art many-objective evolutionary algorithms (MaOEAs) can be grouped into three families:

1. **Reference-vector-based algorithms** (NSGA-III, MOEA/D, RVEA): efficient for regular Pareto fronts, but their fixed reference vectors degrade on irregular fronts (degenerate, disconnected, inverted).
2. **Adaptive normal-vector methods** (NRV-MOEA): estimate the true geometry of the Pareto front via an adaptive normal hyperplane and extreme points. They do not, however, exploit decomposition or decision-variable structure.
3. **Decision-variable-analysis methods** (LERD): identify which decision variables drive convergence/diversity in large-scale problems. They still rely on fixed MOEA/D weight vectors, missing the opportunity to align the search with the discovered front geometry.

The core observation behind GCS-MaOEA is that **the geometry of the Pareto front and the geometry of the decision-variable space are coupled**, and neither fixed reference vectors nor independent adaptive mechanisms fully exploit this coupling. GCS-MaOEA closes this gap by:

- repairing MOEA/D reference vectors using the adaptive normal hyperplane learned from the current archive,
- using decision-variable analysis to assign variables to the repaired directions,
- and balancing exploration, diversity, and convergence through a dynamic multi-archive structure.

---

## 2. Novelty & Contributions

The paper will claim the following explicit contributions:

- **C1 — Adaptive Reference-Vector Repair (ARV):** A mechanism that maps the uniform MOEA/D weight vectors onto the normal hyperplane estimated from adaptive extreme points. This yields reference vectors that follow the true shape of the Pareto front without requiring a pre-defined structural assumption.
- **C2 — Direction-aware Decision-Variable Analysis (DVA):** An extension of LERD-style variable analysis in which variables are scored not only by global convergence improvement, but also by their contribution along each repaired reference direction. This creates direction-specific variable masks.
- **C3 — Dynamic Multi-Archive Framework:** Inspired by CMOEA-CD, three archives (exploration, diversity, convergence) receive dynamically adjusted allocation budgets based on online indicators of spread and convergence. The diversity archive uses the repaired reference vectors for niching.
- **C4 — Unconstrained MAOP Generalization of Cooperative Multi-Archive Search:** The three-archive structure originally devised by CMOEA-CD for constrained problems is generalized to unconstrained many-objective optimization. By removing constraint-Pareto dominance and feasibility archives and coupling the remaining archives with the repaired reference vectors from ARV, GCS-MaOEA shows that multi-archive cooperation is a generic and powerful search paradigm for MAOPs.

These four contributions are combined into a single algorithm: **GCS-MaOEA**.

---

## 3. Algorithm Architecture

GCS-MaOEA is built on top of the `pymoo`/`pymoolab` framework and is organized into five modules:

```text
algorithms/gcs_maoea/
├── __init__.py
├── gcs_maoea.py           # Main algorithm class (pymoo Algorithm)
├── reference_repair.py  # ARV: normal hyperplane, extreme points, vector repair
├── dva.py               # DVA: sparse variable sampling and direction-aware masks
├── archives.py          # Three-archive manager with dynamic budget allocation
└── PLAN.md              # This document
```

### 3.1 Main loop

```text
Input:  Problem with M > 3 objectives, population size N, max evaluations maxFE
Output: Approximated Pareto front

1. Initialize population P with random sampling.
2. Run a short MOEA/D warm-up using uniform reference vectors to obtain a
   well-spread initial front.
3. Repeat until termination:
   a. Update external archive A with non-dominated solutions from P.
   b. Estimate ideal point z_min and nadir point z_nad from A.
   c. Detect extreme points using ASF weights and solve for the normal
      hyperplane H of the current front.
   d. Repair MOEA/D reference vectors W by projecting the uniform vectors
      onto H and re-normalizing (ARV).
   e. If n_var is large, run DVA to obtain direction-aware variable masks M_d
      for each repaired direction d.
   f. Update the three archives:
      - Exploration Archive (EA): non-dominated solutions, promotes discovery.
      - Diversity Archive (DA): niching over repaired reference vectors.
      - Convergence Archive (CA): solutions closest to H, promotes convergence.
      Budget allocation is adaptive based on spread and convergence indicators.
   g. Generate offspring from EA, DA, and CA using DE and GA operators.
   h. Environmental selection merges parents and offspring, keeping N
      non-dominated solutions with preference for under-represented
      reference vectors.
4. Return non-dominated solutions of the final archive.
```

### 3.2 Module: `reference_repair.py`

Key functions:

- `update_ideal_nadir(F, z_min, z_nad)`: tracks ideal and nadir points.
- `find_extreme_points(F, z_min, z_nad)`: ASF-based extreme point detection.
- `normal_hyperplane(F_extreme, z_min)`: solves `H` such that `H^T (f - z_min) = 1`.
- `repair_reference_vectors(W_uniform, H, z_min, z_nad)`: projects each uniform vector onto the hyperplane and re-normalizes, producing `W_repaired`.
- `vertical_projection(F, H, z_min)`: maps objective vectors onto the hyperplane (used by the diversity archive).

### 3.3 Module: `dva.py`

Key functions:

- `sparse_fit(problem, archive, masks, n_samples)`: evaluates variable masks by perturbing selected decision variables from an elite solution.
- `direction_aware_score(mask, F, W_repaired, z_min)`: scores a mask by both convergence improvement and alignment with a repaired reference direction.
- `optimize_masks(...)`: evolves binary masks using NSGA-II-style selection (two objectives: convergence gain and sparsity).

For standard-scale MAOPs (`n_var < 100`), DVA is disabled to reduce overhead; the algorithm still benefits from ARV and the multi-archive framework.

### 3.4 Module: `archives.py`

- `ExplorationArchive`: stores non-dominated solutions; environmental selection uses Pareto dominance + truncation.
- `DiversityArchive`: selects one solution per repaired reference vector using sine-based angular distance.
- `ConvergenceArchive`: selects solutions with smallest distance to the normal hyperplane, preserving extreme points.
- `DynamicAllocator`: adjusts the size of each archive based on:
  - spread indicator (mean pairwise angle on the hyperplane),
  - convergence indicator (mean distance to the hyperplane),
  - generation progress.

---

## 4. Baseline Algorithms

The experimental comparison will include:

| Algorithm | Reason |
|-----------|--------|
| **CMOEA-CD** | Source of the multi-archive idea; its irrestricted version is a strong baseline. |
| **NRV-MOEA** | Source of adaptive normal-vector guidance; must be implemented as the real 2024 algorithm, not the current stub. |
| **LERD** | Source of decision-variable analysis; must be implemented as the real 2024 algorithm, not the current stub. |
| **NSGA-III** | Standard reference-vector MaOEA. |
| **MOEA/D-DE** | Standard decomposition MaOEA. |

> **Note:** The current `pymoolab` implementations of `NRVMOEA` and `LERD` are stubs. To make the comparison scientifically valid, the real algorithms from the `platemoMETISBr` MATLAB code will be ported into `algorithms/nrv_moea/` and `algorithms/lerd/` before the benchmark.

---

## 5. Experimental Roster

### 5.1 Problem suites

| Suite | Problems | Objective counts |
|-------|----------|------------------|
| DTLZ  | DTLZ1–DTLZ7 | M ∈ {5, 8, 10, 15} |
| WFG   | WFG1–WFG9 | M ∈ {5, 8, 10, 15} |
| MaF   | MaF1–MaF15 | M ∈ {5, 8, 10, 15} |

Default decision-variable dimensions are taken from each problem class (e.g., `n_var = M + k - 1` for DTLZ, with `k` from the standard formula). For MaF/WFG the standard constructors in `pymoolab/problems/many/` are used.

### 5.2 Algorithm configuration

- `pop_size`: adapted to the number of reference directions generated by `UniformPoint(pop_size, M)`.
- `max_fe`: 25,000 for M ≤ 10; 50,000 for M = 15.
- `n_runs`: 30 independent runs per (algorithm, problem, M) combination.
- `seed`: fixed seed sequence (1, 2, …, 30) for reproducibility.

### 5.3 Metrics

| Metric | Use |
|--------|-----|
| **IGD** | Primary quality indicator; requires true Pareto front. |
| **HV** | Secondary indicator for M ≤ 10 (becomes expensive for M = 15). |
| **DeltaP** | Distribution/convergence indicator that does not require the true front. |

### 5.4 Statistical analysis

- Wilcoxon rank-sum test (`α = 0.05`) comparing GCS-MaOEA against each baseline.
- Holm-Bonferroni correction for multiple comparisons.
- Summary tables with mean ± std, win/tie/loss counts, and average rank.
- Convergence plots (metric vs. evaluations) for representative problems.

---

## 6. Implementation Milestones

| # | Milestone | Deliverable | Effort |
|---|-----------|-------------|--------|
| 1 | Port real NRV-MOEA | `algorithms/nrv_moea/nrv_moea.py` + helpers | Medium |
| 2 | Port real LERD | `algorithms/lerd/lerd.py` + helpers | Medium |
| 3 | Implement ARV module | `algorithms/gcs_maoea/reference_repair.py` | Low |
| 4 | Implement DVA module | `algorithms/gcs_maoea/dva.py` | Medium |
| 5 | Implement multi-archive engine | `algorithms/gcs_maoea/archives.py` | Medium |
| 6 | Integrate GCS-MaOEA main class | `algorithms/gcs_maoea/gcs_maoea.py` | Medium |
| 7 | Benchmark script | `tests/gcs_maoea_benchmark.py` + CSV outputs | Low |
| 8 | Run experiments & analyze | Tables, plots, statistical markers | High |
| 9 | Paper skeleton | LaTeX/Markdown article draft | High |

---

## 7. Publication Roadmap

1. **Methodology section** will formalize ARV, DVA, and the multi-archive allocator with lemmas on preservation of extreme points and on the repair mapping.
2. **Complexity analysis** will show that GCS-MaOEA keeps the same asymptotic cost per generation as MOEA/D-DE plus an `O(N^2 M)` term for the normal-hyperplane update and reference-vector repair.
3. **Experimental section** will report IGD/HV/DeltaP tables, convergence curves, and statistical significance markers.
4. **Ablation study** will evaluate:
   - GCS-MaOEA without ARV (fixed reference vectors),
   - GCS-MaOEA without DVA,
   - GCS-MaOEA without dynamic allocation (fixed archive sizes),
   isolating the contribution of each module.
5. **Reproducibility:** all code, benchmark scripts, and raw result CSVs will be committed to the repository.

---

## 8. Naming Conventions

- **Algorithm package folder:** `METISBr/pymoolab/algorithms/gcs_maoea/`
- **Main class:** `GCSMaOEA`
- **Display/paper name:** `GCS-MaOEA`
- **Auto-discovered ID:** `local::algorithms.gcs_maoea.gcs_maoea.GCS-MaOEA`
- **Article folder:** `/Users/proftheagos/devSuport/2026_gcs_maoea/`
- **Article working title:** *GCS-MaOEA: Adaptive Reference-Vector Repair via Decision Variable Analysis for Many-Objective Optimization*

---

## 9. References to Cite

- Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach. *IEEE TEVC*.
- Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE TEC*.
- Liu, Z., et al. (2025). Constraint-Pareto dominance and diversity enhancement strategy based CMOEA. *IEEE TEVC*.
- Hua, Y., Liu, Q., & Hao, K. (2024). Adaptive normal vector guided evolutionary multi- and many-objective optimization. *Complex & Intelligent Systems*.
- He, C., Cheng, R., Li, L., Tan, K. C., & Jin, Y. (2024). Large-scale multiobjective optimization via reformulated decision variable analysis. *IEEE TEVC*.
- Cheng, R., et al. (2016). A reference vector guided evolutionary algorithm for many-objective optimization. *IEEE TEVC*.

---

## 10. Implementation Status

- [x] Package stub created and auto-discovered by PymooLab.
- [x] ARV module (`reference_repair.py`) implemented.
- [x] DVA module (`dva.py`) implemented.
- [x] Multi-archive engine (`archives.py`) implemented.
- [x] Main algorithm class (`gcs_maoea.py`) implemented.
- [x] Article guide (`/Users/proftheagos/devSuport/2026_gcs_maoea/ARTICLE_GUIDE.md`) created.
- [x] LaTeX article skeleton (`/Users/proftheagos/devSuport/2026_gcs_maoea/`) created.
- [x] Benchmark script (`tests/gcs_maoea_benchmark.py`) created and smoke-tested.
- [ ] Port real NRV-MOEA and LERD into `pymoolab` for a fair baseline comparison.
- [ ] Run full benchmark (30 runs, M ∈ {5,8,10,15}, DTLZ/WFG/MaF).
- [ ] Perform statistical analysis and generate paper figures/tables.
- [ ] Write the IEEE Access manuscript.

## 11. How to Run

```bash
# Smoke test
python -c "from algorithms.gcs_maoea import GCSMaOEA; print(GCSMaOEA)"

# Small benchmark (quick validation)
python tests/gcs_maoea_benchmark.py \
    --problems DTLZ2 WFG4 MaF1 \
    --m-values 5,8,10 \
    --n-runs 5 \
    --n-evals 5000 \
    --workers 4

# Full paper benchmark
python tests/gcs_maoea_benchmark.py \
    --problems DTLZ1-7 WFG1-9 MaF1-15 LSMOP1-9 \
    --m-values 5,8,10,15 \
    --n-runs 30 \
    --n-evals 25000 \
    --workers 10
```
