![PymooLab Header](app.png)

# PymooLab

**PymooLab** is an open-source visual analytics framework for multi-objective optimization, with integrated benchmark orchestration, local plugin discovery, and LLM-assisted artifact formulation.

> **Author:** Professor Thiago Santos — UFOP, Brazil  
> **Group:** METISBr — Multi-Objective and Many-Objective Optimization Research  
> **Contact:** `santostf+metisbr@ufop.edu.br`

---

## Table of Contents

1. [Environment Baseline](#environment-baseline)
2. [Citation](#citation)
3. [Requirements](#requirements)
4. [Project Architecture](#project-architecture)
5. [Directory Structure](#directory-structure)
6. [Core Modules](#core-modules)
7. [Plugin System](#plugin-system)
8. [Implemented Catalog](#implemented-catalog)
9. [How to Create Plugins](#how-to-create-plugins)
10. [Running PymooLab](#running-pymoolab)

---

## Environment Baseline

| Item | Value |
|---|---|
| PymooLab release | `1.0.1` |
| Catalog snapshot | `2026-06-16` |
| Python | `3.11.x` (recommended) |
| Backend policy | CPU (NumPy) by default; JAX acceleration when requested |
| GUI toolkit | PySide6 + qt-material + qt-material-icons |
| Chart engine | QtCharts + Matplotlib (3D) |

---

## Citation

PymooLab is currently available as an arXiv preprint and is under review in an IEEE journal. If you use PymooLab in academic work, cite:

```bibtex
@misc{santos2026pymoolabopensourcevisualanalytics,
      title={PymooLab: An Open-Source Visual Analytics Framework for Multi-Objective Optimization using LLM-Based Code Generation and MCDM},
      author={Thiago Santos and Sebastiao Xavier and Luiz Gustavo de Oliveira Carneiro and Gustavo de Souza},
      year={2026},
      eprint={2603.01345},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2603.01345},
}
```

---

## Requirements

### Direct Installation (Recommended)

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install all base dependencies from requirements.txt
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Base Runtime

```txt
numpy>=2.3,<3
pymoo>=0.6.1,<0.7
scipy>=1.16,<2
PySide6>=6.10,<7
qt-material>=2.17,<3
qt-material-icons>=0.4,<1
matplotlib>=3.10,<4
psutil>=7.2,<8
```

### Optional Profiles

| Profile | Packages |
|---|---|
| GPU (JAX) | `jax>=0.9,<1`, `jaxlib>=0.9,<1` |
| LLM generation | `anthropic>=0.76,<1` |
| ML algorithms | `scikit-learn>=1.8,<2` |
| Development | `pytest>=9,<10` |

### Installation Examples

```bash
pip install -r requirements.txt
pip install jax jaxlib              # optional GPU/JAX profile
pip install anthropic               # optional LLM profile
pip install scikit-learn            # optional ML-dependent algorithms
pip install pytest                  # optional development profile
```

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      PymooLab.py (GUI)                          │
│   PymooExperimentWindow · AppStyles · Spec dataclasses          │
│   Plugin discovery · Execution engine · Results analysis        │
├──────────────────────┬───────────────────────────────────────────┤
│   pymoolab_core/     │   core/                                  │
│   ├── analysis/      │   ├── algorithm.py  (Algorithm base)     │
│   │   └── stat_tests │   ├── problem.py    (Problem re-export)  │
│   ├── execution/     │   ├── population.py                      │
│   │   ├── backend    │   ├── individual.py                      │
│   │   └── reprod.    │   ├── mating.py                          │
│   ├── llm/           │   ├── parameters.py                      │
│   │   └── formulation│   └── duplicate.py                       │
│   ├── mcdm/          ├───────────────────────────────────────────┤
│   │   └── decision   │   util/                                  │
│   └── registry/      │   ├── array_backend.py                   │
│       ├── backend    │   ├── display/                            │
│       ├── plugin_dirs│   ├── nds/                                │
│       └── rollout    │   ├── ref_dirs.py                         │
│                      │   └── optimum.py                          │
├──────────────────────┴───────────────────────────────────────────┤
│  PLUGIN DIRECTORIES (auto-discovered)                           │
│  algorithms/ (292)  problems/ (47 family modules)               │
│  metrics/ (26 entries across 3 modules)  operators/ (68 modules)│
└──────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
pymoolab/
├── PymooLab.py                  # Main GUI application (16,794 lines)
├── styles.py                    # Centralized design system (colors, typography, spacing)
├── pymoo_metadata.py            # Static metadata for pymoo native algorithms/problems
├── pymoolab_process_worker.py   # Subprocess worker for ProcessPoolExecutor
├── optimize.py                  # Legacy alias for pymoo.optimize
├── __init__.py                  # Package metadata
├── version.py                   # Package semantic version (__version__ = 1.0.1)
├── app.png                      # Application header image
├── pymoolab.png                 # Application logo
│
├── core/                        # Compatibility base classes (pymoo wrappers)
│   ├── __init__.py
│   ├── algorithm.py             # Algorithm base with GPU/backend support
│   ├── problem.py               # Re-exports pymoo.core.problem
│   ├── population.py            # Re-exports pymoo.core.population
│   ├── individual.py            # Re-exports pymoo.core.individual
│   ├── mating.py                # Re-exports pymoo.core.mating
│   ├── parameters.py            # Re-exports pymoo.core.parameters
│   └── duplicate.py             # Re-exports pymoo.core.duplicate
│
├── pymoolab_core/               # Modular core logic (Phase 7)
│   ├── analysis/
│   │   └── stat_tests.py        # Friedman, Wilcoxon, summary statistics
│   ├── execution/
│   │   ├── backend_runtime.py   # GPU/JAX detection and status
│   │   └── reproducibility.py   # Seed plans, execution manifests
│   ├── llm/
│   │   └── formulation.py       # LLM-assisted problem/algorithm formulation
│   ├── mcdm/
│   │   └── decision.py          # MCDM compromise solution selection
│   └── registry/
│       ├── backend_selection.py  # Backend-aware operator selection
│       ├── plugin_dirs.py       # Plugin directory scaffolding
│       └── rollout.py           # Feature rollout gating
│
├── algorithms/                  # Local algorithm plugins (292 implementations)
│   ├── __init__.py              # Auto-discovered by plugin system
│   ├── hyperparameters.py       # Shared hyperparameter utilities
│   ├── larc_nsga3/              # LARC-NSGA3: NSGA-III with local LM Studio controller
│   │   ├── __init__.py
│   │   └── larc_nsga3.py
│   ├── ssw/                     # Example: SSW algorithm
│   │   ├── __init__.py
│   │   └── ssw.py
│   ├── ssw_dnv/                 # SSW-DNV: dynamic normal vectors + EEJ
│   │   ├── __init__.py
│   │   └── ssw_dnv.py
│   ├── ssw_rdpa/                # Example: SSW-RDPA algorithm
│   │   ├── __init__.py
│   │   └── ssw_rdpa.py
│   └── ...                      # 287 more algorithm subdirectories
│
├── problems/                    # Local problem plugins
│   ├── many/                    # Many-objective (≥4 obj): DTLZ, WFG, MAF, ZCAT, ...
│   ├── multi/                   # Multi-objective (2–3 obj): ZDT, CF, UF, RWMOP, ...
│   └── single/                  # Single-objective: BBOB, CEC2008/2013/2020, ...
│
├── metrics/                     # Local metric plugins
│   ├── __init__.py
│   ├── community_metrics.py     # 25 metrics: HV, IGD, GD, Spacing, CPF, ...
│   ├── community_metrics_JAX.py # JAX-accelerated metric variants
│   └── hv_fast_mc.py            # Fast Monte Carlo hypervolume metric
│
├── operators/                   # Local operator plugins
│   ├── crossover/               # SBX, DEX, PCX, PMX, UX, HUX, ...
│   ├── mutation/                # PM, Gaussian, Bitflip, Inversion, ...
│   ├── selection/               # Tournament, Random, Age-based, ...
│   ├── sampling/                # Float/Int/Binary/Permutation random, LHS
│   ├── repair/                  # Bounds repair utilities
│   ├── survival/                # Survival selection operators
│   ├── utility_functions/       # Shared operator utilities
│   ├── control.py               # Operator control logic
│   └── control_JAX.py           # JAX-accelerated operator control
│
├── util/                        # Shared utilities
│   ├── array_backend.py         # NumPy/JAX array module abstraction
│   ├── display/                 # Console output formatters
│   ├── nds/                     # Non-dominated sorting utilities
│   ├── ref_dirs.py              # Reference direction generation
│   └── optimum.py               # Optimum filtering
│
├── tests/                       # Local-only benchmark and validation tests (not pushed in release commits)
│   ├── zdt_ssw_rdpa_vs_nsga3_benchmark.py
│   ├── dtlz_ssw_rdpa_vs_nsga3_benchmark.py
│   ├── wfg_ssw_rdpa_vs_nsga3_benchmark.py
│   ├── temp_bnh_test.py
│   └── test_worker.py
│
├── reports/
│   └── validation/              # Validation reports and results
│
├── pymoo_config.json            # Runtime configuration
├── pymoo_experiment_config.json # Experiment configuration
├── last_experiment_config.json  # Last executed experiment config
├── mypy.ini                     # Type checking configuration
├── clean_pycache.py             # Cache cleanup utility
└── LICENSE                      # MIT License
```

---

## Core Modules

### `PymooLab.py` — Main GUI Application

The main module (`16,794 lines`) contains the entire PySide6 GUI application:

| Component | Description |
|---|---|
| `PymooExperimentWindow` | Main window: algorithm/problem/metric selection, execution, results, charts |
| `AlgorithmSpec` | Dataclass: registered algorithm metadata |
| `ProblemSpec` | Dataclass: registered problem metadata |
| `MetricSpec` | Dataclass: registered metric metadata |
| `OperatorSpec` | Dataclass: registered operator metadata |
| `AppStyles` | Centralized style bridge (delegates to `styles.py`) |
| Plugin discovery | Auto-scans `algorithms/`, `problems/`, `metrics/`, `operators/` for local plugins |
| Execution engine | `ProcessPoolExecutor` with `QThread` bridge for parallel trial execution |
| Analysis workspace | Statistical tests (Friedman, Wilcoxon), convergence charts, Pareto front plots |
| Export system | CSV, LaTeX table export for benchmark results |
| MCDM integration | Multi-criteria decision making for compromise solution selection |
| LLM formulation | Anthropic-powered problem/algorithm code generation |

### `pymoolab_core/` — Modular Core Logic

| Subpackage | Module | Purpose |
|---|---|---|
| `analysis` | `stat_tests.py` | Friedman test, Wilcoxon signed-rank test, statistical summaries |
| `execution` | `backend_runtime.py` | JAX/GPU device detection and status reporting |
| `execution` | `reproducibility.py` | Deterministic seed plans, execution manifests with SHA-256 |
| `llm` | `formulation.py` | LLM-based problem and algorithm code generation service |
| `mcdm` | `decision.py` | Compromise solution selection for Pareto fronts |
| `registry` | `backend_selection.py` | Backend-aware operator resolution (CPU vs JAX) |
| `registry` | `plugin_dirs.py` | Plugin directory scaffolding and migration |
| `registry` | `rollout.py` | Feature rollout stage gating |

### `core/` — Compatibility Base Classes

Thin wrappers around `pymoo.core.*` that provide local algorithm compatibility (GPU flags, array backend selection):

- `Algorithm` extends `pymoo.core.algorithm.Algorithm` with `use_gpu`, `array_backend`, `gpu_dtype` support
- Other modules (`problem.py`, `population.py`, etc.) re-export from pymoo directly

---

## Plugin System

PymooLab uses **automatic directory-based plugin discovery**. At startup, the application scans predefined directories and registers every valid Python module as a plugin.

### Discovery Flow

```
Startup → ensure_plugin_directories()
        → Scan algorithms/ → Each subfolder with __init__.py → AlgorithmSpec
        → Scan problems/   → Each .py file → ProblemSpec
        → Scan metrics/    → community_metrics.py → MetricSpec
        → Scan operators/  → Each subfolder → OperatorSpec
        → Scan pymoo.*     → Native pymoo algorithms/problems/operators
```

### Registration Contracts

| Domain | Required Export | Registration |
|---|---|---|
| Algorithm | `create_algorithm(config) → Algorithm` and/or class inheriting `Algorithm` | `AlgorithmSpec` with `factory` |
| Problem | Class inheriting `pymoo.core.problem.Problem` | `ProblemSpec` with `factory` |
| Metric | Callable `(front: np.ndarray, context: dict) → float` | `MetricSpec` with `factory` |
| Operator | Class inheriting `pymoo.core.{crossover,mutation,selection,sampling}` | `OperatorSpec` with `module` + `class_name` |

---

## Implemented Catalog

- **Snapshot date:** `2026-06-16`
- **Version:** `1.0.1`
- **Release push policy:** local benchmark files under `tests/` are excluded from release commits.

### Algorithms (292)

`ABSAEA`, `ACMMEA`, `AdaW`, `ADSAPSO`, `AENSGAII`, `AFSEA`, `AGEII`, `AGSEA`, `AMGPSL`, `ANSGAIII`, `APSEA`, `ARMOEA`, `AVGSAEA`, `BCEIBEA`, `BCEMOEAD`, `BiCo`, `BiGE`, `BLEAQII`, `BLSAEA`, `C3M`, `CAEAD`, `CAMOEA`, `CCGDE3`, `CCMO`, `cDPEA`, `CGLP`, `CLIA`, `CMaDPPs`, `CMDEIPCM`, `CMEGL`, `CMME`, `CMMO`, `CMOBR`, `CMOCSO`, `CMODEFTR`, `CMODRL`, `CMOEA_CD`, `CMOEAD`, `CMOEAMS`, `CMOEAMSG`, `CMOEBOD`, `CMOEMT`, `CMOES`, `CMOQLMT`, `CMOSMA`, `CNSDEDVC`, `CoMMEA`, `CPSMOEA`, `CSEA`, `CSEMT`, `CTSEA`, `DAEA`, `DBEMTO`, `DCNSGAIII`, `DEAGNG`, `DGEA`, `DirHVEI`, `DISK`, `DISK_2025`, `DISKplus`, `DKCA`, `DMMOEA`, `DMOEAeC`, `dMOPSO`, `DNNSGAII`, `DPCPRA`, `DPPPS`, `DPVAPS`, `DRLOSEMCMO`, `DRLSAEA`, `DSPCMDE`, `DSSEA`, `DVCEA`, `DWU`, `EAGMOEAD`, `EDNARMOEA`, `EFRRR`, `EIMEGO`, `EM_SAEA`, `EMCMMS`, `EMCMO`, `EMMOEA`, `eMOEA`, `EMOSKT`, `EMyOC`, `ENSMOEAD`, `ESBCEO`, `FDV`, `FLEA`, `FRCGM`, `GCNMOEA`, `GDE3`, `GFMMOEA`, `GLMO`, `gNSGAII`, `GPSOM`, `GrEA`, `GWASFGA`, `HEA`, `HeEMOEA`, `HHCMMEA`, `hpaEA`, `HREA`, `HypE`, `IBEA`, `ICMA`, `IDBEA`, `IMCMOEAD`, `IMMOEA`, `IMMOEAD`, `IMTCMO`, `IMTCMO_BS`, `ISIBEA`, `Izui`, `KLEA`, `KLNSGAII`, `KnEA`, `KRVEA`, `KTA2`, `KTS`, `LARC_NSGA3`, `LCMEA`, `LCSA`, `LDSAF`, `LERD`, `LMEA`, `LMOCSO`, `LMOEADS`, `LMPFE`, `LRMOEA`, `LSMOF`, `MaOEACSS`, `MaOEADDFC`, `MaOEAIGD`, `MaOEAIT`, `MaOEARD`, `MCCMO`, `MCEAD`, `MFFS`, `MFOSPEA2`, `MGCEA`, `MGSAEA`, `MMEAPSL`, `MMEAWI`, `MMOPSO`, `MO_Ring_PSO_SCD`, `MOBCA`, `MOCell`, `MOCGDE`, `MOCMA`, `MOEACKF`, `MOEAD2WA`, `MOEADAWA`, `MOEADCMA`, `MOEADCMT`, `MOEADD`, `MOEADDAE`, `MOEADDCWV`, `MOEADDE`, `MOEADDQN`, `MOEADDRA`, `MOEADDU`, `MOEADDYTS`, `MOEADEGO`, `MOEADFRRMAB`, `MOEADM2M`, `MOEADMRDL`, `MOEADPaS`, `MOEADPFE`, `MOEADSTM`, `MOEADUR`, `MOEADURAW`, `MOEADVA`, `MOEADVOV`, `MOEAIGDNS`, `MOEANZD`, `MOEAPC`, `MOEAPSL`, `MOEARE`, `MOEGS`, `MOL2SMEA`, `MOMBIII`, `MOMFEA`, `MOMFEAII`, `MOMFEASADE`, `MONAS`, `MOPSO`, `MOSD`, `MPAES`, `MPMMEA`, `MPSOD`, `MSCEA`, `MSCMO`, `MSEA`, `MSKEA`, `MSOPSII`, `MTCMO`, `MTDEMKTA`, `MTEADDN`, `MTS`, `MultiObjectiveEGO`, `MyODEMR`, `NBLEA`, `NMPSO`, `NNDREAMO`, `NNIA`, `NRVMOEA`, `NSBiDiCo`, `NSGAIIARSBX`, `NSGAIIconflict`, `NSGAIIDTI`, `NSGAIIIEHVI`, `NSGAIISDR`, `NSLS`, `NUCEA`, `onebyoneEA`, `OSPNSDE`, `ParEGO`, `PBNSGAIII`, `PBRVEA`, `PCSAEA`, `PEA`, `PEAplus`, `PeEA`, `PESAII`, `PICEAg`, `PIEA`, `PIMD`, `PMMOEA`, `POCEA`, `PPS`, `PRDH`, `PREA`, `REMO`, `RGA_M1_2`, `RGA_M2_2`, `RMMEDA`, `RMOEADVA`, `RPDNSGAII`, `RPEA`, `RSEA`, `RVEAa`, `RVEAiGNG`, `S3CMAES`, `SAMOEA_TL2M`, `SCDAS`, `SCEA`, `SECSO`, `SFADE`, `SGEA`, `SGECF`, `SIBEA`, `SIBEAkEMOSS`, `SLMEA`, `SMEA`, `SMOA`, `SMPSO`, `SMSEGO`, `SNSGAII`, `SparseEA`, `SparseEA2`, `SPEA2SDE`, `SPEAR`, `SRA`, `SSCEA`, `SSDE`, `SSW`, `SSW_DNV`, `SSW_RDPA`, `SVRNSGAII`, `tDEA`, `tDEACPBI`, `TEA`, `TELSO`, `TiGE2`, `ToP`, `TPCMaO`, `TriMOEATAR`, `TSNSGAII`, `TSSparseEA`, `TSTI`, `Two_Arch2`, `URCMO`, `VaEA`, `WASFGA`, `WOF`, `WVMOEAP`

> Plus native pymoo algorithms: NSGA-II, NSGA-III, U-NSGA-III, RVEA, MOEA/D, AGE-MOEA, AGE-MOEA-II, C-TAEA, SMS-EMOA, SPEA2, etc.

### Metrics (26)

`CPF`, `DeltaP`, `DM`, `Feasible_rate`, `GD`, `HV`, `HV_fast_MC`, `IGD`, `IGDp`, `IGDX`, `Lower_level_Min_value`, `Mean_HV`, `Mean_IGD`, `Min_value`, `PD`, `Spacing`, `Spread`, `Task1_HV`, `Task1_IGD`, `Task1_Min_value`, `Task2_HV`, `Task2_IGD`, `Task2_Min_value`, `Upper_level_Min_value`, `Worst_HV`, `Worst_IGD`

Notes:

- `HV` uses exact pymoo hypervolume for low-dimensional fronts and a deterministic chunked Monte Carlo estimator for many-objective fronts.
- `context["hv_mc_chunk_size"]` can tune Monte Carlo memory use without changing the deterministic seed.
- `HV_fast_MC` is available as a standalone fast Monte Carlo hypervolume plugin.

### Operators (38)

`AgeBasedTournamentSelection`, `BFM`, `BinaryRandomSampling`, `BinomialCrossover`, `BitflipMutation`, `BX`, `ChoiceRandomMutation`, `DEX`, `EdgeRecombinationCrossover`, `ERX`, `ExponentialCrossover`, `FloatRandomSampling`, `GaussianMutation`, `GM`, `HalfUniformCrossover`, `HUX`, `IntegerRandomSampling`, `InversionMutation`, `LatinHypercubeSampling`, `LHS`, `NoCrossover`, `NoMutation`, `OrderCrossover`, `ParentCentricCrossover`, `PCX`, `PermutationRandomSampling`, `PM`, `PointCrossover`, `PolynomialMutation`, `RandomSelection`, `SBX`, `SimulatedBinaryCrossover`, `SinglePointCrossover`, `SPX`, `TournamentSelection`, `TwoPointCrossover`, `UniformCrossover`, `UX`

### Problem Families (47)

`_QIAO_CEC2006`, `BBOB`, `BT`, `CEC2008`, `CEC2013`, `CEC2020`, `CF`, `DASCMOP`, `DOC`, `DSMOP`, `DTLZ`, `FCP`, `FDA`, `GLSMOP`, `IMMOEA`, `IMOP`, `LIRCMOP`, `LRMOP`, `LSCM`, `LSMMOP`, `LSMOP`, `MAF`, `MAOPP`, `MMF`, `MMMOP`, `MOEADDE`, `MOEADM2M`, `MULTITASKING_MOPS`, `MULTITASKING_SOPS`, `MW`, `REALWORLD_MOPS`, `REALWORLD_SOPS`, `RMMEDA`, `RWMOP`, `SDC`, `SIMPLE_SOPS`, `SMD`, `SMMOP`, `SMOP`, `SPARSE_REALWORLD_MOPS`, `TP`, `UF`, `VNT`, `WFG`, `ZCAT`, `ZDT`, `ZXH_CF`

---

## How to Create Plugins

### Creating a New Algorithm

1. Create a subdirectory in `algorithms/`:

```
algorithms/
└── my_algorithm/
    ├── __init__.py
    └── my_algorithm.py
```

2. Implement the algorithm in `my_algorithm.py`:

```python
"""Meu Algoritmo Customizado para PymooLab."""
from __future__ import annotations
import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from core.individual import Individual

# Flags de escopo: indica quais tipos de problema o algoritmo suporta
ALGORITHM_FLAGS = {
    "MyAlgorithm": {"multi", "many"},  # "single", "multi", "many", "neural"
}


class MyAlgorithm(Algorithm):
    """Algoritmo customizado para otimização multiobjetivo."""

    def __init__(self, pop_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size

    def _setup(self, problem, **kwargs):
        """Armazena metadados do problema."""
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.xl = problem.xl
        self.xu = problem.xu

    def _initialize_infill(self):
        """Gera população inicial aleatória."""
        X = np.random.uniform(self.xl, self.xu, (self.pop_size, self.n_var))
        return Population.new("X", X)

    def _infill(self):
        """Gera novas soluções candidatas a cada iteração."""
        # Implementar lógica evolutiva aqui
        X = np.random.uniform(self.xl, self.xu, (self.pop_size, self.n_var))
        return Population.new("X", X)

    def _advance(self, infills=None, **kwargs):
        """Atualiza a população com base nas soluções avaliadas."""
        # Implementar seleção e sobrevivência
        self.pop = infills
```

3. Export in `__init__.py`:

```python
from .my_algorithm import MyAlgorithm

__all__ = ["MyAlgorithm"]
```

4. Alternatively, export a factory function `create_algorithm(config: dict) → Algorithm` for more control.

### Creating a New Problem

1. Create a `.py` file in one of the problem subdirectories (`problems/multi/`, `problems/many/`, `problems/single/`):

```python
"""Meu Problema de Otimização Customizado."""
from __future__ import annotations
import numpy as np
from pymoo.core.problem import Problem


class MyProblem(Problem):
    """Problema bi-objetivo com restrição de caixa."""

    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(
            n_var=int(n_var),
            n_obj=2,           # Número de objetivos
            n_ieq_constr=0,    # Restrições de desigualdade
            xl=0.0,            # Limite inferior
            xu=1.0,            # Limite superior
            vtype=float,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Avalia a função objetivo para toda a população."""
        x = np.asarray(x, dtype=float)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        """Retorna a fronteira de Pareto analítica (se conhecida)."""
        x = np.linspace(0.0, 1.0, n_pareto_points)
        return np.column_stack([x, 1.0 - np.sqrt(x)])


# Para suporte JAX, basta criar variante:
# class MyProblem_JAX(MyProblem):
#     _USE_JAX = True

__all__ = ["MyProblem"]
```

### Creating a New Metric

1. Add the metric function to `metrics/community_metrics.py`, or create a new `.py` file in `metrics/`:

```python
"""Minha Métrica de Qualidade Customizada."""
from __future__ import annotations
import numpy as np
from typing import Any


def _metric_MyMetric(front: np.ndarray, context: dict[str, Any]) -> float:
    """
    Calcula uma métrica customizada sobre a fronteira obtida.

    Parâmetros
    ----------
    front : np.ndarray, forma (N, M)
        Fronteira de Pareto obtida (N soluções, M objetivos).
    context : dict
        Contexto com 'problem', 'reference_front', 'reference_set', etc.

    Retorna
    -------
    float
        Valor da métrica.
    """
    pop_obj = np.asarray(front, dtype=float)
    if pop_obj.ndim == 1:
        pop_obj = pop_obj[np.newaxis, :]

    # Exemplo: distância média ao ponto ideal
    ideal = np.min(pop_obj, axis=0)
    distances = np.sqrt(np.sum((pop_obj - ideal) ** 2, axis=1))
    return float(np.mean(distances))


# Registrar no dicionário METRICS (convenção do plugin system):
METRICS = {
    "MyMetric": _metric_MyMetric,
}
```

### Creating a New Operator

1. Create a `.py` file in the appropriate `operators/` subdirectory (`crossover/`, `mutation/`, `selection/`, `sampling/`):

```python
"""Meu Operador de Crossover Customizado."""
from __future__ import annotations
import numpy as np
from pymoo.core.crossover import Crossover


class MyCrossover(Crossover):
    """Crossover aritmético simples."""

    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
        self.alpha = alpha

    def _do(self, problem, X, **kwargs):
        """Executa o crossover entre pares de pais."""
        n_matings, n_parents, n_var = X.shape
        Y = np.full((n_matings, self.n_offsprings, n_var), np.nan)

        for i in range(n_matings):
            p1, p2 = X[i, 0], X[i, 1]
            Y[i, 0] = self.alpha * p1 + (1 - self.alpha) * p2
            Y[i, 1] = (1 - self.alpha) * p1 + self.alpha * p2

        return Y
```

---

## Running PymooLab

```bash
python PymooLab.py
```

The application starts in maximized mode with a light theme. All plugin directories are automatically created if missing.

---

## Local LLM Integration (LM Studio)

PymooLab supports LLM-guided evolutionary algorithms (such as `LARC_NSGA3`) that use local quantized LLMs to adapt evolutionary operators at runtime. This integration relies on a local server compatible with the OpenAI API structure, such as **LM Studio**.

### Setup Instructions

1. **Install LM Studio**: Download and install LM Studio.
2. **Download a Model**: Download a lightweight instruction-following model, for example `qwen3.5-0.8b` or `qwen2.5-1.5b`.
3. **Start the Local Server**:
   - Click on the **Developer/Server** tab (represented by the `<>` icon on the left sidebar).
   - Select your downloaded model from the dropdown at the top.
   - Click **Start Server**. The server will listen on port `1234` by default (`http://127.0.0.1:1234/v1`).
4. **Configure Environment (Optional)**:
   - PymooLab will automatically query the LM Studio local server models endpoint (`/v1/models`) to detect the loaded model.
   - You can also explicitly configure the model ID or custom port url via environment variables:
     ```bash
     export PYMOOLAB_LARC_MODEL="qwen3.5-0.8b-mlx"
     export PYMOOLAB_LMSTUDIO_URL="http://127.0.0.1:1234/v1/chat/completions"
     ```

### Running LARC-NSGA3

Use `LARC_NSGA3` from the GUI experiment runner after LM Studio is active and a model is loaded. For ad hoc CLI experiments, import the plugin directly from the algorithm package:

```python
from algorithms.larc_nsga3 import LARC_NSGA3

algorithm = LARC_NSGA3(pop_size=100)
```

Benchmark scripts under `tests/` are local validation assets and are not included in release commits.

---

## Manual Continuity Note

This map is intentionally exhaustive and should be preserved as the baseline for the full PDF manual. If the codebase changes, regenerate this map to keep the documentation synchronized.

> **METISBr** — A Brazilian research group dedicated to Multi-Objective and Many-Objective Optimization (MaOPs)  
> UFOP, Brazil
