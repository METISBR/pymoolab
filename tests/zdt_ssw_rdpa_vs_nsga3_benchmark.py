import argparse
import csv
import inspect
import json
import os
import random
import re
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np

# Suprimir avisos de tempo de execução e divisões por zero comuns em benchmarks MOO
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cmoea_gbss.cmoea_gbss import CMOEA_GBSS
from algorithms.cmoea_cd.cmoea_cd import CMOEA_CD
from algorithms.nsga3_local.nsga3_local import NSGA3Local
from optimize import minimize
from pymoo.problems import get_problem
from pymoolab_core.analysis.stat_tests import run_wilcoxon
from util.array_backend import CUPY_AVAILABLE, get_cupy_device_name
from util.nds.non_dominated_sorting import NonDominatedSorting

# Importações unificadas com PymooLab para garantir consistência
from PymooLab import build_reference_dirs, instantiate_algorithm_class

ZDT_ALL_FUNCTIONS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DEFAULT_ZDT_FUNCTIONS = ["zdt1", "zdt2", "zdt3"]

ALGS = {
    "CMOEA_GBSS": CMOEA_GBSS,
    "CMOEA_CD": CMOEA_CD,
    "NSGA3Local": NSGA3Local
}
ALG_NAMES = list(ALGS.keys())

N_OBJ = 2
ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
SEED_MIN_8DIGIT = 10_000_000
SEED_MAX_8DIGIT = 99_999_999


def _get_cupy_device_count_safe() -> int:
    if not bool(CUPY_AVAILABLE):
        return 0
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _parse_zdt_functions(raw: str) -> List[str]:
    token = str(raw).strip().lower()
    if token == "all":
        return list(ZDT_ALL_FUNCTIONS)

    parts = [p.strip().lower() for p in re.split(r"[,\s;|]+", token) if p.strip()]
    if not parts:
        raise ValueError("No ZDT function selected. Use --functions zdt1 or --functions all.")

    selected: List[str] = []
    valid = set(ZDT_ALL_FUNCTIONS)
    for name in parts:
        if name not in valid:
            raise ValueError(
                f"Unknown ZDT function '{name}'. Valid values: {', '.join(ZDT_ALL_FUNCTIONS)} or 'all'."
            )
        if name not in selected:
            selected.append(name)
    return selected


def _random_seed_list_with_rng(n: int, rng) -> List[int]:
    seen = set()
    seeds = []
    while len(seeds) < n:
        s = int(rng.randrange(SEED_MIN_8DIGIT, SEED_MAX_8DIGIT + 1))
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def _scope_rng(seed_master: Optional[int], scope: str):
    if seed_master is None:
        return random.SystemRandom()
    key = f"{int(seed_master)}::{scope}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed)


def _random_seeds_for_runs(
    n: int,
    *,
    seed_master: Optional[int] = None,
    scope: str = "",
) -> List[int]:
    rng = _scope_rng(seed_master, scope)
    return _random_seed_list_with_rng(n, rng)


def _load_json_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSON config file not found: {p}")
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config at {p} must be an object.")
    return data


def _safe_F(F: np.ndarray, n_obj: int) -> np.ndarray:
    arr = np.asarray(F, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return np.full((1, n_obj), np.inf, dtype=float)
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if len(arr) == 0:
        return np.full((1, n_obj), np.inf, dtype=float)
    return arr


def _build_empirical_reference(F_batches: List[np.ndarray], max_points: int = 4000) -> np.ndarray:
    """Constrói referência empírica (mantida como fallback)."""
    F = np.vstack(F_batches)
    finite = np.all(np.isfinite(F), axis=1)
    F = F[finite]
    if len(F) == 0:
        return np.full((1, F_batches[0].shape[1]), np.inf, dtype=float)
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    ref = F[np.asarray(nd_idx, dtype=int)]
    if len(ref) > max_points:
        pick = np.linspace(0, len(ref) - 1, num=max_points, dtype=int)
        ref = ref[pick]
    return ref


def _get_problem_pareto_front(func_name: str, n_var: int, n_obj: int) -> Optional[np.ndarray]:
    """Obtém fronteira Pareto analítica do problema (igual ao PymooLab)."""
    try:
        problem = get_problem(func_name, n_var=n_var)
        pf = None
        for method_name in ["pareto_front", "_calc_pareto_front"]:
            fn = getattr(problem, method_name, None)
            if not callable(fn):
                continue
            for kwargs in [{}, {"n_pareto_points": 500}]:
                try:
                    values = fn(**kwargs)
                except Exception:
                    continue
                if values is not None:
                    arr = np.asarray(values, dtype=float)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    if arr.size > 0 and arr.shape[1] == n_obj:
                        pf = arr
                        break
            if pf is not None:
                break
        return pf
    except Exception:
        return None


from metrics.community_metrics import _deltap_value as pymoolab_deltap_value


def delta_p(approx_set: np.ndarray, reference_set: np.ndarray, p: float = 1.0) -> float:
    _ = p
    return float(pymoolab_deltap_value(approx_set, reference_set))


def _wilcoxon_marker(comp: np.ndarray, base: np.ndarray, alpha: float = 0.05) -> Tuple[str, float]:
    comp = np.asarray(comp, dtype=float)
    base = np.asarray(base, dtype=float)
    if len(comp) != len(base) or len(comp) < 2:
        return "=", 1.0
    if np.allclose(comp, base, atol=1e-15, rtol=0.0):
        return "=", 1.0

    pval = float("nan")
    try:
        result = run_wilcoxon(
            comp,
            base,
            alpha=alpha,
            higher_better=False,
            min_samples=2,
        )
        decision = str(result.get("decision", "?")).strip()
        pval = float(result.get("p_value", float("nan")))
        if decision in {"+", "-", "="} and np.isfinite(pval):
            return decision, pval
    except Exception:
        pass

    if np.mean(comp) < np.mean(base):
        return "+", pval
    if np.mean(comp) > np.mean(base):
        return "-", pval
    return "=", pval


def _write_csv(path: Path, rows: List[Dict], headers: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_debug_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        abs_v = abs(value)
        if abs_v == 0.0:
            return "0.0"
        if abs_v >= 1e3 or abs_v < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def _print_debug_table(title: str, rows: List[Dict], columns: List[Tuple[str, str]]) -> None:
    if not rows:
        print(f"\033[1;33m[DEBUG][TABLE] {title}: no rows\033[0m")
        return

    table_rows: List[List[str]] = []
    widths: List[int] = [len(label) for _, label in columns]

    for row in rows:
        formatted = []
        for col_idx, (key, _) in enumerate(columns):
            text = _fmt_debug_value(row.get(key, ""))
            formatted.append(text)
            widths[col_idx] = max(widths[col_idx], len(text))
        table_rows.append(formatted)

    # Caracteres de desenho de caixa (UTF-8)
    top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
    mid = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"
    header = "│ " + " │ ".join(label.ljust(widths[i]) for i, (_, label) in enumerate(columns)) + " │"

    print(f"\n\033[1;36m[SUMMARY] {title}\033[0m")
    print(top)
    print(header)
    print(mid)
    for row in table_rows:
        print("│ " + " │ ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " │")
    print(bot)


def _algo_kwargs(
    cls: type,
    *,
    ref_dirs: np.ndarray,
    pop_size: int,
    seed: int,
    use_gpu: bool,
    array_backend: str,
    gpu_dtype: str,
) -> Dict:
    """Constrói config dict e usa instantiate_algorithm_class (igual ao PymooLab)."""
    config: Dict = {
        "pop_size": int(pop_size),
        "ref_dirs": ref_dirs,
        "seed": int(seed),
        "use_gpu": bool(use_gpu),
        "array_backend": str(array_backend),
        "gpu_dtype": str(gpu_dtype),
        "n_obj": int(ref_dirs.shape[1]) if ref_dirs is not None else 2,
    }
    return config


def _run_one_seed(
    func_name: str,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    run_id: int,
    seed: int,
    use_gpu: bool = True,
    gpu_dtype: str = "float32",
    config_kwargs: Optional[Dict] = None,
) -> Dict:
    problem = get_problem(func_name, n_var=ZDT_N_VAR[func_name])

    gpu_ok = bool(use_gpu and CUPY_AVAILABLE and _get_cupy_device_count_safe() > 0)
    array_backend = "cupy" if gpu_ok else "numpy"
    gpu_label = get_cupy_device_name(0) or "CUDA GPU"
    
    output = {
        "run_id": int(run_id), 
        "n_var": int(ZDT_N_VAR.get(func_name, 30)), 
        "algo": {},
        "exec": {
            "backend_requested_code": "gpu" if bool(use_gpu) else "cpu",
            "backend_requested_info": f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})" if bool(use_gpu) else "CPU only",
            "cupy_available": bool(CUPY_AVAILABLE),
            "cupy_devices": int(_get_cupy_device_count_safe()),
        }
    }

    config_kwargs = config_kwargs or {}

    for name, cls in ALGS.items():
        # Prepara config dict (igual ao PymooLab)
        config = _algo_kwargs(
            cls,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed),
            use_gpu=gpu_ok,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
        )
        
        # Merge de configurações extra se houver para o algoritmo específico
        if name in config_kwargs:
            config.update(config_kwargs[name])

        # Usa instantiate_algorithm_class (igual ao PymooLab)
        algo_inst = instantiate_algorithm_class(cls, config)
        
        t0 = time.time()
        res = minimize(
            problem,
            algo_inst,
            ("n_eval", n_evals),
            seed=int(seed),
            verbose=False,
        )
        dt = time.time() - t0
        
        actual_gpu = bool(getattr(algo_inst, "use_gpu", False))
        
        output["algo"][name] = {
            "seed": int(seed),
            "time_s": float(dt),
            "F": _safe_F(res.F, N_OBJ),
            "backend_code": "gpu" if actual_gpu else "cpu",
            "backend_info": f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})" if actual_gpu else "CPU only",
        }

    return output


def run_experiment(
    out_dir: Path,
    n_runs: int,
    n_evals: int,
    parallel_workers: int,
    delta_p_order: float,
    pop_size: int,
    zdt_functions: List[str],
    seed_master: Optional[int] = None,
    candidate_kwargs: Optional[Dict] = None,
    use_gpu: bool = True,
    gpu_dtype: str = "float32",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    if pop_size <= 1:
        raise ValueError("pop_size must be greater than 1 for ZDT (m=2).")

    # CAUSA 3: Usa build_reference_dirs (igual ao PymooLab)
    ref_dirs = build_reference_dirs(n_obj=N_OBJ, target=max(12, pop_size))
    real_pop = len(ref_dirs)
    gpu_ready = bool(CUPY_AVAILABLE and _get_cupy_device_count_safe() > 0)

    print(
        f"[ZDT][m={N_OBJ}] setup: pop={real_pop}, runs={n_runs}, evals={n_evals}, "
        f"workers={parallel_workers}, gpu_requested={use_gpu}, "
        f"functions={','.join(zdt_functions)}, seed_master={seed_master}"
    )

    for func_name in zdt_functions:
        print(f"[ZDT][m={N_OBJ}] running {func_name} ...")
        seeds = _random_seeds_for_runs(
            n_runs,
            seed_master=seed_master,
            scope=f"{func_name}|runs={n_runs}|evals={n_evals}",
        )

        for rid, s in enumerate(seeds, start=1):
            seed_rows.append({
                "problem": func_name,
                "m": int(N_OBJ),
                "run_id": int(rid),
                "seed": int(s),
                "seed_master": int(seed_master) if seed_master is not None else "",
            })

        jobs = [
            (func_name, n_evals, ref_dirs, real_pop, rid, s, use_gpu, gpu_dtype, candidate_kwargs)
            for rid, s in enumerate(seeds, start=1)
        ]
        
        results: List[Dict] = []
        with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {ex.submit(_run_one_seed, *job): job[4] for job in jobs}
            for fut in as_completed(futures):
                rid = futures[fut]
                out = fut.result()
                results.append(out)
                t_str = ", ".join(f"{nm}={out['algo'][nm]['time_s']:.2f}s" for nm in ALG_NAMES)
                print(f"[ZDT][m={N_OBJ}][{func_name}] run={rid} done ({t_str})")

        results.sort(key=lambda d: d["run_id"])

        # CAUSA 1: Usa a fronteira Pareto analítica do problema (igual ao PymooLab)
        pf_analytic = _get_problem_pareto_front(
            func_name, 
            n_var=ZDT_N_VAR.get(func_name, 30), 
            n_obj=N_OBJ,
        )
        
        if pf_analytic is not None:
            reference_set = pf_analytic
        else:
            # Fallback: referência empírica (se analítica indisponível)
            F_batches = [out["algo"][alg]["F"] for out in results for alg in ALG_NAMES]
            reference_set = _build_empirical_reference(F_batches, max_points=4000)

        vals = {alg: [] for alg in ALG_NAMES}
        times = {alg: [] for alg in ALG_NAMES}
        for out in results:
            rid = int(out["run_id"])
            for alg in ALG_NAMES:
                dp = delta_p(out["algo"][alg]["F"], reference_set, p=delta_p_order)
                dt = float(out["algo"][alg]["time_s"])
                vals[alg].append(dp)
                times[alg].append(dt)
                run_rows.append({
                    "problem": func_name,
                    "m": int(N_OBJ),
                    "run_id": int(rid),
                    "seed": int(out["algo"][alg]["seed"]),
                    "algorithm": alg,
                    "delta_p": float(dp),
                    "time_s": dt,
                })

        # Calcula o melhor algoritmo por média de Delta_p
        means = {alg: float(np.mean(vals[alg])) for alg in ALG_NAMES}
        best_alg = min(means, key=lambda k: means[k])
        
        row_sum = {
            "problem": func_name,
            "m": int(N_OBJ),
            "pop_size": int(real_pop),
            "n_runs": int(n_runs),
            "n_evals": int(n_evals),
            "best_algo": best_alg,
        }

        for alg in ALG_NAMES:
            m = means[alg]
            s = float(np.std(vals[alg], ddof=1))
            row_sum[f"disp_{alg}"] = f"{m:.2e} ± {s:.1e}"
            row_sum[f"mean_dp_{alg}"] = m
            row_sum[f"std_dp_{alg}"] = s
            row_sum[f"mean_time_{alg}_s"] = float(np.mean(times[alg]))
            
            # Wilcoxon contra o melhor
            if alg == best_alg:
                row_sum[f"{alg}_vs_best_marker"] = "="
                row_sum[f"{alg}_vs_best_pvalue"] = 1.0
            else:
                marker, pval = _wilcoxon_marker(np.asarray(vals[alg]), np.asarray(vals[best_alg]))
                row_sum[f"{alg}_vs_best_marker"] = marker
                row_sum[f"{alg}_vs_best_pvalue"] = pval

        summary_rows.append(row_sum)

        msg = " | ".join(f"{alg}={means[alg]:.4e}" for alg in ALG_NAMES)
        print(f"[ZDT][m={N_OBJ}] {func_name} done -> {msg} | \033[1;32mbest={best_alg}\033[0m")

    best_cols = []
    for alg in ALG_NAMES:
        best_cols.append((f"disp_{alg}", f"{alg} (mean±std)"))
        best_cols.append((f"{alg}_vs_best_marker", "stat"))

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["problem", "m", "run_id", "seed", "seed_master"],
    )
    _write_csv(
        out_dir / "all_runs_delta_p.csv",
        run_rows,
        ["problem", "m", "run_id", "seed", "algorithm", "delta_p", "time_s"],
    )
    
    summary_headers = ["problem", "m", "pop_size", "n_runs", "n_evals", "best_algo"]
    for alg in ALG_NAMES:
        summary_headers.extend([
            f"mean_dp_{alg}", 
            f"std_dp_{alg}", 
            f"disp_{alg}",
            f"mean_time_{alg}_s", 
            f"{alg}_vs_best_marker", 
            f"{alg}_vs_best_pvalue"
        ])
    
    _write_csv(
        out_dir / "summary_delta_p.csv",
        summary_rows,
        summary_headers,
    )
    
    _print_debug_table(
        "ZDT Summary (Delta_p Comparison)",
        summary_rows,
        [("problem", "problem"), ("best_algo", "best_algo")] + best_cols
    )
    print(f"[DONE] Outputs written to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZDT benchmark: CMOEA_GBSS vs CMOEA_CD vs NSGA3Local.",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="Functions to run (default: all). Use comma-separated values (e.g. zdt1,zdt2) or 'all'.",
    )
    parser.add_argument("--n-runs", "--runs", dest="n_runs", type=int, default=10, help="Independent runs per function.")
    parser.add_argument("--n-evals", "--evals", dest="n_evals", type=int, default=25000, help="Max evaluations per run.")
    parser.add_argument(
        "--parallel-workers",
        "--workers",
        dest="parallel_workers",
        type=int,
        default=10,
        help="Number of parallel workers.",
    )
    parser.add_argument("--gpu", action="store_true", dest="gpu", help="Request GPU acceleration (CuPy).")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Force CPU execution.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--gpu-dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Preferred GPU dtype for GPU-capable algorithms.",
    )
    parser.add_argument("--delta-p-order", type=float, default=1.0, help="Order p in Delta_p.")
    parser.add_argument(
        "--pop-size",
        "--pop",
        dest="pop_size",
        type=int,
        default=100,
        help="Population size.",
    )
    parser.add_argument(
        "--paired-seeds",
        action="store_true",
        dest="paired_seeds",
        help="Use paired seeds (same seed for all algorithms in each run).",
    )
    parser.add_argument(
        "--unpaired-seeds",
        action="store_false",
        dest="paired_seeds",
        help="Use independent seeds for each algorithm.",
    )
    parser.set_defaults(paired_seeds=False)
    parser.add_argument(
        "--seed-master",
        type=int,
        default=None,
        help="Master seed for deterministic 8-digit seed plans.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file with algorithm kwargs (e.g. {'CMOEA_GBSS': {...}}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "results",
            "zdt_gbss_cd_nsga3",
        ),
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_functions = _parse_zdt_functions(args.functions)
    candidate_kwargs = _load_json_file(args.config)
    run_experiment(
        out_dir=Path(args.out_dir),
        n_runs=int(args.n_runs),
        n_evals=int(args.n_evals),
        parallel_workers=int(args.parallel_workers),
        delta_p_order=float(args.delta_p_order),
        pop_size=int(args.pop_size),
        zdt_functions=selected_functions,
        seed_master=int(args.seed_master) if args.seed_master is not None else None,
        candidate_kwargs=candidate_kwargs,
        use_gpu=bool(args.gpu),
        gpu_dtype=str(args.gpu_dtype).lower(),
    )


if __name__ == "__main__":
    main()

