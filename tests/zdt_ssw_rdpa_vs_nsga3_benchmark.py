import argparse
import csv
import inspect
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.array_backend import xp as np
from algorithms.moo.nsga3 import NSGA3
from optimize import minimize
from problems import get_problem
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions
from util.array_backend import CUPY_AVAILABLE, get_cupy_device_count, get_cupy_device_name
from util.stats_backend import wilcoxon

from algorithms.ssw_rdpa.ssw_rdpa import SSW_RDPA

ZDT_ALL_FUNCTIONS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6"]
DEFAULT_ZDT_FUNCTIONS = ["zdt1"]
ALGORITHMS = ("NSGA-III", "SSW-RDPA")
N_OBJ = 2
ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt5": 11, "zdt6": 10}


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


def _random_seed_list(n: int) -> List[int]:
    rng = random.SystemRandom()
    seen = set()
    seeds = []
    while len(seeds) < n:
        s = rng.randrange(1, 2_147_483_647)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def _random_seed_pairs(n: int) -> List[Tuple[int, int]]:
    seeds = _random_seed_list(2 * n)
    return [(seeds[2 * i], seeds[2 * i + 1]) for i in range(n)]


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


def _directed_igd_p(source: np.ndarray, target: np.ndarray, p: float = 1.0) -> float:
    if len(source) == 0 or len(target) == 0:
        return float("inf")
    diff = source[:, None, :] - target[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mins = np.min(d, axis=1)
    if p == 1.0:
        return float(np.mean(mins))
    return float(np.mean(mins**p) ** (1.0 / p))


def delta_p(approx_set: np.ndarray, reference_set: np.ndarray, p: float = 1.0) -> float:
    return max(
        _directed_igd_p(approx_set, reference_set, p=p),
        _directed_igd_p(reference_set, approx_set, p=p),
    )


def _wilcoxon_marker(comp: np.ndarray, base: np.ndarray, alpha: float = 0.05) -> Tuple[str, float]:
    comp = np.asarray(comp, dtype=float)
    base = np.asarray(base, dtype=float)
    if len(comp) != len(base) or len(comp) < 2:
        return "=", 1.0
    if np.allclose(comp, base, atol=1e-15, rtol=0.0):
        return "=", 1.0
    try:
        _, pval = wilcoxon(comp, base, zero_method="wilcox", correction=False, alternative="two-sided")
    except Exception:
        if np.mean(comp) < np.mean(base):
            return "+", float("nan")
        if np.mean(comp) > np.mean(base):
            return "-", float("nan")
        return "=", float("nan")
    if pval >= alpha:
        return "=", float(pval)
    return ("+" if np.mean(comp) < np.mean(base) else "-"), float(pval)


def _write_csv(path: Path, rows: List[Dict], headers: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    kwargs: Dict = {"ref_dirs": ref_dirs, "pop_size": int(pop_size)}
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:  # noqa: BLE001
        sig = None

    if sig is None:
        return kwargs

    if "seed" in sig.parameters:
        kwargs["seed"] = int(seed)
    if "use_gpu" in sig.parameters:
        kwargs["use_gpu"] = bool(use_gpu)
    if "array_backend" in sig.parameters:
        kwargs["array_backend"] = str(array_backend)
    if "gpu_dtype" in sig.parameters:
        kwargs["gpu_dtype"] = str(gpu_dtype)

    return kwargs


def _run_one_seed(
    func_name: str,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    run_id: int,
    seed_nsga3: int,
    seed_ssw_rdpa: int,
    use_gpu: bool = True,
    gpu_dtype: str = "float32",
) -> Dict:
    if func_name == "zdt5":
        problem = get_problem(func_name)
    else:
        problem = get_problem(func_name, n_var=ZDT_N_VAR[func_name])

    gpu_ok = bool(use_gpu and CUPY_AVAILABLE and get_cupy_device_count() > 0)
    array_backend = "cupy" if gpu_ok else "numpy"
    gpu_label = get_cupy_device_name(0) or "CUDA GPU"
    output = {"run_id": int(run_id), "n_var": int(ZDT_N_VAR.get(func_name, 11)), "algo": {}}
    output["exec"] = {
        "backend_requested_code": "gpu" if bool(use_gpu) else "cpu",
        "backend_requested_info": (
            f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})" if bool(use_gpu) else "CPU only"
        ),
        "cupy_available": bool(CUPY_AVAILABLE),
        "cupy_devices": int(get_cupy_device_count()),
    }

    algo_nsga = NSGA3(
        **_algo_kwargs(
            NSGA3,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed_nsga3),
            use_gpu=gpu_ok,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
        )
    )
    nsga_gpu = bool(getattr(algo_nsga, "use_gpu", False))

    algo_ssw = SSW_RDPA(
        **_algo_kwargs(
            SSW_RDPA,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed_ssw_rdpa),
            use_gpu=gpu_ok,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
        )
    )
    ssw_gpu = bool(getattr(algo_ssw, "use_gpu", False))
    output["exec"]["backend_code"] = "gpu" if ssw_gpu else "cpu"
    output["exec"]["backend_info"] = (
        f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})" if ssw_gpu else "CPU only"
    )

    t0 = time.time()
    res_nsga = minimize(
        problem,
        algo_nsga,
        ("n_eval", n_evals),
        seed=int(seed_nsga3),
        verbose=False,
    )
    output["algo"]["NSGA-III"] = {
        "seed": int(seed_nsga3),
        "time_s": float(time.time() - t0),
        "F": _safe_F(res_nsga.F, N_OBJ),
        "backend_code": "gpu" if nsga_gpu else "cpu",
        "backend_info": (
            f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})"
            if nsga_gpu
            else "CPU only (no GPU kernels in NSGA-III)"
        ),
    }

    t1 = time.time()
    res_ssw = minimize(
        problem,
        algo_ssw,
        ("n_eval", n_evals),
        seed=int(seed_ssw_rdpa),
        verbose=False,
    )
    output["algo"]["SSW-RDPA"] = {
        "seed": int(seed_ssw_rdpa),
        "time_s": float(time.time() - t1),
        "F": _safe_F(res_ssw.F, N_OBJ),
        "backend_code": "gpu" if ssw_gpu else "cpu",
        "backend_info": (
            f"GPU CUDA via CuPy ({gpu_label}, {gpu_dtype})" if ssw_gpu else "CPU only"
        ),
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
    use_gpu: bool = True,
    gpu_dtype: str = "float32",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    if pop_size <= 1:
        raise ValueError("pop_size must be greater than 1 for ZDT (m=2).")

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=pop_size - 1)
    real_pop = len(ref_dirs)
    gpu_ready = bool(CUPY_AVAILABLE and get_cupy_device_count() > 0)

    print(
        f"[ZDT][m={N_OBJ}] setup: pop={real_pop}, runs={n_runs}, evals={n_evals}, "
        f"backend=process, workers={parallel_workers}, "
        f"gpu_requested={use_gpu}, gpu_ready={gpu_ready}, gpu_dtype={gpu_dtype}, "
        f"functions={','.join(zdt_functions)}"
    )

    for func_name in zdt_functions:
        print(f"[ZDT][m={N_OBJ}] running {func_name} ...")
        seed_pairs = _random_seed_pairs(n_runs)

        for rid, (seed_nsga3, seed_ssw_rdpa) in enumerate(seed_pairs, start=1):
            seed_rows.append(
                {
                    "problem": func_name,
                    "m": int(N_OBJ),
                    "run_id": int(rid),
                    "seed_nsga3": int(seed_nsga3),
                    "seed_ssw_rdpa": int(seed_ssw_rdpa),
                }
            )

        jobs = [
            (func_name, n_evals, ref_dirs, real_pop, rid, seed_nsga3, seed_ssw_rdpa, use_gpu, gpu_dtype)
            for rid, (seed_nsga3, seed_ssw_rdpa) in enumerate(seed_pairs, start=1)
        ]
        results: List[Dict] = []

        with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {ex.submit(_run_one_seed, *job): (job[4], job[5], job[6]) for job in jobs}
            for fut in as_completed(futures):
                run_id, seed_nsga3, seed_ssw_rdpa = futures[fut]
                out = fut.result()
                results.append(out)
                print(
                    f"[ZDT][m={N_OBJ}][{func_name}] run={run_id} done "
                    f"(seed_nsga={seed_nsga3}, seed_ssw={seed_ssw_rdpa}, "
                    f"nsga={out['algo']['NSGA-III']['time_s']:.2f}s, "
                    f"ssw={out['algo']['SSW-RDPA']['time_s']:.2f}s, "
                    f"exec_nsga={out['algo']['NSGA-III']['backend_code']}, "
                    f"exec_ssw={out['algo']['SSW-RDPA']['backend_code']})"
                )

        results.sort(key=lambda d: d["run_id"])

        F_batches = [out["algo"][alg]["F"] for out in results for alg in ALGORITHMS]
        reference_set = _build_empirical_reference(F_batches, max_points=4000)
        reference_source = "empirical_nd"

        vals = {alg: [] for alg in ALGORITHMS}
        times = {alg: [] for alg in ALGORITHMS}
        for out in results:
            rid = int(out["run_id"])
            for alg in ALGORITHMS:
                dp = delta_p(out["algo"][alg]["F"], reference_set, p=delta_p_order)
                dt = float(out["algo"][alg]["time_s"])
                algo_seed = int(out["algo"][alg]["seed"])
                vals[alg].append(dp)
                times[alg].append(dt)
                run_rows.append(
                    {
                        "problem": func_name,
                        "m": int(N_OBJ),
                        "run_id": int(rid),
                        "seed": algo_seed,
                        "n_var": int(out["n_var"]),
                        "pop_size": int(real_pop),
                        "n_evals": int(n_evals),
                        "reference_source": reference_source,
                        "algorithm": alg,
                        "delta_p": float(dp),
                        "time_s": dt,
                        "backend_code": str(out["algo"][alg].get("backend_code", "cpu")),
                        "backend_info": str(out["algo"][alg].get("backend_info", "CPU only")),
                    }
                )

        mean_nsga = float(np.mean(vals["NSGA-III"]))
        mean_ssw = float(np.mean(vals["SSW-RDPA"]))
        std_nsga = float(np.std(vals["NSGA-III"], ddof=1)) if len(vals["NSGA-III"]) > 1 else 0.0
        std_ssw = float(np.std(vals["SSW-RDPA"], ddof=1)) if len(vals["SSW-RDPA"]) > 1 else 0.0
        mark_nsga, p_nsga = _wilcoxon_marker(np.asarray(vals["NSGA-III"]), np.asarray(vals["SSW-RDPA"]))

        if mean_ssw < mean_nsga:
            winner_algo = "SSW-RDPA"
            loser_algo = "NSGA-III"
            winner_val = mean_ssw
            loser_val = mean_nsga
        else:
            winner_algo = "NSGA-III"
            loser_algo = "SSW-RDPA"
            winner_val = mean_nsga
            loser_val = mean_ssw
        winner_margin_pct_over_loser = 100.0 * (loser_val - winner_val) / max(abs(loser_val), 1e-32)

        summary_rows.append(
            {
                "problem": func_name,
                "m": int(N_OBJ),
                "pop_size": int(real_pop),
                "n_runs": int(n_runs),
                "n_evals": int(n_evals),
                "reference_source": reference_source,
                "mean_delta_p_nsga3": mean_nsga,
                "std_delta_p_nsga3": std_nsga,
                "mean_delta_p_ssw": mean_ssw,
                "std_delta_p_ssw": std_ssw,
                "nsga3_vs_ssw_marker": mark_nsga,
                "pvalue_nsga3_vs_ssw": p_nsga,
                "mean_time_nsga3_s": float(np.mean(times["NSGA-III"])),
                "mean_time_ssw_s": float(np.mean(times["SSW-RDPA"])),
                "requested_backend": "gpu" if use_gpu else "cpu",
                "gpu_dtype": str(gpu_dtype),
                "best_algo": winner_algo,
                "winner_algo": winner_algo,
                "loser_algo": loser_algo,
                "winner_delta_p": winner_val,
                "loser_delta_p": loser_val,
                "winner_margin_pct_over_loser": winner_margin_pct_over_loser,
            }
        )

        print(
            f"[ZDT][m={N_OBJ}] {func_name} done -> ref={reference_source}, "
            f"mean_delta_p: NSGA={mean_nsga:.4e}, SSW={mean_ssw:.4e}, "
            f"winner={winner_algo}, margin={winner_margin_pct_over_loser:.2f}%"
        )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["problem", "m", "run_id", "seed_nsga3", "seed_ssw_rdpa"],
    )
    _write_csv(
        out_dir / "all_runs_delta_p.csv",
        run_rows,
        [
            "problem",
            "m",
            "run_id",
            "seed",
            "n_var",
            "pop_size",
            "n_evals",
            "reference_source",
            "algorithm",
            "delta_p",
            "time_s",
            "backend_code",
            "backend_info",
        ],
    )
    _write_csv(
        out_dir / "summary_delta_p.csv",
        summary_rows,
        [
            "problem",
            "m",
            "pop_size",
            "n_runs",
            "n_evals",
            "reference_source",
            "mean_delta_p_nsga3",
            "std_delta_p_nsga3",
            "mean_delta_p_ssw",
            "std_delta_p_ssw",
            "nsga3_vs_ssw_marker",
            "pvalue_nsga3_vs_ssw",
            "mean_time_nsga3_s",
            "mean_time_ssw_s",
            "requested_backend",
            "gpu_dtype",
            "best_algo",
            "winner_algo",
            "loser_algo",
            "winner_delta_p",
            "loser_delta_p",
            "winner_margin_pct_over_loser",
        ],
    )
    print(f"[DONE] Outputs written to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="ZDT-only benchmark: SSW-RDPA vs NSGA-III (Delta_p).")
    parser.add_argument(
        "--functions",
        type=str,
        default="zdt1",
        help="Functions to run (default: zdt1). Use comma-separated values (e.g. zdt1,zdt2) or 'all'.",
    )
    parser.add_argument("--n-runs", "--runs", dest="n_runs", type=int, default=5, help="Independent runs per function.")
    parser.add_argument("--n-evals", "--evals", dest="n_evals", type=int, default=25000, help="Max evaluations per run.")
    parser.add_argument(
        "--parallel-workers",
        "--workers",
        dest="parallel_workers",
        type=int,
        default=8,
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
        "--out-dir",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "artigo",
                "results",
                "zdt_ssw_rdpa_vs_nsga3",
            )
        ),
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_functions = _parse_zdt_functions(args.functions)
    run_experiment(
        out_dir=Path(args.out_dir),
        n_runs=int(args.n_runs),
        n_evals=int(args.n_evals),
        parallel_workers=int(args.parallel_workers),
        delta_p_order=float(args.delta_p_order),
        pop_size=int(args.pop_size),
        zdt_functions=selected_functions,
        use_gpu=bool(args.gpu),
        gpu_dtype=str(args.gpu_dtype).lower(),
    )


if __name__ == "__main__":
    main()
