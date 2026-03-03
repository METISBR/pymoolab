import argparse
import csv
import inspect
import json
import math
import os
import random
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.array_backend import xp as np
from algorithms.moo.nsga3 import NSGA3
from optimize import minimize
from pymoo.problems import get_problem
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions
from util.array_backend import CUPY_AVAILABLE, get_cupy_device_name
from pymoolab_core.analysis.stat_tests import run_wilcoxon

from algorithms.mopso_cd_ls.mopso_cd_ssw import MOPSO_CD_SSW

DTLZ_FUNCTIONS = [f"dtlz{i}" for i in range(1, 8)]
NSGA_LABEL = "NSGA-III"
SSW_LABEL = "MOPSO_CD_SSW"
SEED_MIN_8DIGIT = 10_000_000
SEED_MAX_8DIGIT = 99_999_999

DTLZ_K = {
    "dtlz1": 5,
    "dtlz2": 10,
    "dtlz3": 10,
    "dtlz4": 10,
    "dtlz5": 10,
    "dtlz6": 10,
    "dtlz7": 20,
}


def _parse_dtlz_functions(raw: str) -> List[str]:
    token = str(raw).strip().lower()
    if token == "all":
        return list(DTLZ_FUNCTIONS)
    parts = [p.strip().lower() for p in token.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("No DTLZ function selected. Use --functions all or comma-separated list.")
    valid = set(DTLZ_FUNCTIONS)
    out: List[str] = []
    for name in parts:
        if name not in valid:
            raise ValueError(f"Unknown DTLZ function '{name}'. Valid: {', '.join(DTLZ_FUNCTIONS)} or 'all'.")
        if name not in out:
            out.append(name)
    return out





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


def _das_dennis_points(n_obj: int, n_partitions: int) -> int:
    return math.comb(n_obj + n_partitions - 1, n_partitions)


def choose_partitions(n_obj: int, target_points: int, max_partitions: int = 50) -> int:
    best_p = 1
    best_diff = abs(_das_dennis_points(n_obj, 1) - target_points)
    for p in range(1, max_partitions + 1):
        points = _das_dennis_points(n_obj, p)
        diff = abs(points - target_points)
        if diff < best_diff:
            best_diff = diff
            best_p = p
        if points > target_points and p > 1:
            break
    return best_p


def _build_problem(func_name: str, n_obj: int):
    k = DTLZ_K[func_name]
    n_var = n_obj + k - 1
    problem = get_problem(func_name, n_var=n_var, n_obj=n_obj)
    return problem, {"k": k, "n_var": n_var}


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


def _try_get_true_reference(problem, pf_ref_dirs: np.ndarray) -> Optional[np.ndarray]:
    try:
        sig = inspect.signature(problem._calc_pareto_front)
    except Exception:
        sig = inspect.signature(problem.pareto_front)

    kwargs = {}
    if "ref_dirs" in sig.parameters:
        kwargs["ref_dirs"] = pf_ref_dirs
    elif "n_pareto_points" in sig.parameters:
        kwargs["n_pareto_points"] = len(pf_ref_dirs)

    try:
        pf = problem.pareto_front(**kwargs)
        if pf is None:
            return None
        pf = np.asarray(pf, dtype=float)
        if pf.ndim != 2 or len(pf) == 0:
            return None
        finite = np.all(np.isfinite(pf), axis=1)
        pf = pf[finite]
        return pf if len(pf) > 0 else None
    except Exception:
        return None


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


def _random_seed_list(n: int) -> List[int]:
    rng = random.SystemRandom()
    seen = set()
    seeds = []
    while len(seeds) < n:
        s = rng.randrange(SEED_MIN_8DIGIT, SEED_MAX_8DIGIT + 1)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


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


def _random_seed_pairs(
    n: int,
    *,
    paired_seeds: bool = False,
    seed_master: Optional[int] = None,
    scope: str = "",
) -> List[Tuple[int, int]]:
    rng_nsga = _scope_rng(seed_master, scope + "|nsga3")
    seed_nsga = _random_seed_list_with_rng(n, rng_nsga)
    if paired_seeds:
        return [(s, s) for s in seed_nsga]
    rng_ssw = _scope_rng(seed_master, scope + "|ssw")
    seed_ssw = _random_seed_list_with_rng(n, rng_ssw)
    return list(zip(seed_nsga, seed_ssw))


def _run_one_seed(
    func_name: str,
    n_obj: int,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    run_id: int,
    seed_nsga3: int,
    seed_ssw_rdpa: int,
    use_gpu: bool = False,
    ssw_kwargs: Optional[Dict] = None,
) -> Dict:
    problem, meta = _build_problem(func_name, n_obj)
    gpu_ok = bool(use_gpu and CUPY_AVAILABLE and _get_cupy_device_count_safe() > 0)
    backend_code = "gpu" if gpu_ok else "cpu"
    backend_info = f"GPU CUDA via CuPy ({get_cupy_device_name(0) or 'CUDA GPU'})" if gpu_ok else "CPU only"

    output = {
        "run_id": int(run_id),
        "seed_nsga3": int(seed_nsga3),
        "seed_ssw_rdpa": int(seed_ssw_rdpa),
        "n_var": int(meta["n_var"]),
        "k": int(meta["k"]),
        "algo": {},
    }
    output["exec"] = {"backend_code": backend_code, "backend_info": backend_info}

    t0 = time.time()
    res_nsga = minimize(
        problem,
        NSGA3(ref_dirs=ref_dirs, pop_size=pop_size),
        ("n_eval", n_evals),
        seed=int(seed_nsga3),
        verbose=False,
    )
    output["algo"][NSGA_LABEL] = {
        "seed": int(seed_nsga3),
        "time_s": float(time.time() - t0),
        "F": _safe_F(res_nsga.F, n_obj),
    }

    ssw_user = dict(ssw_kwargs or {})
    for forced in ("ref_dirs", "pop_size", "seed", "use_gpu"):
        ssw_user.pop(forced, None)
    t1 = time.time()
    res_ssw = minimize(
        problem,
        MOPSO_CD_SSW(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed_ssw_rdpa),
            use_gpu=gpu_ok,
            **ssw_user,
        ),
        ("n_eval", n_evals),
        seed=int(seed_ssw_rdpa),
        verbose=False,
    )
    output["algo"][SSW_LABEL] = {
        "seed": int(seed_ssw_rdpa),
        "time_s": float(time.time() - t1),
        "F": _safe_F(res_ssw.F, n_obj),
    }
    return output


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
        print(f"[DEBUG][TABLE] {title}: no rows")
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

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(label.ljust(widths[i]) for i, (_, label) in enumerate(columns)) + " |"

    print("")
    print(f"[DEBUG][TABLE] {title}")
    print(sep)
    print(header)
    print(sep)
    for row in table_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |")
    print(sep)


def _print_latex_table(summary_rows: List[Dict], out_dir: Path, ssw_label: str = "SSW-RDPA") -> None:
    lines = []
    lines.append("\n" + "%" * 80)
    lines.append("% LATEX TABLE OUTPUT - DTLZ BENCHMARK")
    lines.append("% Remember to include \\usepackage{multirow} and \\usepackage{booktabs} in your preamble.")
    lines.append("%" * 80)
    lines.append("\\begin{table*}[htpb]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{Statistical Results Obtained by NSGA-III and {ssw_label} on DTLZ Problems ($\\Delta_p$)}}")
    lines.append("  \\label{tab:dtlz_benchmark}")
    lines.append("  \\begin{tabular}{c c c c}")
    lines.append("    \\toprule")
    lines.append(f"    \\textbf{{Problems}} & $m$ & \\textbf{{NSGA-III}} & \\textbf{{{ssw_label}}} \\\\")
    lines.append("    \\midrule")

    # Agrupar por função
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row['function'].upper()].append(row)

    # Ordenar por nome da função (DTLZ1, 2, ...) e depois pro m
    for func in sorted(grouped.keys()):
        rows_for_func = sorted(grouped[func], key=lambda x: x['m'])
        n_rows = len(rows_for_func)
        
        for i, row in enumerate(rows_for_func):
            m = row['m']
            nsga_mean = row['mean_delta_p_nsga3']
            nsga_std = row['std_delta_p_nsga3']
            ssw_mean = row['mean_delta_p_ssw']
            ssw_std = row['std_delta_p_ssw']
            # nsga3_vs_ssw_marker indica a posição do NSGA-III em relação ao SSW.
            # Se +, NSGA-III é menor (melhor). Se -, NSGA-III é maior (pior).
            marker_nsga = row['nsga3_vs_ssw_marker']

            def fmt_sci(val: float) -> str:
                if not math.isfinite(val):
                    return str(val)
                if val == 0:
                    return "0.0000e+00"
                s = f"{val:.4e}"
                return s

            nsga_val_str = fmt_sci(nsga_mean)
            nsga_std_str = fmt_sci(nsga_std)
            ssw_val_str = fmt_sci(ssw_mean)
            ssw_std_str = fmt_sci(ssw_std)

            # Determinando marcação de significância estatística do SSW-RDPA em relação ao NSGA-III (Baseline)
            ssw_marker = "="
            if marker_nsga == "+":
                ssw_marker = "-" # NSGA-III tem média menor/ganhou o stat-test -> SSW é significativamente pior
            elif marker_nsga == "-":
                ssw_marker = "+" # NSGA-III tem média maior/perdeu o stat-test -> SSW é significativamente melhor
                
            # Verifica qual média é menor (numericamente) para aplicar o negrito (já que queremos minimizar Delta_p)
            if nsga_mean < ssw_mean:
                nsga_str = f"\\textbf{{{nsga_val_str}}} ({nsga_std_str})"
                ssw_str = f"{ssw_val_str} ({ssw_std_str}) {ssw_marker}"
            elif ssw_mean < nsga_mean:
                nsga_str = f"{nsga_val_str} ({nsga_std_str})"
                ssw_str = f"\\textbf{{{ssw_val_str}}} ({ssw_std_str}) {ssw_marker}"
            else:
                nsga_str = f"\\textbf{{{nsga_val_str}}} ({nsga_std_str})"
                ssw_str = f"\\textbf{{{ssw_val_str}}} ({ssw_std_str}) {ssw_marker}"

            if i == 0:
                lines.append(f"    \\multirow{{{n_rows}}}{{*}}{{{func}}} & {m} & {nsga_str} & {ssw_str} \\\\")
            else:
                lines.append(f"    & {m} & {nsga_str} & {ssw_str} \\\\")

        # Linha horizontal sutil entre os problemas
        lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("%" * 80 + "\n")

    latex_str = "\n".join(lines)
    print(latex_str)
    
    latex_file = out_dir / "latex_table.tex"
    with latex_file.open("w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[INFO] LaTeX table also saved to: {latex_file}")

def run_experiment(
    out_dir: Path,
    m_values: List[int],
    n_runs: int,
    n_evals: int,
    parallel_workers: int,
    delta_p_order: float,
    pop_size_override: Optional[int],
    functions: List[str],
    paired_seeds: bool = False,
    seed_master: Optional[int] = None,
    ssw_kwargs: Optional[Dict] = None,
    use_gpu: bool = False,
) -> None:
    algorithms = (NSGA_LABEL, SSW_LABEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for m in m_values:
        pf_p = choose_partitions(m, 2000)
        if pop_size_override is not None and pop_size_override > 1:
            ref_dirs = get_reference_directions("energy", m, int(pop_size_override), seed=1)
            p_algo = "-"
        else:
            p_algo = choose_partitions(m, 126 if m <= 5 else 220)
            ref_dirs = get_reference_directions("das-dennis", m, n_partitions=p_algo)
        pf_ref_dirs = get_reference_directions("das-dennis", m, n_partitions=pf_p)
        pop_size = len(ref_dirs)

        print(
            f"[DTLZ][m={m}] setup: p_algo={p_algo}, pop={pop_size}, p_pf={pf_p}, "
            f"runs={n_runs}, evals={n_evals}, backend=process, workers={parallel_workers}, "
            f"gpu={use_gpu}, seed_mode={'paired' if paired_seeds else 'unpaired'}"
        )

        for func_name in functions:
            print(f"[DTLZ][m={m}] running {func_name} ...")
            seed_scope = f"{func_name}|m={m}|n_runs={n_runs}|n_evals={n_evals}"
            seed_pairs = _random_seed_pairs(
                n_runs,
                paired_seeds=paired_seeds,
                seed_master=seed_master,
                scope=seed_scope,
            )

            for rid, (seed_nsga3, seed_ssw_rdpa) in enumerate(seed_pairs, start=1):
                seed_rows.append(
                    {
                        "function": func_name,
                        "m": int(m),
                        "run_id": int(rid),
                        "seed_nsga3": int(seed_nsga3),
                        "seed_ssw_rdpa": int(seed_ssw_rdpa),
                        "paired": bool(seed_nsga3 == seed_ssw_rdpa),
                        "seed_master": int(seed_master) if seed_master is not None else "",
                    }
                )

            jobs = [
                (
                    func_name,
                    m,
                    n_evals,
                    ref_dirs,
                    pop_size,
                    rid,
                    seed_nsga3,
                    seed_ssw_rdpa,
                    use_gpu,
                    ssw_kwargs,
                )
                for rid, (seed_nsga3, seed_ssw_rdpa) in enumerate(seed_pairs, start=1)
            ]
            results: List[Dict] = []

            with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                futures = {ex.submit(_run_one_seed, *job): (job[5], job[6], job[7]) for job in jobs}
                for fut in as_completed(futures):
                    run_id, seed_nsga3, seed_ssw_rdpa = futures[fut]
                    out = fut.result()
                    results.append(out)
                    print(
                        f"[DTLZ][m={m}][{func_name}] run={run_id} done "
                        f"(seed_nsga={seed_nsga3}, seed_ssw={seed_ssw_rdpa}, "
                        f"nsga={out['algo'][NSGA_LABEL]['time_s']:.2f}s, "
                        f"ssw={out['algo'][SSW_LABEL]['time_s']:.2f}s, "
                        f"exec={out['exec']['backend_code']})"
                    )

            results.sort(key=lambda d: int(d["run_id"]))

            # Always use empirical ND reference built from all algorithm outputs.
            F_batches = [out["algo"][alg]["F"] for out in results for alg in algorithms]
            reference_set = _build_empirical_reference(F_batches, max_points=4000)
            reference_source = "empirical_nd"

            vals = {alg: [] for alg in algorithms}
            times = {alg: [] for alg in algorithms}
            for rid, out in enumerate(results, start=1):
                for alg in algorithms:
                    dp = delta_p(out["algo"][alg]["F"], reference_set, p=delta_p_order)
                    dt = float(out["algo"][alg]["time_s"])
                    seed_used = int(out["algo"][alg]["seed"])
                    vals[alg].append(dp)
                    times[alg].append(dt)
                    run_rows.append(
                        {
                            "function": func_name,
                            "m": int(m),
                            "run_id": int(rid),
                            "seed_nsga3": int(out["seed_nsga3"]),
                            "seed_ssw_rdpa": int(out["seed_ssw_rdpa"]),
                            "paired": bool(int(out["seed_nsga3"]) == int(out["seed_ssw_rdpa"])),
                            "seed_used": seed_used,
                            "n_var": int(out["n_var"]),
                            "pop_size": int(pop_size),
                            "n_evals": int(n_evals),
                            "reference_source": reference_source,
                            "algorithm": alg,
                            "delta_p": float(dp),
                            "time_s": dt,
                        }
                    )

            mean_nsga = float(np.mean(vals[NSGA_LABEL]))
            mean_ssw = float(np.mean(vals[SSW_LABEL]))
            std_nsga = float(np.std(vals[NSGA_LABEL], ddof=1)) if len(vals[NSGA_LABEL]) > 1 else 0.0
            std_ssw = float(np.std(vals[SSW_LABEL], ddof=1)) if len(vals[SSW_LABEL]) > 1 else 0.0
            mark_nsga, p_nsga = _wilcoxon_marker(np.asarray(vals[NSGA_LABEL]), np.asarray(vals[SSW_LABEL]))

            summary_rows.append(
                {
                    "function": func_name,
                    "m": int(m),
                    "pop_size": int(pop_size),
                    "n_runs": int(n_runs),
                    "n_evals": int(n_evals),
                    "reference_source": reference_source,
                    "mean_delta_p_nsga3": mean_nsga,
                    "std_delta_p_nsga3": std_nsga,
                    "mean_delta_p_ssw": mean_ssw,
                    "std_delta_p_ssw": std_ssw,
                    "nsga3_vs_ssw_marker": mark_nsga,
                    "pvalue_nsga3_vs_ssw": p_nsga,
                    "mean_time_nsga3_s": float(np.mean(times[NSGA_LABEL])),
                    "mean_time_ssw_s": float(np.mean(times[SSW_LABEL])),
                    "best_algo": SSW_LABEL if mean_ssw < mean_nsga else NSGA_LABEL,
                    "winner_margin_pct": 100.0 * (max(mean_nsga, mean_ssw) - min(mean_nsga, mean_ssw)) / max(max(mean_nsga, mean_ssw), 1e-12)
                }
            )

            print(
                f"[DTLZ][m={m}] {func_name} done -> ref={reference_source}, "
                f"mean_delta_p: NSGA={mean_nsga:.4e}, SSW={mean_ssw:.4e}"
            )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["function", "m", "run_id", "seed_nsga3", "seed_ssw_rdpa", "paired", "seed_master"],
    )
    _write_csv(
        out_dir / "all_runs_delta_p.csv",
        run_rows,
        [
            "function",
            "m",
            "run_id",
            "seed_nsga3",
            "seed_ssw_rdpa",
            "paired",
            "seed_used",
            "n_var",
            "pop_size",
            "n_evals",
            "reference_source",
            "algorithm",
            "delta_p",
            "time_s",
        ],
    )
    _write_csv(
        out_dir / "summary_delta_p.csv",
        summary_rows,
        [
            "function",
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
            "best_algo",
            "winner_margin_pct"
        ],
    )
    _print_debug_table(
        "DTLZ Summary (Delta_p)",
        summary_rows,
        [
            ("function", "function"),
            ("m", "m"),
            ("mean_delta_p_nsga3", "mean_dP_nsga3"),
            ("mean_delta_p_ssw", "mean_dP_ssw"),
            ("nsga3_vs_ssw_marker", "wilcoxon"),
            ("pvalue_nsga3_vs_ssw", "pvalue"),
            ("mean_time_nsga3_s", "time_nsga_s"),
            ("mean_time_ssw_s", "time_ssw_s"),
            ("best_algo", "best_algo"),
            ("winner_margin_pct", "margin_pct"),
        ],
    )
    print(f"[DONE] Outputs written to: {out_dir}")
    _print_latex_table(summary_rows, out_dir, ssw_label=SSW_LABEL)


def parse_args():
    parser = argparse.ArgumentParser(description="DTLZ-only benchmark: SSW-RDPA variant vs NSGA-III (Delta_p).")
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="DTLZ functions: 'all' or comma-separated list (e.g. dtlz1,dtlz4,dtlz7).",
    )
    parser.add_argument("--m-values", type=str, default="5,10,15", help="Comma-separated m values.")
    parser.add_argument("--n-runs", type=int, default=5, help="Independent runs per function.")
    parser.add_argument("--n-evals", type=int, default=30000, help="Max evaluations per run.")
    parser.add_argument("--parallel-workers", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--delta-p-order", type=float, default=1.0, help="Order p in Delta_p.")
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="Population size (uses energy reference directions if set).",
    )
    parser.add_argument("--gpu", action="store_true", dest="gpu", help="Use GPU acceleration.")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU acceleration.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--paired-seeds",
        action="store_true",
        dest="paired_seeds",
        help="Use paired seeds (same seed for NSGA-III and SSW-RDPA in each run).",
    )
    parser.add_argument(
        "--unpaired-seeds",
        action="store_false",
        dest="paired_seeds",
        help="Use independent seeds for NSGA-III and SSW-RDPA.",
    )
    parser.set_defaults(paired_seeds=False)
    parser.add_argument(
        "--seed-master",
        type=int,
        default=None,
        help="Master seed for deterministic 8-digit seed plans.",
    )
    parser.add_argument(
        "--ssw-config",
        type=str,
        default=None,
        help="JSON file with SSW_RDPA kwargs.",
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
                "dtlz_ssw_rdpa_vs_nsga3",
            )
        ),
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    functions = _parse_dtlz_functions(args.functions)
    ssw_kwargs = _load_json_file(args.ssw_config)
    run_experiment(
        out_dir=Path(args.out_dir),
        m_values=m_values,
        n_runs=int(args.n_runs),
        n_evals=int(args.n_evals),
        parallel_workers=int(args.parallel_workers),
        delta_p_order=float(args.delta_p_order),
        pop_size_override=args.pop_size,
        functions=functions,
        paired_seeds=bool(args.paired_seeds),
        seed_master=int(args.seed_master) if args.seed_master is not None else None,
        ssw_kwargs=ssw_kwargs,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
