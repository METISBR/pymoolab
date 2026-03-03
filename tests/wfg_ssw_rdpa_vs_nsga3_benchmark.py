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

from algorithms.ssw_rdpa.ssw_rdpa import SSW_RDPA

WFG_FUNCTIONS = [f"wfg{i}" for i in range(1, 10)]
ALGORITHMS = ("NSGA-III", "SSW-RDPA")
SEED_MIN_8DIGIT = 10_000_000
SEED_MAX_8DIGIT = 99_999_999


def _parse_wfg_functions(raw: str) -> List[str]:
    token = str(raw).strip().lower()
    if token == "all":
        return list(WFG_FUNCTIONS)
    parts = [p.strip().lower() for p in token.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("No WFG function selected. Use --functions all or comma-separated list.")
    valid = set(WFG_FUNCTIONS)
    out: List[str] = []
    for name in parts:
        if name not in valid:
            raise ValueError(f"Unknown WFG function '{name}'. Valid: {', '.join(WFG_FUNCTIONS)} or 'all'.")
        if name not in out:
            out.append(name)
    return out


def _load_json_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSON config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
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
    # Standard WFG settings: k = 2 * (n_obj - 1), l = 20
    k = 2 * (n_obj - 1)
    l = 20
    n_var = k + l
    problem = get_problem(func_name, n_var=n_var, n_obj=n_obj, k=k)
    return problem, {"k": k, "l": l, "n_var": n_var}

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

    # Keep SSW seeds independent and different from NSGA seeds.
    rng_ssw = _scope_rng(seed_master, scope + "|ssw")
    seen_ssw = set()
    blocked = set(seed_nsga)
    seed_ssw: List[int] = []
    while len(seed_ssw) < n:
        s = int(rng_ssw.randrange(SEED_MIN_8DIGIT, SEED_MAX_8DIGIT + 1))
        if s in seen_ssw or s in blocked:
            continue
        seen_ssw.add(s)
        seed_ssw.append(s)
    return list(zip(seed_nsga, seed_ssw))

def _run_one_seed(
    func_name: str,
    n_obj: int,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    seed_nsga3: int,
    seed_ssw: int,
    use_gpu: bool = False,
    ssw_kwargs: Optional[Dict] = None,
) -> Dict:
    problem, meta = _build_problem(func_name, n_obj)
    gpu_ok = bool(use_gpu and CUPY_AVAILABLE and _get_cupy_device_count_safe() > 0)
    backend_code = "gpu" if gpu_ok else "cpu"
    backend_info = f"GPU CUDA via CuPy ({get_cupy_device_name(0) or 'CUDA GPU'})" if gpu_ok else "CPU only"

    output = {
        "seed_nsga3": int(seed_nsga3),
        "seed_ssw": int(seed_ssw),
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
    output["algo"]["NSGA-III"] = {
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
        SSW_RDPA(ref_dirs=ref_dirs, pop_size=pop_size, seed=int(seed_ssw), use_gpu=gpu_ok, **ssw_user),
        ("n_eval", n_evals),
        seed=int(seed_ssw),
        verbose=False,
    )
    output["algo"]["SSW-RDPA"] = {
        "seed": int(seed_ssw),
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

def _print_latex_table(summary_rows: List[Dict], out_dir: Path) -> None:
    lines = []
    lines.append("\n" + "%" * 80)
    lines.append("% LATEX TABLE OUTPUT - WFG BENCHMARK")
    lines.append("% Remember to include \\usepackage{multirow} and \\usepackage{booktabs} in your preamble.")
    lines.append("%" * 80)
    lines.append("\\begin{table*}[htpb]")
    lines.append("  \\centering")
    lines.append("  \\caption{Statistical Results Obtained by NSGA-III and SSW-RDPA on WFG Problems ($\\Delta_p$)}")
    lines.append("  \\label{tab:wfg_benchmark}")
    lines.append("  \\begin{tabular}{c c c c}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Problems} & $m$ & \\textbf{NSGA-III} & \\textbf{SSW-RDPA} \\\\")
    lines.append("    \\midrule")

    from collections import defaultdict
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row['function'].upper()].append(row)

    for func in sorted(grouped.keys()):
        rows_for_func = sorted(grouped[func], key=lambda x: x['m'])
        n_rows = len(rows_for_func)
        
        for i, row in enumerate(rows_for_func):
            m = row['m']
            nsga_mean = row['mean_delta_p_nsga3']
            nsga_std = row['std_delta_p_nsga3']
            ssw_mean = row['mean_delta_p_ssw']
            ssw_std = row['std_delta_p_ssw']
            marker_nsga = row['nsga3_vs_ssw_marker']

            def fmt_sci(val: float) -> str:
                if not math.isfinite(val):
                    return str(val)
                if val == 0:
                    return "0.0000e+00"
                return f"{val:.4e}"

            nsga_val_str = fmt_sci(nsga_mean)
            nsga_std_str = fmt_sci(nsga_std)
            ssw_val_str = fmt_sci(ssw_mean)
            ssw_std_str = fmt_sci(ssw_std)

            ssw_marker = "="
            if marker_nsga == "+":
                ssw_marker = "-"
            elif marker_nsga == "-":
                ssw_marker = "+"
                
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

        lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("%" * 80 + "\n")

    latex_str = "\n".join(lines)
    latex_file = out_dir / "latex_table.tex"
    with latex_file.open("w", encoding="utf-8") as f:
        f.write(latex_str)


def _load_fixed_reference(cache_file: Path) -> Optional[np.ndarray]:
    if not cache_file.exists():
        return None
    try:
        arr = np.asarray(np.load(cache_file), dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or len(arr) == 0:
        return None
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    return arr if len(arr) > 0 else None


def _save_fixed_reference(cache_file: Path, reference_set: np.ndarray) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, np.asarray(reference_set, dtype=float))


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
    fixed_reference_cache: Optional[Path] = None,
    emit_tuning_metrics: bool = False,
    ssw_kwargs: Optional[Dict] = None,
    seed_master: Optional[int] = None,
    use_gpu: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for m in m_values:
        if pop_size_override is not None and pop_size_override > 1:
            ref_dirs = get_reference_directions("energy", m, int(pop_size_override), seed=1)
        else:
            p_algo = choose_partitions(m, 126 if m <= 5 else 220)
            ref_dirs = get_reference_directions("das-dennis", m, n_partitions=p_algo)
        pop_size = len(ref_dirs)

        print(
            f"[WFG][m={m}] setup: pop={pop_size}, "
            f"runs={n_runs}, evals={n_evals}, backend=process, workers={parallel_workers}, "
            f"gpu={use_gpu}, seed_mode={'paired' if paired_seeds else 'unpaired'}"
        )

        for func_name in functions:
            print(f"[WFG][m={m}] running {func_name} ...")
            seed_scope = f"{func_name}|m={m}|n_runs={n_runs}|n_evals={n_evals}"
            seed_pairs = _random_seed_pairs(
                n_runs,
                paired_seeds=paired_seeds,
                seed_master=seed_master,
                scope=seed_scope,
            )

            for rid, (seed_nsga3, seed_ssw) in enumerate(seed_pairs, start=1):
                seed_rows.append(
                    {
                        "function": func_name,
                        "m": int(m),
                        "run_id": int(rid),
                        "seed_nsga3": int(seed_nsga3),
                        "seed_ssw": int(seed_ssw),
                        "paired": bool(seed_nsga3 == seed_ssw),
                        "seed_master": int(seed_master) if seed_master is not None else "",
                    }
                )

            jobs = [
                (func_name, m, n_evals, ref_dirs, pop_size, int(seed_nsga3), int(seed_ssw), use_gpu, ssw_kwargs)
                for (seed_nsga3, seed_ssw) in seed_pairs
            ]
            results: List[Dict] = []

            with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                futures = {ex.submit(_run_one_seed, *job): (idx + 1, job[5], job[6]) for idx, job in enumerate(jobs)}
                for fut in as_completed(futures):
                    run_id, seed_nsga3, seed_ssw = futures[fut]
                    out = fut.result()
                    out["run_id"] = int(run_id)
                    results.append(out)
                    print(
                        f"[WFG][m={m}][{func_name}] run={run_id} done "
                        f"(seed_nsga={seed_nsga3}, seed_ssw={seed_ssw}, "
                        f"nsga={out['algo']['NSGA-III']['time_s']:.2f}s, "
                        f"ssw={out['algo']['SSW-RDPA']['time_s']:.2f}s, "
                        f"exec={out['exec']['backend_code']})"
                    )

            results.sort(key=lambda d: int(d.get("run_id", 0)))

            reference_set = None
            reference_source = "empirical_nd"
            cache_file = None
            if fixed_reference_cache is not None:
                cache_file = Path(fixed_reference_cache) / f"{func_name}_m{m}.npy"
                reference_set = _load_fixed_reference(cache_file)
                if reference_set is not None:
                    reference_source = "fixed_cache"
            if reference_set is None:
                F_batches = [out["algo"][alg]["F"] for out in results for alg in ALGORITHMS]
                reference_set = _build_empirical_reference(F_batches, max_points=4000)
                if cache_file is not None:
                    _save_fixed_reference(cache_file, reference_set)
                    reference_source = "empirical_nd_cached"

            vals = {alg: [] for alg in ALGORITHMS}
            times = {alg: [] for alg in ALGORITHMS}
            for out in results:
                rid = int(out["run_id"])
                dp_nsga = float(delta_p(out["algo"]["NSGA-III"]["F"], reference_set, p=delta_p_order))
                dp_ssw = float(delta_p(out["algo"]["SSW-RDPA"]["F"], reference_set, p=delta_p_order))
                dt_nsga = float(out["algo"]["NSGA-III"]["time_s"])
                dt_ssw = float(out["algo"]["SSW-RDPA"]["time_s"])

                vals["NSGA-III"].append(dp_nsga)
                vals["SSW-RDPA"].append(dp_ssw)
                times["NSGA-III"].append(dt_nsga)
                times["SSW-RDPA"].append(dt_ssw)

                run_rows.append(
                    {
                        "function": func_name,
                        "m": int(m),
                        "run_id": int(rid),
                        "seed": int(out["algo"]["NSGA-III"]["seed"]),
                        "n_var": int(out["n_var"]),
                        "pop_size": int(pop_size),
                        "n_evals": int(n_evals),
                        "reference_source": reference_source,
                        "algorithm": "NSGA-III",
                        "delta_p": float(dp_nsga),
                        "time_s": dt_nsga,
                    }
                )
                run_rows.append(
                    {
                        "function": func_name,
                        "m": int(m),
                        "run_id": int(rid),
                        "seed": int(out["algo"]["SSW-RDPA"]["seed"]),
                        "n_var": int(out["n_var"]),
                        "pop_size": int(pop_size),
                        "n_evals": int(n_evals),
                        "reference_source": reference_source,
                        "algorithm": "SSW-RDPA",
                        "delta_p": float(dp_ssw),
                        "time_s": dt_ssw,
                    }
                )

                if emit_tuning_metrics:
                    winner = "SSW-RDPA" if dp_ssw < dp_nsga else "NSGA-III"
                    pair_rows.append(
                        {
                            "function": func_name,
                            "m": int(m),
                            "run_id": int(rid),
                            "seed_nsga3": int(out["algo"]["NSGA-III"]["seed"]),
                            "seed_ssw": int(out["algo"]["SSW-RDPA"]["seed"]),
                            "paired": bool(out["algo"]["NSGA-III"]["seed"] == out["algo"]["SSW-RDPA"]["seed"]),
                            "n_evals": int(n_evals),
                            "delta_p_nsga3": dp_nsga,
                            "delta_p_ssw": dp_ssw,
                            "time_nsga3_s": dt_nsga,
                            "time_ssw_s": dt_ssw,
                            "time_ratio_ssw_over_nsga3": float(dt_ssw / max(dt_nsga, 1e-12)),
                            "winner": winner,
                        }
                    )

            mean_nsga = float(np.mean(vals["NSGA-III"]))
            mean_ssw = float(np.mean(vals["SSW-RDPA"]))
            std_nsga = float(np.std(vals["NSGA-III"], ddof=1))
            std_ssw = float(np.std(vals["SSW-RDPA"], ddof=1))
            mark_nsga, p_nsga = _wilcoxon_marker(np.asarray(vals["NSGA-III"]), np.asarray(vals["SSW-RDPA"]))

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
                    "mean_time_nsga3_s": float(np.mean(times["NSGA-III"])),
                    "mean_time_ssw_s": float(np.mean(times["SSW-RDPA"])),
                    "best_algo": "SSW-RDPA" if mean_ssw < mean_nsga else "NSGA-III",
                    "winner_margin_pct": 100.0 * (max(mean_nsga, mean_ssw) - min(mean_nsga, mean_ssw)) / max(max(mean_nsga, mean_ssw), 1e-12)
                }
            )

            print(
                f"[WFG][m={m}] {func_name} done -> ref={reference_source}, "
                f"mean_delta_p: NSGA={mean_nsga:.4e}, SSW={mean_ssw:.4e}"
            )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["function", "m", "run_id", "seed_nsga3", "seed_ssw", "paired", "seed_master"],
    )
    _write_csv(
        out_dir / "all_runs_delta_p.csv",
        run_rows,
        ["function", "m", "run_id", "seed", "n_var", "pop_size", "n_evals", "reference_source", "algorithm", "delta_p", "time_s"],
    )
    _write_csv(
        out_dir / "summary_delta_p.csv",
        summary_rows,
        ["function", "m", "pop_size", "n_runs", "n_evals", "reference_source", "mean_delta_p_nsga3", "std_delta_p_nsga3", "mean_delta_p_ssw", "std_delta_p_ssw", "nsga3_vs_ssw_marker", "pvalue_nsga3_vs_ssw", "mean_time_nsga3_s", "mean_time_ssw_s", "best_algo", "winner_margin_pct"],
    )
    _print_debug_table(
        "WFG Summary (Delta_p)",
        summary_rows,
        [
            ("function", "function"),
            ("m", "m"),
            ("mean_delta_p_nsga3", "mean_dP_nsga3"),
            ("mean_delta_p_ssw", "mean_dP_ssw"),
            ("nsga3_vs_ssw_marker", "wilcoxon"),
            ("pvalue_nsga3_vs_ssw", "pvalue"),
            ("best_algo", "best_algo"),
            ("winner_margin_pct", "margin_pct"),
        ],
    )
    _print_latex_table(summary_rows, out_dir)
    if emit_tuning_metrics:
        _write_csv(
            out_dir / "paired_run_metrics.csv",
            pair_rows,
            [
                "function",
                "m",
                "run_id",
                "seed_nsga3",
                "seed_ssw",
                "paired",
                "n_evals",
                "delta_p_nsga3",
                "delta_p_ssw",
                "time_nsga3_s",
                "time_ssw_s",
                "time_ratio_ssw_over_nsga3",
                "winner",
            ],
        )

def parse_args():
    parser = argparse.ArgumentParser(description="WFG-only benchmark: SSW-RDPA vs NSGA-III.")
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="WFG functions: 'all' or comma-separated list (e.g. wfg1,wfg4,wfg9).",
    )
    parser.add_argument("--m-values", type=str, default="5,10,15", help="Comma-separated m values.")
    parser.add_argument("--n-runs", type=int, default=5, help="Independent runs per function.")
    parser.add_argument("--n-evals", type=int, default=25000, help="Max evaluations per run.")
    parser.add_argument("--parallel-workers", type=int, default=24, help="Number of parallel workers.")
    parser.add_argument("--delta-p-order", type=float, default=1.0, help="Order p in Delta_p.")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size.")
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
        help="Master seed for deterministic 8-digit seed plans (recommended for tuning).",
    )
    parser.add_argument(
        "--fixed-reference-cache",
        type=str,
        default=None,
        help="Directory for per-problem fixed reference cache (.npy).",
    )
    parser.add_argument(
        "--emit-tuning-metrics",
        action="store_true",
        dest="emit_tuning_metrics",
        help="Emit paired_run_metrics.csv for autotuning.",
    )
    parser.add_argument(
        "--no-emit-tuning-metrics",
        action="store_false",
        dest="emit_tuning_metrics",
        help="Do not emit paired_run_metrics.csv.",
    )
    parser.set_defaults(emit_tuning_metrics=False)
    parser.add_argument(
        "--ssw-config",
        type=str,
        default=None,
        help="JSON file with SSW_RDPA kwargs (excluding forced keys).",
    )
    parser.add_argument("--gpu", action="store_true", dest="gpu", help="Use GPU acceleration.")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU acceleration.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artigo", "results", "wfg_ssw_rdpa_vs_nsga3")),
        help="Output directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    functions = _parse_wfg_functions(args.functions)
    ssw_kwargs = _load_json_file(args.ssw_config)
    fixed_reference_cache = (
        Path(args.fixed_reference_cache).expanduser().resolve()
        if args.fixed_reference_cache
        else None
    )
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
        fixed_reference_cache=fixed_reference_cache,
        emit_tuning_metrics=bool(args.emit_tuning_metrics),
        ssw_kwargs=ssw_kwargs,
        seed_master=int(args.seed_master) if args.seed_master is not None else None,
        use_gpu=args.gpu,
    )

if __name__ == "__main__":
    main()
