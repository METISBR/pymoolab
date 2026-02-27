import argparse
import csv
import inspect
import math
import os
import random
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
from problems import get_problem
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions
from util.array_backend import CUPY_AVAILABLE, get_cupy_device_count, get_cupy_device_name
from util.stats_backend import wilcoxon

try:
    from algorithms.cmoea_cd.cmoea_cd import CMOEA_CD
    from algorithms.ssw_rd_n.ssw_rd_n import SSW_RD_N
except ImportError:
    from guiPymoo.algorithms.cmoea_cd.cmoea_cd import CMOEA_CD
    from guiPymoo.algorithms.ssw_rd_n.ssw_rd_n import SSW_RD_N


DTLZ_FUNCTIONS = [f"dtlz{i}" for i in range(1, 8)]
WFG_FUNCTIONS = [f"wfg{i}" for i in range(1, 10)]

DTLZ_K = {
    "dtlz1": 5,
    "dtlz2": 10,
    "dtlz3": 10,
    "dtlz4": 10,
    "dtlz5": 10,
    "dtlz6": 10,
    "dtlz7": 20,
}

ALGORITHMS = ("NSGA-III", "CMOEA-CD", "SSW-RD-N")


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


def _default_algo_target_points(n_obj: int) -> int:
    return 126 if n_obj <= 5 else 220


def _default_pf_target_points(n_obj: int) -> int:
    return 2000


def _build_problem(pack_name: str, func_name: str, n_obj: int):
    pack = pack_name.upper()
    if pack == "DTLZ":
        k = DTLZ_K[func_name]
        n_var = n_obj + k - 1
        problem = get_problem(func_name, n_var=n_var, n_obj=n_obj)
        return problem, {"k": k, "n_var": n_var}

    if pack == "WFG":
        k = 2 * (n_obj - 1)
        l = 20
        n_var = k + l
        problem = get_problem(func_name, n_var=n_var, n_obj=n_obj, k=k)
        return problem, {"k": k, "l": l, "n_var": n_var}

    raise ValueError(f"Unsupported pack_name: {pack_name}")


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
    # Handle problems exposing pareto front with different signatures.
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
        if len(pf) == 0:
            return None
        return pf
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
    return float(np.mean(mins ** p) ** (1.0 / p))


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
        s = rng.randrange(1, 2_147_483_647)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def _run_one_seed(
    pack_name: str,
    func_name: str,
    n_obj: int,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    seed: int,
    use_gpu: bool = False,
) -> Dict:
    problem, meta = _build_problem(pack_name, func_name, n_obj)
    gpu_ok = bool(use_gpu and CUPY_AVAILABLE and get_cupy_device_count() > 0)

    def _algo_kwargs(cls: type) -> dict:
        kwargs = {"ref_dirs": ref_dirs, "pop_size": pop_size}
        try:
            sig = inspect.signature(cls.__init__)
            if "seed" in sig.parameters:
                kwargs["seed"] = seed
            if gpu_ok and "use_gpu" in sig.parameters:
                kwargs["use_gpu"] = True
        except Exception:  # noqa: BLE001
            pass
        return kwargs

    output = {
        "seed": int(seed),
        "n_var": int(meta["n_var"]),
        "k": int(meta["k"]) if "k" in meta else None,
        "l": int(meta["l"]) if "l" in meta else None,
        "algo": {},
    }

    # NSGA-III
    t0 = time.time()
    res_nsga = minimize(
        problem,
        NSGA3(**_algo_kwargs(NSGA3)),
        ("n_eval", n_evals),
        seed=seed,
        verbose=False,
    )
    output["algo"]["NSGA-III"] = {
        "time_s": float(time.time() - t0),
        "F": _safe_F(res_nsga.F, n_obj=n_obj),
    }

    # CMOEA-CD
    t1 = time.time()
    res_cmo = minimize(
        problem,
        CMOEA_CD(**_algo_kwargs(CMOEA_CD)),
        ("n_eval", n_evals),
        seed=seed,
        verbose=False,
    )
    output["algo"]["CMOEA-CD"] = {
        "time_s": float(time.time() - t1),
        "F": _safe_F(res_cmo.F, n_obj=n_obj),
    }

    # SSW-RD-N
    t2 = time.time()
    res_ssw = minimize(
        problem,
        SSW_RD_N(**_algo_kwargs(SSW_RD_N)),
        ("n_eval", n_evals),
        seed=seed,
        verbose=False,
    )
    output["algo"]["SSW-RD-N"] = {
        "time_s": float(time.time() - t2),
        "F": _safe_F(res_ssw.F, n_obj=n_obj),
    }

    return output


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
    if np.mean(comp) < np.mean(base):
        return "+", float(pval)
    return "-", float(pval)


def _fmt_value(mean: float, std: float, marker: Optional[str] = None) -> str:
    txt = f"{mean:.4e} ({std:.2e})"
    if marker is None:
        return txt
    return f"{txt} {marker}"


def _write_csv(path: Path, rows: List[Dict], headers: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _to_latex_table(caption: str, label: str, headers: List[str], rows: List[List[str]]) -> str:
    colspec = "l c l l l c"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        escaped = [x.replace("_", "\\_") for x in r]
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_table_files(out_dir: Path, pack_name: str, summary_rows: List[Dict], table_idx: int) -> None:
    rows_sorted = sorted(summary_rows, key=lambda r: (r["function"], r["m"]))
    headers = ["Problem", "M", "NSGA-III", "CMOEA-CD", "SSW-RD-N", "Best"]
    md_rows = []
    for r in rows_sorted:
        md_rows.append(
            [
                r["function"].upper(),
                str(r["m"]),
                _fmt_value(r["mean_delta_p_nsga3"], r["std_delta_p_nsga3"], r["mark_nsga3_vs_ssw"]),
                _fmt_value(r["mean_delta_p_cmoea"], r["std_delta_p_cmoea"], r["mark_cmoea_vs_ssw"]),
                _fmt_value(r["mean_delta_p_ssw"], r["std_delta_p_ssw"], None),
                r["best_algo"],
            ]
        )

    caption = (
        "Statistical results (Delta_p) on DTLZ problems."
        if pack_name.upper() == "DTLZ"
        else "Statistical results (Delta_p) on WFG problems."
    )
    label = f"tab:table{table_idx}_{pack_name.lower()}_delta_p"

    md_text = [
        f"# TABLE {table_idx}",
        "",
        caption,
        "",
        _to_markdown_table(headers, md_rows),
        "",
        "Note: '+' / '=' / '-' in NSGA-III and CMOEA-CD columns denote "
        "statistically better / equivalent / worse than SSW-RD-N "
        "(paired Wilcoxon, alpha=0.05, lower Delta_p is better).",
        "",
    ]
    (out_dir / f"table{table_idx}_{pack_name.lower()}_delta_p.md").write_text(
        "\n".join(md_text), encoding="utf-8"
    )

    latex_text = _to_latex_table(caption=caption, label=label, headers=headers, rows=md_rows)
    (out_dir / f"table{table_idx}_{pack_name.lower()}_delta_p.tex").write_text(
        latex_text, encoding="utf-8"
    )


def run_experiment(
    out_dir: Path,
    m_values: List[int],
    n_runs: int,
    n_evals: int,
    parallel_workers: int,
    delta_p_order: float,
    use_gpu: bool,
    pop_size_override: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_gpu:
        gpu_ok = bool(CUPY_AVAILABLE and get_cupy_device_count() > 0)
        print(f"[INFO] GPU requested via CuPy CUDA: available={gpu_ok}")
        if gpu_ok:
            print(f"[INFO] CUDA device: {get_cupy_device_name(0) or 'CUDA GPU'}")

    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for pack_name, functions in [("DTLZ", DTLZ_FUNCTIONS), ("WFG", WFG_FUNCTIONS)]:
        for m in m_values:
            pf_target_points = _default_pf_target_points(m)
            p_pf = choose_partitions(m, pf_target_points)
            if pop_size_override is not None and pop_size_override > 1:
                ref_dirs = get_reference_directions("energy", m, int(pop_size_override), seed=1)
                p_algo = "-"
            else:
                algo_target_points = _default_algo_target_points(m)
                p_algo = choose_partitions(m, algo_target_points)
                ref_dirs = get_reference_directions("das-dennis", m, n_partitions=p_algo)
            pf_ref_dirs = get_reference_directions("das-dennis", m, n_partitions=p_pf)
            pop_size = len(ref_dirs)

            print(
                f"[{pack_name}][m={m}] setup: p_algo={p_algo}, pop={pop_size}, "
                f"p_pf={p_pf}, runs={n_runs}, evals={n_evals}, workers={parallel_workers}"
            )

            for func_name in functions:
                print(f"[{pack_name}][m={m}] running {func_name} ...")

                seeds = _random_seed_list(n_runs)
                for rid, seed in enumerate(seeds, start=1):
                    seed_rows.append(
                        {
                            "pack": pack_name,
                            "function": func_name,
                            "m": int(m),
                            "run_id": int(rid),
                            "seed": int(seed),
                        }
                    )

                jobs = [
                    (pack_name, func_name, m, n_evals, ref_dirs, pop_size, seed, use_gpu)
                    for seed in seeds
                ]

                results: List[Dict] = []
                if parallel_workers > 1:
                    with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                        futures = {ex.submit(_run_one_seed, *job): job[6] for job in jobs}
                        for fut in as_completed(futures):
                            seed = futures[fut]
                            out = fut.result()
                            results.append(out)
                            print(
                                f"[{pack_name}][m={m}][{func_name}] seed={seed} done "
                                f"(nsga={out['algo']['NSGA-III']['time_s']:.2f}s, "
                                f"cmo={out['algo']['CMOEA-CD']['time_s']:.2f}s, "
                                f"ssw={out['algo']['SSW-RD-N']['time_s']:.2f}s)"
                            )
                else:
                    for job in jobs:
                        out = _run_one_seed(*job)
                        results.append(out)
                        print(
                            f"[{pack_name}][m={m}][{func_name}] seed={job[-1]} done "
                            f"(nsga={out['algo']['NSGA-III']['time_s']:.2f}s, "
                            f"cmo={out['algo']['CMOEA-CD']['time_s']:.2f}s, "
                            f"ssw={out['algo']['SSW-RD-N']['time_s']:.2f}s)"
                        )

                # Keep output ordering stable by seed sequence generated above.
                order = {s: i for i, s in enumerate(seeds)}
                results.sort(key=lambda d: order[d["seed"]])

                problem, meta = _build_problem(pack_name, func_name, m)
                true_pf = _try_get_true_reference(problem, pf_ref_dirs=pf_ref_dirs)
                if true_pf is not None:
                    reference_set = true_pf
                    reference_source = "true_pf"
                else:
                    F_batches = []
                    for out in results:
                        for alg in ALGORITHMS:
                            F_batches.append(out["algo"][alg]["F"])
                    reference_set = _build_empirical_reference(F_batches, max_points=4000)
                    reference_source = "empirical_nd"

                delta_values = {alg: [] for alg in ALGORITHMS}
                time_values = {alg: [] for alg in ALGORITHMS}

                for rid, out in enumerate(results, start=1):
                    for alg in ALGORITHMS:
                        F = out["algo"][alg]["F"]
                        dp = delta_p(F, reference_set, p=delta_p_order)
                        dt = float(out["algo"][alg]["time_s"])

                        delta_values[alg].append(dp)
                        time_values[alg].append(dt)

                        run_rows.append(
                            {
                                "pack": pack_name,
                                "function": func_name,
                                "m": int(m),
                                "run_id": int(rid),
                                "seed": int(out["seed"]),
                                "n_var": int(out["n_var"]),
                                "pop_size": int(pop_size),
                                "n_evals": int(n_evals),
                                "reference_source": reference_source,
                                "algorithm": alg,
                                "delta_p": float(dp),
                                "time_s": float(dt),
                            }
                        )

                mean_nsga = float(np.mean(delta_values["NSGA-III"]))
                std_nsga = float(np.std(delta_values["NSGA-III"], ddof=1))
                mean_cmo = float(np.mean(delta_values["CMOEA-CD"]))
                std_cmo = float(np.std(delta_values["CMOEA-CD"], ddof=1))
                mean_ssw = float(np.mean(delta_values["SSW-RD-N"]))
                std_ssw = float(np.std(delta_values["SSW-RD-N"], ddof=1))

                mark_nsga, p_nsga = _wilcoxon_marker(
                    np.asarray(delta_values["NSGA-III"]),
                    np.asarray(delta_values["SSW-RD-N"]),
                )
                mark_cmo, p_cmo = _wilcoxon_marker(
                    np.asarray(delta_values["CMOEA-CD"]),
                    np.asarray(delta_values["SSW-RD-N"]),
                )

                mean_map = {
                    "NSGA-III": mean_nsga,
                    "CMOEA-CD": mean_cmo,
                    "SSW-RD-N": mean_ssw,
                }
                best_algo = min(mean_map.items(), key=lambda kv: kv[1])[0]

                summary_rows.append(
                    {
                        "pack": pack_name,
                        "function": func_name,
                        "m": int(m),
                        "n_var": int(meta["n_var"]),
                        "k": int(meta["k"]) if "k" in meta else "",
                        "l": int(meta["l"]) if "l" in meta else "",
                        "pop_size": int(pop_size),
                        "n_runs": int(n_runs),
                        "n_evals": int(n_evals),
                        "delta_p_order": float(delta_p_order),
                        "reference_source": reference_source,
                        "mean_delta_p_nsga3": mean_nsga,
                        "std_delta_p_nsga3": std_nsga,
                        "mean_delta_p_cmoea": mean_cmo,
                        "std_delta_p_cmoea": std_cmo,
                        "mean_delta_p_ssw": mean_ssw,
                        "std_delta_p_ssw": std_ssw,
                        "mark_nsga3_vs_ssw": mark_nsga,
                        "mark_cmoea_vs_ssw": mark_cmo,
                        "pvalue_nsga3_vs_ssw": p_nsga,
                        "pvalue_cmoea_vs_ssw": p_cmo,
                        "mean_time_nsga3_s": float(np.mean(time_values["NSGA-III"])),
                        "mean_time_cmoea_s": float(np.mean(time_values["CMOEA-CD"])),
                        "mean_time_ssw_s": float(np.mean(time_values["SSW-RD-N"])),
                        "best_algo": best_algo,
                    }
                )

                print(
                    f"[{pack_name}][m={m}] {func_name} done -> ref={reference_source}, "
                    f"best={best_algo}, mean_delta_p: "
                    f"NSGA={mean_nsga:.4e}, CMO={mean_cmo:.4e}, SSW={mean_ssw:.4e}"
                )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["pack", "function", "m", "run_id", "seed"],
    )
    _write_csv(
        out_dir / "all_runs_delta_p.csv",
        run_rows,
        [
            "pack",
            "function",
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
        ],
    )
    _write_csv(
        out_dir / "summary_delta_p.csv",
        summary_rows,
        [
            "pack",
            "function",
            "m",
            "n_var",
            "k",
            "l",
            "pop_size",
            "n_runs",
            "n_evals",
            "delta_p_order",
            "reference_source",
            "mean_delta_p_nsga3",
            "std_delta_p_nsga3",
            "mean_delta_p_cmoea",
            "std_delta_p_cmoea",
            "mean_delta_p_ssw",
            "std_delta_p_ssw",
            "mark_nsga3_vs_ssw",
            "mark_cmoea_vs_ssw",
            "pvalue_nsga3_vs_ssw",
            "pvalue_cmoea_vs_ssw",
            "mean_time_nsga3_s",
            "mean_time_cmoea_s",
            "mean_time_ssw_s",
            "best_algo",
        ],
    )

    dtlz_summary = [r for r in summary_rows if r["pack"] == "DTLZ"]
    wfg_summary = [r for r in summary_rows if r["pack"] == "WFG"]
    _write_table_files(out_dir=out_dir, pack_name="DTLZ", summary_rows=dtlz_summary, table_idx=1)
    _write_table_files(out_dir=out_dir, pack_name="WFG", summary_rows=wfg_summary, table_idx=2)

    print(f"[DONE] Outputs written to: {out_dir}")
    print("[DONE] Table I: table1_dtlz_delta_p.md / .tex")
    print("[DONE] Table II: table2_wfg_delta_p.md / .tex")


def parse_args():
    parser = argparse.ArgumentParser(description="MAOP benchmark with Delta_p (DTLZ + WFG).")
    parser.add_argument(
        "--m-values",
        type=str,
        default="5,10",
        help="Comma-separated objective counts (default: 5,10)",
    )
    parser.add_argument("--n-runs", type=int, default=30, help="Independent runs per function.")
    parser.add_argument("--n-evals", type=int, default=25000, help="Max evaluations per run.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel workers for seed-level runs.",
    )
    parser.add_argument(
        "--delta-p-order",
        type=float,
        default=1.0,
        help="Order p used in Delta_p = max(IGD_p(A,R), IGD_p(R,A)).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Request CuPy CUDA mode info in logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "artigo", "results", "maop_delta_p")
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Override population size and reference-direction count (uses energy method).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    run_experiment(
        out_dir=out_dir,
        m_values=m_values,
        n_runs=int(args.n_runs),
        n_evals=int(args.n_evals),
        parallel_workers=max(1, int(args.parallel_workers)),
        delta_p_order=float(args.delta_p_order),
        use_gpu=bool(args.use_gpu),
        pop_size_override=args.pop_size,
    )


if __name__ == "__main__":
    main()
