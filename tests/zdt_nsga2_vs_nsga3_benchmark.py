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

import numpy as np
from scipy.stats import wilcoxon

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import (
    NSGA3,
    ReferenceDirectionSurvival,
    associate_to_niches,
    calc_niche_count,
    niching,
)
from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from problems import get_problem
from pymoo.util.misc import intersect
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

ZDT_ALL_FUNCTIONS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6"]
DEFAULT_ZDT_FUNCTIONS = ["zdt1"]
ALGORITHMS = ("NSGA-II", "NSGA-III")
N_OBJ = 2
ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt5": 11, "zdt6": 10}


class FastReferenceDirectionSurvival(ReferenceDirectionSurvival):
    """NSGA-III survival variant with faster non-dominated sorting."""

    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):  # type: ignore[override]
        F = pop.get("F")
        fronts, rank = NonDominatedSorting(method="efficient_non_dominated_sort").do(
            F, return_rank=True, n_stop_if_ranked=n_survive
        )
        non_dominated = fronts[0]

        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F, self.ref_dirs, ideal, nadir)
        pop.set("rank", rank, "niche", niche_of_individuals, "dist_to_niche", dist_to_niche)

        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop[fronts[0]]

        if len(pop) > n_survive:
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(
                pop[last_front],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front],
                dist_to_niche[last_front],
                random_state=random_state,
            )
            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


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


def _safe_F(F, n_obj: int):
    if F is None:
        return np.full((1, n_obj), np.inf, dtype=float)
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
    fast_nsga3: bool,
) -> Dict:
    kwargs: Dict = {"pop_size": int(pop_size), "seed": int(seed)}
    if cls == NSGA3:
        kwargs["ref_dirs"] = ref_dirs
        if fast_nsga3:
            kwargs.update(
                {
                    "selection": RandomSelection(),
                    "eliminate_duplicates": False,
                    "n_offsprings": int(pop_size),
                    "survival": FastReferenceDirectionSurvival(ref_dirs),
                }
            )
    return kwargs


def _run_one_seed(
    func_name: str,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    run_id: int,
    seed_nsga2: int,
    seed_nsga3: int,
    fast_nsga3: bool,
) -> Dict:
    if func_name == "zdt5":
        problem = get_problem(func_name)
    else:
        problem = get_problem(func_name, n_var=ZDT_N_VAR[func_name])

    output = {"run_id": int(run_id), "n_var": int(ZDT_N_VAR.get(func_name, 11)), "algo": {}}
    output["exec"] = {
        "backend_requested_code": "cpu",
        "backend_requested_info": "CPU only",
        "cupy_available": False,
        "cupy_devices": 0,
    }

    algo_nsga2 = NSGA2(
        **_algo_kwargs(
            NSGA2,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed_nsga2),
            fast_nsga3=bool(fast_nsga3),
        )
    )

    algo_nsga3 = NSGA3(
        **_algo_kwargs(
            NSGA3,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=int(seed_nsga3),
            fast_nsga3=bool(fast_nsga3),
        )
    )

    output["exec"]["backend_code"] = "cpu"
    output["exec"]["backend_info"] = "CPU only"

    t0 = time.time()
    res_nsga2 = minimize(
        problem,
        algo_nsga2,
        ("n_eval", n_evals),
        verbose=False,
        seed=int(seed_nsga2),
        copy_algorithm=False,
        save_history=False,
    )
    output["algo"]["NSGA-II"] = {
        "seed": int(seed_nsga2),
        "time_s": float(time.time() - t0),
        "F": _safe_F(res_nsga2.F, N_OBJ),
        "backend_code": "cpu",
        "backend_info": "CPU only",
    }

    t1 = time.time()
    res_nsga3 = minimize(
        problem,
        algo_nsga3,
        ("n_eval", n_evals),
        verbose=False,
        seed=int(seed_nsga3),
        copy_algorithm=False,
        save_history=False,
    )
    output["algo"]["NSGA-III"] = {
        "seed": int(seed_nsga3),
        "time_s": float(time.time() - t1),
        "F": _safe_F(res_nsga3.F, N_OBJ),
        "backend_code": "cpu",
        "backend_info": "CPU only",
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
    fast_nsga3: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict] = []
    run_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    if pop_size <= 1:
        raise ValueError("pop_size must be greater than 1 for ZDT (m=2).")

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=pop_size - 1)
    real_pop = len(ref_dirs)

    print(
        f"[ZDT][m={N_OBJ}] setup: pop={real_pop}, runs={n_runs}, evals={n_evals}, "
        f"backend=process, workers={parallel_workers}, backend=cpu, fast_nsga3={bool(fast_nsga3)}"
    )

    for func_name in zdt_functions:
        print(f"[ZDT][m={N_OBJ}] running {func_name} ...")
        seed_pairs = _random_seed_pairs(n_runs)

        for rid, (seed_nsga2, seed_nsga3) in enumerate(seed_pairs, start=1):
            seed_rows.append(
                {
                    "problem": func_name,
                    "m": int(N_OBJ),
                    "run_id": int(rid),
                    "seed_nsga2": int(seed_nsga2),
                    "seed_nsga3": int(seed_nsga3),
                }
            )

        jobs = [
            (func_name, n_evals, ref_dirs, real_pop, rid, seed_nsga2, seed_nsga3, bool(fast_nsga3))
            for rid, (seed_nsga2, seed_nsga3) in enumerate(seed_pairs, start=1)
        ]
        results: List[Dict] = []

        with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {ex.submit(_run_one_seed, *job): (job[4], job[5], job[6]) for job in jobs}
            for fut in as_completed(futures):
                run_id, seed_nsga2, seed_nsga3 = futures[fut]
                out = fut.result()
                results.append(out)
                print(
                    f"[ZDT][m={N_OBJ}][{func_name}] run={run_id} done "
                    f"(seed_nsga2={seed_nsga2}, seed_nsga3={seed_nsga3}, "
                    f"nsga2={out['algo']['NSGA-II']['time_s']:.2f}s, "
                    f"nsga3={out['algo']['NSGA-III']['time_s']:.2f}s, "
                    f"exec=cpu)"
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
                        "backend_code": "cpu",
                        "backend_info": "CPU only",
                    }
                )

        mean_nsga2 = float(np.mean(vals["NSGA-II"]))
        mean_nsga3 = float(np.mean(vals["NSGA-III"]))
        std_nsga2 = float(np.std(vals["NSGA-II"], ddof=1)) if len(vals["NSGA-II"]) > 1 else 0.0
        std_nsga3 = float(np.std(vals["NSGA-III"], ddof=1)) if len(vals["NSGA-III"]) > 1 else 0.0
        mark_nsga2, p_nsga2 = _wilcoxon_marker(np.asarray(vals["NSGA-II"]), np.asarray(vals["NSGA-III"]))

        if mean_nsga3 < mean_nsga2:
            winner_algo = "NSGA-III"
            loser_algo = "NSGA-II"
            winner_val = mean_nsga3
            loser_val = mean_nsga2
        else:
            winner_algo = "NSGA-II"
            loser_algo = "NSGA-III"
            winner_val = mean_nsga2
            loser_val = mean_nsga3
        winner_margin_pct_over_loser = 100.0 * (loser_val - winner_val) / max(abs(loser_val), 1e-32)

        summary_rows.append(
            {
                "problem": func_name,
                "m": int(N_OBJ),
                "pop_size": int(real_pop),
                "n_runs": int(n_runs),
                "n_evals": int(n_evals),
                "reference_source": reference_source,
                "mean_delta_p_nsga2": mean_nsga2,
                "std_delta_p_nsga2": std_nsga2,
                "mean_delta_p_nsga3": mean_nsga3,
                "std_delta_p_nsga3": std_nsga3,
                "nsga2_vs_nsga3_marker": mark_nsga2,
                "pvalue_nsga2_vs_nsga3": p_nsga2,
                "mean_time_nsga2_s": float(np.mean(times["NSGA-II"])),
                "mean_time_nsga3_s": float(np.mean(times["NSGA-III"])),
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
            f"mean_delta_p: NSGA2={mean_nsga2:.4e}, NSGA3={mean_nsga3:.4e}, "
            f"winner={winner_algo}, margin={winner_margin_pct_over_loser:.2f}%"
        )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["problem", "m", "run_id", "seed_nsga2", "seed_nsga3"],
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
            "mean_delta_p_nsga2",
            "std_delta_p_nsga2",
            "mean_delta_p_nsga3",
            "std_delta_p_nsga3",
            "nsga2_vs_nsga3_marker",
            "pvalue_nsga2_vs_nsga3",
            "mean_time_nsga2_s",
            "mean_time_nsga3_s",
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
    parser = argparse.ArgumentParser(description="ZDT-only benchmark: NSGA-II vs NSGA-III (Delta_p).")
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
                "zdt_nsga2_vs_nsga3",
            )
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--fast-nsga3",
        dest="fast_nsga3",
        action="store_true",
        default=True,
        help="Enable NSGA-III fast path (faster survival + random selection + no duplicate elimination).",
    )
    parser.add_argument(
        "--no-fast-nsga3",
        dest="fast_nsga3",
        action="store_false",
        help="Disable NSGA-III fast path and use default NSGA-III settings.",
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
        fast_nsga3=bool(args.fast_nsga3),
    )


if __name__ == "__main__":
    main()
