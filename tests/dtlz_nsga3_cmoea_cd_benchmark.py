import argparse
import csv
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Force JAX execution on CPU to benchmark NumPy baseline vs JAX-on-CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Side-effect import: installs legacy aliases (core/util/operators/algorithms.moo)
# required by local algorithms such as CMOEA_CD.
try:
    import guiPymoo  # noqa: F401
except Exception:  # noqa: BLE001
    import PymooLab  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wilcoxon
from pymoo.algorithms.moo.nsga3 import (
    NSGA3,
    ReferenceDirectionSurvival,
    associate_to_niches,
    calc_niche_count,
    niching,
)
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from problems import get_problem
from pymoo.util.misc import intersect
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

try:
    from algorithms.cmoea_cd.cmoea_cd import CMOEA_CD
except Exception:  # noqa: BLE001
    from guiPymoo.algorithms.cmoea_cd.cmoea_cd import CMOEA_CD


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)


DTLZ_ALL_FUNCTIONS = ["dtlz1", "dtlz2", "dtlz3"]
DEFAULT_DTLZ_FUNCTIONS = ["dtlz1", "dtlz2", "dtlz3"]
ALGORITHMS = ("NSGA-III", "CMOEA-CD")
DTLZ_K = {"dtlz1": 5, "dtlz2": 10, "dtlz3": 10}


class FastReferenceDirectionSurvival(ReferenceDirectionSurvival):
    """NSGA-III survival variant with faster non-dominated sorting."""

    def _do(  # type: ignore[override]
        self,
        problem: Any,
        pop: Any,
        n_survive: int,
        D: Any = None,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
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


class DTLZJAXProblem(Problem):
    """JAX version of DTLZ1/2/3 for CPU-mode JAX runtime benchmarking."""

    def __init__(self, func_name: str, m: int, n_var: int) -> None:
        self.func_name = str(func_name).strip().lower()
        self.m = int(m)
        super().__init__(n_var=int(n_var), n_obj=int(m), n_ieq_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        _x = jnp.asarray(x)
        out["F"] = np.asarray(self._eval_F(_x), dtype=float)

    @partial(jax.jit, static_argnums=0)
    def _eval_F(self, x: Any) -> Any:
        m = self.m
        xm = x[:, m - 1 :]
        if self.func_name in {"dtlz1", "dtlz3"}:
            k = x.shape[1] - m + 1
            g = 100.0 * (
                k
                + jnp.sum(
                    jnp.power(xm - 0.5, 2.0) - jnp.cos(20.0 * jnp.pi * (xm - 0.5)),
                    axis=1,
                )
            )
        else:
            g = jnp.sum(jnp.power(xm - 0.5, 2.0), axis=1)

        f_values = []
        for i in range(m):
            if self.func_name == "dtlz1":
                val = 0.5 * (1.0 + g)
                n_mul = m - i - 1
                if n_mul > 0:
                    val = val * jnp.prod(x[:, :n_mul], axis=1)
                if i > 0:
                    val = val * (1.0 - x[:, n_mul])
            else:
                val = 1.0 + g
                n_cos = m - i - 1
                if n_cos > 0:
                    val = val * jnp.prod(jnp.cos(x[:, :n_cos] * jnp.pi / 2.0), axis=1)
                if i > 0:
                    val = val * jnp.sin(x[:, n_cos] * jnp.pi / 2.0)
            f_values.append(val)

        return jnp.stack(f_values, axis=1)


def _parse_dtlz_functions(raw: str) -> List[str]:
    token = str(raw).strip().lower()
    if token == "all":
        return list(DTLZ_ALL_FUNCTIONS)

    parts = [p.strip().lower() for p in re.split(r"[,\s;|]+", token) if p.strip()]
    if not parts:
        raise ValueError("No DTLZ function selected. Use --functions dtlz1 or --functions all.")

    selected: List[str] = []
    valid = set(DTLZ_ALL_FUNCTIONS)
    for name in parts:
        if name not in valid:
            raise ValueError(
                f"Unknown DTLZ function '{name}'. Valid values: {', '.join(DTLZ_ALL_FUNCTIONS)} or 'all'."
            )
        if name not in selected:
            selected.append(name)
    return selected


def _random_seed_list(n: int) -> List[int]:
    rng = random.SystemRandom()
    seen = set()
    seeds: List[int] = []
    while len(seeds) < n:
        value = rng.randrange(1, 2_147_483_647)
        if value not in seen:
            seen.add(value)
            seeds.append(value)
    return seeds


def _random_seed_pairs(n: int) -> List[Tuple[int, int]]:
    seeds = _random_seed_list(2 * n)
    return [(seeds[2 * i], seeds[2 * i + 1]) for i in range(n)]


def _safe_F(F: Any, n_obj: int) -> np.ndarray:
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
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mins = np.min(dist, axis=1)
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


def _write_csv(path: Path, rows: List[Dict[str, Any]], headers: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _das_dennis_points(n_obj: int, n_partitions: int) -> int:
    return math.comb(n_obj + n_partitions - 1, n_partitions)


def _choose_partitions(n_obj: int, target_points: int, max_partitions: int = 60) -> int:
    best_p = 1
    best_diff = abs(_das_dennis_points(n_obj, 1) - target_points)
    for p in range(1, max_partitions + 1):
        pts = _das_dennis_points(n_obj, p)
        diff = abs(pts - target_points)
        if diff < best_diff:
            best_diff = diff
            best_p = p
        if pts > target_points and p > 1:
            break
    return best_p


def _build_ref_dirs(n_obj: int, target_points: int) -> np.ndarray:
    partitions = _choose_partitions(n_obj=n_obj, target_points=max(2, int(target_points)))
    return np.asarray(get_reference_directions("das-dennis", n_obj, n_partitions=partitions), dtype=float)


def _build_problem_pair(func_name: str, m: int) -> Tuple[Any, Any, int]:
    k = DTLZ_K[str(func_name).strip().lower()]
    n_var = int(m + k - 1)
    problem_cpu = get_problem(func_name, n_var=n_var, n_obj=m)
    problem_jax = DTLZJAXProblem(func_name=func_name, m=m, n_var=n_var)
    return problem_cpu, problem_jax, n_var


def _new_algorithm(
    name: str,
    ref_dirs: np.ndarray,
    pop_size: int,
    seed: int,
    *,
    fast_nsga3: bool = True,
) -> Any:
    algo_name = str(name).strip().upper()
    if algo_name == "NSGA-III":
        kwargs: Dict[str, Any] = {"ref_dirs": ref_dirs, "pop_size": int(pop_size), "seed": int(seed)}
        if fast_nsga3:
            # Unconstrained DTLZ: avoid duplicate checks and CV-based tournament overhead.
            kwargs.update(
                {
                    "selection": RandomSelection(),
                    "eliminate_duplicates": False,
                    "n_offsprings": int(pop_size),
                    "survival": FastReferenceDirectionSurvival(ref_dirs),
                }
            )
        return NSGA3(**kwargs)
    if algo_name == "CMOEA-CD":
        return CMOEA_CD(ref_dirs=ref_dirs, pop_size=int(pop_size), seed=int(seed), use_gpu=False, array_backend="numpy")
    raise ValueError(f"Unknown algorithm: {name}")


def _run_one_seed(
    func_name: str,
    m: int,
    n_evals: int,
    ref_dirs: np.ndarray,
    pop_size: int,
    run_id: int,
    seed_nsga3: int,
    seed_cmoea_cd: int,
    fast_nsga3: bool,
) -> Dict[str, Any]:
    problem_cpu, problem_jax, n_var = _build_problem_pair(func_name=func_name, m=m)
    seed_map = {"NSGA-III": int(seed_nsga3), "CMOEA-CD": int(seed_cmoea_cd)}

    out: Dict[str, Any] = {"run_id": int(run_id), "m": int(m), "n_var": int(n_var), "algo": {}}
    for algo_name in ALGORITHMS:
        seed = seed_map[algo_name]

        algo_cpu = _new_algorithm(
            algo_name,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=seed,
            fast_nsga3=bool(fast_nsga3),
        )
        t0 = time.perf_counter()
        res_cpu = minimize(
            problem_cpu,
            algo_cpu,
            ("n_eval", int(n_evals)),
            verbose=False,
            seed=int(seed),
            copy_algorithm=False,
            save_history=False,
        )
        time_cpu = time.perf_counter() - t0

        algo_jax = _new_algorithm(
            algo_name,
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            seed=seed,
            fast_nsga3=bool(fast_nsga3),
        )
        t1 = time.perf_counter()
        res_jax = minimize(
            problem_jax,
            algo_jax,
            ("n_eval", int(n_evals)),
            verbose=False,
            seed=int(seed),
            copy_algorithm=False,
            save_history=False,
        )
        time_jax = time.perf_counter() - t1

        out["algo"][algo_name] = {
            "seed": int(seed),
            "time_cpu_s": float(time_cpu),
            "time_jax_s": float(time_jax),
            "speedup_cpu_over_jax": float(time_cpu / time_jax) if time_jax > 0.0 else float("nan"),
            "F_cpu": _safe_F(res_cpu.F, m),
            "F_jax": _safe_F(res_jax.F, m),
        }

    return out


def run_experiment(
    out_dir: Path,
    n_runs: int,
    n_evals: int,
    parallel_workers: int,
    delta_p_order: float,
    target_pop_size: int,
    dtlz_functions: List[str],
    n_obj_values: List[int],
    fast_nsga3: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    devices = [f"{d.platform}:{getattr(d, 'device_kind', d)}" for d in jax.devices()]
    print(
        "[DTLZ] setup: "
        f"runs={n_runs}, evals={n_evals}, workers={parallel_workers}, "
        f"target_pop={target_pop_size}, functions={','.join(dtlz_functions)}, "
        f"m={','.join(map(str, n_obj_values))}, jax_devices={devices}, fast_nsga3={bool(fast_nsga3)}"
    )

    for func_name in dtlz_functions:
        for m in n_obj_values:
            ref_dirs = _build_ref_dirs(n_obj=int(m), target_points=int(target_pop_size))
            real_pop = int(len(ref_dirs))
            print(f"[DTLZ][{func_name}][m={m}] ref_dirs={real_pop} n_evals={n_evals}")

            seed_pairs = _random_seed_pairs(int(n_runs))
            for rid, (seed_nsga3, seed_cmoea_cd) in enumerate(seed_pairs, start=1):
                seed_rows.append(
                    {
                        "problem": f"{func_name}_m{m}",
                        "m": int(m),
                        "run_id": int(rid),
                        "seed_nsga3": int(seed_nsga3),
                        "seed_cmoea_cd": int(seed_cmoea_cd),
                    }
                )

            jobs = [
                (func_name, int(m), int(n_evals), ref_dirs, real_pop, rid, seed_nsga3, seed_cmoea_cd, bool(fast_nsga3))
                for rid, (seed_nsga3, seed_cmoea_cd) in enumerate(seed_pairs, start=1)
            ]

            results: List[Dict[str, Any]] = []
            if int(parallel_workers) > 1:
                with ProcessPoolExecutor(max_workers=int(parallel_workers)) as ex:
                    futures = {ex.submit(_run_one_seed, *job): job for job in jobs}
                    for fut in as_completed(futures):
                        job = futures[fut]
                        out = fut.result()
                        results.append(out)
                        print(
                            f"[DTLZ][{func_name}][m={m}] run={job[5]} done "
                            f"(nsga3 cpu={out['algo']['NSGA-III']['time_cpu_s']:.3f}s, "
                            f"jax={out['algo']['NSGA-III']['time_jax_s']:.3f}s | "
                            f"cmoea cpu={out['algo']['CMOEA-CD']['time_cpu_s']:.3f}s, "
                            f"jax={out['algo']['CMOEA-CD']['time_jax_s']:.3f}s)"
                        )
            else:
                for job in jobs:
                    out = _run_one_seed(*job)
                    results.append(out)
                    print(
                        f"[DTLZ][{func_name}][m={m}] run={job[5]} done "
                        f"(nsga3 cpu={out['algo']['NSGA-III']['time_cpu_s']:.3f}s, "
                        f"jax={out['algo']['NSGA-III']['time_jax_s']:.3f}s | "
                        f"cmoea cpu={out['algo']['CMOEA-CD']['time_cpu_s']:.3f}s, "
                        f"jax={out['algo']['CMOEA-CD']['time_jax_s']:.3f}s)"
                    )

            results.sort(key=lambda item: int(item["run_id"]))

            cpu_fronts = [out["algo"][alg]["F_cpu"] for out in results for alg in ALGORITHMS]
            reference_set = _build_empirical_reference(cpu_fronts, max_points=4000)
            reference_source = "empirical_nd_cpu"

            delta_cpu: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
            delta_jax: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
            time_cpu: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
            time_jax: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}

            for out in results:
                rid = int(out["run_id"])
                for alg in ALGORITHMS:
                    info = out["algo"][alg]
                    dp_cpu = float(delta_p(info["F_cpu"], reference_set, p=delta_p_order))
                    dp_jax = float(delta_p(info["F_jax"], reference_set, p=delta_p_order))
                    dt_cpu = float(info["time_cpu_s"])
                    dt_jax = float(info["time_jax_s"])
                    speedup = float(info["speedup_cpu_over_jax"])

                    delta_cpu[alg].append(dp_cpu)
                    delta_jax[alg].append(dp_jax)
                    time_cpu[alg].append(dt_cpu)
                    time_jax[alg].append(dt_jax)

                    run_rows.append(
                        {
                            "problem": f"{func_name}_m{m}",
                            "function": func_name,
                            "m": int(m),
                            "run_id": int(rid),
                            "algorithm": alg,
                            "seed": int(info["seed"]),
                            "n_var": int(out["n_var"]),
                            "pop_size": int(real_pop),
                            "n_evals": int(n_evals),
                            "reference_source": reference_source,
                            "delta_p_cpu": dp_cpu,
                            "delta_p_jax": dp_jax,
                            "time_cpu_s": dt_cpu,
                            "time_jax_s": dt_jax,
                            "speedup_cpu_over_jax": speedup,
                            "backend_code": "cpu_vs_jax_cpu_only",
                        }
                    )

            marker_nsga3, p_nsga3 = _wilcoxon_marker(np.asarray(time_jax["NSGA-III"]), np.asarray(time_cpu["NSGA-III"]))
            marker_cmoea, p_cmoea = _wilcoxon_marker(np.asarray(time_jax["CMOEA-CD"]), np.asarray(time_cpu["CMOEA-CD"]))

            mean_delta_cpu_nsga3 = float(np.mean(delta_cpu["NSGA-III"]))
            mean_delta_cpu_cmoea = float(np.mean(delta_cpu["CMOEA-CD"]))
            std_delta_cpu_nsga3 = float(np.std(delta_cpu["NSGA-III"], ddof=1)) if len(delta_cpu["NSGA-III"]) > 1 else 0.0
            std_delta_cpu_cmoea = float(np.std(delta_cpu["CMOEA-CD"], ddof=1)) if len(delta_cpu["CMOEA-CD"]) > 1 else 0.0

            marker_alg, p_alg = _wilcoxon_marker(np.asarray(delta_cpu["CMOEA-CD"]), np.asarray(delta_cpu["NSGA-III"]))
            best_algo_cpu_delta = "CMOEA-CD" if mean_delta_cpu_cmoea < mean_delta_cpu_nsga3 else "NSGA-III"

            summary_rows.append(
                {
                    "problem": f"{func_name}_m{m}",
                    "function": func_name,
                    "m": int(m),
                    "n_var": int(results[0]["n_var"]) if results else "",
                    "pop_size": int(real_pop),
                    "n_runs": int(n_runs),
                    "n_evals": int(n_evals),
                    "reference_source": reference_source,
                    "mean_delta_p_cpu_nsga3": mean_delta_cpu_nsga3,
                    "std_delta_p_cpu_nsga3": std_delta_cpu_nsga3,
                    "mean_delta_p_cpu_cmoea_cd": mean_delta_cpu_cmoea,
                    "std_delta_p_cpu_cmoea_cd": std_delta_cpu_cmoea,
                    "cmoea_vs_nsga3_delta_p_marker": marker_alg,
                    "pvalue_cmoea_vs_nsga3_delta_p": p_alg,
                    "mean_time_cpu_nsga3_s": float(np.mean(time_cpu["NSGA-III"])),
                    "std_time_cpu_nsga3_s": float(np.std(time_cpu["NSGA-III"], ddof=1))
                    if len(time_cpu["NSGA-III"]) > 1
                    else 0.0,
                    "mean_time_jax_nsga3_s": float(np.mean(time_jax["NSGA-III"])),
                    "std_time_jax_nsga3_s": float(np.std(time_jax["NSGA-III"], ddof=1))
                    if len(time_jax["NSGA-III"]) > 1
                    else 0.0,
                    "cpu_vs_jax_marker_nsga3": marker_nsga3,
                    "pvalue_cpu_vs_jax_nsga3": p_nsga3,
                    "mean_speedup_cpu_over_jax_nsga3": float(np.mean(np.asarray(time_cpu["NSGA-III"]) / np.asarray(time_jax["NSGA-III"]))),
                    "mean_time_cpu_cmoea_cd_s": float(np.mean(time_cpu["CMOEA-CD"])),
                    "std_time_cpu_cmoea_cd_s": float(np.std(time_cpu["CMOEA-CD"], ddof=1))
                    if len(time_cpu["CMOEA-CD"]) > 1
                    else 0.0,
                    "mean_time_jax_cmoea_cd_s": float(np.mean(time_jax["CMOEA-CD"])),
                    "std_time_jax_cmoea_cd_s": float(np.std(time_jax["CMOEA-CD"], ddof=1))
                    if len(time_jax["CMOEA-CD"]) > 1
                    else 0.0,
                    "cpu_vs_jax_marker_cmoea_cd": marker_cmoea,
                    "pvalue_cpu_vs_jax_cmoea_cd": p_cmoea,
                    "mean_speedup_cpu_over_jax_cmoea_cd": float(
                        np.mean(np.asarray(time_cpu["CMOEA-CD"]) / np.asarray(time_jax["CMOEA-CD"]))
                    ),
                    "best_algo_cpu_delta_p": best_algo_cpu_delta,
                }
            )

            print(
                f"[DTLZ][{func_name}][m={m}] done "
                f"(cpu_vs_jax nsga3={marker_nsga3}, cmoea={marker_cmoea}, "
                f"best_cpu_delta_p={best_algo_cpu_delta})"
            )

    _write_csv(
        out_dir / "seed_plan.csv",
        seed_rows,
        ["problem", "m", "run_id", "seed_nsga3", "seed_cmoea_cd"],
    )
    _write_csv(
        out_dir / "all_runs_speed.csv",
        run_rows,
        [
            "problem",
            "function",
            "m",
            "run_id",
            "algorithm",
            "seed",
            "n_var",
            "pop_size",
            "n_evals",
            "reference_source",
            "delta_p_cpu",
            "delta_p_jax",
            "time_cpu_s",
            "time_jax_s",
            "speedup_cpu_over_jax",
            "backend_code",
        ],
    )
    _write_csv(
        out_dir / "summary_speed.csv",
        summary_rows,
        [
            "problem",
            "function",
            "m",
            "n_var",
            "pop_size",
            "n_runs",
            "n_evals",
            "reference_source",
            "mean_delta_p_cpu_nsga3",
            "std_delta_p_cpu_nsga3",
            "mean_delta_p_cpu_cmoea_cd",
            "std_delta_p_cpu_cmoea_cd",
            "cmoea_vs_nsga3_delta_p_marker",
            "pvalue_cmoea_vs_nsga3_delta_p",
            "mean_time_cpu_nsga3_s",
            "std_time_cpu_nsga3_s",
            "mean_time_jax_nsga3_s",
            "std_time_jax_nsga3_s",
            "cpu_vs_jax_marker_nsga3",
            "pvalue_cpu_vs_jax_nsga3",
            "mean_speedup_cpu_over_jax_nsga3",
            "mean_time_cpu_cmoea_cd_s",
            "std_time_cpu_cmoea_cd_s",
            "mean_time_jax_cmoea_cd_s",
            "std_time_jax_cmoea_cd_s",
            "cpu_vs_jax_marker_cmoea_cd",
            "pvalue_cpu_vs_jax_cmoea_cd",
            "mean_speedup_cpu_over_jax_cmoea_cd",
            "best_algo_cpu_delta_p",
        ],
    )
    print(f"[DONE] Outputs written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTLZ CPU baseline vs JAX (CPU) benchmark with NSGA-III and CMOEA-CD.")
    parser.add_argument(
        "--functions",
        type=str,
        default="dtlz1,dtlz2,dtlz3",
        help="Functions to run (comma-separated) or 'all'.",
    )
    parser.add_argument("--n-runs", "--runs", dest="n_runs", type=int, default=3, help="Independent runs per function.")
    parser.add_argument("--n-evals", "--evals", dest="n_evals", type=int, default=25000, help="Max evaluations per run.")
    parser.add_argument(
        "--parallel-workers",
        "--workers",
        dest="parallel_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel workers.",
    )
    parser.add_argument("--delta-p-order", type=float, default=1.0, help="Order p in Delta_p.")
    parser.add_argument(
        "--pop-size",
        "--pop",
        dest="pop_size",
        type=int,
        default=300,
        help="Target number of reference directions / population size.",
    )
    parser.add_argument(
        "--m",
        type=str,
        default="5,10",
        help="Objective counts (comma-separated), e.g. 5,10.",
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
                "dtlz_nsga3_cmoea_cd_speed_cpu_vs_jax",
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


def main() -> None:
    args = parse_args()
    selected_functions = _parse_dtlz_functions(args.functions)
    n_obj_values = [int(x.strip()) for x in str(args.m).split(",") if x.strip()]
    run_experiment(
        out_dir=Path(args.out_dir),
        n_runs=int(args.n_runs),
        n_evals=int(args.n_evals),
        parallel_workers=max(1, int(args.parallel_workers)),
        delta_p_order=float(args.delta_p_order),
        target_pop_size=max(2, int(args.pop_size)),
        dtlz_functions=selected_functions,
        n_obj_values=n_obj_values,
        fast_nsga3=bool(args.fast_nsga3),
    )


if __name__ == "__main__":
    main()
