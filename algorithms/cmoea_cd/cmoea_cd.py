# -*- coding: utf-8 -*-
"""
CMOEA-CD (pymoo-compatible)
===========================

Reference publication:
Z. Liu, F. Han, Q. Ling, H. Han, and J. Jiang,
"Constraint-Pareto Dominance and Diversity Enhancement Strategy-Based
Evolutionary Algorithm for Solving Constrained Multiobjective Optimization
Problems,"
IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2771-2784.
DOI: 10.1109/TEVC.2024.3525153

This implementation follows the original three-archive structure:
1) Forward exploration archive (FA)
2) Diversity enhancement archive (DA)
3) Feasibility exploitation archive (FEA)
"""

from __future__ import annotations

from typing import Optional

from util.array_backend import xp as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.UniformPoint import UniformPoint
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.array_backend import (
    get_array_module as _backend_get_array_module,
    to_device as _backend_to_device,
    to_numpy as _backend_to_numpy,
)

ALGORITHM_FLAGS = {
    "CMOEA_CD": {"multi", "many"},
}


def _to_numpy(x):
    out = _backend_to_numpy(x)
    if out is None:
        return None
    return np.asarray(out)


def _to_device(x, *, use_gpu: bool = False, dtype=None):
    return _backend_to_device(x, use_gpu=use_gpu, dtype=dtype)


def _uniform_points(target: int, n_obj: int) -> np.ndarray:
    w, _n_eff = UniformPoint(int(max(target, 1)), int(n_obj))
    w = np.asarray(w, dtype=float)
    return w


def _uniform_points_h(target: int, n_obj: int) -> float:
    w = _uniform_points(target, n_obj)
    cos_ww = _cosine_similarity(w, w)
    angle_ww = np.arccos(cos_ww)
    np.fill_diagonal(angle_ww, np.inf)
    return float(np.mean(np.min(angle_ww, axis=1)))


def _safe_span(zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    xp = _backend_get_array_module(zmin)
    span = zmax - zmin
    span = xp.where(xp.abs(span) <= 1e-12, 1.0, span)
    return span


def _normalize_obj(F: np.ndarray, zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    return (F - zmin) / _safe_span(zmin, zmax)


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    xp = _backend_get_array_module(A)
    AA = xp.sum(A * A, axis=1, keepdims=True)
    BB = xp.sum(B * B, axis=1, keepdims=True).T
    D2 = xp.maximum(AA + BB - 2.0 * (A @ B.T), 0.0)
    return xp.sqrt(D2)


def _cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    xp = _backend_get_array_module(A)
    na = xp.linalg.norm(A, axis=1, keepdims=True)
    nb = xp.linalg.norm(B, axis=1, keepdims=True)
    denom = xp.maximum(na * nb.T, 1e-32)
    C = (A @ B.T) / denom
    return xp.clip(C, -1.0, 1.0)


def _constraint_violation(pop: Population) -> np.ndarray:
    g = pop.get("G")
    if g is not None:
        xp = _backend_get_array_module(g)
        g = xp.asarray(g, dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        return xp.sum(xp.maximum(g, 0.0), axis=1)

    cv = pop.get("CV")
    if cv is not None:
        xp = _backend_get_array_module(cv)
        cv = xp.asarray(cv, dtype=float).reshape(-1)
        if len(cv) == len(pop):
            return xp.maximum(cv, 0.0)

    return np.zeros(len(pop), dtype=float)


def _round_obj(F: np.ndarray, decimals: int = 10) -> np.ndarray:
    xp = _backend_get_array_module(F)
    return xp.round(xp.asarray(F, dtype=float), decimals=decimals)


def _constraint_pareto_nondominated(pop: Population, add_constraint_rule: bool) -> np.ndarray:
    F = _round_obj(pop.get("F"), decimals=10)
    cv = _constraint_violation(pop)

    n, _m = F.shape
    dominated = np.zeros(n, dtype=bool)

    for i in range(n - 1):
        if dominated[i]:
            continue

        # Keep compatibility with the previous Python behavior:
        # comparisons skip already dominated j-indices.
        rem = np.arange(i + 1, n, dtype=int)
        rem = rem[~dominated[rem]]
        if rem.size == 0:
            continue

        diff = F[i] - F[rem]
        max_err = np.max(diff, axis=1)
        min_err = np.min(diff, axis=1)
        eq = np.all(diff == 0.0, axis=1)

        if not add_constraint_rule:
            dom_i = (~eq) & (min_err >= 0.0)
            dom_j = eq | ((~eq) & (~(min_err >= 0.0)) & (max_err <= 0.0))
        else:
            cvi = cv[i]
            cvj = cv[rem]
            dom_i = (eq & (cvi > cvj)) | ((~eq) & (min_err >= 0.0) & ((cvj <= 0.0) | (cvj <= cvi)))
            dom_j = (eq & (cvi <= cvj)) | (
                (~eq) & (~(min_err >= 0.0)) & (max_err <= 0.0) & ((cvi <= 0.0) | (cvi <= cvj))
            )

        # Equivalent to loop break: once dom_i occurs at position k, only positions < k can dom_j.
        hit = np.flatnonzero(dom_i)
        cut = int(hit[0]) if hit.size > 0 else rem.size

        if cut > 0:
            rem_p = rem[:cut]
            dom_j_p = dom_j[:cut]
            if np.any(dom_j_p):
                dominated[rem_p[dom_j_p]] = True

        if hit.size > 0:
            dominated[i] = True

    return ~dominated


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    cd = np.zeros(n, dtype=float)
    for k in range(m):
        idx = np.argsort(F[:, k])
        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf

        f_min = F[idx[0], k]
        f_max = F[idx[-1], k]
        denom = f_max - f_min
        if denom <= 1e-32:
            continue

        interior = idx[1:-1]
        prev_idx = idx[:-2]
        next_idx = idx[2:]
        cd[interior] += (F[next_idx, k] - F[prev_idx, k]) / denom
    return cd


def _truncation(F: np.ndarray, k_delete: int) -> np.ndarray:
    n = len(F)
    if k_delete <= 0 or n == 0:
        return np.zeros(n, dtype=bool)

    D = _pairwise_dist(F, F)
    np.fill_diagonal(D, np.inf)

    deleted = np.zeros(n, dtype=bool)
    while int(np.sum(deleted)) < int(k_delete):
        remain = np.where(~deleted)[0]
        temp = np.sort(D[np.ix_(remain, remain)], axis=1)
        # Equivalent to MATLAB sortrows(temp): lexicographic ascending by columns.
        rank = np.lexsort(temp.T[::-1])
        deleted[remain[rank[0]]] = True
    return deleted


def _modified_nsga3_select(
    pop: Population,
    target_size: int,
    zmin: np.ndarray,
    zmax: np.ndarray,
    *,
    use_gpu: bool = False,
) -> Population:
    F = _to_device(pop.get("F"), use_gpu=bool(use_gpu), dtype=float)
    n, m = F.shape
    if n == 0:
        return pop

    W = _to_device(_uniform_points(target_size, m), use_gpu=bool(use_gpu), dtype=float)
    n_ref = len(W)

    zmin_dev = _to_device(zmin, use_gpu=bool(use_gpu), dtype=float)
    zmax_dev = _to_device(zmax, use_gpu=bool(use_gpu), dtype=float)
    Fn = _normalize_obj(F, zmin_dev, zmax_dev)
    xp = _backend_get_array_module(Fn)
    dist = xp.linalg.norm(Fn, axis=1)

    cos_wp = _cosine_similarity(W, Fn)
    angle_sin = xp.sqrt(xp.maximum(1.0 - cos_wp * cos_wp, 0.0))

    fitness = angle_sin * dist[None, :]
    chosen = _to_numpy(xp.argmin(fitness, axis=1)).astype(int, copy=False)
    return pop[chosen]


class CMOEA_CD(Algorithm):
    """
    Constraint-Pareto dominance + diversity enhancement CMOEA.
    """

    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int = 100,
        e1: int = 1,
        e2: int = 1,
        de_cr: float = 1.0,
        de_f: float = 0.5,
        seed: Optional[int] = None,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )

        self.ref_dirs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.pop_size = int(max(pop_size, 2))
        self.e1 = int(np.clip(e1, 1, 3))
        self.e2 = int(np.clip(e2, 1, 3))
        self.de_cr = float(np.clip(de_cr, 0.0, 1.0))
        self.de_f = float(max(0.0, de_f))

        self.ns = int(max(1, self.pop_size // 3))
        self.nds = NonDominatedSorting()

        self.fa: Population = Population.empty()
        self.da: Population = Population.empty()
        self.fea: Population = Population.empty()
        self.zmin: Optional[np.ndarray] = None

    def _setup(self, problem, **kwargs):
        self.ns = int(max(1, self.pop_size // 3))

    def _initialize_infill(self):
        X = self._sample_random(self.pop_size)
        return Population.new("X", X)

    def _initialize_advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            self.pop = Population.empty()
            self.fea = Population.empty()
            self.opt = self.pop
            return

        self.fa = Population.empty()
        self.da = Population.empty()
        self.fea = infills
        F_init = _to_device(infills.get("F"), use_gpu=bool(self.use_gpu), dtype=float)
        xp = _backend_get_array_module(F_init)
        self.zmin = xp.min(F_init, axis=0) - 1e-6
        self._update_archives(infills)
        self.pop = self.fea
        self._set_optimum()

    def _infill(self):
        if len(self.fea) == 0:
            X = self._sample_random(self.pop_size)
            return Population.new("X", X)

        pop1 = self.fa if len(self.fa) > 0 else self.fea
        pop2 = self.da if len(self.da) > 0 else self.fea

        n3 = int(max(1, self.pop_size // 3))
        idx3 = self.random_state.integers(0, len(self.fea), size=n3)
        pop3 = self.fea[np.asarray(idx3, dtype=int)]

        mp11 = self.random_state.permutation(len(pop1))
        mp12 = self.random_state.permutation(len(pop1))
        mp21 = self.random_state.permutation(len(pop2))
        mp22 = self.random_state.permutation(len(pop2))
        mp31 = self.random_state.permutation(len(pop3))
        mp32 = self.random_state.permutation(len(pop3))

        use_de = bool(self.random_state.random() < 0.5)
        if use_de:
            off1 = self._operator_de(pop1, mp11, mp12)
            off2 = self._operator_de(pop2, mp21, mp22)
            off3 = self._operator_de(pop3, mp31, mp32)
        else:
            off1 = self._operator_ga(pop1, mp11)
            off2 = self._operator_ga(pop2, mp21)
            off3 = self._operator_ga(pop3, mp31)

        off = Population.merge(off1, off2, off3)
        if len(off) == 0:
            return Population.new("X", self._sample_random(self.pop_size))
        return off

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        self._update_archives(infills)
        self.pop = self.fea
        self._set_optimum()

    def _update_archives(self, offspring: Population):
        F_off = _to_device(offspring.get("F"), use_gpu=bool(self.use_gpu), dtype=float)
        xp = _backend_get_array_module(F_off)
        if self.zmin is None:
            self.zmin = xp.min(F_off, axis=0) - 1e-6
        else:
            self.zmin = xp.minimum(self.zmin, xp.min(F_off, axis=0) - 1e-6)

        self.fa = self._forward_exploration_archive(self.fa, offspring, self.zmin, self.ns, self.e1)
        self.da = self._diversity_enhancement_archive(self.da, offspring, self.zmin, self.ns)
        self.fea = self._feasibility_exploitation_archive(self.fea, offspring, self.pop_size, self.e2)

    def _forward_exploration_archive(
        self,
        fa: Population,
        offspring: Population,
        zmin: np.ndarray,
        ns: int,
        add_mode: int,
    ) -> Population:
        pool = Population.merge(fa, offspring) if len(fa) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=False)
        pool = pool[np.where(nd_mask)[0]]

        p = len(pool)
        if p == 0:
            return pool
        if p <= ns:
            extra = ns - p
            if extra > 0:
                idx = self.random_state.integers(0, p, size=extra)
                pool = pool[np.concatenate([np.arange(p), idx])]
            return pool

        F = np.asarray(pool.get("F"), dtype=float)
        zmax = np.max(F, axis=0)
        Fn = _normalize_obj(F, zmin, zmax)

        if add_mode == 1:
            deleted = _truncation(Fn, p - ns)
            idx = np.where(~deleted)[0]
            return pool[idx]
        if add_mode == 2:
            cd = _crowding_distance(Fn)
            rank = np.argsort(-cd)
            return pool[rank[:ns]]
        return _modified_nsga3_select(pool, ns, zmin, zmax, use_gpu=bool(self.use_gpu))

    def _feasibility_exploitation_archive(
        self,
        fea: Population,
        offspring: Population,
        n_target: int,
        add_mode: int,
    ) -> Population:
        pool = Population.merge(fea, offspring) if len(fea) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=True)
        pool = pool[np.where(nd_mask)[0]]

        p = len(pool)
        if p == 0:
            return pool

        if p <= n_target:
            extra = n_target - p
            if extra > 0:
                idx = self.random_state.integers(0, p, size=extra)
                pool = pool[np.concatenate([np.arange(p), idx])]
            return pool

        cv = _constraint_violation(pool)
        feasible = cv <= 0.0
        n_feasible = int(np.sum(feasible))

        if n_feasible <= n_target:
            idx = np.argsort(cv)[:n_target]
            return pool[idx]

        pool_f = pool[np.where(feasible)[0]]
        F = np.asarray(pool_f.get("F"), dtype=float)
        zmin = np.min(F, axis=0) - 1e-6
        zmax = np.max(F, axis=0)
        Fn = _normalize_obj(F, zmin, zmax)

        if add_mode == 1:
            deleted = _truncation(Fn, len(pool_f) - n_target)
            idx = np.where(~deleted)[0]
            return pool_f[idx]
        if add_mode == 2:
            cd = _crowding_distance(Fn)
            rank = np.argsort(-cd)
            return pool_f[rank[:n_target]]
        return _modified_nsga3_select(pool_f, n_target, zmin, zmax, use_gpu=bool(self.use_gpu))

    def _diversity_enhancement_archive(
        self,
        da: Population,
        offspring: Population,
        zmin: np.ndarray,
        ns: int,
    ) -> Population:
        pool = Population.merge(da, offspring) if len(da) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=True)
        pool = pool[np.where(nd_mask)[0]]
        if len(pool) == 0:
            return pool

        F = np.asarray(pool.get("F"), dtype=float)
        cv = _constraint_violation(pool)
        n, m = F.shape

        W = _uniform_points(ns, m)
        ns_eff = len(W)
        h = _uniform_points_h(int(max(ns, 2)), int(m))

        feasible = cv <= 0.0
        if np.any(feasible):
            zmax = np.max(F[feasible], axis=0)
        else:
            zmax = F[int(np.argmin(cv))]

        Fn = _normalize_obj(F, zmin, zmax)
        cos_ws = _cosine_similarity(W, Fn)
        angle_sin = np.sqrt(np.maximum(1.0 - cos_ws * cos_ws, 0.0))

        chosen = np.empty(ns_eff, dtype=int)
        for i in range(ns_eff):
            angle = angle_sin[i]
            mask = angle <= h
            if not np.any(mask):
                mask[int(np.argmin(angle))] = True

            t = np.full(n, np.inf, dtype=float)
            t[mask] = cv[mask]

            feas_mask = t <= 0.0
            if not np.any(feas_mask):
                idx = int(np.argmin(t))
            else:
                t2 = np.full(n, np.inf, dtype=float)
                t2[feas_mask] = angle[feas_mask]
                idx = int(np.argmin(t2))
            chosen[i] = idx
        return pool[chosen]

    def _operator_de(self, pop: Population, idx1: np.ndarray, idx2: np.ndarray) -> Population:
        n = len(pop)
        if n == 0:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        off = OperatorDE(
            self.problem,
            X,
            X[np.asarray(idx1, dtype=int)],
            X[np.asarray(idx2, dtype=int)],
            Parameter=[self.de_cr, self.de_f, 1, 1],
            rng=self.random_state,
        )
        X = _to_device(off, use_gpu=bool(self.use_gpu), dtype=float)
        xp = _backend_get_array_module(X)
        xl = _to_device(self.problem.xl, use_gpu=bool(self.use_gpu), dtype=float)
        xu = _to_device(self.problem.xu, use_gpu=bool(self.use_gpu), dtype=float)
        X = xp.clip(X, xl, xu)
        return Population.new("X", X)

    def _operator_ga(self, pop: Population, idx: np.ndarray) -> Population:
        n = len(pop)
        if n == 0:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        off = OperatorGA(
            self.problem,
            X[np.asarray(idx, dtype=int)],
            Parameter=[1, 20, 1, 1],
            rng=self.random_state,
        )
        X = _to_device(off, use_gpu=bool(self.use_gpu), dtype=float)
        xp = _backend_get_array_module(X)
        xl = _to_device(self.problem.xl, use_gpu=bool(self.use_gpu), dtype=float)
        xu = _to_device(self.problem.xu, use_gpu=bool(self.use_gpu), dtype=float)
        X = xp.clip(X, xl, xu)
        return Population.new("X", X)

    def _sample_random(self, n: int) -> np.ndarray:
        n = int(max(0, n))
        if n == 0:
            return np.empty((0, self.problem.n_var), dtype=float)
        xl = _to_device(self.problem.xl, use_gpu=bool(self.use_gpu), dtype=float)
        xu = _to_device(self.problem.xu, use_gpu=bool(self.use_gpu), dtype=float)
        X = xl + self.random_state.random((n, self.problem.n_var)) * (xu - xl)
        xp = _backend_get_array_module(xl)
        return xp.asarray(X, dtype=float)

    def _set_optimum(self):
        if self.pop is None or len(self.pop) == 0:
            self.opt = self.pop
            return

        cv = _constraint_violation(self.pop)
        feasible_idx = np.where(cv <= 0.0)[0]
        if len(feasible_idx) > 0:
            feasible = self.pop[feasible_idx]
            nd = self.nds.do(feasible.get("F"), only_non_dominated_front=True)
            self.opt = feasible[np.asarray(nd, dtype=int)]
        else:
            self.opt = self.pop[np.asarray([int(np.argmin(cv))], dtype=int)]
