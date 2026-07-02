# pymoolab 2026
"""Direction-aware Decision-Variable Analysis (DVA) module for GCSMaOEA.

This module implements the second core contribution of GCSMaOEA:
**Direction-aware Decision-Variable Analysis**.  Conventional decision-variable
analysis treats convergence and diversity as scalar properties of the whole
objective space.  In GCSMaOEA, variables are scored by their contribution to
*each repaired reference direction*.  This produces direction-specific
variable masks, so the algorithm can assign different decision variables to
different regions of the Pareto front.

For an article, this corresponds to Section 3.2: the proposed DVA mechanism
couples variable importance with the repaired reference vectors from ARV.

Mathematical notation:
    X          : decision vectors, shape (N, D)
    F          : objective vectors, shape (N, M)
    masks      : binary matrix, shape (K, D); 1 = variable active
    W_repaired : repaired reference vectors, shape (L, M)
    alpha      : perturbation magnitude for sparse sampling
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymoo.core.population import Population

from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.TournamentSelection import TournamentSelection


# ---------------------------------------------------------------------------
# Counted evaluation
# ---------------------------------------------------------------------------

def _evaluate_F(problem, X: np.ndarray, evaluator=None) -> np.ndarray:
    """Evaluate decision vectors and return the objective matrix.

    When an ``evaluator`` is supplied every evaluation is routed through it so
    that it increments the algorithm's ``n_eval`` counter.  This is essential
    for a fair stopping criterion: probe/sample evaluations performed by DVA
    must be charged to the budget exactly like ordinary offspring.  The direct
    ``problem.evaluate`` path is kept only as a fallback for standalone/unit
    use where no evaluator exists (and is *not* used inside the algorithm).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if evaluator is not None:
        pop = Population.new("X", X)
        evaluator.eval(problem, pop)
        return np.asarray(pop.get("F"), dtype=float)
    return np.asarray(problem.evaluate(X, return_values_of=["F"]), dtype=float)


# ---------------------------------------------------------------------------
# Convergence helper
# ---------------------------------------------------------------------------

def _cal_con(range_obj: np.ndarray, pop_obj: np.ndarray) -> np.ndarray:
    """Sum-normalized convergence indicator used by sparse-sampling DVA.

    The objective values are linearly normalized to [0, 1] inside the range
    and then summed.  A smaller value means better convergence toward the
    ideal point.
    """
    pop_obj = np.asarray(pop_obj, dtype=float)
    range_obj = np.asarray(range_obj, dtype=float)

    if range_obj.shape[0] != 2:
        raise ValueError("range_obj must have shape (2, M).")

    lo = range_obj[0]
    hi = range_obj[1]
    denom = np.maximum(hi - lo, 1e-12)
    normalized = (pop_obj - lo) / denom
    return normalized.sum(axis=1)


# ---------------------------------------------------------------------------
# Sparse variable sampling
# ---------------------------------------------------------------------------

def sparse_fit(
    problem,
    archive: Population,
    masks: np.ndarray,
    n_samples: int,
    alpha: float = 0.25,
    rng: Optional[np.random.Generator] = None,
    evaluator=None,
) -> tuple[Population, np.ndarray]:
    """Evaluate a set of binary variable masks by sparse perturbation.

    For each mask, a base decision vector (the archive solution with best
    ``cal_con``) is replicated ``n_samples`` times, the variables marked 1 in
    the mask are replaced by uniform noise inside the decision-box range, and
    the resulting trial vectors are evaluated.  The fitness of the mask is a
    2-D vector:
        fitness[i, 0] = best convergence among the n_samples trials
        fitness[i, 1] = number of active variables (sparsity penalty)

    This is the sparse mask evaluation; in GCSMaOEA it is later combined
    with a direction-aware term (see ``direction_aware_score``).

    Parameters
    ----------
    problem : pymoo Problem
        The optimization problem.
    archive : Population
        Archive used to choose the base decision vector.
    masks : np.ndarray, shape (K, D)
        Binary masks; each row defines a group of variables to perturb.
    n_samples : int
        Number of trial vectors evaluated per mask.
    alpha : float
        Noise scaling factor relative to the decision-variable range.
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    population : Population
        All evaluated trial solutions (concatenated over masks).
    fitness : np.ndarray, shape (K, 2)
        Fitness of each mask.
    """
    if rng is None:
        rng = np.random.default_rng()

    masks = np.asarray(masks, dtype=bool)
    K, D = masks.shape

    if len(archive) == 0:
        raise ValueError("sparse_fit requires a non-empty archive.")

    X_all = np.asarray(archive.get("X"), dtype=float)
    F_all = np.asarray(archive.get("F"), dtype=float)

    range_obj = np.vstack([F_all.min(axis=0), F_all.max(axis=0)])
    con = _cal_con(range_obj, F_all)
    base_idx = int(np.argmin(con))
    base_x = X_all[base_idx]

    x_min = np.asarray(problem.xl, dtype=float).reshape(1, -1)
    x_max = np.asarray(problem.xu, dtype=float).reshape(1, -1)
    x_range = np.maximum(x_max - x_min, 1e-12)

    all_trials = []
    all_F = []
    fitness = np.zeros((K, 2), dtype=float)

    for i in range(K):
        mask = masks[i]
        if not np.any(mask):
            fitness[i] = [np.inf, 0.0]
            continue

        trials = np.tile(base_x, (n_samples, 1))
        noise = rng.uniform(0.0, 1.0, size=(n_samples, D))
        trials[:, mask] = x_min[:, mask] + alpha * noise[:, mask] * x_range[:, mask]
        trials = np.clip(trials, x_min, x_max)

        F_trials = _evaluate_F(problem, trials, evaluator)
        all_trials.append(trials)
        all_F.append(F_trials)

        fitness[i, 0] = float(np.min(_cal_con(range_obj, F_trials)))
        fitness[i, 1] = float(np.sum(mask))

    if all_trials:
        X_pop = np.vstack(all_trials)
        F_pop = np.vstack(all_F)
        pop = Population.new("X", X_pop)
        pop.set("F", F_pop)
        return pop, fitness

    return Population.empty(), fitness


# ---------------------------------------------------------------------------
# Direction-aware scoring
# ---------------------------------------------------------------------------

def direction_aware_score(
    F_samples: np.ndarray,
    W_repaired: np.ndarray,
    z_min: np.ndarray,
) -> float:
    """Compute a direction-awareness score for a set of sampled objective vectors.

    A mask is good not only if it improves global convergence, but also if
    the perturbed solutions spread across multiple repaired reference
    directions.  The score is the negative of the average minimum angle to
    the repaired vectors: the more directions covered, the lower (better) the
    score.

    Parameters
    ----------
    F_samples : np.ndarray, shape (S, M)
        Objective vectors obtained by sparse sampling with the mask.
    W_repaired : np.ndarray, shape (L, M)
        Repaired reference vectors.
    z_min : np.ndarray, shape (M,)
        Ideal point.

    Returns
    -------
    score : float
        Direction-awareness score (lower is better).
    """
    F_samples = np.asarray(F_samples, dtype=float)
    W_repaired = np.asarray(W_repaired, dtype=float)
    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)

    Fn = F_samples - z_min
    Wn = W_repaired - z_min

    fn_norm = np.linalg.norm(Fn, axis=1, keepdims=True)
    wn_norm = np.linalg.norm(Wn, axis=1, keepdims=True)
    fn_norm = np.where(fn_norm < 1e-12, 1.0, fn_norm)
    wn_norm = np.where(wn_norm < 1e-12, 1.0, wn_norm)

    cosine = (Fn @ Wn.T) / (fn_norm * wn_norm.T)
    cosine = np.clip(cosine, -1.0, 1.0)
    sine = np.sqrt(np.maximum(1.0 - cosine ** 2, 0.0))

    return float(np.mean(sine.min(axis=1)))


# ---------------------------------------------------------------------------
# Binary variation for masks
# ---------------------------------------------------------------------------

def _binary_variation(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    pro_c: float = 1.0,
    pro_m: float = 1.0,
) -> np.ndarray:
    """One-point crossover + bit-flip mutation for binary masks."""
    p1 = np.asarray(parent1, dtype=bool)
    p2 = np.asarray(parent2, dtype=bool)
    D = len(p1)

    # One-point crossover.
    if rng.random() < pro_c:
        k = rng.integers(1, D)
        child = np.concatenate([p1[:k], p2[k:]])
    else:
        child = p1.copy()

    # Bit-flip mutation.
    mutate = rng.random(D) < (pro_m / D)
    child = np.where(mutate, ~child, child)

    return child


# ---------------------------------------------------------------------------
# Mask optimization
# ---------------------------------------------------------------------------

def optimize_masks(
    problem,
    archive: Population,
    W_repaired: np.ndarray,
    z_min: np.ndarray,
    n_masks: int = 10,
    n_samples: int = 3,
    n_generations: int = 20,
    rng: Optional[np.random.Generator] = None,
    evaluator=None,
) -> tuple[np.ndarray, np.ndarray, Population]:
    """Evolve a population of binary variable masks using dominance-based selection.

    The masks are optimized with three objectives:
        1. best convergence obtained by sparse perturbation,
        2. sparsity (number of active variables),
        3. direction-aware coverage of the repaired reference vectors.

    For simplicity and stability, objectives 1 and 3 are aggregated into a
    single scalar using a weighted sum; sparsity remains the second objective.

    Parameters
    ----------
    problem : pymoo Problem
        The optimization problem.
    archive : Population
        Archive used to pick the base solution.
    W_repaired : np.ndarray, shape (L, M)
        Repaired reference vectors.
    z_min : np.ndarray, shape (M,)
        Ideal point.
    n_masks : int
        Number of masks in the DVA population.
    n_samples : int
        Trial evaluations per mask per generation.
    n_generations : int
        Generations of mask evolution.
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    masks : np.ndarray, shape (n_masks, D)
        Final optimized masks.
    fitness : np.ndarray, shape (n_masks, 2)
        Final (convergence, sparsity) values.
    evaluated_population : Population
        All extra solutions evaluated during mask optimization.
    """
    if rng is None:
        rng = np.random.default_rng()

    D = int(problem.n_var)

    # Initialize random masks with ~25% active variables on average.
    masks = rng.random((n_masks, D)) < 0.25
    # Ensure every mask has at least one active variable.
    for i in range(n_masks):
        if not np.any(masks[i]):
            masks[i, rng.integers(0, D)] = True

    evaluated_solutions = Population.empty()

    for _ in range(n_generations):
        pop, base_fitness = sparse_fit(
            problem,
            archive,
            masks,
            n_samples,
            alpha=0.25,
            rng=rng,
            evaluator=evaluator,
        )
        evaluated_solutions = Population.merge(evaluated_solutions, pop)

        # Compute direction-aware term for each mask from the current batch.
        F_batch = np.asarray(pop.get("F"), dtype=float)
        n_per_mask = n_samples
        dir_scores = np.zeros(n_masks, dtype=float)
        for i in range(n_masks):
            start = i * n_per_mask
            end = start + n_per_mask
            if end <= len(F_batch):
                dir_scores[i] = direction_aware_score(
                    F_batch[start:end], W_repaired, z_min
                )
            else:
                dir_scores[i] = 1.0

        # Aggregate convergence and direction-awareness; keep sparsity.
        # Lower is better for both objectives.
        con_norm = base_fitness[:, 0]
        dir_norm = dir_scores
        f1 = con_norm + 0.5 * dir_norm
        f2 = base_fitness[:, 1]
        fitness = np.column_stack([f1, f2])

        # Dominance + crowding environmental selection on the mask population.
        front_no, max_f_no = NDSort(fitness, n_masks)
        front_no = np.asarray(front_no, dtype=float).reshape(-1)
        next_mask = front_no < float(max_f_no)
        crowd = np.asarray(CrowdingDistance(fitness, front_no), dtype=float)
        last = np.where(front_no == float(max_f_no))[0]
        need = int(n_masks - np.sum(next_mask))
        if need > 0 and last.size > 0:
            rank = np.argsort(-crowd[last])
            next_mask[last[rank[:need]]] = True

        selected_indices = np.where(next_mask)[0]
        if len(selected_indices) > n_masks:
            selected_indices = selected_indices[:n_masks]
        elif len(selected_indices) < n_masks:
            extras = n_masks - len(selected_indices)
            selected_indices = np.concatenate([
                selected_indices,
                rng.integers(0, n_masks, size=extras),
            ])
        selected_masks = masks[selected_indices]

        # Generate offspring masks by binary variation.
        fitness_for_tournament = np.column_stack([front_no, -crowd])
        mating = np.asarray(
            TournamentSelection(2, n_masks, fitness_for_tournament, rng=rng),
            dtype=int,
        ) - 1
        mating = np.mod(mating, len(selected_masks))
        offspring = []
        for i in range(n_masks):
            p1 = selected_masks[mating[i]]
            p2 = selected_masks[mating[(i + 1) % n_masks]]
            child = _binary_variation(p1, p2, rng)
            if not np.any(child):
                child[rng.integers(0, D)] = True
            offspring.append(child)
        masks = np.asarray(offspring, dtype=bool)

    # Final evaluation of the returned masks.
    pop, fitness = sparse_fit(problem, archive, masks, n_samples, alpha=0.25, rng=rng, evaluator=evaluator)
    evaluated_solutions = Population.merge(evaluated_solutions, pop)

    return masks, fitness, evaluated_solutions


# ---------------------------------------------------------------------------
# Directional variable partition (Geometry-Coupled Search core)
# ---------------------------------------------------------------------------

def directional_variable_analysis(
    problem,
    base_x: np.ndarray,
    W_repaired: np.ndarray,
    z_min: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    max_probe: int = 200,
    delta: float = 0.05,
    evaluator=None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, Population]:
    """Assign variables to directions and learn convergence targets.

    This is the geometric coupling at the heart of GCS-MaOEA: the *same* repaired
    reference geometry that adapts the decomposition is used to structure the
    decision space.  Each variable ``j`` is perturbed around an elite solution and
    the resulting objective-space displacement is projected onto every repaired
    direction; ``j`` is assigned to the direction of maximal aligned displacement.

    If all coordinate probes for a variable stay in the same nearest direction,
    the variable is treated as a convergence/distance variable.  Its target value
    is the probed coordinate value that minimizes the local objective norm.  This
    is a problem-agnostic coordinate repair: it does not know whether a problem is
    DTLZ5, DTLZ6, or any other degenerate-front instance.

    Cost is bounded: at most ``max_probe`` single-coordinate evaluations (a random
    subset of variables for large-scale problems). Unprobed variables stay
    unassigned (direction ``-1``) and reproduction treats them globally.

    Returns
    -------
    dir_to_vars : list[np.ndarray]
        For each repaired direction, the indices of variables assigned to it.
    var_to_dir : np.ndarray, shape (D,)
        Direction index per variable (``-1`` = unassigned/global).
    var_targets : np.ndarray, shape (D,)
        Learned coordinate targets for convergence variables; NaN if unknown.
    evaluated : Population
        The probe solutions (re-usable as extra evaluated individuals).
    """
    if rng is None:
        rng = np.random.default_rng()

    base_x = np.asarray(base_x, dtype=float).reshape(-1)
    D = base_x.size
    W = np.asarray(W_repaired, dtype=float)
    L = len(W)
    z = np.asarray(z_min, dtype=float).reshape(1, -1)

    var_to_dir = np.full(D, -1, dtype=int)
    var_targets = np.full(D, np.nan, dtype=float)
    if L == 0 or D == 0:
        return (
            [np.array([], dtype=int) for _ in range(max(L, 0))],
            var_to_dir,
            var_targets,
            Population.empty(),
        )

    xl = np.asarray(problem.xl, dtype=float).reshape(-1)
    xu = np.asarray(problem.xu, dtype=float).reshape(-1)
    x_range = np.maximum(xu - xl, 1e-12)

    probe = np.arange(D) if D <= max_probe else rng.choice(D, size=max_probe, replace=False)

    Wn = W - z
    wn = np.linalg.norm(Wn, axis=1, keepdims=True)
    wn = np.where(wn < 1e-12, 1.0, wn)
    Wn = Wn / wn

    def _nearest_direction(F_rows: np.ndarray) -> np.ndarray:
        Fn = F_rows - z
        nrm = np.linalg.norm(Fn, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        cos = (Fn / nrm) @ Wn.T
        return np.argmax(cos, axis=1)

    f0 = _evaluate_F(
        problem, np.clip(base_x, xl, xu).reshape(1, -1), evaluator
    ).reshape(-1)
    base_assoc = int(_nearest_direction(f0.reshape(1, -1))[0])

    # Multi-sample control-variable analysis: sample each probed variable across
    # its full range (other variables fixed at the elite). A variable is a
    # *position/diversity* variable if its samples reach >=2 distinct nearest
    # directions (it steers movement ALONG the front); it is then linked to every
    # direction it can reach. A variable whose samples keep one association is a
    # *convergence/distance* variable -> global (shared by all directions), and
    # gets a coordinate target from the best local probe.
    samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    S = samples.size
    P = len(probe)
    trials = np.repeat(base_x.reshape(1, -1), P * S, axis=0)
    cols = np.repeat(probe, S)
    vals = xl[cols] + np.tile(samples, P) * x_range[cols]
    trials[np.arange(P * S), cols] = vals
    trials = np.clip(trials, xl, xu)

    Ft = _evaluate_F(problem, trials, evaluator)
    assoc = _nearest_direction(Ft).reshape(P, S)
    local_score = np.linalg.norm(np.maximum(Ft - z, 0.0), axis=1).reshape(P, S)

    base_score = float(np.linalg.norm(np.maximum(f0 - z.reshape(-1), 0.0)))

    def _probe_target_confidence(F_rows: np.ndarray, scores: np.ndarray) -> bool:
        """Return whether coordinate probes look like radial convergence."""
        M = F_rows.shape[1]
        shifted = np.maximum(F_rows - z, 0.0)
        norms = np.linalg.norm(shifted, axis=1, keepdims=True)
        unit = shifted / np.where(norms < 1e-12, 1.0, norms)
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        angular_spread = float(np.max(np.arccos(cosine)))

        component_range = F_rows.max(axis=0) - F_rows.min(axis=0)
        component_scale = np.maximum(
            np.maximum(np.max(np.abs(F_rows), axis=0), np.abs(f0)),
            1e-12,
        )
        relative_range = component_range / component_scale
        active_components = int(np.sum(relative_range > 0.05))
        max_relative_range = float(np.max(relative_range))
        isolated_component = (
            max_relative_range >= 0.8
            and active_components <= max(1, int(np.ceil(0.25 * M)))
        )
        if isolated_component:
            return False

        gain = (base_score - float(np.min(scores))) / max(base_score, 1e-12)
        radial_scale = (
            angular_spread <= 0.05
            and active_components > 0
            and max_relative_range <= 0.60
        )

        return radial_scale or (gain >= 0.02 and angular_spread <= 0.20)

    def _is_monotone_convergence(scores: np.ndarray) -> bool:
        """Geometry-independent distance-variable test.

        A variable is a convergence/distance variable if sweeping it makes the
        objective-norm ``|F - z|`` change monotonically with a non-trivial gain,
        regardless of which reference direction is reached.  This does NOT rely
        on the (unreliable) repaired geometry, so it still detects distance
        variables on degenerate or strongly-biased fronts where the angular
        association is pure noise.
        """
        s = np.asarray(scores, dtype=float).reshape(-1)
        if s.size < 3:
            return False
        diffs = np.diff(s)
        monotone = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
        scale = max(float(np.max(np.abs(s))), 1e-12)
        gain = float(np.max(s) - np.min(s)) / scale
        return monotone and gain >= 0.02

    dir_members: list[list[int]] = [[] for _ in range(L)]
    for p in range(P):
        j = int(probe[p])
        reached = set(int(a) for a in assoc[p])
        best = int(np.argmin(local_score[p]))
        target_confident = _probe_target_confidence(
            Ft[p * S : (p + 1) * S],
            local_score[p],
        )
        monotone_convergence = _is_monotone_convergence(local_score[p])
        if len(reached) <= 1 and (target_confident or monotone_convergence):
            var_to_dir[j] = -1  # convergence variable -> global
            var_targets[j] = float(vals[p * S + best])
        else:
            var_to_dir[j] = int(assoc[p, best])  # representative (for inspection)
            for d in reached:
                dir_members[d].append(j)

    known_primary = np.isfinite(var_targets)
    secondary_trials = []
    secondary_F = []
    if known_primary.any():
        target_base = base_x.copy()
        target_base[known_primary] = var_targets[known_primary]
        unresolved = [int(j) for j in probe if not np.isfinite(var_targets[int(j)])]
        if unresolved:
            trials2 = np.repeat(target_base.reshape(1, -1), len(unresolved) * S, axis=0)
            cols2 = np.repeat(np.asarray(unresolved, dtype=int), S)
            vals2 = xl[cols2] + np.tile(samples, len(unresolved)) * x_range[cols2]
            trials2[np.arange(len(unresolved) * S), cols2] = vals2
            trials2 = np.clip(trials2, xl, xu)
            Ft2 = _evaluate_F(problem, trials2, evaluator)
            base_target_F = _evaluate_F(
                problem, np.clip(target_base, xl, xu).reshape(1, -1), evaluator
            ).reshape(-1)

            for p2, j in enumerate(unresolved):
                F_rows = Ft2[p2 * S : (p2 + 1) * S]
                component_range = F_rows.max(axis=0) - F_rows.min(axis=0)
                component_scale = np.maximum(
                    np.maximum(np.max(np.abs(F_rows), axis=0), np.abs(base_target_F)),
                    1e-12,
                )
                relative_range = component_range / component_scale
                max_relative_range = float(np.max(relative_range))
                if 0.0 < max_relative_range <= 1e-5:
                    scores = np.linalg.norm(np.maximum(F_rows - z, 0.0), axis=1)
                    best = int(np.argmin(scores))
                    var_targets[j] = float(vals2[p2 * S + best])
                    var_to_dir[j] = -1

            secondary_trials.append(trials2)
            secondary_F.append(Ft2)

    dir_to_vars = [np.asarray(sorted(set(m)), dtype=int) for m in dir_members]

    if secondary_trials:
        extra_X = np.vstack([trials] + secondary_trials)
        extra_F = np.vstack([Ft] + secondary_F)
    else:
        extra_X = trials
        extra_F = Ft
    extra = Population.new("X", extra_X)
    extra.set("F", extra_F)
    return dir_to_vars, var_to_dir, var_targets, extra


def directional_variable_partition(
    problem,
    base_x: np.ndarray,
    W_repaired: np.ndarray,
    z_min: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    max_probe: int = 200,
    delta: float = 0.05,
    evaluator=None,
) -> tuple[list[np.ndarray], np.ndarray, Population]:
    """Backward-compatible wrapper returning only direction partition data."""
    dir_to_vars, var_to_dir, _, extra = directional_variable_analysis(
        problem,
        base_x,
        W_repaired,
        z_min,
        rng=rng,
        max_probe=max_probe,
        delta=delta,
        evaluator=evaluator,
    )
    return dir_to_vars, var_to_dir, extra


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class DVAManager:
    """State container for direction-aware decision-variable analysis.

    The manager decides whether DVA should run (large-scale problems) and
    stores the resulting variable masks.  For standard-scale MAOPs
    (``n_var < 100``) DVA is skipped to save function evaluations.
    """

    def __init__(
        self,
        n_var: int,
        n_masks: int = 10,
        n_samples: int = 3,
        n_generations: int = 20,
        enabled: bool = True,
        period: int = 1,
        max_fe: Optional[int] = None,
        fe_budget_ratio: float = 0.10,
    ):
        self.n_var = int(n_var)
        self.n_masks = int(n_masks)
        self.n_samples = int(n_samples)
        self.n_generations = int(n_generations)
        self.enabled = bool(enabled) and (n_var >= 100)
        self.masks: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None

        # Budget control: DVA evaluations are bounded to a fraction of the total
        # budget and the analysis runs only every ``period`` generations, so it
        # cannot starve the main search of function evaluations.
        self.period = int(max(1, period))
        self.max_fe = None if max_fe is None else int(max_fe)
        self.fe_budget_ratio = float(np.clip(fe_budget_ratio, 0.0, 1.0))
        self._age = 0
        self._fe_used = 0

    def _cost_estimate(self) -> int:
        """Approximate number of evaluations a single DVA call consumes."""
        return self.n_masks * self.n_samples * (self.n_generations + 1)

    def run(
        self,
        problem,
        archive: Population,
        W_repaired: np.ndarray,
        z_min: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        evaluator=None,
    ) -> Population:
        """Run DVA and return all extra evaluated solutions.

        If DVA is disabled, off its periodic schedule, or over its evaluation
        budget, an empty population is returned.
        """
        if not self.enabled:
            return Population.empty()

        # Periodic gate.
        self._age += 1
        if self._age < self.period:
            return Population.empty()
        self._age = 0

        # Budget gate: stop spending FE on DVA once its share is exhausted.
        if self.max_fe is not None and self.fe_budget_ratio < 1.0:
            allowed = int(self.fe_budget_ratio * self.max_fe)
            if self._fe_used + self._cost_estimate() > max(allowed, 0):
                return Population.empty()

        self.masks, self.fitness, extra = optimize_masks(
            problem,
            archive,
            W_repaired,
            z_min,
            n_masks=self.n_masks,
            n_samples=self.n_samples,
            n_generations=self.n_generations,
            rng=rng,
            evaluator=evaluator,
        )
        self._fe_used += int(len(extra))
        return extra

    def active_variable_groups(self) -> list[np.ndarray]:
        """Return the list of active variable indices for each mask."""
        if self.masks is None:
            return []
        return [np.where(m)[0] for m in self.masks]


# ---------------------------------------------------------------------------
# Linkage learning: budgeted differential-grouping interaction detection
# ---------------------------------------------------------------------------

def detect_variable_interactions(
    problem,
    x_base: np.ndarray,
    rng: np.random.Generator,
    evaluator=None,
    pair_budget: int = 200,
    delta_frac: float = 0.1,
    eps_rel: float = 1e-4,
) -> tuple[list[np.ndarray], np.ndarray, Population]:
    """Budgeted differential-grouping test for variable interactions.

    Implements the finite-difference interaction criterion of differential
    grouping (Omidvar et al.; refined in DG2/ERDG) under a strict evaluation
    budget, for use on nonseparable problems (WFG transformations, LSMOP3/5+):
    variables ``i`` and ``j`` interact when

        f(x + d e_i + d e_j) - f(x + d e_j)  !=  f(x + d e_i) - f(x)

    beyond a relative tolerance, tested component-wise on the objective vector
    (a pair interacts if ANY objective exhibits non-additivity).  A full test
    is O(D^2) evaluations, which is unaffordable at D=100M; instead a random
    subset of ``pair_budget`` pairs is tested and the interaction graph closed
    into connected components (linkage groups).  Cost:
    ``1 + |touched vars| + pair_budget`` evaluations, all charged to the
    common budget through ``evaluator``.

    Returns
    -------
    (groups, var_group_of, extra) :
        ``groups``       list of variable-index arrays (size >= 2 only);
        ``var_group_of`` array (D,) mapping each variable to its group id,
                         or -1 when the variable is in no detected group;
        ``extra``        evaluated probe Population (mergeable, no FE waste).
    """
    x_base = np.asarray(x_base, dtype=float).reshape(-1)
    D = x_base.shape[0]
    xl = np.asarray(problem.xl, dtype=float).reshape(-1)
    xu = np.asarray(problem.xu, dtype=float).reshape(-1)
    span = np.maximum(xu - xl, 1e-12)
    step = delta_frac * span

    # Perturbation direction: move toward whichever bound is farther.
    toward_upper = (xu - x_base) >= (x_base - xl)
    delta = np.where(toward_upper, step, -step)

    pair_budget = int(max(0, pair_budget))
    if pair_budget == 0 or D < 3:
        return [], -np.ones(D, dtype=int), Population.empty()

    # Sample distinct random pairs.
    n_pairs = min(pair_budget, D * (D - 1) // 2)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    guard = 0
    while len(pairs) < n_pairs and guard < 20 * n_pairs:
        guard += 1
        i = int(rng.integers(0, D))
        j = int(rng.integers(0, D))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    touched = sorted({v for p in pairs for v in p})
    var_pos = {v: k for k, v in enumerate(touched)}

    # Batch-evaluate: base, single perturbations, pair perturbations.
    X_batch = [x_base]
    for v in touched:
        xv = x_base.copy()
        xv[v] = np.clip(xv[v] + delta[v], xl[v], xu[v])
        X_batch.append(xv)
    for (i, j) in pairs:
        xij = x_base.copy()
        xij[i] = np.clip(xij[i] + delta[i], xl[i], xu[i])
        xij[j] = np.clip(xij[j] + delta[j], xl[j], xu[j])
        X_batch.append(xij)

    X_batch = np.vstack([x.reshape(1, -1) for x in X_batch])
    pop = Population.new("X", X_batch)
    if evaluator is not None:
        evaluator.eval(problem, pop)
        F_batch = np.asarray(pop.get("F"), dtype=float)
    else:
        F_batch = np.asarray(problem.evaluate(X_batch, return_values_of=["F"]), dtype=float)
        pop = Population.new("X", X_batch, F=F_batch)

    f0 = F_batch[0]
    f_single = F_batch[1 : 1 + len(touched)]
    f_pair = F_batch[1 + len(touched) :]
    scale = np.maximum(np.abs(f0), 1.0)

    # Union-find over interacting pairs.
    parent = np.arange(D, dtype=int)

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    interacting = np.zeros(D, dtype=bool)
    for k, (i, j) in enumerate(pairs):
        d1 = f_single[var_pos[i]] - f0
        d2 = f_pair[k] - f_single[var_pos[j]]
        if np.any(np.abs(d1 - d2) > eps_rel * scale):
            _union(i, j)
            interacting[i] = True
            interacting[j] = True

    groups_map: dict[int, list[int]] = {}
    for v in np.where(interacting)[0]:
        groups_map.setdefault(_find(int(v)), []).append(int(v))

    groups = [np.asarray(sorted(g), dtype=int) for g in groups_map.values() if len(g) >= 2]
    var_group_of = -np.ones(D, dtype=int)
    for gid, g in enumerate(groups):
        var_group_of[g] = gid
    return groups, var_group_of, pop
