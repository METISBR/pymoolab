import numpy as np

def HV_fast_MC(pop_obj: np.ndarray, optimum: np.ndarray, sample_num: int = 1_000_000) -> float:
    """
    Fast Monte Carlo Hypervolume estimation ported from PlatEMO's HV.m.
    This method dramatically speeds up HV calculation for M > 5 by
    dynamically removing dominated samples from the test pool.
    """
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")

    pop_obj = np.atleast_2d(pop_obj)
    optimum = np.atleast_2d(optimum)

    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")

    N, M = pop_obj.shape

    # 1. Normalization (matching PlatEMO's fmin / fmax scaling)
    fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(M))
    fmax = np.max(optimum, axis=0)

    den = (fmax - fmin) * 1.1
    # Avoid div by zero
    den = np.where(np.abs(den) <= 1e-12, 1.0, den)

    pop_obj = (pop_obj - fmin) / den

    # Remove solutions out of bounds
    pop_obj = pop_obj[~np.any(pop_obj > 1.0, axis=1)]

    if pop_obj.size == 0:
        return 0.0

    ref_point = np.ones(M)
    max_value = ref_point
    min_value = np.min(pop_obj, axis=0)

    if np.any(max_value < min_value):
        return 0.0

    # 2. Monte Carlo Estimation with Dynamic Sample Pruning
    rng = np.random.default_rng(1)
    samples = rng.uniform(low=min_value, high=max_value, size=(sample_num, M))

    for i in range(pop_obj.shape[0]):
        if samples.shape[0] == 0:
            break

        # Match PlatEMO's short-circuit domination check
        domi = np.ones(samples.shape[0], dtype=bool)
        for m in range(M):
            if not np.any(domi):
                break
            domi &= (pop_obj[i, m] <= samples[:, m])

        # Dynamically shrink the sample pool
        samples = samples[~domi]

    score = np.prod(max_value - min_value) * (1.0 - (samples.shape[0] / sample_num))
    return float(score)

def _pymoolab_wrapper(front: np.ndarray, context: dict) -> float:
    pf = context.get("pareto_front")
    if pf is None:
        return float("nan")
    return HV_fast_MC(front, pf)

METRICS = {
    "HV_fast_MC": _pymoolab_wrapper
}
