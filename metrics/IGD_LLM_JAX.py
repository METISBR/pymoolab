# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
# igd_llm_JAX.py
# PymooLab metric plugin: IGD with configurable p-norm - JAX variant (igd_llm_JAX)
#
# Formula (Convention A - arithmetic mean, jMetal/jMetalPy/MOEA-D standard):
#   IGD_p(S, PF) = (1/|PF|) * sum_{z in PF} min_{s in S} ||z - s||_p
#
# References:
#   - Zhang & Li, MOEA/D, IEEE TEC 11(6):712-731, 2007  (IGD canonical citation)
#   - Van Veldhuizen & Lamont, TR-98-03, 1998            (GD lineage)
#   - jMetal InvertedGenerationalDistance.java           (Java reference)
#   - jMetalPy quality_indicator.py                      (Python reference)
#   - Sierra & Coello Coello, 2004                       (p-norm generalisation)
#
# JAX strategy:
#   - Uses jax.numpy (jnp) for vectorised pairwise distance computation.
#   - Falls back to NumPy if JAX is not available.
#   - Numerically consistent with the CPU variant under float64 inputs.
#   - jax.jit compilation is optional; the returned metric callable is plain Python
#     (not jitted by default) to stay compatible with PymooLab's dispatch.
#
# Outer aggregation: plain arithmetic mean over |PF| reference points.
# p-norm: used ONLY in the inner per-point Minkowski distance ||z-s||_p.
# Does NOT wrap pymoo.indicators.igd.IGD (per policy).

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    # Enable 64-bit precision in JAX (critical for metric accuracy)
    jax.config.update('jax_enable_x64', True)
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False
    import numpy as jnp  # type: ignore[assignment]  # NumPy fallback alias


# ---------------------------------------------------------------------------
# Core JAX computation (vectorised, no Python loops over points)
# ---------------------------------------------------------------------------

def _compute_igd_jax(approx_front, ref_front, p: float) -> float:
    """
    Compute IGD_p using JAX-vectorised broadcasting.

    Pairwise distance tensor: (|PF|, |S|) computed via broadcasting over
    the (|PF|, |S|, n_obj) difference tensor - O(|PF|*|S|*n_obj) arithmetic.

    Parameters
    ----------
    approx_front : array-like, shape (n_solutions, n_obj)
    ref_front    : array-like, shape (n_ref, n_obj)
    p            : float, Minkowski exponent >= 1.0

    Returns
    -------
    float
        Returns np.nan for degenerate inputs.
    """
    # Clamp p to valid range [1.0, 100.0] silently
    p_safe = max(1.0, min(float(p), 100.0))

    if _JAX_AVAILABLE:
        S  = jnp.array(approx_front, dtype=jnp.float64)   # (n_s, n_obj)
        PF = jnp.array(ref_front,    dtype=jnp.float64)   # (n_r, n_obj)

        # Guard: degenerate inputs -> return nan
        if S.ndim != 2 or PF.ndim != 2:
            return float('nan')
        if S.shape[0] == 0 or PF.shape[0] == 0:
            return float('nan')
        if S.shape[1] != PF.shape[1]:
            return float('nan')

        # Broadcasting: PF[:, None, :] - S[None, :, :] -> (n_r, n_s, n_obj)
        diff = PF[:, None, :] - S[None, :, :]              # (n_r, n_s, n_obj)

        if p_safe == 2.0:
            # Euclidean: stable via squared sum then sqrt
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (n_r, n_s)
        elif p_safe == 1.0:
            # Manhattan: sum of absolute values
            dist = jnp.sum(jnp.abs(diff), axis=-1)        # (n_r, n_s)
        else:
            # General Minkowski: sum(|d|^p)^(1/p)
            dist = jnp.sum(jnp.abs(diff) ** p_safe, axis=-1) ** (1.0 / p_safe)  # (n_r, n_s)

        min_dists = jnp.min(dist, axis=1)                 # (n_r,)
        igd_value = float(jnp.mean(min_dists))
    else:
        # Pure NumPy fallback (identical semantics, no JAX dependency)
        S  = np.asarray(approx_front, dtype=np.float64)
        PF = np.asarray(ref_front,    dtype=np.float64)

        # Guard: degenerate inputs -> return nan
        if S.ndim != 2 or PF.ndim != 2:
            return float('nan')
        if S.shape[0] == 0 or PF.shape[0] == 0:
            return float('nan')
        if S.shape[1] != PF.shape[1]:
            return float('nan')

        diff = PF[:, None, :] - S[None, :, :]             # (n_r, n_s, n_obj)
        if p_safe == 2.0:
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        elif p_safe == 1.0:
            dist = np.sum(np.abs(diff), axis=-1)
        else:
            dist = np.sum(np.abs(diff) ** p_safe, axis=-1) ** (1.0 / p_safe)
        min_dists = dist.min(axis=1)
        igd_value = float(np.mean(min_dists))

    return igd_value


# ---------------------------------------------------------------------------
# Context helper (mirrors CPU module)
# ---------------------------------------------------------------------------

def _extract_context(context: dict) -> dict:
    """Support nested context['config'] as well as flat context dicts."""
    if context is None:
        return {}
    cfg = context.get('config', context)
    return cfg if cfg is not None else {}


def _get_ref_front(cfg: dict):
    """Resolve reference front from known PymooLab context keys.
    Returns np.ndarray if found, else None.
    """
    for key in ('pareto_front', 'ref_pf', 'reference_front', 'pf'):
        pf = cfg.get(key)
        if pf is not None:
            return np.asarray(pf, dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Plugin factory
# ---------------------------------------------------------------------------

def create_metric(context: dict):
    """
    PymooLab JAX metric plugin factory for IGD with configurable p-norm.

    Parameters read from context (or context['config']):
        p           : float, default 2.0  - Minkowski p-norm exponent (>= 1.0)
        pareto_front / ref_pf / reference_front / pf
                    : 2-D array-like - reference front PF

    Returns
    -------
    callable
        metric(front) -> float
        where `front` is the approximation front S as a 2-D array of shape
        (n_solutions, n_obj) in objective space.
        Returns np.nan for degenerate inputs.
    """
    cfg = _extract_context(context)

    # Hyperparameter: p - clamp to [1.0, 100.0] silently
    p = float(cfg.get('p', 2.0))
    _p = max(1.0, min(p, 100.0))

    # Resolve and cache the reference front; fall back to empty sentinel
    ref_front = _get_ref_front(cfg)
    if ref_front is None:
        # No reference front in context: metric will always return nan
        ref_front = np.empty((0, 0), dtype=np.float64)
    elif ref_front.ndim != 2 or ref_front.shape[0] == 0:
        ref_front = np.empty((0, 0), dtype=np.float64)

    _ref_front = ref_front

    def metric(front) -> float:
        """
        Compute IGD_p(front, PF) using JAX-vectorised broadcasting.

        Parameters
        ----------
        front : array-like, shape (n_solutions, n_obj)
            Approximation front in objective space.

        Returns
        -------
        float
            IGD_p value in [0, +inf).  Lower is better (direction='minimize').
            Returns np.nan for degenerate inputs.
        """
        S = np.asarray(front, dtype=np.float64)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        return _compute_igd_jax(S, _ref_front, _p)

    # Attach metadata for PymooLab introspection
    metric.__name__        = 'igd_llm_JAX'
    metric.direction       = 'minimize'
    metric.lower_is_better = True
    metric.p               = _p
    metric.jax_available   = _JAX_AVAILABLE
    metric.ref_front_shape = _ref_front.shape
    metric.__doc_extra__   = (
        f"IGD_p (JAX) with p={_p}, ref_front shape={_ref_front.shape}. "
        "Formula: mean_{z in PF}(min_{s in S} ||z-s||_p). "
        "Convention A (arithmetic mean outer aggregation, jMetal/jMetalPy standard). "
        f"JAX backend: {'available' if _JAX_AVAILABLE else 'unavailable - NumPy fallback active'}."
    )
    return metric
