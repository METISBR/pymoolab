# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
from typing import Any

import numpy as np
try:
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover - JAX is optional
    jnp = np  # Fallback so the module still imports without JAX

from pymoo.core.problem import Problem


class ZCAT2_LLM_JAX(Problem):
    '''
    JAX-oriented version of the ZCAT2_LLM test problem.

    This class mirrors ZCAT2_LLM but uses `jnp` in `_evaluate` so it can be
    compatible with JAX-based workflows when JAX is available.
    '''

    def __init__(self, n_var: int = 30, n_obj: int = 3, **kwargs: Any) -> None:
        # Box constraints on the decision variables.
        xl = np.zeros(n_var, dtype=float)
        xu = np.ones(n_var, dtype=float)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        """Analytical Pareto front for the DTLZ2-style formulation.

        The PF of DTLZ2 is the portion of the unit hypersphere located in the
        first (non-negative) orthant: sum(F_i^2) = 1, F_i >= 0.
        """
        m = int(self.n_obj)
        n_points = max(1, int(n_pareto_points))

        rng = np.random.default_rng(1)
        z = rng.standard_normal(size=(n_points, m))
        z = np.abs(z)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        f = z / norms
        return f

    def _calc_pareto_set(self, n_pareto_points: int = 200):
        """Analytical Pareto set for the DTLZ2-style formulation.

        For DTLZ2, the Pareto-optimal set is obtained when the distance
        variables x_{m},...,x_{D} are fixed to 0.5, and the remaining
        variables x_1,...,x_{m-1} can take any value in [0, 1].
        """
        m = int(self.n_obj)
        d = int(self.n_var)
        n_points = max(1, int(n_pareto_points))

        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 1.0, size=(n_points, d))
        if d > m - 1:
            x[:, m - 1 :] = 0.5
        return x

    def _evaluate(self, X: Any, out: dict, *args: Any, **kwargs: Any) -> None:
        '''Vectorized evaluation using JAX where available.

        Parameters
        ----------
        X : (N, n_var) array_like or DeviceArray
            Batch of decision vectors.
        out : dict
            Output dictionary; will contain key 'F' with shape (N, n_obj).
        '''
        Xj = jnp.asarray(X, dtype=float)
        if Xj.ndim == 1:
            Xj = Xj[jnp.newaxis, :]

        N = Xj.shape[0]
        D = Xj.shape[1]
        M = int(self.n_obj)

        # If the dimensionality is inconsistent, return NaNs instead of raising.
        if D != int(self.n_var):
            out['F'] = jnp.full((N, M), jnp.nan, dtype=float)
            return

        # DTLZ2-style formulation.
        if self.n_var > M - 1:
            xm = Xj[:, M - 1 :]
            g = jnp.sum((xm - 0.5) ** 2, axis=1)
        else:
            g = jnp.zeros(N, dtype=float)

        one_plus_g = 1.0 + g

        def single_obj(i: int) -> Any:
            prod = jnp.ones(N, dtype=float)
            # Product of cos terms
            for j in range(0, M - i - 1):
                prod = prod * jnp.cos(0.5 * jnp.pi * Xj[:, j])
            # Final sin term if i > 0
            if i > 0:
                idx = M - i - 1
                prod = prod * jnp.sin(0.5 * jnp.pi * Xj[:, idx])
            return one_plus_g * prod

        cols = [single_obj(i) for i in range(M)]
        F = jnp.stack(cols, axis=1)

        out['F'] = F
