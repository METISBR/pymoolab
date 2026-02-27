# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
# Made by PymooLab 2026.
from typing import Any

import numpy as np
from pymoo.core.problem import Problem


class ZCAT2_LLM(Problem):
    '''
    Smooth scalable multi-objective test problem labeled "ZCAT2_LLM".

    This implementation uses a DTLZ2-style benchmark with an arbitrary
    number of objectives (n_obj >= 2) and decision variables (n_var >= n_obj - 1).
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
        # Sample from a normal distribution and project onto the unit sphere
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

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        '''Vectorized evaluation.

        Parameters
        ----------
        X : (N, n_var) array_like
            Batch of decision vectors.
        out : dict
            Output dictionary; will contain key 'F' with shape (N, n_obj).
        '''
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]

        N, D = X.shape
        M = int(self.n_obj)

        # If the dimensionality is inconsistent, return NaNs instead of raising.
        if D != int(self.n_var):
            out['F'] = np.full((N, M), np.nan, dtype=float)
            return

        # DTLZ2-style formulation.
        if self.n_var > M - 1:
            xm = X[:, M - 1 :]
            g = np.sum((xm - 0.5) ** 2, axis=1)
        else:
            g = np.zeros(N, dtype=float)

        one_plus_g = 1.0 + g
        F = np.empty((N, M), dtype=float)

        for i in range(M):
            # Product of cos terms
            prod = np.ones(N, dtype=float)
            for j in range(0, M - i - 1):
                prod *= np.cos(0.5 * np.pi * X[:, j])

            # Final sin term if i > 0
            if i > 0:
                idx = M - i - 1
                prod *= np.sin(0.5 * np.pi * X[:, idx])

            F[:, i] = one_plus_g * prod

        out['F'] = F
