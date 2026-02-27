# pymoolab 2026
"""AdaW (Evolutionary algorithm with adaptive weights).

Reference:
M. Li and X. Yao. What weights work for you? Adapting weights for any Pareto front shape in decomposition-based evolutionary multiobjective optimisation. Evolutionary Computation, 2020, 28(2): 227-253.
"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'AdaW': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class AdaW(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, adapt_weights=True, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.adapt_weights = adapt_weights
