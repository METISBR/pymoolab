# pymoolab 2026
"""Decomposition based evolutionary algorithm guided by growing neural gas.\n\nReference:\nY. Liu, H. Ishibuchi, N. Masuyama, and Y. Nojima. Adapting reference vectors and scalarizing functions by growing neural gas to handle irregular Pareto fronts. IEEE Transactions on Evolutionary Computation, 2020, 24(3): 439-453. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'DEAGNG': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class DEAGNG(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
