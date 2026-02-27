# pymoolab 2026
"""Large-scale evolutionary algorithm with reformulated decision variable analysis.\n\nReference:\nC. He, R. Cheng, L. Li, K. C. Tan, and Y. Jin. Large-scale multiobjective optimization via reformulated decision variable analysis. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 47-61.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'LERD': {'large', 'many', 'multi', 'real'}}


class LERD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
