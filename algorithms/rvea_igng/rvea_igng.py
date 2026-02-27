# pymoolab 2026
"""RVEA based on improved growing neural gas.\n\nReference:\nQ. Liu, Y. Jin, M. Heiderich, T. Rodemann, and G. Yu. An adaptive reference vector-guided evolutionary algorithm using growing neural gas for many-objective optimization of irregular problems. IEEE Transactions on Cybernetics, 2022, 52(5): 2698-2711. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'RVEAiGNG': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class RVEAiGNG(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
