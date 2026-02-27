# pymoolab 2026
"""Adaptive normal reference vector-based multi- and many-objective evolutionary algorithm.\n\nReference:\nY. Hua, Q. Liu, and K. Hao. Adaptive normal vector guided evolutionary multi- and many-objective optimization. Complex & Intelligent Systems, 2024, 10: 3709-3726.\n"""

from __future__ import annotations

from pymoo.algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NRVMOEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class NRVMOEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
