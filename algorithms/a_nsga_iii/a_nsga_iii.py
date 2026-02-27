# pymoolab 2026
"""Adaptive NSGA-III.\n\nReference:\nH. Jain and K. Deb. An evolutionary many-objective optimization algorithm using reference-point based non-dominated sorting approach, part II: Handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation, 2014, 18(4): 602-622.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'ANSGAIII': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class ANSGAIII(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
