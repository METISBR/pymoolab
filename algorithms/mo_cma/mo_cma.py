# pymoolab 2026
"""Multi-objective covariance matrix adaptation evolution strategy.\n\nReference:\nC. Igel, N. Hansen, and S. Roth. Covariance matrix adaptation for multi- objective optimization. Evolutionary computation, 2007, 15(1): 1-28.\n"""

from __future__ import annotations

from pymoo.algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOCMA': {'integer', 'multi', 'real'}}


class MOCMA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
