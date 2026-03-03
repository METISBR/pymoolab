import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD

from algorithms.mopso_cd_ls.mopso_cd_ssw import MOPSO_CD_SSW

def run_test():
    problem = get_problem("bnh")
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    algos = {
        "NSGA-II": NSGA2(pop_size=100),
        "MOPSO-CD-SSW (Podado)": MOPSO_CD_SSW(pop_size=100, ref_dirs=ref_dirs)
    }

    results = {}
    metric = IGD(problem.pareto_front())

    for name, algo in algos.items():
        print(f"\nRodando {name}...")
        res = minimize(
            problem,
            algo,
            ('n_gen', 200),
            seed=42,
            verbose=False
        )
        if res.F is not None:
            igd_val = metric(res.F)
            results[name] = igd_val
            print(f"[{name}] IGD: {igd_val:.4f} | Soluções Finais: {len(res.F)}")
        else:
            print(f"[{name}] Falha ao encontrar frente Pareto factível.")

    print("\nResumo Final (Menor IGD é melhor):")
    for name, igd in results.items():
        print(f"{name}: {igd:.4f}")

if __name__ == "__main__":
    run_test()
