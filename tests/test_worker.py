"""Teste de ProcessPoolExecutor com o worker de processos."""
import sys
sys.path.insert(0, r"d:\devsuport\pymoolab")

import concurrent.futures
import time

from pymoolab_process_worker import run_trial_in_process

PROJECT_ROOT = r"d:\devsuport\pymoolab"


def main():
    # Prepara 4 trials (2 algoritmos x 2 seeds)
    trials = [
        {"algo_module": "algorithms.cmoea_cd.cmoea_cd", "algo_class_name": "CMOEA_CD", "algo_spec_name": "CMOEA_CD", "seed": 42},
        {"algo_module": "algorithms.cmoea_cd.cmoea_cd", "algo_class_name": "CMOEA_CD", "algo_spec_name": "CMOEA_CD", "seed": 123},
        {"algo_module": "algorithms.nsga3_local.nsga3_local", "algo_class_name": "NSGA3Local", "algo_spec_name": "NSGA3Local", "seed": 42},
        {"algo_module": "algorithms.nsga3_local.nsga3_local", "algo_class_name": "NSGA3Local", "algo_spec_name": "NSGA3Local", "seed": 123},
    ]

    print(f"Submetendo {len(trials)} trials via ProcessPoolExecutor (2 workers)...")
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, t in enumerate(trials):
            f = executor.submit(
                run_trial_in_process,
                algo_module=t["algo_module"],
                algo_class_name=t["algo_class_name"],
                algo_spec_id=f"test_{i}",
                algo_spec_name=t["algo_spec_name"],
                problem_source="pymoo",
                problem_name="zdt1",
                problem_module="pymoo.problems.multi.zdt",
                problem_class_name="ZDT1",
                problem_kwargs={"n_var": 30},
                runtime_cfg={"pop_size": 50, "n_obj": 2, "seed": t["seed"]},
                seed=t["seed"],
                max_fe=500,
                run_index=i + 1,
                n_runs=len(trials),
                project_root=PROJECT_ROOT,
            )
            futures[f] = t

        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                print(
                    f"  {t['algo_spec_name']} seed={t['seed']}: "
                    f"time={result['time_s']:.3f}s "
                    f"front={len(result['final_front'])} pontos"
                )
            except Exception as exc:
                print(f"  ERRO {t['algo_spec_name']} seed={t['seed']}: {exc}")

    elapsed = time.perf_counter() - start
    print(f"\nTotal ProcessPool: {elapsed:.3f}s (4 trials em 2 workers)")

    # Agora testa sequencial para comparar
    print("\nAgora sequencial...")
    start2 = time.perf_counter()
    for t in trials:
        run_trial_in_process(
            algo_module=t["algo_module"],
            algo_class_name=t["algo_class_name"],
            algo_spec_id="seq",
            algo_spec_name=t["algo_spec_name"],
            problem_source="pymoo",
            problem_name="zdt1",
            problem_module="pymoo.problems.multi.zdt",
            problem_class_name="ZDT1",
            problem_kwargs={"n_var": 30},
            runtime_cfg={"pop_size": 50, "n_obj": 2, "seed": t["seed"]},
            seed=t["seed"],
            max_fe=500,
            run_index=1,
            n_runs=1,
            project_root=PROJECT_ROOT,
        )

    elapsed2 = time.perf_counter() - start2
    print(f"Total sequencial: {elapsed2:.3f}s")
    print(f"\nSpeedup: {elapsed2/elapsed:.2f}x")


if __name__ == "__main__":
    main()
