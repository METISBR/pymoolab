from __future__ import annotations

from pathlib import Path

import numpy as np

import PymooLab
from problems.single.bbob import BBOB_F20, BBOB_F20_JAX


def _find_spec_by_name(specs, name: str):
    for spec in specs.values():
        if spec.name == name:
            return spec
    return None


def test_eval_matlab_positive_int_expr_is_safe_and_compatible():
    ctx = {"D": 10}
    assert PymooLab._eval_matlab_positive_int_expr("ceil(obj.D/3)", ctx) == 4
    assert PymooLab._eval_matlab_positive_int_expr("max(1, floor(obj.D/4)+1)", ctx) == 3
    assert PymooLab._eval_matlab_positive_int_expr("__import__('os').system('whoami')", ctx) is None


def test_discover_problem_specs_has_local_zdt1_defaults_and_factory():
    warnings: list[str] = []
    specs = PymooLab.discover_problem_specs(Path("."), warnings)
    zdt1 = _find_spec_by_name(specs, "ZDT1")
    zdt1_jax = _find_spec_by_name(specs, "ZDT1_JAX")

    assert zdt1 is not None
    assert zdt1_jax is not None
    assert zdt1.source == "local"
    assert zdt1.default_n_var == 30
    assert zdt1.default_n_obj == 2

    problem = zdt1.factory({"n_var": zdt1.default_n_var})
    x = np.random.uniform(0.0, 1.0, size=(4, problem.n_var))
    out: dict[str, np.ndarray] = {}
    problem._evaluate(x, out)
    assert np.asarray(out["F"]).shape == (4, problem.n_obj)


def test_discover_problem_specs_keeps_family_specific_default_objectives():
    warnings: list[str] = []
    specs = PymooLab.discover_problem_specs(Path("."), warnings)

    expected_m3 = {
        "IMMOEA_F4",
        "IMMOEA_F4_JAX",
        "IMMOEA_F8",
        "IMMOEA_F8_JAX",
        "MOEADDE_F6",
        "MOEADDE_F6_JAX",
        "MOEADM2M_F6",
        "MOEADM2M_F6_JAX",
        "MOEADM2M_F7",
        "MOEADM2M_F7_JAX",
        "RMMEDA_F4",
        "RMMEDA_F4_JAX",
        "RMMEDA_F8",
        "RMMEDA_F8_JAX",
    }

    for name in sorted(expected_m3):
        spec = _find_spec_by_name(specs, name)
        assert spec is not None, name
        assert spec.default_n_obj == 3, name


def test_discover_metric_specs_local_only_and_deltap_runs():
    warnings: list[str] = []
    specs = PymooLab.discover_metric_specs(Path("."), warnings)

    assert specs
    assert all(spec.source == "local" for spec in specs.values())

    delta_p = _find_spec_by_name(specs, "DeltaP")
    delta_p_jax = _find_spec_by_name(specs, "DeltaP_JAX")
    assert delta_p is not None
    assert delta_p_jax is not None

    ref_pf = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=float)
    context = {"pareto_front": ref_pf, "n_obj": 2}
    front = ref_pf.copy()

    delta_cpu = float(delta_p.factory(context)(front))
    delta_jax = float(delta_p_jax.factory(context)(front))

    assert np.isfinite(delta_cpu)
    assert np.isfinite(delta_jax)
    assert abs(delta_cpu) <= 1e-9
    assert abs(delta_jax) <= 1e-9


def test_bbob_f20_cpu_and_jax_smoke():
    for cls in (BBOB_F20, BBOB_F20_JAX):
        problem = cls()
        x = np.random.uniform(np.asarray(problem.xl), np.asarray(problem.xu), size=(3, problem.n_var))
        out: dict[str, np.ndarray] = {}
        problem._evaluate(x, out)
        f = np.asarray(out["F"], dtype=float)
        assert f.shape == (3, 1)
        assert np.all(np.isfinite(f))
