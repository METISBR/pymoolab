from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from problems import get_problem as get_local_problem
from pymoolab_core.registry.backend_selection import (
    map_selected_ids_to_backend,
    resolve_available_operator_target,
)
from pymoolab_core.registry.rollout import (
    ENV_BACKEND_AWARE_LOADING,
    ENV_BACKEND_AWARE_STAGE,
    rollout_allows_domain,
    rollout_stage,
)


SYNC_MANIFEST = BASE_DIR / "tools" / "manifests" / "pymoo_sync_manifest.json"
JAX_MANIFEST = BASE_DIR / "tools" / "manifests" / "pymoo_jax_generation_manifest.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _EnvContext:
    def __init__(self, mapping: dict[str, str | None]) -> None:
        self.mapping = dict(mapping)
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.mapping.items():
            self.previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(value)

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(value)


@dataclass(frozen=True)
class _FakeMetricSpec:
    id: str
    name: str
    module: str


class SyncAndGenerationManifestTests(unittest.TestCase):
    def test_sync_manifest_integrity(self) -> None:
        self.assertTrue(SYNC_MANIFEST.exists(), f"Missing manifest: {SYNC_MANIFEST}")
        manifest = _read_json(SYNC_MANIFEST)
        sync = manifest.get("sync", {})
        self.assertIsInstance(sync, dict)

        for domain in ("problems", "operators"):
            payload = sync.get(domain, {})
            self.assertIsInstance(payload, dict)
            stats = payload.get("stats", {})
            files = payload.get("files", [])
            self.assertIsInstance(stats, dict)
            self.assertIsInstance(files, list)
            self.assertEqual(int(stats.get("total_synced", -1)), len(files))

            for entry in files:
                local_rel = str(entry.get("local_path", "")).strip()
                expected_sha = str(entry.get("sha256", "")).strip().lower()
                self.assertTrue(local_rel, f"{domain}: local_path missing in manifest entry")
                self.assertTrue(expected_sha, f"{domain}: sha256 missing for {local_rel}")

                target = BASE_DIR / local_rel
                self.assertTrue(target.exists(), f"{domain}: missing local file {local_rel}")
                text = target.read_text(encoding="utf-8-sig", errors="ignore")
                self.assertEqual(
                    _sha256_text(text).lower(),
                    expected_sha,
                    f"{domain}: hash mismatch for {local_rel}",
                )

    def test_jax_generation_manifest_consistency(self) -> None:
        self.assertTrue(JAX_MANIFEST.exists(), f"Missing manifest: {JAX_MANIFEST}")
        manifest = _read_json(JAX_MANIFEST)
        stats = manifest.get("stats", {})
        files = manifest.get("files", [])
        self.assertIsInstance(stats, dict)
        self.assertIsInstance(files, list)
        self.assertEqual(int(stats.get("total_sources", -1)), len(files))

        for entry in files:
            source = str(entry.get("source", "")).strip()
            target = str(entry.get("target", "")).strip()
            status = str(entry.get("status", "")).strip().lower()
            self.assertTrue(source)
            self.assertTrue(target)
            self.assertIn(status, {"created", "updated", "unchanged", "failed"})
            self.assertTrue((BASE_DIR / source).exists(), f"source missing: {source}")
            if status != "failed":
                self.assertTrue((BASE_DIR / target).exists(), f"target missing: {target}")


class BackendSelectionAcceptanceTests(unittest.TestCase):
    def test_problem_backend_selection_cpu_vs_gpu(self) -> None:
        cases = [
            ("zdt1", {"n_var": 30}),
            ("dtlz1", {"n_var": 7, "n_obj": 3}),
            ("wfg1", {"n_var": 24, "n_obj": 3}),
            ("bnh", {}),
        ]
        for name, kwargs in cases:
            cpu_problem = get_local_problem(name, backend="cpu", **kwargs)
            gpu_problem = get_local_problem(name, backend="gpu", **kwargs)
            cpu_module = str(type(cpu_problem).__module__).lower()
            gpu_module = str(type(gpu_problem).__module__).lower()
            self.assertNotIn("_jax", cpu_module, f"{name}: CPU path should be non-JAX")
            if name == "wfg1":
                # Known fallback path: some generated JAX variants still rely on in-place writes.
                self.assertTrue(
                    "_jax" in gpu_module or gpu_module.startswith("problems."),
                    f"{name}: GPU path should use *_JAX or fallback local CPU safely.",
                )
            else:
                self.assertIn("_jax", gpu_module, f"{name}: GPU path should prefer *_JAX module")

    def test_operator_backend_selection_cpu_vs_gpu(self) -> None:
        cpu_target = resolve_available_operator_target(
            "pymoo.operators.crossover.sbx",
            "SBX",
            prefer_jax=False,
        )
        gpu_target = resolve_available_operator_target(
            "pymoo.operators.crossover.sbx",
            "SBX",
            prefer_jax=True,
        )
        self.assertIsNotNone(cpu_target)
        self.assertIsNotNone(gpu_target)
        assert cpu_target is not None
        assert gpu_target is not None
        self.assertNotIn("_jax", cpu_target[0].lower())
        self.assertNotIn("_jax", cpu_target[1].lower())
        self.assertTrue(
            "_jax" in gpu_target[0].lower() or "_jax" in gpu_target[1].lower(),
            "GPU operator resolution should prefer *_JAX target when available.",
        )

    def test_metric_mapping_cpu_vs_gpu(self) -> None:
        cpu_spec = _FakeMetricSpec(
            id="local::metrics.delta_p_metric.Delta P",
            name="Delta P",
            module="metrics.delta_p_metric",
        )
        jax_spec = _FakeMetricSpec(
            id="local::metrics.delta_p_metric_JAX.Delta P JAX",
            name="Delta P JAX",
            module="metrics.delta_p_metric_JAX",
        )
        specs = {cpu_spec.id: cpu_spec, jax_spec.id: jax_spec}

        gpu_selected = map_selected_ids_to_backend(
            [cpu_spec.id],
            specs_by_id=specs,
            prefer_jax=True,
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )
        self.assertEqual(gpu_selected, [jax_spec.id])

        cpu_selected = map_selected_ids_to_backend(
            [jax_spec.id],
            specs_by_id=specs,
            prefer_jax=False,
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )
        self.assertEqual(cpu_selected, [cpu_spec.id])


class NumericParityAcceptanceTests(unittest.TestCase):
    def _sample_with_bounds(self, problem: object, n_samples: int = 8) -> np.ndarray:
        n_var = int(getattr(problem, "n_var", 2))
        rng = np.random.default_rng(1234)
        xl_raw = getattr(problem, "xl", None)
        xu_raw = getattr(problem, "xu", None)
        if xl_raw is None or xu_raw is None:
            return rng.random((n_samples, n_var))
        xl = np.asarray(xl_raw, dtype=float)
        xu = np.asarray(xu_raw, dtype=float)
        if xl.ndim == 0:
            xl = np.full((n_var,), float(xl))
        if xu.ndim == 0:
            xu = np.full((n_var,), float(xu))
        return xl + (xu - xl) * rng.random((n_samples, n_var))

    def _evaluate_f(self, problem: object, x: np.ndarray) -> np.ndarray:
        out: dict[str, np.ndarray] = {}
        problem._evaluate(x, out)
        f = np.asarray(out.get("F"), dtype=float)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        return f

    def test_problem_parity_zdt_dtlz_wfg_bnh(self) -> None:
        cases = [
            ("zdt1", {"n_var": 30}, 1e-6),
            ("dtlz1", {"n_var": 7, "n_obj": 3}, 1e-6),
            ("wfg1", {"n_var": 24, "n_obj": 3}, 1e-6),
            ("bnh", {}, 1e-6),
        ]

        for name, kwargs, tol in cases:
            cpu_problem = get_local_problem(name, backend="cpu", **kwargs)
            gpu_problem = get_local_problem(name, backend="gpu", **kwargs)
            x = self._sample_with_bounds(cpu_problem)
            f_cpu = self._evaluate_f(cpu_problem, x)
            f_gpu = self._evaluate_f(gpu_problem, x)
            self.assertEqual(f_cpu.shape, f_gpu.shape, f"{name}: shape mismatch")
            max_abs = float(np.max(np.abs(f_cpu - f_gpu)))
            self.assertLessEqual(max_abs, tol, f"{name}: max|CPU-JAX|={max_abs} > {tol}")

    def test_delta_p_metric_parity(self) -> None:
        import importlib

        cpu_mod = importlib.import_module("metrics.delta_p_metric")
        jax_mod = importlib.import_module("metrics.delta_p_metric_JAX")

        if not bool(getattr(jax_mod, "_HAS_JAX", False)):
            self.skipTest("JAX runtime is not available for delta_p_metric_JAX parity test.")

        rng = np.random.default_rng(2026)
        approx = rng.random((64, 3))
        ref = rng.random((256, 3))

        cpu_value = float(cpu_mod.calc_delta_p(approx, ref, p=1))
        jax_value = float(jax_mod.calc_delta_p_JAX(approx, ref, p=1))
        self.assertAlmostEqual(cpu_value, jax_value, delta=1e-6)


class FallbackAcceptanceTests(unittest.TestCase):
    def test_operator_fallback_when_jax_module_is_missing(self) -> None:
        target = resolve_available_operator_target("math", "sqrt", prefer_jax=True)
        self.assertEqual(target, ("math", "sqrt"))

    def test_metric_mapping_fallback_when_only_cpu_variant_exists(self) -> None:
        cpu_spec = _FakeMetricSpec(
            id="local::metrics.only_cpu.OnlyCPU",
            name="OnlyCPU",
            module="metrics.only_cpu",
        )
        specs = {cpu_spec.id: cpu_spec}
        selected = map_selected_ids_to_backend(
            [cpu_spec.id],
            specs_by_id=specs,
            prefer_jax=True,
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )
        self.assertEqual(selected, [cpu_spec.id])

    def test_rollout_feature_flag_and_stage(self) -> None:
        with _EnvContext(
            {
                ENV_BACKEND_AWARE_LOADING: "1",
                ENV_BACKEND_AWARE_STAGE: "problems",
            }
        ):
            self.assertEqual(rollout_stage(), "problems")
            self.assertTrue(rollout_allows_domain("problems"))
            self.assertFalse(rollout_allows_domain("operators"))
            self.assertFalse(rollout_allows_domain("metrics"))

        with _EnvContext({ENV_BACKEND_AWARE_LOADING: "0", ENV_BACKEND_AWARE_STAGE: "all"}):
            self.assertEqual(rollout_stage(), "off")
            self.assertFalse(rollout_allows_domain("problems"))
            self.assertFalse(rollout_allows_domain("operators"))
            self.assertFalse(rollout_allows_domain("metrics"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
