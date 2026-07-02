from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

import numpy as np

EXPERIMENT_MANIFEST_VERSION = 1
SEED_MODE_RANDOM = "random"
SEED_MODE_FIXED = "fixed"
SEED_MODE_SEQUENCE = "sequence"


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _random_seed() -> int:
    return int(np.random.default_rng().integers(1, 2_147_483_647))


def _normalize_seed_value(value: Any, default: int = 1) -> int:
    seed = _positive_int(value, default, minimum=1)
    return int(max(1, min(seed, 2_147_483_647)))


def _canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_config_for_manifest(config: dict[str, Any]) -> dict[str, Any]:
    safe = dict(config)
    safe.pop("__single_problem_mode__", None)
    safe.pop("__suppress_progress__", None)
    return safe


def build_seed_plan(config: dict[str, Any], total_slots: int) -> dict[str, Any]:
    slots = max(1, int(total_slots))
    mode_raw = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
    if mode_raw not in {SEED_MODE_RANDOM, SEED_MODE_FIXED, SEED_MODE_SEQUENCE}:
        mode_raw = SEED_MODE_RANDOM

    base_seed = _normalize_seed_value(config.get("seed_base", config.get("seed", 1)), default=1)
    step = _positive_int(config.get("seed_step", 1), 1, minimum=1)
    sequence_raw = config.get("seed_sequence", [])

    sequence: list[int] = []
    if isinstance(sequence_raw, str):
        chunks = re.split(r"[,;\s]+", sequence_raw.strip())
        sequence = [_normalize_seed_value(chunk, default=1) for chunk in chunks if chunk.strip()]
    elif isinstance(sequence_raw, (list, tuple)):
        for item in sequence_raw:
            try:
                sequence.append(_normalize_seed_value(item, default=1))
            except Exception:  # noqa: BLE001
                continue

    if mode_raw == SEED_MODE_FIXED:
        seeds = [base_seed for _ in range(slots)]
    elif mode_raw == SEED_MODE_SEQUENCE:
        if sequence:
            seeds = [sequence[idx % len(sequence)] for idx in range(slots)]
        else:
            seeds = [_normalize_seed_value(base_seed + idx * step, default=1) for idx in range(slots)]
    else:
        seeds = [_random_seed() for _ in range(slots)]

    return {
        "mode": mode_raw,
        "deterministic": mode_raw in {SEED_MODE_FIXED, SEED_MODE_SEQUENCE},
        "base_seed": base_seed,
        "step": int(step),
        "provided_sequence": sequence,
        "seeds": seeds,
    }


def build_execution_manifest(
    *,
    config: dict[str, Any],
    seed_plan: dict[str, Any],
    selected_problem_ids: list[str],
    selected_algorithm_ids: list[str],
    selected_metric_ids: list[str],
    execution_backend: str,
    execution_backend_label: str,
) -> dict[str, Any]:
    if execution_backend == "mlx":
        try:
            from core.execution.backend_runtime import detect_mlx_runtime
            info = detect_mlx_runtime()
            if info.get("mlx_ok"):
                # detect_mlx_runtime() exposes the device under the "device" key
                # (e.g. "Device(gpu, 0)"); fall back to a generic label only when
                # the device cannot be queried.
                device_name = info.get("device") or "Apple Silicon"
                if device_name not in execution_backend_label:
                    execution_backend_label = f"{execution_backend_label} ({device_name})"
        except Exception:
            pass

    manifest = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "timestamp_iso": datetime.now().astimezone().isoformat(),
        "config": _sanitize_config_for_manifest(config),
        "selection": {
            "problem_ids": list(selected_problem_ids),
            "algorithm_ids": list(selected_algorithm_ids),
            "metric_ids": list(selected_metric_ids),
        },
        "seed_plan": {
            "mode": seed_plan.get("mode", SEED_MODE_RANDOM),
            "deterministic": bool(seed_plan.get("deterministic", False)),
            "base_seed": int(seed_plan.get("base_seed", 1)),
            "step": int(seed_plan.get("step", 1)),
            "provided_sequence": list(seed_plan.get("provided_sequence", [])),
            "seeds": list(seed_plan.get("seeds", [])),
        },
        "execution_backend": execution_backend,
        "execution_backend_label": execution_backend_label,
    }
    manifest_json = _canonical_json_dumps(manifest)
    manifest["manifest_sha256"] = _sha256_text(manifest_json)
    return manifest
