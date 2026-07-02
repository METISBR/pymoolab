from __future__ import annotations

import re
from typing import Any

import numpy as np


def select_compromise_solution(
    front: np.ndarray,
    method: str,
    weights_text: str = "",
) -> dict[str, Any]:
    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Pareto front is empty or invalid.")

    n_obj = int(values.shape[1])
    raw_weights = str(weights_text or "").strip()
    if raw_weights:
        chunks = [c for c in re.split(r"[,;\s]+", raw_weights) if c]
        weights = np.asarray([float(c) for c in chunks], dtype=float)
        if weights.size != n_obj:
            raise ValueError(f"Provide exactly {n_obj} weights.")
        if np.sum(weights) <= 0:
            raise ValueError("Weights must sum to a positive value.")
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(n_obj, dtype=float) / float(n_obj)

    f_min = np.min(values, axis=0)
    f_max = np.max(values, axis=0)
    denom = np.where((f_max - f_min) > 0, (f_max - f_min), 1.0)
    norm = (values - f_min) / denom

    method_name = str(method or "topsis").strip().lower()
    if method_name == "topsis":
        weighted = norm * weights
        ideal_best = np.min(weighted, axis=0)
        ideal_worst = np.max(weighted, axis=0)
        d_best = np.linalg.norm(weighted - ideal_best, axis=1)
        d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
        score = d_worst / np.maximum(d_best + d_worst, 1e-12)
        best_idx = int(np.argmax(score))
        best_score = float(score[best_idx])
    else:
        score = np.sum(norm * weights, axis=1)
        best_idx = int(np.argmin(score))
        best_score = float(score[best_idx])

    return {
        "index": best_idx,
        "score": best_score,
        "selected": values[best_idx, :],
        "weights": weights,
        "method": method_name,
    }
