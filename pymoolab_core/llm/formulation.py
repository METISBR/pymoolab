from __future__ import annotations

import ast
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast
try:
    from anthropic import Anthropic
except Exception as _anthropic_import_exc:  # noqa: BLE001
    Anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR = _anthropic_import_exc
else:
    _ANTHROPIC_IMPORT_ERROR = None


class LLMFormulationService:
    """LLM-assisted formulation helper for PymooLab artifact generation and validation."""

    TEMPLATE_PROVIDER = "template"
    ANTHROPIC_PROVIDER = "anthropic_messages"
    DEFAULT_PROVIDER = ANTHROPIC_PROVIDER

    DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
    ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_API_KEYS_URL = "https://console.anthropic.com/settings/keys"
    DEFAULT_GROUNDING_MODEL = DEFAULT_ANTHROPIC_MODEL
    API_KEY_FILENAME = ".pymoolab_anthropic_api_key"
    METRIC_MODE_CANONICAL_WRAPPER = "canonical_wrapper"
    METRIC_MODE_GITHUB_CONVERTED = "github_converted"

    _BLOCKED_AST_NODES = (
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Lambda,
    )
    _ALLOWED_IMPORT_ROOTS = {"numpy", "pymoo", "jax", "typing", "math", "metrics", "problems"}
    _PROMPT_SIGNATURE_MARKER = "pymoolab prompt engineering"
    _KEY_VALIDATION_CACHE: dict[str, dict[str, Any]] = {}
    _API_CALL_MIN_INTERVAL_S = 4.0
    _API_CALL_GATE_LOCK = threading.Lock()
    _API_LAST_CALL_STARTED_AT_MONO = 0.0
    _BLOCKED_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
    _BLOCKED_ATTR_CALLS = {
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_output"),
        ("requests", "get"),
        ("requests", "post"),
    }

    @staticmethod
    def _slugify_problem_name(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
        if not cleaned:
            cleaned = "GeneratedProblem"
        if cleaned[0].isdigit():
            cleaned = f"P_{cleaned}"
        return cleaned

    @classmethod
    def _problem_symbol_upper(cls, name: str, *, default: str = "GENERATEDPROBLEM") -> str:
        raw = cls._slugify_problem_name(name or default)
        return raw.upper()

    @staticmethod
    def _slugify_module_name(name: str, *, default: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "")).strip("_")
        if not cleaned:
            cleaned = default
        if cleaned[0].isdigit():
            cleaned = f"m_{cleaned}"
        return cleaned.lower()

    @staticmethod
    def _camelize(name: str, *, default: str) -> str:
        raw = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "")).strip("_")
        if not raw:
            raw = default
        parts = [p for p in raw.split("_") if p]
        if not parts:
            parts = [default]
        out = "".join(p[:1].upper() + p[1:] for p in parts)
        if out and out[0].isdigit():
            out = f"P{out}"
        return out

    @staticmethod
    def _coerce_positive_int_or_default(value: Any, *, default: int) -> int:
        try:
            if value is None:
                raise ValueError("none")
            parsed = int(value)
            if parsed <= 0:
                raise ValueError("non-positive")
            return parsed
        except Exception:
            return int(max(1, int(default)))

    @classmethod
    def _normalize_problem_ui_defaults(cls, n_var: Any, n_obj: Any) -> tuple[int, int]:
        """Use caller-provided defaults when present; fall back to 30/2 only for missing/invalid values."""
        return (
            cls._coerce_positive_int_or_default(n_var, default=30),
            cls._coerce_positive_int_or_default(n_obj, default=2),
        )

    @classmethod
    def _normalize_metric_generation_mode(cls, mode: Any) -> str:
        _ = mode
        # Metric generation is fixed to GitHub-converted mode.
        return cls.METRIC_MODE_GITHUB_CONVERTED

    @staticmethod
    def _coerce_optional_positive_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            parsed = int(value)
            if parsed <= 0:
                return None
            return parsed
        except Exception:
            return None

    @staticmethod
    def _prompt_has_explicit_n_var(prompt: str) -> bool:
        text = str(prompt or "")
        if not text:
            return False
        patterns = (
            r"\bn_var\s*=\s*\d+\b",
            r"\b\d+\s+decision\s+variables?\b",
            r"\buse\s+\d+\s+decision\s+variables?\b",
            r"\bwith\s+\d+\s+variables?\b",
            r"\bdimension(?:s)?\s*[:=]?\s*\d+\b",
        )
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _prompt_has_explicit_n_obj_exact(prompt: str) -> bool:
        text = str(prompt or "")
        if not text:
            return False
        patterns = (
            r"\bn_obj\s*=\s*\d+\b",
            r"\bm\s*=\s*\d+\b",
            r"\b\d+\s+objectives?\b",
            r"\bwith\s+\d+\s+objectives?\b",
        )
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    @classmethod
    def _apply_spec_suggested_problem_defaults(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        spec_report: dict[str, Any] | None,
        n_var_default: int,
        n_obj_default: int,
    ) -> tuple[int, int]:
        if str(artifact_type or "").strip().lower() != "problem":
            return int(n_var_default), int(n_obj_default)
        spec = spec_report if isinstance(spec_report, dict) else {}
        suggested_n_var = cls._coerce_optional_positive_int(spec.get("suggested_n_var_default"))
        suggested_n_obj = cls._coerce_optional_positive_int(spec.get("suggested_n_obj_default"))

        out_n_var = int(n_var_default)
        out_n_obj = int(n_obj_default)

        if suggested_n_var is not None and (not cls._prompt_has_explicit_n_var(prompt)):
            out_n_var = int(suggested_n_var)
        if suggested_n_obj is not None and (not cls._prompt_has_explicit_n_obj_exact(prompt)):
            out_n_obj = int(suggested_n_obj)
        return out_n_var, out_n_obj

    @staticmethod
    def _ensure_numpy_import(code: str) -> str:
        text = str(code or "")
        if re.search(r"^\s*import\s+numpy\s+as\s+np\s*$", text, flags=re.MULTILINE):
            return text
        return "import numpy as np\n" + text

    @staticmethod
    def _force_problem_class_name(code: str, expected_class_name: str) -> str:
        text = str(code or "")
        expected = str(expected_class_name or "").strip()
        if not text or not expected:
            return text
        return re.sub(
            r"(?m)^(\s*class\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*\()",
            lambda m: f"{m.group(1)}{expected}{m.group(3)}",
            text,
            count=1,
        )

    @staticmethod
    def _normalize_problem_init_keywords(code: str) -> str:
        text = str(code or "")
        # Older/generated snippets may use n_constr; pymoo current API expects n_ieq_constr.
        text = re.sub(r"(?<![A-Za-z0-9_])n_constr\s*=", "n_ieq_constr=", text)
        return text

    @classmethod
    def _prompt_engineering_signature_comment(cls) -> str:
        year = int(datetime.now().year)
        return f"# Made by PymooLab {year}."

    @classmethod
    def _apply_prompt_signature(cls, code: str) -> str:
        text = str(code or "").replace("\r\n", "\n").strip()
        if not text:
            return text
        if cls._PROMPT_SIGNATURE_MARKER in text.lower():
            return text
        sig = cls._prompt_engineering_signature_comment()
        # Keep any existing reference/comment block from conversions; prepend only one signature line.
        return f"{sig}\n{text}"

    @classmethod
    def _inject_method_before_evaluate(cls, code: str, method_src: str) -> str:
        text = str(code or "")
        if "_calc_pareto_front" in text:
            return text
        marker = re.search(r"(?m)^(\s*)def\s+_evaluate\s*\(", text)
        if not marker:
            return text
        indent = marker.group(1)
        method = str(method_src).rstrip() + "\n\n"
        # Ensure method indentation matches class body indentation.
        if not method.startswith(indent):
            method = "\n".join((indent + line if line.strip() else line) for line in method.splitlines()) + "\n\n"
        pos = marker.start()
        return text[:pos] + method + text[pos:]
    @staticmethod
    def _generic_pf_method_source() -> str:
        return (
            "def _calc_pareto_front(self, n_pareto_points=200):\n"
            "    # Approximate PF fallback inserted by PymooLab for metric compatibility.\n"
            "    # Replace with a true/reference Pareto front whenever available.\n"
            "    n_pf = int(max(20, n_pareto_points))\n"
            "    n_samples = int(max(2000, n_pf * 40))\n"
            "    rng = np.random.default_rng(1)\n"
            "    xl = np.asarray(getattr(self, 'xl', 0.0), dtype=float)\n"
            "    xu = np.asarray(getattr(self, 'xu', 1.0), dtype=float)\n"
            "    if xl.ndim == 0:\n"
            "        xl = np.full(int(self.n_var), float(xl))\n"
            "    if xu.ndim == 0:\n"
            "        xu = np.full(int(self.n_var), float(xu))\n"
            "    X = rng.uniform(xl, xu, size=(n_samples, int(self.n_var)))\n"
            "    out = {}\n"
            "    self._evaluate(X, out)\n"
            "    F = np.asarray(out.get('F', []), dtype=float)\n"
            "    if F.ndim != 2 or F.shape[0] == 0:\n"
            "        return np.empty((0, int(getattr(self, 'n_obj', 0))), dtype=float)\n"
            "    if 'G' in out and out['G'] is not None:\n"
            "        G = np.asarray(out['G'], dtype=float)\n"
            "        if G.ndim == 1:\n"
            "            G = G[:, None]\n"
            "        if G.shape[0] == F.shape[0]:\n"
            "            feasible = np.all(G <= 0.0, axis=1)\n"
            "            if np.any(feasible):\n"
            "                F = F[feasible]\n"
            "    if F.shape[0] == 0:\n"
            "        return np.empty((0, int(getattr(self, 'n_obj', 0))), dtype=float)\n"
            "    keep = np.ones(F.shape[0], dtype=bool)\n"
            "    for i in range(F.shape[0]):\n"
            "        if not keep[i]:\n"
            "            continue\n"
            "        fi = F[i]\n"
            "        for j in range(F.shape[0]):\n"
            "            if i == j or not keep[j]:\n"
            "                continue\n"
            "            fj = F[j]\n"
            "            if np.all(fj <= fi) and np.any(fj < fi):\n"
            "                keep[i] = False\n"
            "                break\n"
            "    PF = F[keep]\n"
            "    if PF.shape[0] <= n_pf:\n"
            "        return PF\n"
            "    order = np.argsort(PF[:, 0], kind='mergesort')\n"
            "    PF = PF[order]\n"
            "    idx = np.linspace(0, PF.shape[0] - 1, n_pf).astype(int)\n"
            "    return PF[idx]"
        )

    @staticmethod
    def _looks_like_native_pymoo_problem_wrapper(code: str) -> bool:
        text = str(code or "")
        tl = text.lower()
        if not tl.strip():
            return False
        # Detect thin wrappers around local canonical benchmark modules or pymoo benchmark modules.
        wrapper_import_hints = (
            "from problems.multi.zdt import",
            "from problems.many.dtlz import",
            "from problems.many.zcat import",
            "from pymoo.problems",
        )
        if any(hint in tl for hint in wrapper_import_hints):
            if "_canonicalproblem" in tl or re.search(r"class\s+\w+\s*\(\s*\w+\s*\)\s*:", text):
                return True
        if "_canonicalproblem" in tl:
            return True
        return False

    @staticmethod
    def _looks_like_placeholder_benchmark_fallback_text(code: str) -> bool:
        tl = str(code or "").lower()
        if not tl.strip():
            return False
        placeholder_markers = (
            "placeholder semantics",
            "could not be reliably recovered",
            "keeps the existing placeholder semantics",
            "fallback semantics",
        )
        if not any(tok in tl for tok in placeholder_markers):
            return False
        # Keep the rule narrow to benchmark/problem-generation scenarios.
        benchmark_markers = ("zcat", "zdt", "dtlz", "wfg", "cec", "uf", "cf", "maf", "lz")
        return any(tok in tl for tok in benchmark_markers)

    @staticmethod
    def _emit_stream_event(
        cb: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any] | None,
    ) -> None:
        if cb is None or not isinstance(payload, dict):
            return
        try:
            cb(payload)
        except Exception:
            pass

    @classmethod
    def _augment_problem_code_for_metrics(
        cls,
        code: str,
        *,
        base_name: str,
        class_name: str,
        n_obj: int | None = None,
    ) -> str:
        text = cls._force_problem_class_name(code, class_name)
        text = cls._normalize_problem_init_keywords(text)
        # Generic fallback PF approximation for PF-based metrics (DeltaP, GD/IGD variants, HV, etc.)
        try:
            nobj_val = int(n_obj) if n_obj is not None else None
        except Exception:
            nobj_val = None
        if (nobj_val is None or nobj_val >= 2) and "_calc_pareto_front" not in text:
            text = cls._ensure_numpy_import(text)
            text = cls._inject_method_before_evaluate(text, cls._generic_pf_method_source())
        return text

    @classmethod
    def _build_problem_template(
        cls,
        class_name: str,
        n_var: int,
        n_obj: int,
        has_constraints: bool,
    ) -> str:
        n_ieq = 1 if has_constraints else 0
        objective_lines: list[str] = []
        for idx in range(max(1, int(n_obj))):
            shift = 0.15 + 0.7 * (idx / max(1, int(n_obj) - 1)) if int(n_obj) > 1 else 0.5
            objective_lines.append(
                f"        f{idx+1} = np.sum((X - {shift:.4f}) ** 2, axis=1) + {0.05*idx:.4f} * np.sum(X, axis=1)"
            )
        stack_expr = ", ".join(f"f{i+1}" for i in range(max(1, int(n_obj))))
        constr = (
            "        g1 = np.sum(X, axis=1) - (0.5 * self.n_var)\n"
            "        out['G'] = g1[:, None]\n"
            if has_constraints
            else ""
        )
        return (
            "import numpy as np\n"
            "from pymoo.core.problem import Problem\n\n\n"
            f"class {class_name}(Problem):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__(\n"
            f"            n_var={int(n_var)},\n"
            f"            n_obj={int(n_obj)},\n"
            f"            n_ieq_constr={int(n_ieq)},\n"
            "            xl=0.0,\n"
            "            xu=1.0,\n"
            "        )\n\n"
            "    def _evaluate(self, X, out, *args, **kwargs):\n"
            "        X = np.asarray(X, dtype=float)\n"
            + "\n".join(objective_lines)
            + "\n"
            f"        out['F'] = np.column_stack([{stack_expr}])\n"
            + constr
        )

    @classmethod
    def _build_problem_jax_template(
        cls,
        class_name: str,
        n_var: int,
        n_obj: int,
        has_constraints: bool,
    ) -> str:
        n_ieq = 1 if has_constraints else 0
        objective_lines: list[str] = []
        for idx in range(max(1, int(n_obj))):
            shift = 0.15 + 0.7 * (idx / max(1, int(n_obj) - 1)) if int(n_obj) > 1 else 0.5
            objective_lines.append(
                f"        f{idx+1} = jnp.sum((Xj - {shift:.4f}) ** 2, axis=1) + {0.05*idx:.4f} * jnp.sum(Xj, axis=1)"
            )
        stack_expr = ", ".join(f"f{i+1}" for i in range(max(1, int(n_obj))))
        constr = (
            "        g1 = jnp.sum(Xj, axis=1) - (0.5 * self.n_var)\n"
            "        out['G'] = np.asarray(g1[:, None], dtype=float)\n"
            if has_constraints
            else ""
        )
        return (
            "import numpy as np\n"
            "from pymoo.core.problem import Problem\n\n"
            "try:\n"
            "    import jax.numpy as jnp\n"
            "except Exception:  # noqa: BLE001\n"
            "    import numpy as jnp  # type: ignore\n\n\n"
            f"class {class_name}(Problem):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__(\n"
            f"            n_var={int(n_var)},\n"
            f"            n_obj={int(n_obj)},\n"
            f"            n_ieq_constr={int(n_ieq)},\n"
            "            xl=0.0,\n"
            "            xu=1.0,\n"
            "        )\n\n"
            "    def _evaluate(self, X, out, *args, **kwargs):\n"
            "        Xj = jnp.asarray(X, dtype=jnp.float32)\n"
            + "\n".join(objective_lines)
            + "\n"
            f"        out['F'] = np.asarray(jnp.column_stack([{stack_expr}]), dtype=float)\n"
            + constr
        )

    @staticmethod
    def _build_metric_template(module_name: str) -> str:
        metric_name = module_name
        return (
            "import numpy as np\n\n\n"
            "def create_metric(context):\n"
            f"    \"\"\"Auto-generated metric '{metric_name}'.\"\"\"\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            "        if F.size == 0:\n"
            "            return float('nan')\n"
            "        # Placeholder metric: average value of the first objective.\n"
            "        return float(np.mean(F[:, 0]))\n"
            "    return metric\n"
        )

    @staticmethod
    def _build_metric_jax_template(module_name: str) -> str:
        metric_name = module_name + "_JAX"
        return (
            "import numpy as np\n\n"
            "try:\n"
            "    import jax.numpy as jnp\n"
            "    _HAS_JAX = True\n"
            "except Exception:  # noqa: BLE001\n"
            "    import numpy as jnp  # type: ignore\n"
            "    _HAS_JAX = False\n\n\n"
            "def create_metric(context):\n"
            f"    \"\"\"Auto-generated JAX metric '{metric_name}'.\"\"\"\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            "        if F.size == 0:\n"
            "            return float('nan')\n"
            "        if not _HAS_JAX:\n"
            "            return float(np.mean(F[:, 0]))\n"
            "        Fj = jnp.asarray(F, dtype=jnp.float32)\n"
            "        return float(jnp.mean(Fj[:, 0]))\n"
            "    return metric\n"
        )

    @staticmethod
    def _augment_metric_code_for_framework(code: str) -> str:
        text = str(code or "")
        if "def create_metric(context)" not in text:
            return text

        helper_name = "_pymoolab_unwrap_metric_context"
        if helper_name not in text:
            helper = (
                "\n\n"
                f"def {helper_name}(context):\n"
                "    if not isinstance(context, dict):\n"
                "        return {}\n"
                "    cfg = context.get('config')\n"
                "    if isinstance(cfg, dict):\n"
                "        merged = dict(cfg)\n"
                "        for key, value in context.items():\n"
                "            if key == 'config':\n"
                "                continue\n"
                "            merged.setdefault(key, value)\n"
                "        return merged\n"
                "    return context\n"
            )
            # Insert helper after imports when possible.
            marker = re.search(r"(?m)^(from\\s+.+|import\\s+.+)$", text)
            if marker:
                last_import_end = 0
                for m in re.finditer(r"(?m)^(from\\s+.+|import\\s+.+)$", text):
                    last_import_end = m.end()
                text = text[:last_import_end] + helper + text[last_import_end:]
            else:
                text = helper.lstrip("\n") + "\n\n" + text

        if helper_name in text and f"context = {helper_name}(context)" not in text:
            text = re.sub(
                r"def\s+create_metric\(context\):\s*\n",
                "def create_metric(context):\n    context = _pymoolab_unwrap_metric_context(context)\n",
                text,
                count=1,
            )
        return text

    @staticmethod
    def _looks_like_hv_montecarlo_request(prompt: str, base_name: str) -> bool:
        text = f"{str(prompt or '')} {str(base_name or '')}".lower()
        hv_hint = ("hypervolume" in text) or re.search(r"\bhv\b", text) is not None
        mc_hint = ("monte" in text) or ("montecarlo" in text) or ("monte-carlo" in text)
        return bool(hv_hint and mc_hint)

    @staticmethod
    def _looks_like_novel_metric_variant_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        return any(
            tok in text
            for tok in (
                "novel",
                "new metric",
                "custom metric",
                "variant",
                "inspired",
                "adapted",
                "like",
                "similar to",
                "p-norm",
                "pnorm",
                "p norm",
                "minkowski",
                "lp-norm",
                "l_p",
                "norm order",
                "order p",
                "parameter p",
                "weighted distance",
                "generalized distance",
            )
        )

    @staticmethod
    def _looks_like_parameterized_known_metric_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        # Parameterization cues for known indicators that should not collapse into canonical proxies.
        return any(
            tok in text
            for tok in (
                "p-norm",
                "pnorm",
                "p norm",
                "minkowski",
                "lp norm",
                "lp-norm",
                "l_p",
                "norm order",
                "order p",
                "parameter p",
                "weighted distance",
                "generalized distance",
                "distance exponent",
            )
        )

    @staticmethod
    def _looks_like_novel_problem_variant_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        return any(
            tok in text
            for tok in (
                "novel",
                "new problem",
                "custom problem",
                "variant",
                "inspired",
                "adapted",
                "like",
                "similar to",
            )
        )

    @classmethod
    def analyze_request_intent(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str = "",
    ) -> dict[str, Any]:
        """Classify whether the request fits plugin generation or asks for a benchmark survey/research task."""
        artifact = str(artifact_type or "problem").strip().lower()
        prompt_text = str(prompt or "")
        base_text = str(base_name or "")
        text = f"{base_text} {prompt_text}".lower()

        external_source_terms = (
            "gecco",
            "cec",
            "congress",
            "conference",
            "journal",
            "proceedings",
            "paper",
            "papers",
            "article",
            "literature",
            "survey",
            "review",
            "doi",
        )
        listing_terms = (
            "provide all",
            "list all",
            "all benchmark",
            "which benchmarks",
            "used in",
            "used by",
            "adopted in",
            "including the correct dimensions",
            "including dimensions",
        )
        benchmark_terms = (
            "benchmark",
            "benchmarks",
            "benchmark function",
            "benchmark functions",
            "test problem",
            "test suite",
            "suite",
            "multi-objective algorithms",
        )
        dimension_terms = (
            "dimension",
            "dimensions",
            "decision variables",
            "objectives",
            "n_var",
            "n_obj",
        )
        codegen_terms = (
            "generate",
            "implement",
            "write code",
            "python module",
            "plugin",
            "pymoo problem",
            "problem subclass",
            "create_metric",
            "cpu and jax",
            "jax variant",
        )

        external_hits = [tok for tok in external_source_terms if tok in text]
        listing_hits = [tok for tok in listing_terms if tok in text]
        benchmark_hits = [tok for tok in benchmark_terms if tok in text]
        dimension_hits = [tok for tok in dimension_terms if tok in text]
        codegen_hits = [tok for tok in codegen_terms if tok in text]
        known_family_tokens = sorted({m.group(1).upper() for m in re.finditer(r"\b(ZDT[1-6]|DTLZ[1-7]|WFG[1-9])\b", text, flags=re.IGNORECASE)})

        strong_survey_markers = bool(external_hits and (listing_hits or ("used in" in text)))
        benchmark_catalog_intent = bool(benchmark_hits and (listing_hits or dimension_hits))
        needs_external_sources = bool(external_hits or ("gecco" in text) or ("proceedings" in text))
        benchmark_survey = bool(
            artifact == "problem"
            and (strong_survey_markers or benchmark_catalog_intent)
            and (
                needs_external_sources
                or "used in" in text
                or "congress" in text
                or "conference" in text
            )
        )

        task_kind = "benchmark_survey" if benchmark_survey else "plugin_generation"
        fit_for_llm_agent = not benchmark_survey
        unsupported_for_generation = benchmark_survey

        reasons: list[str] = []
        if benchmark_survey:
            reasons.append("Prompt asks for benchmark list/survey instead of a single plugin implementation.")
            if needs_external_sources:
                reasons.append("Prompt references conference/papers/proceedings, requiring external sources.")
            if dimension_hits:
                reasons.append("Prompt requests dimensions across benchmarks/papers, which is a structured extraction task.")
        else:
            reasons.append("Prompt is compatible with single plugin generation workflow.")
            if codegen_hits:
                reasons.append("Code-generation cues detected in request.")

        return {
            "task_kind": task_kind,
            "artifact_type": artifact,
            "fit_for_llm_agent": bool(fit_for_llm_agent),
            "unsupported_for_generation": bool(unsupported_for_generation),
            "needs_external_sources": bool(needs_external_sources),
            "detected_known_families": known_family_tokens,
            "signals": {
                "external_source_terms": external_hits,
                "listing_terms": listing_hits,
                "benchmark_terms": benchmark_hits,
                "dimension_terms": dimension_hits,
                "codegen_terms": codegen_hits,
            },
            "reasons": reasons,
        }

    @classmethod
    def _apply_request_intent_flags_to_spec_report(
        cls,
        spec_report: dict[str, Any] | None,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
    ) -> dict[str, Any] | None:
        if not isinstance(spec_report, dict):
            return spec_report
        intent = cls.analyze_request_intent(prompt=prompt, artifact_type=artifact_type, base_name=base_name)
        payload = dict(spec_report)
        payload["task_kind"] = str(intent.get("task_kind", "plugin_generation"))
        payload["needs_external_sources"] = bool(intent.get("needs_external_sources", False))
        payload["fit_for_llm_agent"] = bool(intent.get("fit_for_llm_agent", True))
        payload["unsupported_for_generation"] = bool(intent.get("unsupported_for_generation", False))
        payload["intent_reasons"] = [str(x) for x in (intent.get("reasons", []) or [])]
        if intent.get("detected_known_families"):
            payload["detected_known_families"] = list(intent.get("detected_known_families") or [])
        return payload

    @classmethod
    def _infer_known_metric_kind_from_request(cls, prompt: str, base_name: str) -> str:
        probe = {"base_name": str(base_name or ""), "_prompt": str(prompt or ""), "cpu_code": ""}
        kind = cls._metric_kind_hint(probe)
        return kind if kind in {"hv", "gd", "igd", "igdp", "deltap"} else ""

    @staticmethod
    def _infer_known_problem_kind_from_request(prompt: str, base_name: str) -> str:
        text = f"{str(base_name or '')} {str(prompt or '')}"
        m = re.search(r"\b(ZDT([1-6])|DTLZ([1-7]))\b", text, flags=re.IGNORECASE)
        if not m:
            return ""
        token = str(m.group(1) or "").strip().lower()
        return token if token else ""

    @staticmethod
    def _build_canonical_metric_proxy_code(*, metric_kind: str, jax: bool) -> str:
        metric_kind = str(metric_kind).strip().lower()
        metric_map_cpu = {
            "hv": "_metric_HV",
            "gd": "_metric_GD",
            "igd": "_metric_IGD",
            "igdp": "_metric_IGDp",
            "deltap": "_metric_DeltaP",
        }
        metric_map_jax = {
            "hv": "_metric_HV_JAX",
            "gd": "_metric_GD_JAX",
            "igd": "_metric_IGD_JAX",
            "igdp": "_metric_IGDp_JAX",
            "deltap": "_metric_DeltaP_JAX",
        }
        fn_name = metric_map_jax.get(metric_kind) if jax else metric_map_cpu.get(metric_kind)
        if not fn_name:
            raise ValueError(f"Unsupported canonical metric kind: {metric_kind}")
        module_name = "metrics.community_metrics_JAX" if jax else "metrics.community_metrics"
        return (
            "import numpy as np\n"
            f"from {module_name} import {fn_name}\n\n"
            "def create_metric(context):\n"
            "    local_context = dict(context or {}) if isinstance(context, dict) else {}\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            f"        return float({fn_name}(F, local_context))\n"
            "    return metric\n"
        )

    @staticmethod
    def _build_canonical_hv_montecarlo_metric_code(*, jax: bool) -> str:
        jax_import = (
            "try:\n"
            "    import jax.numpy as jnp  # noqa: F401\n"
            "except Exception:  # noqa: BLE001\n"
            "    jnp = None  # type: ignore[assignment]\n\n"
            if jax
            else ""
        )
        return (
            "import numpy as np\n"
            "from metrics.community_metrics import _get_front, _get_reference_front, _as_2d, _safe_divisor, _non_dominated_front\n\n"
            + jax_import
            + "def create_metric(context):\n"
            "    \"\"\"Monte Carlo Hypervolume (project-compatible, MaOP-focused, approximate for all m>=2).\"\"\"\n"
            "    local_context = dict(context or {}) if isinstance(context, dict) else {}\n"
            "    cfg = local_context.get('config') if isinstance(local_context, dict) else None\n"
            "    cfg_map = dict(cfg) if isinstance(cfg, dict) else {}\n"
            "    hv_samples = cfg_map.get('hv_samples', local_context.get('hv_samples', None))\n"
            "    hv_mc_samples = cfg_map.get('hv_mc_samples', local_context.get('hv_mc_samples', None))\n"
            "    if hv_mc_samples is None and hv_samples is not None:\n"
            "        try:\n"
            "            hv_mc_samples = int(hv_samples)\n"
            "        except Exception:  # noqa: BLE001\n"
            "            hv_mc_samples = None\n"
            "    if hv_mc_samples is None:\n"
            "        hv_mc_samples = 100000\n"
            "    if isinstance(cfg, dict):\n"
            "        cfg_map['hv_mc_samples'] = int(max(1, hv_mc_samples))\n"
            "        local_context['config'] = cfg_map\n"
            "    local_context['hv_mc_samples'] = int(max(1, hv_mc_samples))\n\n"
            "    def _hv_mc_only(pop_obj, optimum, ctx):\n"
            "        pop_obj = _as_2d(pop_obj)\n"
            "        optimum = _as_2d(optimum)\n"
            "        if pop_obj.size == 0:\n"
            "            return 0.0\n"
            "        if pop_obj.shape[1] != optimum.shape[1]:\n"
            "            return float('nan')\n"
            "        _, m = pop_obj.shape\n"
            "        if m < 2:\n"
            "            return float('nan')\n"
            "        fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(m, dtype=float))\n"
            "        fmax = np.max(optimum, axis=0)\n"
            "        den = _safe_divisor((fmax - fmin) * 1.1)\n"
            "        norm_pop = (pop_obj - fmin) / den\n"
            "        norm_pop = norm_pop[~np.any(norm_pop > 1.0, axis=1)]\n"
            "        if norm_pop.size == 0:\n"
            "            return 0.0\n"
            "        norm_pop = _non_dominated_front(norm_pop)\n"
            "        ref_point = np.ones(m, dtype=float)\n"
            "        min_value = np.min(norm_pop, axis=0)\n"
            "        max_value = ref_point\n"
            "        if np.any(max_value < min_value):\n"
            "            return 0.0\n"
            "        seed = int(ctx.get('seed', 1) or 1)\n"
            "        rng = np.random.default_rng(seed)\n"
            "        sample_num = int(max(1, ctx.get('hv_mc_samples', 100000)))\n"
            "        samples = rng.uniform(low=min_value, high=max_value, size=(sample_num, m))\n"
            "        dominated = np.zeros(sample_num, dtype=bool)\n"
            "        chunk = 2048\n"
            "        for i in range(0, sample_num, chunk):\n"
            "            s = samples[i:i+chunk]\n"
            "            dominated[i:i+len(s)] = np.any(np.all(norm_pop[None, :, :] <= s[:, None, :], axis=2), axis=1)\n"
            "        return float(np.prod(max_value - min_value) * np.mean(dominated))\n\n"
            "    def metric(front):\n"
            "        pop_obj = _get_front(front)\n"
            "        optimum = _get_reference_front(local_context)\n"
            "        if optimum is None:\n"
            "            return float('nan')\n"
            "        return float(_hv_mc_only(pop_obj, optimum, local_context))\n\n"
            "    return metric\n"
        )

    @classmethod
    def _build_canonical_problem_proxy_code(
        cls,
        *,
        problem_kind: str,
        jax: bool,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> str:
        kind = str(problem_kind or "").strip().lower()
        if not kind:
            raise ValueError("problem_kind is required")
        class_name = cls._problem_symbol_upper(base_name or "GeneratedProblem", default="GENERATEDPROBLEM")
        if jax and not class_name.endswith("_JAX"):
            class_name = f"{class_name}_JAX"

        if kind.startswith("zdt"):
            canonical = kind.upper()
            module_name = "problems.multi.zdt"
            import_name = f"{canonical}_JAX" if jax else canonical
            return (
                "import numpy as np\n"
                f"from {module_name} import {import_name} as _CanonicalProblem\n\n"
                f"class {class_name}(_CanonicalProblem):\n"
                f"    \"\"\"Canonical {canonical} proxy generated by the PymooLab Llm Agent.\"\"\"\n"
                f"    def __init__(self, n_var={int(max(1, n_var))}, **kwargs):\n"
                f"        super().__init__(n_var=n_var, **kwargs)\n\n"
                "    def _evaluate(self, X, out, *args, **kwargs):\n"
                "        # Delegates vectorized out[\"F\"] computation to the canonical local implementation.\n"
                "        X = np.asarray(X)\n"
                "        return super()._evaluate(X, out, *args, **kwargs)\n\n"
                "    def _calc_pareto_front(self, *args, **kwargs):\n"
                "        return super()._calc_pareto_front(*args, **kwargs)\n"
            )

        if kind.startswith("dtlz"):
            canonical = kind.upper()
            module_name = "problems.many.dtlz"
            import_name = f"{canonical}_JAX" if jax else canonical
            return (
                "import numpy as np\n"
                f"from {module_name} import {import_name} as _CanonicalProblem\n\n"
                f"class {class_name}(_CanonicalProblem):\n"
                f"    \"\"\"Canonical {canonical} proxy generated by the PymooLab Llm Agent.\"\"\"\n"
                f"    def __init__(self, n_var={int(max(1, n_var))}, n_obj={int(max(1, n_obj))}, **kwargs):\n"
                f"        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)\n\n"
                "    def _evaluate(self, X, out, *args, **kwargs):\n"
                "        # Delegates vectorized out[\"F\"] computation to the canonical local implementation.\n"
                "        X = np.asarray(X)\n"
                "        return super()._evaluate(X, out, *args, **kwargs)\n\n"
                "    def _calc_pareto_front(self, *args, **kwargs):\n"
                "        return super()._calc_pareto_front(*args, **kwargs)\n"
            )

        raise ValueError(f"Unsupported canonical problem kind: {problem_kind}")

    @classmethod
    def _apply_known_metric_request_overrides(cls, bundle: dict[str, Any], *, prompt: str) -> None:
        if str(bundle.get("artifact_type", "")).strip().lower() != "metric":
            return
        intent = cls.analyze_request_intent(
            prompt=str(prompt or ""),
            artifact_type="metric",
            base_name=str(bundle.get("base_name", "")),
        )
        if bool(intent.get("unsupported_for_generation", False)):
            return
        base_name = str(bundle.get("base_name", ""))
        if cls._looks_like_hv_montecarlo_request(prompt, base_name):
            bundle["cpu_code"] = cls._build_canonical_hv_montecarlo_metric_code(jax=False)
            bundle["jax_code"] = cls._build_canonical_hv_montecarlo_metric_code(jax=True)
            return

        if cls._looks_like_novel_metric_variant_request(prompt):
            return

        kind = cls._infer_known_metric_kind_from_request(prompt, base_name)
        if kind and cls._looks_like_parameterized_known_metric_request(prompt):
            return
        if kind:
            bundle["cpu_code"] = cls._build_canonical_metric_proxy_code(metric_kind=kind, jax=False)
            bundle["jax_code"] = cls._build_canonical_metric_proxy_code(metric_kind=kind, jax=True)

    @classmethod
    def _apply_known_problem_request_overrides(cls, bundle: dict[str, Any], *, prompt: str) -> None:
        if str(bundle.get("artifact_type", "")).strip().lower() != "problem":
            return
        intent = cls.analyze_request_intent(
            prompt=str(prompt or ""),
            artifact_type="problem",
            base_name=str(bundle.get("base_name", "")),
        )
        if bool(intent.get("unsupported_for_generation", False)):
            return
        if cls._looks_like_novel_problem_variant_request(prompt):
            return
        base_name = str(bundle.get("base_name", ""))
        kind = cls._infer_known_problem_kind_from_request(prompt, base_name)
        if not kind:
            return
        n_var = int(max(1, int(bundle.get("n_var", 30) or 30)))
        n_obj = int(max(1, int(bundle.get("n_obj", 2) or 2)))
        bundle["cpu_code"] = cls._build_canonical_problem_proxy_code(
            problem_kind=kind,
            jax=False,
            base_name=base_name,
            n_var=n_var,
            n_obj=n_obj,
        )
        bundle["jax_code"] = cls._build_canonical_problem_proxy_code(
            problem_kind=kind,
            jax=True,
            base_name=base_name,
            n_var=n_var,
            n_obj=n_obj,
        )

    @classmethod
    def _build_local_spec_report(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> dict[str, Any]:
        prompt_text = str(prompt or "").strip()
        text_l = prompt_text.lower()
        intent = cls.analyze_request_intent(prompt=prompt_text, artifact_type=artifact_type, base_name=base_name)
        limitations = [
            "Local template path uses heuristic defaults and cannot infer domain-specific formulas from natural language."
        ]
        if bool(intent.get("unsupported_for_generation", False)):
            limitations.append(
                "Prompt requests a benchmark survey/listing (conference/paper coverage + dimensions), which requires web/literature extraction instead of plugin code generation."
            )
        payload = {
            "mode": "local_template_spec",
            "artifact_type": str(artifact_type),
            "base_name": str(base_name),
            "summary": f"Local spec summary for {artifact_type} artifact '{base_name}' with n_var={n_var}, n_obj={n_obj}.",
            "assumptions": [
                "Use vectorized NumPy/JAX-compatible implementation.",
                "Preserve PymooLab interface contracts for problem/metric plugins.",
            ],
            "invariants": [
                "CPU and JAX variants should be numerically consistent under the same inputs.",
                "Generated code must pass AST, compile, and runtime smoke validation.",
            ],
            "limitations": limitations,
            "suggested_n_var_default": None,
            "suggested_n_obj_default": None,
            "dimension_defaults_source_note": "Local template path cannot infer benchmark defaults from web sources.",
            "novel_variant_hint": cls._looks_like_novel_metric_variant_request(prompt_text) if str(artifact_type).lower() == "metric" else False,
            "known_metric_hint": cls._infer_known_metric_kind_from_request(prompt_text, str(base_name)) if str(artifact_type).lower() == "metric" else "",
            "keywords": [tok for tok in ("constraint", "hypervolume", "igd", "gd", "delta", "dtlz", "zdt", "wfg") if tok in text_l],
        }
        if str(artifact_type).lower() == "problem":
            try:
                payload["assumptions"] = list(payload.get("assumptions", [])) + [
                    "Use pymoo.core.problem.Problem with vectorized _evaluate(self, X, out, *args, **kwargs) and out['F'].",
                    "Use pymoo constraint API n_ieq_constr/n_eq_constr with out['G']/out['H'] when constraints are present.",
                ]
                payload["invariants"] = list(payload.get("invariants", [])) + [
                    "Return plugin modules only (no __main__ demo/test harness).",
                ]
            except Exception:
                pass
        return cast(dict[str, Any], cls._apply_request_intent_flags_to_spec_report(payload, prompt=prompt_text, artifact_type=artifact_type, base_name=base_name))

    @classmethod
    def _generate_anthropic_spec_report(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        api_key: str,
        timeout_s: float,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)
        model = cls.select_anthropic_model(api_key, timeout_s=timeout_s)
        system_prompt = (
            "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks. "
            "Produce a compact specification JSON for code generation. Return strict JSON only."
        )
        problem_spec_rules = ""
        metric_spec_rules = ""
        if str(artifact_type or "").strip().lower() == "problem":
            problem_spec_rules = (
                "\nProblem-generation policy (pymoo/PymooLab): "
                "assume pymoo Problem subclasses (`from pymoo.core.problem import Problem`) with vectorized "
                "`_evaluate(self, X, out, *args, **kwargs)` and `out['F']`. "
                "If constraints exist, use `n_ieq_constr` / `n_eq_constr` and write `out['G']` / `out['H']` "
                "(do not use deprecated `n_constr`). "
                "Require explicit `xl`/`xu`, no markdown fences, no `if __name__ == '__main__':`, and no demo/test harness. "
                f"PymooLab UI defaults are n_var={n_var_default}, n_obj={n_obj_default}; treat them as constructor defaults/presets (not forced hardcoded values) unless the user explicitly fixes dimensions."
                " Always use web_search in GitHub-only mode first to find trustworthy benchmark implementations/references (official/academic repos, including MATLAB/Python reference code such as PlatEMO-style implementations)."
                " If repositories cite the original article/reference, preserve that benchmark semantics while converting to pymoo."
                " When the request does not explicitly fix n_var/n_obj (or uses parametric forms like m>=2), infer benchmark default constructor dimensions from the reference/source when possible and return them in suggested_n_var_default / suggested_n_obj_default."
                " For named benchmarks, do not suggest placeholder or substitute semantics; preserve the requested family/problem semantics."
            )
        elif str(artifact_type or "").strip().lower() == "metric":
            if metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED:
                metric_spec_rules = (
                    "\nMetric-generation policy (GitHub-converted mode): "
                    "use web_search in GitHub-only mode to recover trustworthy metric implementations/references "
                    "(official/academic Python/MATLAB code) and preserve exact metric semantics when converting to PymooLab metric plugins. "
                    "Do not propose canonical wrapper/proxy mode in this spec unless the user explicitly asks for it."
                )
            else:
                metric_spec_rules = (
                    "\nMetric-generation policy (canonical-wrapper mode): "
                    "prefer project-compatible canonical wrappers for known indicators when applicable; "
                    "preserve custom semantics only when the request explicitly asks for a variant/novel metric."
                )
        user_prompt = (
            f"Create a specification for generating a {artifact_type} plugin for PymooLab.\n"
            f"Base name: {base_name}\n"
            f"PymooLab UI defaults/presets: n_var_default={n_var_default}, n_obj_default={n_obj_default}\n"
            f"Request:\n{str(prompt or '').strip()}\n"
            f"{problem_spec_rules}{metric_spec_rules}\n\n"
            "Return JSON with keys: summary (string), assumptions (list[string]), invariants (list[string]), "
            "limitations (list[string]), known_family_hint (string), metric_kind_hint (string), ambiguity_notes (list[string]), "
            "suggested_n_var_default (integer|null), suggested_n_obj_default (integer|null), dimension_defaults_source_note (string), "
            "task_kind (string), needs_external_sources (bool), fit_for_llm_agent (bool), unsupported_for_generation (bool)."
        )
        spec_api_call_debug: dict[str, Any] = {}
        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "spec_first",
                "event": "stage_start",
                "message": "Spec-first: querying Anthropic with web_search for refined generation spec.",
            },
        )
        raw_text = cls._request_anthropic_text(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=min(float(timeout_s), 45.0),
            enable_web_search=(
                (str(artifact_type or "").strip().lower() == "problem")
                or (
                    str(artifact_type or "").strip().lower() == "metric"
                    and metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED
                )
            ),
            debug_out=spec_api_call_debug,
            stream_event_cb=stream_event_cb,
            stream_phase="spec_first",
            web_search_profile="github_only",
        )
        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "spec_first",
                "event": "stage_end",
                "message": "Spec-first completed.",
            },
        )
        payload = cls._extract_json_from_response(raw_text)
        if not isinstance(payload, dict):
            raise RuntimeError("Spec-first step returned invalid JSON.")
        payload = dict(payload)
        payload["mode"] = "anthropic_spec_first"
        payload["model"] = model
        payload["_raw_text"] = raw_text
        payload["_api_call_debug"] = spec_api_call_debug
        return cast(
            dict[str, Any],
            cls._apply_request_intent_flags_to_spec_report(
                payload,
                prompt=prompt,
                artifact_type=artifact_type,
                base_name=base_name,
            ),
        )

    @classmethod
    def generate_artifact_bundle(
        cls,
        prompt: str,
        *,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        timeout_s: float = 90.0,
        spec_first: bool = False,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        artifact = str(artifact_type or "problem").strip().lower()
        if artifact not in {"problem", "metric"}:
            raise ValueError("artifact_type must be 'problem' or 'metric'.")

        provider_token = str(provider or cls.DEFAULT_PROVIDER).strip().lower()
        prompt_text = str(prompt or "")
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        request_intent = cls.analyze_request_intent(prompt=prompt_text, artifact_type=artifact, base_name=base_name)
        spec_report: dict[str, Any] | None = None
        if bool(spec_first):
            try:
                if provider_token == cls.ANTHROPIC_PROVIDER:
                    key = cls.resolve_anthropic_api_key(api_key)
                    if key:
                        spec_report = cls._generate_anthropic_spec_report(
                            prompt=prompt_text,
                            artifact_type=artifact,
                            base_name=base_name,
                            n_var=n_var_default,
                            n_obj=n_obj_default,
                            api_key=key,
                            timeout_s=timeout_s,
                            metric_generation_mode=metric_mode,
                            stream_event_cb=stream_event_cb,
                        )
                    else:
                        spec_report = cls._build_local_spec_report(
                            prompt=prompt_text,
                            artifact_type=artifact,
                            base_name=base_name,
                            n_var=n_var_default,
                            n_obj=n_obj_default,
                        )
                else:
                    spec_report = cls._build_local_spec_report(
                        prompt=prompt_text,
                        artifact_type=artifact,
                        base_name=base_name,
                        n_var=n_var_default,
                        n_obj=n_obj_default,
                    )
            except Exception as exc:  # noqa: BLE001
                spec_report = {
                    "mode": "spec_first_failed",
                    "error": str(exc),
                }
            spec_report = cls._apply_request_intent_flags_to_spec_report(
                spec_report,
                prompt=prompt_text,
                artifact_type=artifact,
                base_name=base_name,
            )
        n_var_default, n_obj_default = cls._apply_spec_suggested_problem_defaults(
            prompt=prompt_text,
            artifact_type=artifact,
            spec_report=spec_report,
            n_var_default=n_var_default,
            n_obj_default=n_obj_default,
        )
        generation_prompt = prompt_text
        if isinstance(spec_report, dict) and spec_report and "summary" in spec_report:
            generation_prompt = (
                prompt_text
                + "\n\nGeneration spec (validated pre-step, use this as a constraint, not as prose output):\n"
                + json.dumps(spec_report, ensure_ascii=False)
            )
        if provider_token == cls.ANTHROPIC_PROVIDER:
            bundle = cls._generate_artifact_bundle_anthropic(
                prompt=generation_prompt,
                artifact_type=artifact,
                base_name=base_name,
                n_var=n_var_default,
                n_obj=n_obj_default,
                api_key=api_key,
                timeout_s=timeout_s,
                metric_generation_mode=metric_mode,
                stream_event_cb=stream_event_cb,
            )
        else:
            bundle = cls._generate_artifact_bundle_template(
                prompt=generation_prompt,
                artifact_type=artifact,
                base_name=base_name,
                n_var=n_var_default,
                n_obj=n_obj_default,
            )

        bundle.setdefault("artifact_type", artifact)
        bundle.setdefault("provider", provider_token)
        bundle.setdefault("n_var", int(n_var_default))
        bundle.setdefault("n_obj", int(n_obj_default))
        bundle["_prompt"] = prompt_text
        bundle["_request_intent"] = request_intent
        bundle["_spec_first"] = bool(spec_first)
        bundle["_metric_generation_mode"] = metric_mode
        if spec_report is not None:
            bundle["_spec_report"] = spec_report
        # Problem generation should rely on the refined internal prompt + ChatGPT response
        # instead of collapsing to local canonical proxies via request-name heuristics.
        if artifact == "metric" and metric_mode == cls.METRIC_MODE_CANONICAL_WRAPPER:
            cls._apply_known_metric_request_overrides(bundle, prompt=prompt_text)
        cls._normalize_bundle_metadata(bundle)
        report = cls.validate_artifact_bundle_detailed(bundle)
        # One-step repair loop for LLM-generated bundles (skip when canonical override already passed).
        if (not report.get("ok", False)) and provider_token == cls.ANTHROPIC_PROVIDER:
            repaired = cls._attempt_anthropic_repair(
                bundle=bundle,
                api_key=api_key,
                timeout_s=timeout_s,
                max_attempts=2,
                metric_generation_mode=metric_mode,
                stream_event_cb=stream_event_cb,
            )
            if repaired is not None:
                bundle = repaired
                bundle.setdefault("_prompt", prompt_text)
                bundle.setdefault("_spec_first", bool(spec_first))
                if spec_report is not None:
                    bundle.setdefault("_spec_report", spec_report)
                if artifact == "metric" and metric_mode == cls.METRIC_MODE_CANONICAL_WRAPPER:
                    cls._apply_known_metric_request_overrides(bundle, prompt=prompt_text)
                cls._normalize_bundle_metadata(bundle)
                cls.validate_artifact_bundle_detailed(bundle)
        return bundle

    @classmethod
    def _attempt_anthropic_repair(
        cls,
        *,
        bundle: dict[str, Any],
        api_key: str | None,
        timeout_s: float,
        max_attempts: int = 1,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        key = cls.resolve_anthropic_api_key(api_key)
        if not key:
            return None

        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        report = dict(bundle.get("_validation_report", {}) or {})
        issues = list(report.get("issues", []) or [])
        if not issues:
            return None

        cpu_code = str(bundle.get("cpu_code", "")).strip()
        jax_code = str(bundle.get("jax_code", "")).strip()
        base_name = str(bundle.get("base_name", "")).strip()
        n_var = int(max(1, int(bundle.get("n_var", 30) or 30)))
        n_obj = int(max(1, int(bundle.get("n_obj", 2) or 2)))
        prompt = str(bundle.get("_prompt", "")).strip()
        model = str(bundle.get("model", "")).strip() or cls.select_anthropic_model(key, timeout_s=timeout_s)
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)

        if artifact == "problem":
            system_prompt = (
                "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks. "
                "You are fixing invalid pymoo Problem plugin code. "
                "Keep pymoo current APIs (`out['F']`, `n_ieq_constr`/`n_eq_constr`, `out['G']`/`out['H']`) and no `if __name__ == '__main__':`. "
                "Return strict JSON with cpu_code and jax_code only."
            )
            user_contract = (
                f"Repair these two PymooLab problem modules for the original request:\n{prompt}\n\n"
                f"PymooLab runtime defaults/presets (use as constructor defaults unless request says otherwise): n_var_default={n_var}, n_obj_default={n_obj}, base_name={base_name}.\n"
                "Keep pymoo Problem compatibility, vectorized `_evaluate(self, X, out, *args, **kwargs)`, `out['F']`, explicit bounds, and CPU/JAX parity.\n"
                "Do not overwrite constructor parameters with hardcoded local assignments (e.g., `n_obj = 2` inside `__init__`) unless the user explicitly requested a fixed dimension.\n"
                "If constrained, use `n_ieq_constr` / `n_eq_constr` with `out['G']` / `out['H']` (no deprecated `n_constr`).\n"
                "Use web_search in GitHub-only mode to re-check benchmark formulas/semantics from trustworthy repositories when the request names a benchmark family/problem.\n"
                "Prioritize official/academic GitHub implementations (such as PlatEMO or reference MATLAB/Python code repos) before any other secondary source.\n"
                "If web_search/reference sources reveal canonical default dimensions and the original request did not explicitly fix them, restore those defaults in the constructor.\n"
                "When possible, restore a true/reference `_calc_pareto_front` (and `_calc_pareto_set` when meaningful) instead of leaving a generic approximate PF fallback.\n"
                "Do not keep placeholder fallback comments/text such as 'canonical definition could not be reliably recovered' or 'placeholder semantics'; recover the requested benchmark semantics via web_search and implement them directly.\n"
                "Do not repair by replacing the code with a thin wrapper/proxy around native pymoo problems or local canonical benchmark classes.\n"
                "Do not add markdown, demo/test harness, plotting, or `if __name__ == '__main__':`.\n"
                "Fix the validation failures listed below.\n\n"
            )
        else:
            system_prompt = (
                "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks. "
                "You are fixing invalid PymooLab metric plugin code. Return strict JSON with cpu_code and jax_code only."
            )
            if metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED:
                user_contract = (
                    f"Repair these two PymooLab metric modules for the original request:\n{prompt}\n\n"
                    f"Constraints: n_obj={n_obj}, base_name={base_name}.\n"
                    "Keep create_metric(context) -> metric(front)->float, CPU/JAX parity, and PymooLab context compatibility.\n"
                    "Use web_search in GitHub-only mode to recover/re-check trustworthy metric implementations (official/academic Python/MATLAB repositories) and preserve metric semantics while converting to PymooLab.\n"
                    "Do not fall back to placeholder metric semantics/comments.\n"
                    "Do not repair by replacing the code with a trivial project canonical wrapper/proxy; preserve the requested metric semantics via GitHub-sourced implementations.\n"
                    "Fix the validation failures listed below.\n\n"
                )
            else:
                user_contract = (
                    f"Repair these two PymooLab metric modules for the original request:\n{prompt}\n\n"
                    f"Constraints: n_obj={n_obj}, base_name={base_name}.\n"
                    "Keep create_metric(context) -> metric(front)->float, CPU/JAX parity, and PymooLab context compatibility.\n"
                    "If the metric is known (HV/GD/IGD/DeltaP), match PymooLab metric semantics.\n"
                    "Fix the validation failures listed below.\n\n"
                )

        attempts = max(1, int(max_attempts))
        current_cpu = cpu_code
        current_jax = jax_code
        current_issues = issues
        repair_attempt_log: list[dict[str, Any]] = []
        for attempt in range(1, attempts + 1):
            issue_block = "\n".join(f"- {msg}" for msg in current_issues[:20])
            issue_categories = sorted(
                {
                    "compile" if "Compile error" in msg else
                    "ast" if ("Unsupported AST" in msg or "Unsupported import" in msg or "Unsupported call" in msg) else
                    "shape" if "shape" in msg.lower() else
                    "parity" if "parity" in msg.lower() else
                    "oracle" if "oracle" in msg.lower() else
                    "runtime" if "runtime" in msg.lower() else
                    "validation"
                    for msg in current_issues
                }
            ) or ["validation"]
            repair_prompt = (
                user_contract
                + "Validation failures:\n"
                + issue_block
                + "\n\nCurrent CPU code:\n"
                + current_cpu
                + "\n\nCurrent JAX code:\n"
                + current_jax
                + "\n\nReturn JSON only with corrected cpu_code and jax_code."
            )
            attempt_entry: dict[str, Any] = {
                "attempt": attempt,
                "model": model,
                "issue_count": len(current_issues),
                "issue_categories": issue_categories,
            }
            repair_attempt_log.append(attempt_entry)
            try:
                repair_api_call_debug: dict[str, Any] = {}
                cls._emit_stream_event(
                    stream_event_cb,
                    {
                        "kind": "llm_stream",
                        "phase": f"repair_{attempt}",
                        "event": "stage_start",
                        "message": (
                            (
                                f"Repair attempt {attempt}: querying Anthropic with web_search (GitHub-only)."
                                if (artifact == "problem" or (artifact == "metric" and metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED))
                                else f"Repair attempt {attempt}: querying Anthropic without web_search."
                            )
                        ),
                    },
                )
                raw_text = cls._request_anthropic_text(
                    api_key=key,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    timeout_s=timeout_s,
                    enable_web_search=(
                        (artifact == "problem")
                        or (artifact == "metric" and metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED)
                    ),
                    debug_out=repair_api_call_debug,
                    stream_event_cb=stream_event_cb,
                    stream_phase=f"repair_{attempt}",
                    web_search_profile="github_only",
                )
                cls._emit_stream_event(
                    stream_event_cb,
                    {
                        "kind": "llm_stream",
                        "phase": f"repair_{attempt}",
                        "event": "stage_end",
                        "message": f"Repair attempt {attempt} completed.",
                    },
                )
                attempt_entry["api_call_debug"] = repair_api_call_debug
                payload = cls._extract_json_from_response(raw_text)
                if not isinstance(payload, dict):
                    continue
                repaired = dict(bundle)
                repaired["cpu_code"] = str(payload.get("cpu_code", "")).strip() or current_cpu
                repaired["jax_code"] = str(payload.get("jax_code", "")).strip() or current_jax
                # Quick normalize + validate in-loop.
                cls._normalize_bundle_metadata(repaired)
                rep = cls.validate_artifact_bundle_detailed(repaired)
                if rep.get("ok", False):
                    repaired["_repair_attempts"] = attempt
                    repaired["_repair_report"] = {
                        "used": True,
                        "final_status": "success",
                        "attempts": repair_attempt_log,
                    }
                    attempt_entry["result"] = "success"
                    return repaired
                current_cpu = str(repaired.get("cpu_code", current_cpu))
                current_jax = str(repaired.get("jax_code", current_jax))
                current_issues = list(rep.get("issues", []) or current_issues)
                attempt_entry["result"] = "invalid_after_repair"
                attempt_entry["post_issue_count"] = len(current_issues)
            except Exception as exc:
                attempt_entry["result"] = "request_error"
                attempt_entry["error"] = str(exc)
                continue
        bundle["_repair_report"] = {
            "used": True,
            "final_status": "failed",
            "attempts": repair_attempt_log,
            "remaining_issue_count": len(current_issues),
        }
        return None

    @classmethod
    def _normalize_bundle_metadata(cls, bundle: dict[str, Any]) -> None:
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        n_var = int(max(1, int(bundle.get("n_var", 30))))
        n_obj = int(max(1, int(bundle.get("n_obj", 2))))
        raw_base = str(bundle.get("base_name", "")).strip()

        if artifact == "problem":
            base_class = cls._problem_symbol_upper(raw_base or "GeneratedProblem", default="GENERATEDPROBLEM")
            cpu_class = cls._problem_symbol_upper(str(bundle.get("cpu_symbol", base_class)), default=base_class)
            jax_class = cls._problem_symbol_upper(str(bundle.get("jax_symbol", f"{base_class}_JAX")), default=f"{base_class}_JAX")
            if not jax_class.endswith("_JAX"):
                jax_class = f"{jax_class}_JAX"
            base_module = cls._slugify_module_name(raw_base or base_class, default="generated_problem")
            bundle["base_name"] = base_class
            bundle["cpu_symbol"] = cpu_class
            bundle["jax_symbol"] = jax_class
            bundle["cpu_file"] = f"{base_module}.py"
            bundle["jax_file"] = f"{base_module}_JAX.py"
        else:
            base_module = cls._slugify_module_name(raw_base, default="generated_metric").upper()
            bundle["base_name"] = base_module
            bundle["cpu_symbol"] = "create_metric"
            bundle["jax_symbol"] = "create_metric"
            bundle["cpu_file"] = f"{base_module}.py"
            bundle["jax_file"] = f"{base_module}_JAX.py"

        bundle["n_var"] = n_var
        bundle["n_obj"] = n_obj
        cpu_code = str(bundle.get("cpu_code", "")).replace("\r\n", "\n").strip()
        jax_code = str(bundle.get("jax_code", "")).replace("\r\n", "\n").strip()
        if artifact == "problem":
            cpu_code = cls._augment_problem_code_for_metrics(
                cpu_code,
                base_name=str(bundle.get("base_name", "")),
                class_name=str(bundle.get("cpu_symbol", "")),
                n_obj=n_obj,
            )
            jax_code = cls._augment_problem_code_for_metrics(
                jax_code,
                base_name=str(bundle.get("base_name", "")),
                class_name=str(bundle.get("jax_symbol", "")),
                n_obj=n_obj,
            )
        else:
            cpu_code = cls._augment_metric_code_for_framework(cpu_code)
            jax_code = cls._augment_metric_code_for_framework(jax_code)
        cpu_code = cls._apply_prompt_signature(cpu_code)
        jax_code = cls._apply_prompt_signature(jax_code)
        bundle["cpu_code"] = cpu_code + "\n"
        bundle["jax_code"] = jax_code + "\n"

    @classmethod
    def _generate_artifact_bundle_template(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> dict[str, Any]:
        text = str(prompt or "").lower()
        if artifact_type == "problem":
            has_constraints = any(tok in text for tok in ("constraint", "restri", "g(", "<=", ">="))
            base_class = cls._problem_symbol_upper(base_name or "GeneratedProblem", default="GENERATEDPROBLEM")
            return {
                "artifact_type": "problem",
                "base_name": base_class,
                "cpu_symbol": base_class,
                "jax_symbol": f"{base_class}_JAX",
                "cpu_code": cls._build_problem_template(base_class, max(1, int(n_var)), max(1, int(n_obj)), has_constraints),
                "jax_code": cls._build_problem_jax_template(f"{base_class}_JAX", max(1, int(n_var)), max(1, int(n_obj)), has_constraints),
            }

        base_module = cls._slugify_module_name(base_name, default="generated_metric").upper()
        return {
            "artifact_type": "metric",
            "base_name": base_module,
            "cpu_symbol": "create_metric",
            "jax_symbol": "create_metric",
            "cpu_code": cls._build_metric_template(base_module),
            "jax_code": cls._build_metric_jax_template(base_module),
        }

    @classmethod
    def _generate_artifact_bundle_anthropic(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        api_key: str | None,
        timeout_s: float,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        key = cls.resolve_anthropic_api_key(api_key)
        if not key:
            raise RuntimeError("Anthropic API key not found. Provide a key or set ANTHROPIC_API_KEY.")

        model_candidates = cls.select_anthropic_model_candidates(key, timeout_s=timeout_s)
        if not model_candidates:
            model_candidates = [cls.DEFAULT_ANTHROPIC_MODEL]
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)
        raw_base = str(base_name or "").strip()
        if artifact_type == "problem":
            safe_base = cls._problem_symbol_upper(raw_base or "GeneratedProblem", default="GENERATEDPROBLEM")
            cpu_symbol = safe_base
            jax_symbol = f"{safe_base}_JAX"
            contract = (
                f"Generate two Python modules for pymoo problems based on this requirement:\n{str(prompt or '').strip()}\n\n"
                "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks.\n"
                f"PymooLab UI runtime defaults/presets (not mandatory hard constraints unless the user explicitly fixes dimensions): "
                f"n_var_default={n_var_default}, n_obj_default={n_obj_default}.\n"
                f"CPU class name: {cpu_symbol}. JAX class name: {jax_symbol}.\n"
                "Both modules must import `Problem` from `pymoo.core.problem` and subclass it.\n"
                "Both modules must use vectorized `_evaluate(self, X, out, *args, **kwargs)` over a batch of decision vectors and set `out['F']` with shape (N, n_obj).\n"
                "Define explicit variable bounds (`xl`, `xu`) and keep constructor defaults (`n_var`, `n_obj`) consistent with the request and UI presets.\n"
                "If web_search/reference sources provide canonical benchmark default dimensions and the request does not explicitly fix them, prefer those canonical defaults over the UI presets.\n"
                "If the request is parametric (e.g., `m>=2`, variable objectives, benchmark-family wrapper), preserve constructor parameters and do not hardcode local reassignment that ignores them.\n"
                "If the problem is constrained, use pymoo's current constraint API (`n_ieq_constr`, `n_eq_constr`) and write `out['G']` / `out['H']` (do not use deprecated `n_constr`).\n"
                "JAX module may import jax.numpy as jnp with NumPy fallback.\n"
                "Do not invent hidden normalization, scaling, or objective transforms unless explicitly required by the request.\n"
                "State assumptions only in comments when the request is ambiguous; do not silently guess semantics.\n"
                "Ensure CPU and JAX versions are mathematically equivalent up to floating-point differences.\n"
                "When possible, include the true/reference Pareto front via `_calc_pareto_front` "
                "and `_calc_pareto_set` when meaningful (prefer exact analytical/reference PF over approximations).\n"
                "If the true PF cannot be derived from the article/source, only then provide an explicitly labeled approximation fallback.\n"
                "When possible, include Pareto-front helpers (_calc_pareto_front, and _calc_pareto_set when meaningful) "
                "to maximize compatibility with PF-based metrics (HV, GD/IGD variants, DeltaP).\n"
                "If converting an established benchmark family, preserve mathematical semantics and references.\n"
                "Always use the available web_search tool in GitHub-only mode first to locate trustworthy benchmark implementations/references for the requested benchmark/problem.\n"
                "Use official/academic GitHub implementations (including reference MATLAB/Python code such as PlatEMO-style implementations) and convert them to pymoo-compatible code while preserving semantics.\n"
                "Never emit placeholder fallback semantics/comments for a named benchmark (e.g., 'canonical definition could not be reliably recovered' or 'placeholder semantics'); continue searching and implement the requested benchmark family/problem semantics.\n"
                "Do not replace the request with a different benchmark family just because it is more common.\n"
                "Do NOT return a thin wrapper/proxy around native pymoo benchmark problems or local canonical benchmark modules; implement/adapt the requested problem equations/logic directly in the generated plugin.\n"
                "Return production plugin modules only: no markdown fences, no notebook code, no plotting, no CLI parser, no `if __name__ == '__main__':`, and no test harness/demo block.\n"
                "Self-check before returning: shapes, finite outputs, and CPU/JAX parity on a small random batch.\n"
                "Return JSON only with keys cpu_code and jax_code (optional: assumptions, invariants, limitations, validation_notes). "
                "No markdown fences."
            )
            system_prompt = (
                "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks. "
                "You write safe, vectorized Python code for pymoo Problem subclasses. "
                "Use pymoo's current Problem/constraint APIs (`out['F']`, `n_ieq_constr`/`n_eq_constr`, `out['G']`/`out['H']`). "
                "A web_search tool is available in GitHub-only mode: use it to find trusted benchmark implementations/references (official/academic repos, MATLAB/Python code) and preserve semantics. "
                "Do not emit thin wrappers around native pymoo problems or local canonical benchmark classes. "
                "Do not include `if __name__ == '__main__':` or demo code. "
                "Return a strict JSON object with cpu_code and jax_code as strings. "
                "No prose, no markdown, no file I/O, no subprocess, no network code."
            )
        else:
            safe_base = cls._slugify_module_name(raw_base, default="generated_metric")
            if metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED:
                contract = (
                    f"Generate two Python metric modules for PymooLab based on this requirement:\n{str(prompt or '').strip()}\n\n"
                    "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks.\n"
                    "Each module must expose create_metric(context) returning a callable metric(front)->float.\n"
                    "Use web_search in GitHub-only mode to find trustworthy metric implementations/references (official/academic Python/MATLAB repositories) and convert them to PymooLab metric plugins while preserving metric semantics.\n"
                    "Do not replace the requested metric with a different indicator just because it is more common.\n"
                    "Do not emit placeholder semantics/comments or incomplete pseudo-implementations.\n"
                    "Support PymooLab metric contexts where the actual context may be nested under context['config'].\n"
                    "Use available context keys such as pareto_front/ref_pf, pareto_set/ref_ps, ref_point, current_population_F/X when relevant.\n"
                    "CPU module uses NumPy. JAX module uses jax.numpy as jnp with NumPy fallback.\n"
                    "CPU and JAX variants should be numerically consistent under the same inputs.\n"
                    "Self-check before returning: finite values, stable shape handling, and CPU/JAX parity on a small synthetic front.\n"
                    f"Module base name: {safe_base}; JAX file will use suffix _JAX.\n"
                    "Return JSON only with keys cpu_code and jax_code (optional: assumptions, invariants, limitations, validation_notes). No markdown fences."
                )
            else:
                contract = (
                    f"Generate two Python metric modules for PymooLab based on this requirement:\n{str(prompt or '').strip()}\n\n"
                    "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks.\n"
                    "Each module must expose create_metric(context) returning a callable metric(front)->float.\n"
                    "The metric MUST be numerically compatible with PymooLab/pymoo conventions and scale whenever an equivalent "
                    "indicator exists (e.g., HV, GD, IGD, IGD+, DeltaP, spacing, spread).\n"
                    "Prefer robust implementations and trusted pymoo indicator primitives when available, instead of fragile "
                    "hand-translated recursive code.\n"
                    "Support PymooLab metric contexts where the actual context may be nested under context['config'].\n"
                    "Use available context keys such as pareto_front/ref_pf, pareto_set/ref_ps, ref_point, current_population_F/X "
                    "when relevant to the metric.\n"
                    "CPU module uses NumPy. JAX module uses jax.numpy as jnp with NumPy fallback.\n"
                    "CPU and JAX variants should produce statistically close values under the same inputs.\n"
                    "For known indicators (HV/GD/IGD/DeltaP and variants), preserve standard semantics; do not change normalization "
                    "or reference handling unless explicitly requested.\n"
                    "If the request asks for an approximation (e.g., Monte Carlo), approximate only the target computational step, "
                    "not the metric definition itself.\n"
                    "Self-check before returning: run a tiny thought-level test mentally and ensure output scale is plausible.\n"
                    f"Module base name: {safe_base}; JAX file will use suffix _JAX.\n"
                    "Return JSON only with keys cpu_code and jax_code (optional: assumptions, invariants, limitations, validation_notes). "
                    "No markdown fences."
                )
            system_prompt = (
                "Act as a senior software engineer specialized in pymoo, PlatEMO, and optimization frameworks. "
                "You write safe Python metric plugins for PymooLab. "
                + (
                    "A web_search tool is available in GitHub-only mode: use it to recover trusted metric implementations/references and preserve semantics. "
                    if metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED
                    else ""
                )
                + "Return a strict JSON object with cpu_code and jax_code as strings. "
                + "No prose, no markdown, no file I/O, no subprocess, no network code."
            )

        raw_text = ""
        selected_model = ""
        generation_api_call_debug: dict[str, Any] = {}
        last_exc: Exception | None = None
        for candidate_model in model_candidates:
            try:
                generation_api_call_debug = {}
                cls._emit_stream_event(
                    stream_event_cb,
                    {
                        "kind": "llm_stream",
                        "phase": "generation",
                        "event": "stage_start",
                        "message": (
                            f"Generation: querying Anthropic model {candidate_model} "
                            f"(web_search {'on' if (artifact_type == 'problem' or (artifact_type == 'metric' and metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED)) else 'off'})."
                        ),
                    },
                )
                raw_text = cls._request_anthropic_text(
                    api_key=key,
                    model=str(candidate_model),
                    system_prompt=system_prompt,
                    user_prompt=contract,
                    timeout_s=timeout_s,
                    enable_web_search=(
                        (artifact_type == "problem")
                        or (artifact_type == "metric" and metric_mode == cls.METRIC_MODE_GITHUB_CONVERTED)
                    ),
                    debug_out=generation_api_call_debug,
                    stream_event_cb=stream_event_cb,
                    stream_phase="generation",
                    web_search_profile="github_only",
                )
                cls._emit_stream_event(
                    stream_event_cb,
                    {
                        "kind": "llm_stream",
                        "phase": "generation",
                        "event": "stage_end",
                        "message": f"Generation call completed for model {candidate_model}.",
                    },
                )
                selected_model = str(candidate_model)
                last_exc = None
                break
            except RuntimeError as exc:
                last_exc = exc
                # Free models can be temporarily rate-limited upstream. Try the next candidate.
                if "HTTP 429" in str(exc):
                    continue
                raise
        if last_exc is not None and not raw_text:
            raise last_exc

        payload = cls._extract_json_from_response(raw_text)
        if not isinstance(payload, dict):
            raise RuntimeError("Anthropic returned an invalid JSON payload for artifact bundle generation.")
        cpu_code = str(payload.get("cpu_code", "")).strip()
        jax_code = str(payload.get("jax_code", "")).strip()
        if not cpu_code or not jax_code:
            raise RuntimeError("Anthropic response is missing cpu_code or jax_code.")
        if artifact_type == "problem":
            cpu_code = cls._augment_problem_code_for_metrics(
                cpu_code,
                base_name=safe_base,
                class_name=cpu_symbol,
                n_obj=n_obj_default,
            )
            jax_code = cls._augment_problem_code_for_metrics(
                jax_code,
                base_name=safe_base,
                class_name=jax_symbol,
                n_obj=n_obj_default,
            )
        return {
            "artifact_type": artifact_type,
            "base_name": safe_base,
            "cpu_code": cpu_code,
            "jax_code": jax_code,
            "provider": cls.ANTHROPIC_PROVIDER,
            "model": selected_model,
            "_api_raw_text": raw_text,
            "_api_call_debug": generation_api_call_debug,
        }

    @classmethod
    def _wait_for_api_call_slot(cls, *, label: str = "anthropic_api") -> None:
        """
        Enforce a minimum interval between provider API calls across threads.

        The interval is measured between the *start* times of consecutive API requests.
        """
        _ = label  # reserved for future logging/debug hooks
        min_interval = float(max(0.0, float(getattr(cls, "_API_CALL_MIN_INTERVAL_S", 0.0) or 0.0)))
        if min_interval <= 0.0:
            return
        with cls._API_CALL_GATE_LOCK:
            now = time.monotonic()
            last_started = float(getattr(cls, "_API_LAST_CALL_STARTED_AT_MONO", 0.0) or 0.0)
            wait_s = min_interval - (now - last_started)
            if wait_s > 0.0:
                time.sleep(wait_s)
                now = time.monotonic()
            cls._API_LAST_CALL_STARTED_AT_MONO = now

    @staticmethod
    def _anthropic_client(*, api_key: str, timeout_s: float) -> Any:
        if Anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Install with: pip install anthropic")
        return Anthropic(api_key=str(api_key), timeout=float(max(1.0, timeout_s)))

    @staticmethod
    def _anthropic_message_to_dict(resp_obj: Any) -> dict[str, Any]:
        if isinstance(resp_obj, dict):
            return dict(resp_obj)
        if hasattr(resp_obj, "model_dump"):
            try:
                dumped = resp_obj.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        if hasattr(resp_obj, "to_dict"):
            try:
                dumped = resp_obj.to_dict()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        if hasattr(resp_obj, "model_dump_json"):
            try:
                raw = resp_obj.model_dump_json()
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        try:
            return json.loads(json.dumps(resp_obj, default=str))
        except Exception:
            return {}

    @staticmethod
    def _anthropic_message_to_raw_text(resp_obj: Any) -> str:
        payload = LLMFormulationService._anthropic_message_to_dict(resp_obj)
        if payload:
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                pass
        return str(resp_obj)

    @staticmethod
    def _anthropic_exception_status_and_body(exc: Exception) -> tuple[int | None, str]:
        status = getattr(exc, "status_code", None)
        try:
            status_i = int(status) if status is not None else None
        except Exception:
            status_i = None
        body = ""
        raw_body = getattr(exc, "body", None)
        if raw_body is not None:
            try:
                body = raw_body if isinstance(raw_body, str) else json.dumps(raw_body, ensure_ascii=False)
            except Exception:
                body = str(raw_body)
        if not body:
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = getattr(resp, "text", "") or body
                except Exception:
                    body = body
        if not body:
            body = str(exc)
        return status_i, body

    @staticmethod
    def _anthropic_default_thinking_payload() -> dict[str, Any]:
        # Keep thinking disabled by default for latency/cost parity with the previous setup.
        return {}

    @classmethod
    def _anthropic_web_search_tool_payload(cls, *, profile: str = "github_only") -> dict[str, Any]:
        p = str(profile or "open").strip().lower()
        tool: dict[str, Any] = {
            "name": "web_search",
            "type": "web_search_20250305",
            # Anthropic requires at least one location field in addition to type.
            "user_location": {"type": "approximate", "country": "US"},
            "max_uses": 5,
        }
        if p == "github_only":
            tool["allowed_domains"] = ["github.com"]
        return tool

    @classmethod
    def _anthropic_stream_event_to_dict(cls, event_obj: Any) -> dict[str, Any]:
        data = cls._anthropic_message_to_dict(event_obj)
        if isinstance(data, dict) and data:
            return data
        return {}

    @classmethod
    def _anthropic_stream_event_type(cls, event_obj: Any) -> str:
        data = cls._anthropic_stream_event_to_dict(event_obj)
        et = data.get("type")
        if isinstance(et, str) and et.strip():
            return et.strip()
        try:
            raw = getattr(event_obj, "type", "")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        except Exception:
            pass
        return ""

    @classmethod
    def _extract_anthropic_stream_text_delta(cls, event_obj: Any) -> str:
        data = cls._anthropic_stream_event_to_dict(event_obj)
        if not isinstance(data, dict) or not data:
            return ""
        # Anthropic normalized text event emitted by MessageStream helper
        if str(data.get("type", "") or "").strip().lower() == "text":
            txt = data.get("text")
            if isinstance(txt, str) and txt:
                return txt
        for key in ("text",):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        delta = data.get("delta")
        if isinstance(delta, dict):
            if str(delta.get("type", "") or "").strip().lower() == "text_delta":
                txt = delta.get("text")
                if isinstance(txt, str) and txt:
                    return txt
        return ""

    @classmethod
    def _extract_anthropic_stream_web_search_query(cls, event_obj: Any) -> str:
        data = cls._anthropic_stream_event_to_dict(event_obj)
        if not isinstance(data, dict) or not data:
            return ""
        # Anthropic emits web_search activity via server_tool_use blocks; query lives in `input`.
        candidates: list[Any] = [data]
        for key in ("content_block", "delta", "message", "content"):
            val = data.get(key)
            if isinstance(val, (dict, list)):
                candidates.append(val)
        stack = candidates[:]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                q = cur.get("query")
                if isinstance(q, str) and q.strip():
                    return q.strip()
                for v in cur.values():
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend([v for v in cur if isinstance(v, (dict, list))])
        return ""

    @classmethod
    def _select_anthropic_web_search_model(cls, api_key: str, *, timeout_s: float = 30.0) -> str:
        """Use a fixed Claude model for web_search to avoid dynamic routing drift."""
        _ = (api_key, timeout_s)
        return cls.DEFAULT_GROUNDING_MODEL

    @classmethod
    def _request_anthropic_web_search_text(
        cls,
        *,
        api_key: str,
        model: str,
        prompt: str,
        timeout_s: float,
        web_search_profile: str = "github_only",
    ) -> dict[str, Any]:
        request_timeout = float(max(2.0, timeout_s))
        payload: dict[str, Any] = {
            "model": str(model),
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": str(prompt)}],
            "tool_choice": {"type": "auto"},
            "tools": [cls._anthropic_web_search_tool_payload(profile=web_search_profile)],
        }
        thinking = cls._anthropic_default_thinking_payload()
        if isinstance(thinking, dict) and thinking:
            payload["thinking"] = thinking
        try:
            cls._wait_for_api_call_slot(label="anthropic_messages_web_search")
            client = cls._anthropic_client(api_key=api_key, timeout_s=request_timeout)
            with client.messages.stream(**payload, timeout=request_timeout) as stream:
                stream.until_done()
                resp = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001
            status_code, body = cls._anthropic_exception_status_and_body(exc)
            if status_code is not None:
                raise RuntimeError(f"Anthropic Messages web search HTTP {status_code}: {body[:700]}")
            raise RuntimeError(f"Anthropic Messages web search request failed: {exc}") from exc

        data = cls._anthropic_message_to_dict(resp)
        raw = cls._anthropic_message_to_raw_text(resp)
        if not isinstance(data, dict) or not data:
            raise RuntimeError(f"Anthropic Messages web search returned non-JSON response: {raw[:700]}")

        payload_dict = data
        text = cls._extract_anthropic_messages_output_text(payload_dict)
        if not text:
            raise RuntimeError(f"Anthropic Messages web search response has no usable text content: {raw[:700]}")

        web_search_calls: list[dict[str, Any]] = []
        sources_flat: list[dict[str, Any]] = []
        call_index_by_id: dict[str, int] = {}
        content_items = payload_dict.get("content", []) if isinstance(payload_dict.get("content", []), list) else []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip().lower()
            if item_type in {"server_tool_use", "tool_use"} and str(item.get("name", "") or "").strip().lower() == "web_search":
                tool_input = item.get("input", {}) if isinstance(item.get("input", {}), dict) else {}
                query = str(tool_input.get("query", "") or "").strip()
                call_payload = {
                    "id": item.get("id"),
                    "status": "completed",
                    "query": query,
                    "sources": [],
                    "action": {"query": query, "sources": []},
                }
                web_search_calls.append(call_payload)
                tool_id = str(item.get("id", "") or "").strip()
                if tool_id:
                    call_index_by_id[tool_id] = len(web_search_calls) - 1
                continue
            if item_type != "web_search_tool_result":
                continue
            tool_use_id = str(item.get("tool_use_id", "") or "").strip()
            raw_content = item.get("content")
            normalized_sources: list[dict[str, Any]] = []
            if isinstance(raw_content, list):
                for src in raw_content:
                    if not isinstance(src, dict):
                        continue
                    if str(src.get("type", "") or "").strip().lower() != "web_search_result":
                        continue
                    src_norm = {
                        "type": src.get("type"),
                        "title": src.get("title"),
                        "url": src.get("url"),
                        "page_age": src.get("page_age"),
                        "encrypted_content": src.get("encrypted_content"),
                    }
                    normalized_sources.append(src_norm)
                    sources_flat.append(src_norm)
            elif isinstance(raw_content, dict):
                err_payload = dict(raw_content)
                normalized_sources.append(err_payload)
            if tool_use_id and tool_use_id in call_index_by_id:
                idx = call_index_by_id[tool_use_id]
                call = web_search_calls[idx]
                merged = list(call.get("sources", [])) + normalized_sources
                call["sources"] = merged
                action = call.get("action", {}) if isinstance(call.get("action", {}), dict) else {}
                action["sources"] = merged
                call["action"] = action
            else:
                web_search_calls.append(
                    {
                        "id": tool_use_id or item.get("tool_use_id"),
                        "status": "completed",
                        "query": "",
                        "sources": normalized_sources,
                        "action": {"query": "", "sources": normalized_sources},
                    }
                )
        return {
            "text": text,
            "grounding_metadata": {
                "provider": "anthropic_web_search",
                "web_search_calls": web_search_calls,
                "sources": sources_flat,
            },
            "raw_response": payload_dict,
            "raw_text": raw,
            "model": str(model),
        }

    @classmethod
    def run_benchmark_survey_web(
        cls,
        prompt: str,
        *,
        base_name: str = "benchmark_survey",
        provider: str = ANTHROPIC_PROVIDER,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        search_limit: int = 10,
        fetch_limit: int = 6,
    ) -> dict[str, Any]:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise ValueError("Benchmark survey prompt is empty.")

        intent = cls.analyze_request_intent(prompt=prompt_text, artifact_type="problem", base_name=base_name)
        provider_token = str(provider or cls.DEFAULT_PROVIDER).strip().lower()
        if provider_token != cls.ANTHROPIC_PROVIDER:
            raise ValueError("Benchmark Survey (Web) currently requires Anthropic provider.")

        key = cls.resolve_anthropic_api_key(api_key)
        if not key:
            raise RuntimeError("Anthropic API key not found. Provide a key or set ANTHROPIC_API_KEY.")
        # `search_limit` and `fetch_limit` are kept in the signature for compatibility with older callers.
        _ = (search_limit, fetch_limit)

        model = cls._select_anthropic_web_search_model(key, timeout_s=min(float(timeout_s), 30.0))
        grounding_prompt = (
            "Use the Claude/Anthropic web_search tool to answer the following benchmark-survey request with current web data.\n\n"
            f"Request:\n{prompt_text}\n\n"
            "Return a STRICT JSON object as plain text (do NOT rely on API JSON mode) with keys:\n"
            "- summary (string)\n"
            "- scope_assessment (string)\n"
            "- benchmark_entries (array of objects)\n"
            "- sources_used (array of objects)\n"
            "- missing_information (array[string])\n"
            "- warnings (array[string])\n"
            "- recommended_next_steps (array[string])\n\n"
            "Each benchmark_entries item must contain:\n"
            "name (string), family (string|null), source_title (string), source_url (string), "
            "n_var (string|null), n_obj (string|null), dimension_statement (string|null), "
            "evidence (string|null), confidence (string), notes (string|null).\n\n"
            "Rules:\n"
            "1) Use grounded web information only; no guesses.\n"
            "2) If dimensions are not explicit or vary, set n_var/n_obj to null and explain in notes.\n"
            "3) Cite source_url for every benchmark entry.\n"
            "4) Prefer partial but accurate output over speculative completeness.\n"
        )
        grounded = cls._request_anthropic_web_search_text(
            api_key=key,
            model=model,
            prompt=grounding_prompt,
            timeout_s=float(max(20.0, timeout_s)),
        )
        extraction_raw = str(grounded.get("text", "") or "")
        parsed = cls._extract_json_from_response(extraction_raw)
        if not isinstance(parsed, dict):
            parsed = {
                "summary": "Benchmark survey extraction returned invalid JSON.",
                "scope_assessment": "Grounded extraction failed to parse model response.",
                "benchmark_entries": [],
                "sources_used": [],
                "missing_information": ["Claude/Anthropic web_search extraction response was not valid JSON."],
                "warnings": ["Review raw extraction response and grounding metadata."],
                "recommended_next_steps": ["Narrow the request (specific conference track / year / paper set) and retry."],
            }

        grounding_metadata = grounded.get("grounding_metadata", {})
        if not isinstance(grounding_metadata, dict):
            grounding_metadata = {}
        web_search_calls = grounding_metadata.get("web_search_calls", [])
        if not isinstance(web_search_calls, list):
            web_search_calls = []
        raw_sources = grounding_metadata.get("sources", [])
        if not isinstance(raw_sources, list):
            raw_sources = []
        web_queries: list[str] = []
        for call in web_search_calls:
            if not isinstance(call, dict):
                continue
            q = str(call.get("query", "") or "").strip()
            if q:
                web_queries.append(q)

        web_sources: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for idx, src in enumerate(raw_sources, start=1):
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "") or src.get("uri", "") or "").strip()
            title = str(src.get("title", "") or src.get("name", "") or "").strip()
            if not url and not title:
                continue
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            web_sources.append(
                {
                    "id": idx,
                    "query": "",
                    "title": title,
                    "url": url,
                    "snippet": str(src.get("snippet", "") or "").strip(),
                    "excerpt": "",
                    "fetch_ok": True,
                    "status_code": 200,
                    "fetch_error": "",
                }
            )

        report = dict(parsed)
        for key_name, default_value in (
            ("benchmark_entries", []),
            ("sources_used", []),
            ("missing_information", []),
            ("warnings", []),
            ("recommended_next_steps", []),
        ):
            report.setdefault(key_name, default_value)
        report["task_kind"] = "benchmark_survey"
        report["intent"] = intent
        report["search_queries"] = [str(q) for q in web_queries if str(q).strip()]
        report["search_result_count"] = len(web_sources)
        report["fetched_source_count"] = len(web_sources)
        report["grounding_support_count"] = len(web_search_calls)
        if not web_sources and isinstance(report.get("warnings"), list):
            report["warnings"].append("No web_search sources were returned by Anthropic Messages API for this request.")

        debug_payload = {
            "web_search_queries": web_queries,
            "web_search_calls": web_search_calls[:20],
            "sources": raw_sources[:50],
            "raw_response_keys": list(grounded.get("raw_response", {}).keys()) if isinstance(grounded.get("raw_response", {}), dict) else [],
        }
        return {
            "artifact_type": "benchmark_survey",
            "base_name": str(base_name or "benchmark_survey"),
            "provider": cls.ANTHROPIC_PROVIDER,
            "model": str(grounded.get("model", model)),
            "survey_report": report,
            "web_sources": web_sources,
            "_request_intent": intent,
            "_search_queries": report.get("search_queries", []),
            "_search_results": web_sources,
            "_api_raw_text": extraction_raw,
            "_grounding_metadata": grounding_metadata,
            "_grounded_raw_response": grounded.get("raw_response", {}),
            "_search_debug_text": json.dumps(debug_payload, indent=2, ensure_ascii=False),
        }

    @classmethod
    @staticmethod
    def _extract_anthropic_messages_output_text(data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        chunks: list[str] = []
        content = data.get("content", [])
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type", "") or "").strip().lower()
                if ptype == "text":
                    txt = part.get("text")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
        return "\n".join(chunks).strip()

    @classmethod
    def _request_anthropic_text(
        cls,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float,
        enable_web_search: bool = False,
        debug_out: dict[str, Any] | None = None,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
        stream_phase: str = "generation",
        web_search_profile: str = "github_only",
    ) -> str:
        request_timeout = float(max(1.0, timeout_s))
        payload: dict[str, Any] = {
            "model": str(model),
            "max_tokens": 8192,
            "system": str(system_prompt),
            "messages": [{"role": "user", "content": str(user_prompt)}],
            "tool_choice": {"type": "auto"},
        }
        thinking = cls._anthropic_default_thinking_payload()
        if isinstance(thinking, dict) and thinking:
            payload["thinking"] = thinking
        if bool(enable_web_search):
            payload["tools"] = [cls._anthropic_web_search_tool_payload(profile=web_search_profile)]
        if isinstance(debug_out, dict):
            try:
                debug_out.clear()
                debug_out.update(
                    {
                        "provider": "anthropic_messages",
                        "endpoint": cls.ANTHROPIC_MESSAGES_URL,
                        "method": "POST",
                        "enable_web_search": bool(enable_web_search),
                        "web_search_profile": str(web_search_profile),
                        "stream": True,
                        "timeout_s": request_timeout,
                        "payload": json.loads(json.dumps(payload)),
                        "payload_chars": {
                            "system": len(str(system_prompt or "")),
                            "user_prompt": len(str(user_prompt or "")),
                        },
                    }
                )
            except Exception:
                pass
        try:
            cls._wait_for_api_call_slot(label="anthropic_messages_text_web" if bool(enable_web_search) else "anthropic_messages_text")
            client = cls._anthropic_client(api_key=api_key, timeout_s=request_timeout)
            with client.messages.stream(**payload, timeout=request_timeout) as stream:
                saw_web_search_event = False
                last_emit_ts = time.monotonic()
                pending_chunks: list[str] = []
                pending_chars = 0

                def _flush_pending() -> None:
                    nonlocal pending_chunks, pending_chars, last_emit_ts
                    if not pending_chunks:
                        return
                    text_chunk = "".join(pending_chunks)
                    pending_chunks = []
                    pending_chars = 0
                    last_emit_ts = time.monotonic()
                    cls._emit_stream_event(
                        stream_event_cb,
                        {
                            "kind": "llm_stream",
                            "phase": str(stream_phase or "generation"),
                            "event": "text_delta",
                            "text": text_chunk,
                        },
                    )

                try:
                    for event in stream:
                        event_type = cls._anthropic_stream_event_type(event)
                        event_dict = cls._anthropic_stream_event_to_dict(event)
                        event_dict_json = ""
                        try:
                            event_dict_json = json.dumps(event_dict, ensure_ascii=False)
                        except Exception:
                            event_dict_json = str(event_dict)
                        if "web_search" in event_type.lower() or "web_search" in event_dict_json.lower():
                            query = cls._extract_anthropic_stream_web_search_query(event)
                            if (not saw_web_search_event) or query or ("server_tool_use" in event_dict_json.lower()):
                                saw_web_search_event = True
                                cls._emit_stream_event(
                                    stream_event_cb,
                                    {
                                        "kind": "llm_stream",
                                        "phase": str(stream_phase or "generation"),
                                        "event": "web_search",
                                        "event_type": event_type,
                                        "query": query,
                                        "message": (
                                            f"web_search query: {query}"
                                            if query
                                            else "web_search tool invoked during generation."
                                        ),
                                    },
                                )
                        delta = cls._extract_anthropic_stream_text_delta(event)
                        if isinstance(delta, str) and delta:
                            pending_chunks.append(delta)
                            pending_chars += len(delta)
                            now_ts = time.monotonic()
                            if ("\n" in delta) or pending_chars >= 240 or (now_ts - last_emit_ts) >= 0.25:
                                _flush_pending()
                    _flush_pending()
                except TypeError:
                    stream.until_done()
                resp = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001
            status_code, body = cls._anthropic_exception_status_and_body(exc)
            if isinstance(debug_out, dict):
                try:
                    debug_out["request_error"] = str(exc)
                    if status_code is not None:
                        debug_out["response_status_code"] = status_code
                    if body:
                        debug_out["response_body_preview"] = body[:700]
                except Exception:
                    pass
            if status_code is not None:
                raise RuntimeError(f"Anthropic Messages API HTTP {status_code}: {body[:700]}")
            raise RuntimeError(f"Anthropic Messages API request failed: {exc}") from exc

        raw = cls._anthropic_message_to_raw_text(resp)
        data = cls._anthropic_message_to_dict(resp)
        if not isinstance(data, dict) or not data:
            if isinstance(debug_out, dict):
                try:
                    debug_out["response_body_preview"] = raw[:700]
                except Exception:
                    pass
            raise RuntimeError(f"Anthropic Messages API returned non-JSON response: {raw[:700]}")
        if isinstance(debug_out, dict):
            try:
                debug_out["response_status_code"] = 200
                debug_out["response_content_type"] = "application/json (anthropic sdk)"
                debug_out["response_url"] = cls.ANTHROPIC_MESSAGES_URL
                debug_out["response_id"] = data.get("id")
                debug_out["response_model"] = data.get("model")
                content_items = data.get("content", [])
                debug_out["response_content_items"] = int(len(content_items)) if isinstance(content_items, list) else 0
                if "payload" in debug_out and isinstance(debug_out.get("payload"), dict):
                    debug_out["payload"]["stream_phase"] = str(stream_phase or "generation")
            except Exception:
                pass

        text = cls._extract_anthropic_messages_output_text(data if isinstance(data, dict) else {})
        if text:
            return text
        raise RuntimeError(f"Anthropic Messages API response does not contain usable output text: {raw[:700]}")

    @classmethod
    def validate_anthropic_key_access(
        cls,
        explicit_key: str | None = None,
        *,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        key = cls.resolve_anthropic_api_key(explicit_key)
        if not key:
            return {
                "has_key": False,
                "auth_ok": False,
                "route_ok": False,
                "message": "Anthropic API key is empty.",
            }

        cache_key = str(hash(key))
        cached = cls._KEY_VALIDATION_CACHE.get(cache_key)
        if isinstance(cached, dict):
            try:
                age_s = time.time() - float(cached.get("_ts", 0.0))
            except Exception:
                age_s = 1e9
            if age_s <= 120.0:
                out = dict(cached)
                out.pop("_ts", None)
                out["cached"] = True
                return out

        try:
            cls._wait_for_api_call_slot(label="anthropic_models_list")
            client = cls._anthropic_client(api_key=key, timeout_s=float(max(1.0, timeout_s)))
            page = client.models.list(timeout=float(max(1.0, timeout_s)))
            data = cls._anthropic_message_to_dict(page)
            model_items = data.get("data", []) if isinstance(data.get("data", []), list) else []
            model_count = len(model_items)
            auth_ok = True
        except Exception as exc:  # noqa: BLE001
            body = str(exc)
            result = {
                "has_key": True,
                "auth_ok": False,
                "route_ok": False,
                "message": f"Anthropic key authentication failed: {body[:300]}",
            }
            cls._KEY_VALIDATION_CACHE[cache_key] = {**result, "_ts": time.time()}
            return result

        try:
            route_errors: list[str] = []
            for selected_model in cls.select_anthropic_model_candidates(key, timeout_s=timeout_s)[:5]:
                try:
                    reply = cls._request_anthropic_text(
                        api_key=key,
                        model=selected_model,
                        system_prompt="Reply with exactly OK.",
                        user_prompt="OK",
                        timeout_s=min(float(timeout_s), 20.0),
                    )
                    result = {
                        "has_key": True,
                        "auth_ok": auth_ok,
                        "route_ok": True,
                        "message": (
                            f"Anthropic key validated. Models endpoint OK ({model_count} models listed). "
                            f"Messages API route OK for model {selected_model}."
                        ),
                        "reply_preview": str(reply)[:120],
                        "model": selected_model,
                    }
                    cls._KEY_VALIDATION_CACHE[cache_key] = {**result, "_ts": time.time()}
                    return result
                except Exception as route_exc:  # noqa: BLE001
                    route_errors.append(f"{selected_model}: {route_exc}")
                    continue
            raise RuntimeError("; ".join(route_errors[:3]) or "No candidate model route succeeded.")
        except Exception as exc:  # noqa: BLE001
            result = {
                "has_key": True,
                "auth_ok": auth_ok,
                "route_ok": False,
                "message": (
                    f"Anthropic key auth is OK (models endpoint, {model_count} models), "
                    f"but Messages API route for {cls.DEFAULT_ANTHROPIC_MODEL} failed: {exc}"
                ),
            }
            cls._KEY_VALIDATION_CACHE[cache_key] = {**result, "_ts": time.time()}
            return result

    @classmethod
    def select_anthropic_model(cls, api_key: str, *, timeout_s: float = 30.0) -> str:
        candidates = cls.select_anthropic_model_candidates(api_key, timeout_s=timeout_s)
        return candidates[0] if candidates else cls.DEFAULT_ANTHROPIC_MODEL

    @classmethod
    def select_anthropic_model_candidates(cls, api_key: str, *, timeout_s: float = 30.0) -> list[str]:
        _ = (api_key, timeout_s)
        # Fixed model policy requested by the application: avoid env overrides and dynamic model discovery.
        return [cls.DEFAULT_ANTHROPIC_MODEL]

    @staticmethod
    def _extract_code_from_response(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        fence_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        return cleaned

    @classmethod
    def _extract_json_from_response(cls, text: str) -> Any:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
        return None

    @classmethod
    def default_anthropic_model(cls) -> str:
        return cls.DEFAULT_ANTHROPIC_MODEL

    @classmethod
    def resolve_anthropic_api_key(cls, explicit_key: str | None = None) -> str:
        for candidate in (
            explicit_key,
            os.getenv("ANTHROPIC_API_KEY"),
        ):
            value = str(candidate or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def api_key_path(cls, base_dir: Path) -> Path:
        return Path(base_dir) / cls.API_KEY_FILENAME

    @classmethod
    def load_saved_api_key(cls, *, base_dir: Path) -> str:
        candidate = cls.api_key_path(base_dir)
        if not candidate.exists():
            return ""
        try:
            return candidate.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            return ""

    @classmethod
    def save_api_key(cls, *, base_dir: Path, api_key: str) -> Path:
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("API key is empty.")
        key_path = cls.api_key_path(base_dir)
        key_path.write_text(value + "\n", encoding="utf-8")
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass
        return key_path

    @classmethod
    def api_key_portal_url(cls) -> str:
        return cls.ANTHROPIC_API_KEYS_URL

    @classmethod
    def _validate_common_python_code(cls, code: str, *, require_class: bool, require_create_metric: bool) -> tuple[bool, list[str]]:
        issues: list[str] = []
        try:
            tree = ast.parse(code)
        except Exception as exc:  # noqa: BLE001
            return False, [f"Syntax error: {exc}"]

        for node in ast.walk(tree):
            if isinstance(node, cls._BLOCKED_AST_NODES):
                issues.append(f"Unsupported AST node: {type(node).__name__}")
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = str(alias.name or "").split(".")[0]
                    if root not in cls._ALLOWED_IMPORT_ROOTS:
                        issues.append(f"Unsupported import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = str(getattr(node, "module", "") or "")
                root = module.split(".")[0] if module else ""
                if root not in cls._ALLOWED_IMPORT_ROOTS:
                    issues.append(f"Unsupported import-from: {module or '<empty module>'}")
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    if fn.id in cls._BLOCKED_CALL_NAMES:
                        issues.append(f"Unsupported call: {fn.id}(...)")
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    if (fn.value.id, fn.attr) in cls._BLOCKED_ATTR_CALLS:
                        issues.append(f"Unsupported call: {fn.value.id}.{fn.attr}(...)")

        class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        fn_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if require_class and not class_defs:
            issues.append("No class definition found.")
        if require_class:
            has_evaluate = False
            for cls_node in class_defs:
                for item in cls_node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_evaluate":
                        has_evaluate = True
                        break
            if not has_evaluate:
                issues.append("Class must implement _evaluate(self, X, out, *args, **kwargs).")

        if require_create_metric and not any(fn.name == "create_metric" for fn in fn_defs):
            issues.append("Module must expose create_metric(context).")

        try:
            compile(code, "<llm_artifact>", "exec")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Compile error: {exc}")

        return len(issues) == 0, issues

    @classmethod
    def validate_problem_code(cls, code: str) -> tuple[bool, list[str]]:
        ok, issues = cls._validate_common_python_code(code, require_class=True, require_create_metric=False)
        vectorization_hints = ("axis=1", "column_stack", "np.asarray", "out['F']", 'out["F"]')
        if not any(tok in code for tok in vectorization_hints):
            issues.append("Vectorization hints not found (expected operations over population matrices).")
        if cls._looks_like_placeholder_benchmark_fallback_text(code):
            issues.append(
                "Placeholder benchmark fallback text detected (e.g., 'placeholder semantics' / 'could not be reliably recovered'). "
                "LLM Agent must recover and implement the requested benchmark semantics via web_search sources, not ship placeholder semantics."
            )
        if cls._looks_like_native_pymoo_problem_wrapper(code):
            issues.append(
                "Native/local benchmark wrapper pattern detected. LLM Agent problem generation must implement/adapt the requested problem semantics directly (no thin wrapper around pymoo or local canonical problem classes)."
            )
        return len(issues) == 0, issues

    @classmethod
    def validate_metric_code(cls, code: str) -> tuple[bool, list[str]]:
        ok, issues = cls._validate_common_python_code(code, require_class=False, require_create_metric=True)
        if "create_metric" in code and "front" not in code.lower():
            issues.append("Metric code should reference front values (expected 'front' usage).")
        tl = str(code or "").lower()
        if "placeholder metric" in tl or ("placeholder semantics" in tl and "metric" in tl):
            issues.append(
                "Placeholder metric fallback text detected. LLM Agent metric generation must produce complete metric semantics via GitHub-sourced implementations."
            )
        return len(issues) == 0, issues

    @staticmethod
    def _metric_kind_hint(bundle: dict[str, Any]) -> str:
        text_raw = " ".join(
            [
                str(bundle.get("base_name", "")),
                str(bundle.get("_prompt", "")),
                str(bundle.get("cpu_code", ""))[:4000],
            ]
        ).lower()
        # Normalize common separators so names like "HV_LLM_MONTECARLO" are recognized.
        text = re.sub(r"[_/\\\\-]+", " ", text_raw)
        if ("hypervolume" in text) or re.search(r"\bhv\b", text):
            mc_hint = (
                ("monte" in text)
                or ("montecarlo" in text)
                or ("monte carlo" in text)
                or re.search(r"\bhv\s*mc\b", text) is not None
                or re.search(r"\bmc\s*hv\b", text) is not None
            )
            return "hv_mc" if mc_hint else "hv"
        if "deltap" in text or "delta p" in text:
            return "deltap"
        if re.search(r"\bigd\+\b", text) or "igdp" in text:
            return "igdp"
        if re.search(r"\bigd\b", text):
            return "igd"
        if re.search(r"\bgd\b", text):
            return "gd"
        return ""

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return out

    @classmethod
    def _exec_metric_factory(cls, code: str) -> tuple[Any, list[str]]:
        issues: list[str] = []
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(compile(str(code), "<llm_metric>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return None, [f"Runtime exec failed: {exc}"]
        fn = ns.get("create_metric")
        if not callable(fn):
            issues.append("create_metric(context) not found after exec.")
            return None, issues
        return fn, issues

    @classmethod
    def _exec_problem_class(cls, code: str) -> tuple[type[Any] | None, list[str]]:
        issues: list[str] = []
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(compile(str(code), "<llm_problem>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return None, [f"Runtime exec failed: {exc}"]
        imported_problem = ns.get("Problem")
        for name, obj in ns.items():
            try:
                if not isinstance(obj, type):
                    continue
                if obj is imported_problem or name == "Problem":
                    continue
                if hasattr(obj, "_evaluate"):
                    return obj, issues
            except Exception:
                continue
        return None, ["No Problem class found after exec."]

    @classmethod
    def _metric_oracle_value(cls, metric_kind: str, front: Any, context: dict[str, Any]) -> float:
        try:
            from metrics import community_metrics as cm
        except Exception:  # noqa: BLE001
            return float("nan")
        metric_map = {
            "hv": getattr(cm, "_metric_HV", None),
            "hv_mc": getattr(cm, "_metric_HV", None),
            "gd": getattr(cm, "_metric_GD", None),
            "igd": getattr(cm, "_metric_IGD", None),
            "igdp": getattr(cm, "_metric_IGDp", None),
            "deltap": getattr(cm, "_metric_DeltaP", None),
        }
        fn = metric_map.get(metric_kind)
        if not callable(fn):
            return float("nan")
        try:
            return cls._safe_float(fn(front, dict(context)))
        except Exception:  # noqa: BLE001
            return float("nan")

    @classmethod
    def _build_metric_validation_case(cls, n_obj: int) -> tuple[Any, dict[str, Any]]:
        import numpy as _np

        m = int(max(2, n_obj))
        rng = _np.random.default_rng(7)
        # Synthetic PF on simplex-like manifold in [0,1]^m.
        pf = rng.random((256, m))
        pf /= _np.clip(pf.sum(axis=1, keepdims=True), 1e-12, None)
        # Candidate front = perturbed subset, clipped.
        idx = rng.choice(_np.arange(len(pf)), size=96, replace=False)
        F = _np.clip(pf[idx] + 0.03 * rng.normal(size=(96, m)), 0.0, 1.2)
        context = {
            "pareto_front": pf,
            "ref_pf": pf,
            "n_obj": m,
            "hv_mc_samples": 30000,
            "seed": 11,
            "case_name": f"synthetic_m{m}",
        }
        return F, context

    @classmethod
    def _build_metric_validation_problem_case_zdt1(cls) -> tuple[Any, dict[str, Any]] | None:
        import numpy as _np

        try:
            from problems.multi.zdt import ZDT1 as _ZDT1
        except Exception:
            return None

        try:
            problem = _ZDT1(n_var=30)
            n_var = int(max(1, int(getattr(problem, "n_var", 30) or 30)))
            rng = _np.random.default_rng(23)
            X = rng.random((96, n_var), dtype=float)
            F = None
            evaluate = getattr(problem, "evaluate", None)
            if callable(evaluate):
                F = evaluate(X, return_values_of=["F"])
                if isinstance(F, (tuple, list)):
                    F = F[0]
            if F is None:
                out: dict[str, Any] = {}
                problem._evaluate(X, out)
                F = out.get("F")
            F_arr = _np.asarray(F, dtype=float)
            if F_arr.ndim == 1:
                F_arr = F_arr.reshape(1, -1)
            if F_arr.ndim != 2 or F_arr.shape[1] != 2 or F_arr.size == 0 or not _np.all(_np.isfinite(F_arr)):
                return None

            pf = None
            for fn_name in ("pareto_front", "_calc_pareto_front"):
                fn = getattr(problem, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    pf = fn(n_pareto_points=256)
                except TypeError:
                    try:
                        pf = fn()
                    except Exception:
                        pf = None
                except Exception:
                    pf = None
                if pf is not None:
                    break
            if pf is None:
                return None
            pf_arr = _np.asarray(pf, dtype=float)
            if pf_arr.ndim == 1:
                pf_arr = pf_arr.reshape(1, -1)
            if pf_arr.ndim != 2 or pf_arr.shape[1] != 2 or pf_arr.size == 0 or not _np.all(_np.isfinite(pf_arr)):
                return None

            nested_cfg = {
                "pareto_front": pf_arr,
                "ref_pf": pf_arr,
                "n_obj": 2,
                "hv_mc_samples": 8000,
                "seed": 29,
                "problem": problem,
            }
            context = {
                "pareto_front": pf_arr,
                "ref_pf": pf_arr,
                "n_obj": 2,
                "hv_mc_samples": 8000,
                "seed": 29,
                "problem": problem,
                "config": dict(nested_cfg),
                "case_name": "zdt1_real_context",
            }
            return F_arr, context
        except Exception:
            return None

    @classmethod
    def _build_metric_validation_cases(cls, metric_kind: str, n_obj: int) -> list[tuple[Any, dict[str, Any]]]:
        dims: list[int] = []
        target = int(max(2, n_obj))
        dims.append(target)
        if metric_kind in {"hv", "hv_mc", "gd", "igd", "igdp", "deltap"}:
            dims.extend([2, 3])
            if target > 3:
                dims.append(min(10, target))
            else:
                dims.append(5)
        uniq_dims: list[int] = []
        for d in dims:
            d = int(max(2, d))
            if d not in uniq_dims:
                uniq_dims.append(d)
        cases: list[tuple[Any, dict[str, Any]]] = []
        for d in uniq_dims[:4]:
            F, context = cls._build_metric_validation_case(d)
            # Keep validation fast while still meaningful across dimensions.
            context["hv_mc_samples"] = int(min(20000, max(4000, context.get("hv_mc_samples", 8000))))
            cases.append((F, context))
            # Also test wrapped PymooLab-style metric context (`context['config']`) to catch missing unwrapping.
            wrapped_cfg = {
                key: value
                for key, value in context.items()
                if key not in {"case_name", "config"}
            }
            wrapped_context = {
                "config": dict(wrapped_cfg),
                "n_obj": int(context.get("n_obj", d) or d),
                "case_name": f"wrapped_{context.get('case_name', f'synthetic_m{d}')}",
            }
            cases.append((F, wrapped_context))

        if metric_kind in {"hv", "hv_mc", "gd", "igd", "igdp", "deltap"}:
            zdt1_case = cls._build_metric_validation_problem_case_zdt1()
            if zdt1_case is not None:
                cases.append(zdt1_case)
        return cases

    @staticmethod
    def _problem_pf_mode_from_code(code: str) -> str:
        text = str(code or "")
        if "_calc_pareto_front" not in text:
            return "unavailable"
        if "n_samples = int(max(2000, n_pf * 40))" in text:
            return "approximate_generic"
        if "np.linspace(0.2807753191" in text or "regions = [" in text or "1.0 - np.sqrt(x)" in text:
            return "exact_family"
        return "custom_or_unknown"

    @classmethod
    def _probe_problem_pareto_front(cls, problem: Any, *, n_obj: int) -> tuple[str, dict[str, Any]]:
        import numpy as _np
        report: dict[str, Any] = {}
        fn = getattr(problem, "_calc_pareto_front", None)
        if not callable(fn):
            return "unavailable", report
        try:
            pf = fn(n_pareto_points=32)
        except TypeError:
            try:
                from pymoo.util.ref_dirs import get_reference_directions as _get_ref_dirs

                ref_dirs = _get_ref_dirs("energy", int(max(2, n_obj)), 32)
                pf = fn(ref_dirs=ref_dirs)
            except Exception as exc:  # noqa: BLE001
                report["pf_probe_error"] = str(exc)
                return "available_but_probe_failed", report
        except Exception as exc:  # noqa: BLE001
            report["pf_probe_error"] = str(exc)
            return "available_but_probe_failed", report

        try:
            pf_arr = _np.asarray(pf, dtype=float)
            report["pf_probe_shape"] = list(pf_arr.shape)
            report["pf_probe_finite"] = bool(_np.all(_np.isfinite(pf_arr))) if pf_arr.size else True
            if pf_arr.ndim == 2 and pf_arr.shape[1] == int(max(1, n_obj)):
                return "available", report
            return "available_shape_mismatch", report
        except Exception as exc:  # noqa: BLE001
            report["pf_probe_error"] = str(exc)
            return "available_but_probe_failed", report

    @classmethod
    def _validate_metric_bundle_runtime(cls, bundle: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
        import numpy as _np
        issues: list[str] = []
        report: dict[str, Any] = {"kind": "metric", "checks": [], "validation_cases": []}

        cpu_factory_fn, cpu_exec_issues = cls._exec_metric_factory(str(bundle.get("cpu_code", "")))
        jax_factory_fn, jax_exec_issues = cls._exec_metric_factory(str(bundle.get("jax_code", "")))
        issues.extend([f"CPU runtime: {x}" for x in cpu_exec_issues])
        issues.extend([f"JAX runtime: {x}" for x in jax_exec_issues])
        if not callable(cpu_factory_fn) or not callable(jax_factory_fn):
            return False, issues, report

        metric_kind = cls._metric_kind_hint(bundle)
        report["metric_kind_hint"] = metric_kind or "generic"

        try:
            # Smoke instantiate once on default dimension first.
            base_n_obj = int(max(2, int(bundle.get("n_obj", 2) or 2)))
            _, base_context = cls._build_metric_validation_case(base_n_obj)
            cpu_metric = cpu_factory_fn(dict(base_context))
            jax_metric = jax_factory_fn(dict(base_context))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Metric factory instantiation failed: {exc}")
            return False, issues, report

        if not callable(cpu_metric) or not callable(jax_metric):
            issues.append("create_metric(context) did not return callable metric(front).")
            return False, issues, report

        prompt_text = str(bundle.get("_prompt", "") or "")
        skip_oracle_for_variant = bool(metric_kind and cls._looks_like_novel_metric_variant_request(prompt_text))
        if skip_oracle_for_variant:
            report["oracle_skipped_reason"] = "variant_or_novel_metric_request"

        worst_cpu_jax_abs = 0.0
        worst_cpu_jax_rel = 0.0
        worst_oracle_abs = 0.0
        worst_oracle_rel = 0.0
        first_cpu_val = float("nan")
        first_jax_val = float("nan")
        first_oracle_val = float("nan")

        cases = cls._build_metric_validation_cases(metric_kind or "generic", int(max(2, int(bundle.get("n_obj", 2) or 2))))

        for case_idx, (F, context) in enumerate(cases):
            case_report: dict[str, Any] = {"idx": case_idx, "m": int(context.get("n_obj", 0))}
            if context.get("case_name") is not None:
                case_report["case_name"] = str(context.get("case_name"))
            try:
                cpu_metric_case = cpu_factory_fn(dict(context))
                jax_metric_case = jax_factory_fn(dict(context))
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Metric factory instantiation failed on case m={case_report['m']}: {exc}")
                report["validation_cases"].append(case_report)
                continue

            try:
                cpu_val = cls._safe_float(cpu_metric_case(F))
            except Exception as exc:  # noqa: BLE001
                cpu_val = float("nan")
                issues.append(f"CPU metric(front) failed on case m={case_report['m']}: {exc}")
            try:
                jax_val = cls._safe_float(jax_metric_case(F))
            except Exception as exc:  # noqa: BLE001
                jax_val = float("nan")
                issues.append(f"JAX metric(front) failed on case m={case_report['m']}: {exc}")

            case_report["cpu_value"] = cpu_val
            case_report["jax_value"] = jax_val

            if case_idx == 0:
                first_cpu_val = cpu_val
                first_jax_val = jax_val

            if not math.isfinite(cpu_val):
                issues.append(f"CPU metric returned non-finite value in runtime check (m={case_report['m']}).")
            if not math.isfinite(jax_val):
                issues.append(f"JAX metric returned non-finite value in runtime check (m={case_report['m']}).")

            if math.isfinite(cpu_val) and math.isfinite(jax_val):
                abs_err = abs(cpu_val - jax_val)
                rel_err = abs_err / max(abs(cpu_val), abs(jax_val), 1e-12)
                case_report["cpu_jax_abs_err"] = abs_err
                case_report["cpu_jax_rel_err"] = rel_err
                worst_cpu_jax_abs = max(worst_cpu_jax_abs, abs_err)
                worst_cpu_jax_rel = max(worst_cpu_jax_rel, rel_err)
                rel_tol = 0.25 if metric_kind == "hv_mc" else 0.05
                abs_tol = 1e-6 if metric_kind != "hv_mc" else 5e-3
                if not (abs_err <= abs_tol or rel_err <= rel_tol):
                    issues.append(
                        f"CPU/JAX parity check failed on m={case_report['m']} (abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                    )

            if metric_kind and not skip_oracle_for_variant:
                oracle = cls._metric_oracle_value(metric_kind, F, dict(context))
                case_report["oracle_value"] = oracle
                if case_idx == 0:
                    first_oracle_val = oracle
                if math.isfinite(oracle) and math.isfinite(cpu_val):
                    abs_err = abs(cpu_val - oracle)
                    rel_err = abs_err / max(abs(oracle), 1e-12)
                    case_report["oracle_abs_err"] = abs_err
                    case_report["oracle_rel_err"] = rel_err
                    worst_oracle_abs = max(worst_oracle_abs, abs_err)
                    worst_oracle_rel = max(worst_oracle_rel, rel_err)
                    if metric_kind == "hv_mc":
                        oracle_scale = abs(oracle)
                        hv_mc_ok = (rel_err <= 0.35) if oracle_scale > 1e-6 else (abs_err <= 5e-3)
                        if not hv_mc_ok:
                            issues.append(
                                f"Known-metric sanity check failed for HV Monte Carlo on m={case_report['m']} "
                                f"(abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                            )
                    else:
                        if not (abs_err <= 1e-4 or rel_err <= 5e-2):
                            issues.append(
                                f"Known-metric compatibility check failed on m={case_report['m']} "
                                f"(abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                            )

            # Metamorphic checks (record for all; enforce for known metrics only)
            try:
                perm = _np.random.default_rng(101 + case_idx).permutation(F.shape[0])
                F_perm = F[perm]
                cpu_perm_val = cls._safe_float(cpu_metric_case(F_perm))
                case_report["cpu_perm_value"] = cpu_perm_val
                if math.isfinite(cpu_val) and math.isfinite(cpu_perm_val):
                    perm_abs = abs(cpu_val - cpu_perm_val)
                    perm_rel = perm_abs / max(abs(cpu_val), abs(cpu_perm_val), 1e-12)
                    case_report["perm_abs_err"] = perm_abs
                    case_report["perm_rel_err"] = perm_rel
                    if metric_kind and perm_rel > 1e-6 and perm_abs > 1e-8:
                        issues.append(
                            f"Permutation invariance failed for known metric on m={case_report['m']} "
                            f"(abs_err={perm_abs:.6g}, rel_err={perm_rel:.6g})."
                        )
                cpu_repeat_val = cls._safe_float(cpu_metric_case(F))
                case_report["cpu_repeat_value"] = cpu_repeat_val
                if math.isfinite(cpu_val) and math.isfinite(cpu_repeat_val):
                    rep_abs = abs(cpu_val - cpu_repeat_val)
                    rep_rel = rep_abs / max(abs(cpu_val), abs(cpu_repeat_val), 1e-12)
                    case_report["repeat_abs_err"] = rep_abs
                    case_report["repeat_rel_err"] = rep_rel
                    if metric_kind and rep_rel > 1e-6 and rep_abs > 1e-8:
                        issues.append(
                            f"Determinism check failed for known metric on m={case_report['m']} "
                            f"(abs_err={rep_abs:.6g}, rel_err={rep_rel:.6g})."
                        )
            except Exception as exc:  # noqa: BLE001
                case_report["metamorphic_error"] = str(exc)
                if metric_kind:
                    issues.append(f"Metamorphic checks failed on m={case_report['m']}: {exc}")

            report["validation_cases"].append(case_report)

        report["cpu_value"] = first_cpu_val
        report["jax_value"] = first_jax_val
        if math.isfinite(worst_cpu_jax_abs) or math.isfinite(worst_cpu_jax_rel):
            report["cpu_jax_abs_err"] = worst_cpu_jax_abs
            report["cpu_jax_rel_err"] = worst_cpu_jax_rel
        if metric_kind and not skip_oracle_for_variant:
            report["oracle_value"] = first_oracle_val
            report["oracle_abs_err"] = worst_oracle_abs
            report["oracle_rel_err"] = worst_oracle_rel
        report["multi_scenario_dims"] = [int(c[1].get("n_obj", 0)) for c in cases]

        return len(issues) == 0, issues, report

    @classmethod
    def _validate_problem_bundle_runtime(cls, bundle: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
        import numpy as _np

        issues: list[str] = []
        report: dict[str, Any] = {"kind": "problem", "checks": []}
        report["pf_mode_from_code"] = cls._problem_pf_mode_from_code(str(bundle.get("cpu_code", "")))
        cpu_cls, cpu_exec_issues = cls._exec_problem_class(str(bundle.get("cpu_code", "")))
        jax_cls, jax_exec_issues = cls._exec_problem_class(str(bundle.get("jax_code", "")))
        issues.extend([f"CPU runtime: {x}" for x in cpu_exec_issues])
        issues.extend([f"JAX runtime: {x}" for x in jax_exec_issues])
        if cpu_cls is None or jax_cls is None:
            return False, issues, report

        try:
            cpu_problem = cpu_cls()
            jax_problem = jax_cls()
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Problem instantiation failed: {exc}")
            return False, issues, report

        n_var = int(max(1, int(getattr(cpu_problem, "n_var", bundle.get("n_var", 30)) or 30)))
        n_obj = int(max(1, int(getattr(cpu_problem, "n_obj", bundle.get("n_obj", 2)) or 2)))
        rng = _np.random.default_rng(17)
        xl = _np.asarray(getattr(cpu_problem, "xl", 0.0), dtype=float)
        xu = _np.asarray(getattr(cpu_problem, "xu", 1.0), dtype=float)
        if xl.ndim == 0:
            xl = _np.full(n_var, float(xl))
        else:
            xl = _np.ravel(xl).astype(float)
            if xl.size != n_var:
                xl = _np.resize(xl, n_var)
        if xu.ndim == 0:
            xu = _np.full(n_var, float(xu))
        else:
            xu = _np.ravel(xu).astype(float)
            if xu.size != n_var:
                xu = _np.resize(xu, n_var)

        # Some generated problems may expose placeholder/invalid bounds (nan/inf or reversed bounds).
        xl = _np.where(_np.isfinite(xl), xl, 0.0)
        xu = _np.where(_np.isfinite(xu), xu, 1.0)
        lo = _np.minimum(xl, xu)
        hi = _np.maximum(xl, xu)
        same = _np.isclose(lo, hi)
        hi = _np.where(same, lo + 1.0, hi)

        try:
            X = rng.uniform(lo, hi, size=(12, n_var))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Runtime input sampling failed with problem bounds ({exc}); using [0,1] fallback.")
            X = rng.uniform(0.0, 1.0, size=(12, n_var))

        out_cpu: dict[str, Any] = {}
        out_jax: dict[str, Any] = {}
        try:
            cpu_problem._evaluate(X, out_cpu)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"CPU _evaluate runtime failed: {exc}")
        try:
            jax_problem._evaluate(X, out_jax)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"JAX _evaluate runtime failed: {exc}")

        Fc = _np.asarray(out_cpu.get("F", []), dtype=float)
        Fj = _np.asarray(out_jax.get("F", []), dtype=float)
        report["cpu_F_shape"] = list(Fc.shape) if Fc.ndim == 2 else [*Fc.shape]
        report["jax_F_shape"] = list(Fj.shape) if Fj.ndim == 2 else [*Fj.shape]
        if Fc.ndim != 2 or Fc.shape != (X.shape[0], n_obj):
            issues.append(f"CPU _evaluate produced invalid F shape {Fc.shape}, expected ({X.shape[0]}, {n_obj}).")
        if Fj.ndim != 2 or Fj.shape != (X.shape[0], n_obj):
            issues.append(f"JAX _evaluate produced invalid F shape {Fj.shape}, expected ({X.shape[0]}, {n_obj}).")
        if Fc.size and not _np.all(_np.isfinite(Fc)):
            issues.append("CPU _evaluate produced non-finite F values.")
        if Fj.size and not _np.all(_np.isfinite(Fj)):
            issues.append("JAX _evaluate produced non-finite F values.")

        if Fc.shape == Fj.shape and Fc.size:
            abs_err = float(_np.max(_np.abs(Fc - Fj)))
            denom = float(max(_np.max(_np.abs(Fc)), _np.max(_np.abs(Fj)), 1e-12))
            rel_err = abs_err / denom
            report["cpu_jax_max_abs_err"] = abs_err
            report["cpu_jax_max_rel_err"] = rel_err
            if not (abs_err <= 1e-5 or rel_err <= 1e-3):
                issues.append(
                    f"Problem CPU/JAX parity check failed (max_abs_err={abs_err:.6g}, max_rel_err={rel_err:.6g})."
                )

        pf_probe_status, pf_probe_report = cls._probe_problem_pareto_front(cpu_problem, n_obj=n_obj)
        report["pf_probe_status"] = pf_probe_status
        report.update(pf_probe_report)
        if report.get("pf_mode_from_code") == "unavailable":
            report["pf_mode"] = "unavailable"
        elif report.get("pf_mode_from_code") == "approximate_generic":
            report["pf_mode"] = "approximate"
        elif report.get("pf_mode_from_code") == "exact_family":
            report["pf_mode"] = "exact"
        else:
            report["pf_mode"] = "custom"
        if pf_probe_status in {"available", "available_shape_mismatch", "available_but_probe_failed"}:
            # Keep explicit runtime observation separated from code inference.
            report["pf_probe_available"] = True
        else:
            report["pf_probe_available"] = False

        return len(issues) == 0, issues, report

    @classmethod
    def validate_artifact_bundle_detailed(cls, bundle: dict[str, Any]) -> dict[str, Any]:
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        cpu_code = str(bundle.get("cpu_code", ""))
        jax_code = str(bundle.get("jax_code", ""))
        issues: list[str] = []
        report: dict[str, Any] = {"artifact_type": artifact, "ok": False, "issues": issues, "checks": []}
        if not cpu_code.strip():
            issues.append("CPU code is empty.")
        if not jax_code.strip():
            issues.append("JAX code is empty.")
        if issues:
            bundle["_validation_report"] = report
            return report

        if artifact == "metric":
            ok_cpu, cpu_issues = cls.validate_metric_code(cpu_code)
            ok_jax, jax_issues = cls.validate_metric_code(jax_code)
        else:
            ok_cpu, cpu_issues = cls.validate_problem_code(cpu_code)
            ok_jax, jax_issues = cls.validate_problem_code(jax_code)
        issues.extend([f"CPU: {msg}" for msg in cpu_issues])
        issues.extend([f"JAX: {msg}" for msg in jax_issues])

        runtime_ok = False
        runtime_report: dict[str, Any] = {}
        if artifact == "metric":
            runtime_ok, runtime_issues, runtime_report = cls._validate_metric_bundle_runtime(bundle)
        else:
            runtime_ok, runtime_issues, runtime_report = cls._validate_problem_bundle_runtime(bundle)
        issues.extend(runtime_issues)
        report["runtime"] = runtime_report
        report["ok"] = bool(ok_cpu and ok_jax and runtime_ok and not issues)
        bundle["_validation_report"] = report
        return report

    @classmethod
    def validate_artifact_bundle(cls, bundle: dict[str, Any]) -> tuple[bool, list[str]]:
        report = cls.validate_artifact_bundle_detailed(bundle)
        return bool(report.get("ok", False)), list(report.get("issues", []))

    @classmethod
    def _problem_target_dir(cls, base_dir: Path, n_obj: int) -> Path:
        n = int(max(1, int(n_obj)))
        bucket = "single" if n == 1 else ("multi" if n <= 3 else "many")
        return Path(base_dir) / "problems" / bucket

    @classmethod
    def save_artifact_bundle(cls, *, base_dir: Path, bundle: dict[str, Any]) -> tuple[Path, Path]:
        cls._normalize_bundle_metadata(bundle)
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        cpu_file = str(bundle.get("cpu_file", "generated.py"))
        jax_file = str(bundle.get("jax_file", "generated_JAX.py"))
        cpu_code = str(bundle.get("cpu_code", ""))
        jax_code = str(bundle.get("jax_code", ""))

        if artifact == "metric":
            out_dir = Path(base_dir) / "metrics"
        else:
            out_dir = cls._problem_target_dir(base_dir, int(bundle.get("n_obj", 2)))
        out_dir.mkdir(parents=True, exist_ok=True)

        cpu_path = out_dir / cpu_file
        jax_path = out_dir / jax_file
        cpu_path.write_text(cpu_code, encoding="utf-8")
        jax_path.write_text(jax_code, encoding="utf-8")
        return cpu_path, jax_path

    # Convenience wrappers.
    @classmethod
    def generate_problem_code(
        cls,
        prompt: str,
        *,
        class_name: str,
        n_var: int,
        n_obj: int,
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> str:
        _ = model
        bundle = cls.generate_artifact_bundle(
            prompt,
            artifact_type="problem",
            base_name=class_name,
            n_var=n_var,
            n_obj=n_obj,
            provider=provider,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        return str(bundle.get("cpu_code", ""))

    @classmethod
    def save_problem_code(cls, *, base_dir: Path, class_name: str, code: str) -> Path:
        bundle = cls._generate_artifact_bundle_template(
            prompt="",
            artifact_type="problem",
            base_name=class_name,
            n_var=30,
            n_obj=2,
        )
        bundle["cpu_code"] = str(code)
        cls._normalize_bundle_metadata(bundle)
        out_dir = cls._problem_target_dir(base_dir, int(bundle.get("n_obj", 2)))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(bundle.get("cpu_file", f"{class_name}.py"))
        out_path.write_text(str(code), encoding="utf-8")
        return out_path
