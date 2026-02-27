from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Sequence, TypeVar


T = TypeVar("T")


def looks_like_jax_identifier(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        text.endswith("_jax")
        or "_jax." in text
        or "_jax_" in text
        or ".jax." in text
        or ".jax_" in text
    )


def strip_jax_suffix(value: str) -> str:
    return re.sub(r"_jax$", "", str(value).strip(), flags=re.IGNORECASE)


def backend_prefers_jax(
    *,
    backend: str | None = None,
    array_backend: str | None = None,
    use_gpu: bool | None = None,
) -> bool:
    if use_gpu is True:
        return True
    if use_gpu is False:
        return False
    token = str(backend or array_backend or "").strip().lower()
    return token in {"gpu", "jax"}


def normalize_backend_token(value: str) -> str:
    base = str(value).strip().lower()
    base = re.sub(r"[\s\-_]+jax$", "", base)
    base = strip_jax_suffix(base)
    base = re.sub(r"\.jax$", "", base)
    return re.sub(r"[^a-z0-9]+", "", base)


def is_jax_spec(
    name: Any,
    module: Any = None,
    spec_id: Any = None,
) -> bool:
    return (
        looks_like_jax_identifier(name)
        or looks_like_jax_identifier(module)
        or looks_like_jax_identifier(spec_id)
    )


def select_specs_for_backend(
    specs: Sequence[T],
    *,
    prefer_jax: bool,
    name_getter: Callable[[T], Any],
    module_getter: Callable[[T], Any] | None = None,
    id_getter: Callable[[T], Any] | None = None,
) -> list[T]:
    def _spec_is_jax(spec: T) -> bool:
        return is_jax_spec(
            name_getter(spec),
            module_getter(spec) if module_getter is not None else None,
            id_getter(spec) if id_getter is not None else None,
        )

    all_specs = list(specs)
    if prefer_jax:
        jax_specs = [spec for spec in all_specs if _spec_is_jax(spec)]
        if not jax_specs:
            return all_specs
        jax_names = {str(name_getter(spec)).strip().lower() for spec in jax_specs}
        return [
            spec
            for spec in jax_specs
            if str(name_getter(spec)).strip().lower().endswith("_jax")
            or f"{str(name_getter(spec)).strip().lower()}_jax" not in jax_names
        ]

    cpu_specs = [spec for spec in all_specs if not _spec_is_jax(spec)]
    return cpu_specs if cpu_specs else all_specs


def map_selected_ids_to_backend(
    selected_ids: Sequence[str],
    *,
    specs_by_id: dict[str, T],
    prefer_jax: bool,
    name_getter: Callable[[T], Any],
    module_getter: Callable[[T], Any] | None = None,
    id_getter: Callable[[T], Any] | None = None,
) -> list[str]:
    def _spec_is_jax(spec: T) -> bool:
        sid = id_getter(spec) if id_getter is not None else None
        return is_jax_spec(
            name_getter(spec),
            module_getter(spec) if module_getter is not None else None,
            sid,
        )

    token_index: dict[str, list[str]] = {}
    for sid, spec in specs_by_id.items():
        token = normalize_backend_token(str(name_getter(spec)))
        if not token:
            continue
        token_index.setdefault(token, []).append(str(sid))

    resolved: list[str] = []
    seen: set[str] = set()

    for raw_id in selected_ids:
        sid = str(raw_id)
        selected_spec: T | None = specs_by_id.get(sid)
        if selected_spec is None:
            continue

        chosen_id = sid
        spec_is_jax = _spec_is_jax(selected_spec)
        token = normalize_backend_token(str(name_getter(selected_spec)))
        candidates = [specs_by_id[cid] for cid in token_index.get(token, []) if cid in specs_by_id]

        desired = [c for c in candidates if _spec_is_jax(c) == prefer_jax]
        if desired:
            desired_ids = [
                str(id_getter(c)) if id_getter is not None else next(
                    (k for k, v in specs_by_id.items() if v is c),
                    "",
                )
                for c in desired
            ]
            desired_ids = [d for d in desired_ids if d]
            if desired_ids:
                # Prefer explicit *_JAX naming for jax mode, and non-_JAX otherwise.
                desired_ids.sort(
                    key=lambda did: (
                        0
                        if (
                            (prefer_jax and str(name_getter(specs_by_id[did])).strip().lower().endswith("_jax"))
                            or (not prefer_jax and not str(name_getter(specs_by_id[did])).strip().lower().endswith("_jax"))
                        )
                        else 1,
                        str(name_getter(specs_by_id[did])).strip().lower(),
                    )
                )
                chosen_id = desired_ids[0]
        elif spec_is_jax != prefer_jax:
            chosen_id = sid

        if chosen_id not in seen:
            seen.add(chosen_id)
            resolved.append(chosen_id)

    return resolved


def iter_operator_runtime_candidates(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool,
) -> list[tuple[str, str]]:
    module_clean = str(module_path).strip()
    class_clean = str(class_name).strip()
    module_cpu = strip_jax_suffix(module_clean)
    class_cpu = strip_jax_suffix(class_clean)

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(mod: str, cls: str) -> None:
        mod_s = str(mod).strip()
        cls_s = str(cls).strip()
        if not mod_s or not cls_s:
            return
        key = (mod_s, cls_s)
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    if prefer_jax:
        _add(f"{module_cpu}_JAX", f"{class_cpu}_JAX")
        _add(f"{module_cpu}_JAX", class_cpu)
        _add(module_cpu, f"{class_cpu}_JAX")

        if module_cpu.startswith("pymoo.operators."):
            local_base = f"operators.{module_cpu[len('pymoo.operators.'):]}"
            _add(f"{local_base}_JAX", class_cpu)
            _add(f"{local_base}_JAX", f"{class_cpu}_JAX")
            _add(local_base, f"{class_cpu}_JAX")
            _add(local_base, class_cpu)
        _add(module_clean, class_clean)
    else:
        _add(module_cpu, class_cpu)
        _add(module_clean, class_clean)
        _add(module_clean, class_cpu)
        _add(module_cpu, class_clean)

        if module_cpu.startswith("operators."):
            pymoo_base = f"pymoo.operators.{module_cpu[len('operators.'):]}"
            _add(pymoo_base, class_cpu)

    return candidates


def resolve_available_operator_target(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool,
) -> tuple[str, str] | None:
    for cand_module, cand_class in iter_operator_runtime_candidates(
        module_path,
        class_name,
        prefer_jax=prefer_jax,
    ):
        try:
            module = importlib.import_module(cand_module)
            _ = getattr(module, cand_class)
            return cand_module, cand_class
        except Exception:  # noqa: BLE001
            continue
    return None
