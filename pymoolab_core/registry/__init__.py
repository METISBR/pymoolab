"""Registry modules."""

from .backend_selection import (
    backend_prefers_jax,
    is_jax_spec,
    iter_operator_runtime_candidates,
    looks_like_jax_identifier,
    map_selected_ids_to_backend,
    normalize_backend_token,
    resolve_available_operator_target,
    select_specs_for_backend,
    strip_jax_suffix,
)
from .rollout import (
    backend_aware_loading_enabled,
    rollout_allows_domain,
    rollout_stage,
)

__all__ = [
    "backend_prefers_jax",
    "is_jax_spec",
    "iter_operator_runtime_candidates",
    "looks_like_jax_identifier",
    "map_selected_ids_to_backend",
    "normalize_backend_token",
    "resolve_available_operator_target",
    "backend_aware_loading_enabled",
    "rollout_stage",
    "rollout_allows_domain",
    "select_specs_for_backend",
    "strip_jax_suffix",
]
