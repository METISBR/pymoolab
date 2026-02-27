from __future__ import annotations

from typing import Any


def detect_gpu_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "jax_ok": False,
        "jax_version": None,
        "cuda_ok": False,
        "cuda_device_name": None,
        "device_count": 0,
        "all_devices": [],
        "error": None,
    }
    try:
        import jax
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"JAX not available: {exc}"
        return info

    info["jax_ok"] = True
    info["jax_version"] = str(getattr(jax, "__version__", "unknown"))
    try:
        devices = list(jax.devices())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not query JAX devices: {exc}"
        return info

    info["device_count"] = len(devices)
    info["all_devices"] = [f"{getattr(d, 'platform', '?')}:{getattr(d, 'device_kind', d)}" for d in devices]

    gpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() in {"gpu", "cuda", "rocm"}]
    if gpu_devices:
        info["cuda_ok"] = True
        info["cuda_device_name"] = str(getattr(gpu_devices[0], "device_kind", gpu_devices[0]))

    return info


def build_gpu_status_text(info: dict[str, Any]) -> str:
    if info.get("cuda_ok"):
        return f"JAX GPU available: {info.get('cuda_device_name', 'GPU device')}"
    if info.get("jax_ok"):
        return "JAX available, but no GPU device detected. CPU execution will be used."
    if info.get("error"):
        return str(info.get("error"))
    return "JAX runtime unavailable. CPU execution will be used."

