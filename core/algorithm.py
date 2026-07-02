"""Legacy Algorithm compatibility adapter (pymoolab 2026)."""

from __future__ import annotations

from typing import Any

from pymoo.core.algorithm import Algorithm as PymooAlgorithm

from util import array_backend as _array_backend


class Algorithm(PymooAlgorithm):
    """Compatibility layer used by local algorithms ported from MATLAB frameworks."""

    def __init__(
        self,
        *args: Any,
        use_gpu: bool = False,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        self.use_gpu_requested = bool(use_gpu)
        self.array_backend_requested = str(array_backend).strip().lower() or "auto"
        self.gpu_dtype = str(gpu_dtype).strip().lower() or "float32"

        from core.execution.backend_runtime import apple_silicon_available, detect_mlx_runtime
        
        req = self.array_backend_requested
        if req == "auto":
            req = "mlx" if apple_silicon_available() else ("jax" if self.use_gpu_requested else "numpy")
            
        self.array_backend_effective = req
        
        if req == "mlx":
            info = detect_mlx_runtime()
            if not info.get("mlx_ok"):
                self.array_backend_effective = "numpy"

        self.array_backend = self.array_backend_effective
        self.use_gpu = self.array_backend_effective in {"jax", "mlx"}

        self.backend_state = {
            "requested_backend": self.array_backend_requested,
            "effective_backend": self.array_backend_effective,
            "use_gpu": self.use_gpu,
            "gpu_dtype": self.gpu_dtype,
        }
        super().__init__(*args, **kwargs)

    def get_array_module(self) -> Any:
        if self.array_backend_effective == "mlx":
            try:
                import mlx.core as mx
                return mx
            except ImportError:
                pass
        return _array_backend.xp


__all__ = ["Algorithm"]

