"""Hardware detection and target-machine specification.

The *target* machine for this project is an **Apple M4 with 24 GB unified
memory** (MPS backend). All plots and reports are labelled with this spec.

At runtime the engine auto-detects the actual host so the same code runs on
the dev box, CI, or the target Mac. Detected facts always take priority over
the declared target spec for live metrics; the target spec is only used as a
label / fallback when a detail can't be probed (e.g. GPU utilisation on MPS,
which Apple does not expose through a public counter).
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Optional

# ---------------------------------------------------------------------------
# Declared target machine. Edit here if you move to different hardware.
# ---------------------------------------------------------------------------
TARGET_SPEC = {
    "machine": "Apple MacBook (M4)",
    "chip": "Apple M4",
    "cpu": "10-core CPU (4P + 6E)",      # base M4 layout
    "gpu": "10-core GPU",
    "neural_engine": "16-core ANE",
    "memory_gb": 24,
    "memory_type": "unified",
    "memory_bandwidth_gbs": 120,          # base M4 ~120 GB/s
    "os": "macOS",
    "label": "Apple M4 · 24 GB unified memory",
}


@dataclass
class HostInfo:
    """Facts probed from the machine the code is actually running on."""
    platform: str = ""
    machine: str = ""
    processor: str = ""
    python: str = ""
    cpu_count: int = 0
    total_memory_gb: float = 0.0
    torch_version: Optional[str] = None
    has_mps: bool = False
    has_cuda: bool = False
    device: str = "cpu"
    is_apple_silicon: bool = False
    target_label: str = TARGET_SPEC["label"]
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _sysctl(key: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def detect_host() -> HostInfo:
    info = HostInfo()
    info.platform = platform.platform()
    info.machine = platform.machine()
    info.processor = platform.processor() or _sysctl("machdep.cpu.brand_string") or ""
    info.python = platform.python_version()

    # CPU / memory
    try:
        import psutil

        info.cpu_count = psutil.cpu_count(logical=True) or 0
        info.total_memory_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        import os

        info.cpu_count = os.cpu_count() or 0

    # torch / accelerators
    try:
        import torch

        info.torch_version = torch.__version__
        info.has_cuda = torch.cuda.is_available()
        info.has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        info.notes.append("torch not importable; engines will not run here")

    info.is_apple_silicon = info.machine in ("arm64", "aarch64") and platform.system() == "Darwin"

    if info.has_cuda:
        info.device = "cuda"
    elif info.has_mps:
        info.device = "mps"
    else:
        info.device = "cpu"

    return info


def best_device(prefer: str = "auto") -> str:
    """Resolve a device string. prefer in {auto, mps, cuda, cpu}."""
    host = detect_host()
    if prefer != "auto":
        return prefer
    return host.device


def gpu_utilization(device: str) -> Optional[float]:
    """Best-effort GPU utilisation percentage.

    - CUDA: read via torch / nvidia-ml if available.
    - MPS (Apple): Apple does not expose a public per-process GPU-busy counter,
      so this returns None and callers should display "n/a (MPS)".
    """
    if device == "cuda":
        try:
            import torch

            # torch.cuda.utilization needs pynvml; guard it.
            return float(torch.cuda.utilization())
        except Exception:
            return None
    return None  # MPS / CPU: not available


if __name__ == "__main__":
    import json

    print("TARGET_SPEC:")
    print(json.dumps(TARGET_SPEC, indent=2))
    print("\nDETECTED HOST:")
    print(json.dumps(detect_host().as_dict(), indent=2))
