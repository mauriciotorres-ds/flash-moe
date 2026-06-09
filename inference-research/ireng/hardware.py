"""hardware.py — Hardware detection for the Apple M4 target machine."""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Optional

TARGET_SPEC = {
    "machine":           "Apple MacBook (M4)",
    "chip":              "Apple M4",
    "cpu":               "10-core CPU (4P + 6E)",
    "gpu":               "10-core GPU (Metal)",
    "neural_engine":     "16-core ANE",
    "memory_gb":         24,
    "memory_type":       "unified",
    "memory_bandwidth_gbs": 120,
    "os":                "macOS",
    "label":             "Apple M4 · 24 GB unified memory",
    "inference_backend": "llama-cpp-python + Metal (GGUF Q4_K_M)",
}


@dataclass
class HostInfo:
    platform:        str   = ""
    machine:         str   = ""
    processor:       str   = ""
    python:          str   = ""
    cpu_count:       int   = 0
    total_memory_gb: float = 0.0
    llama_version:   Optional[str] = None
    has_metal:       bool  = False
    has_cuda:        bool  = False
    device:          str   = "cpu"
    is_apple_silicon: bool = False
    target_label:    str   = TARGET_SPEC["label"]
    notes:           list  = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _sysctl(key: str) -> Optional[str]:
    try:
        out = subprocess.run(["sysctl", "-n", key],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def detect_host() -> HostInfo:
    info = HostInfo()
    info.platform  = platform.platform()
    info.machine   = platform.machine()
    info.processor = platform.processor() or _sysctl("machdep.cpu.brand_string") or ""
    info.python    = platform.python_version()

    try:
        import psutil
        info.cpu_count       = psutil.cpu_count(logical=True) or 0
        info.total_memory_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info.cpu_count = os.cpu_count() or 0

    try:
        import llama_cpp
        info.llama_version = getattr(llama_cpp, "__version__", "installed")
    except ImportError:
        info.notes.append("llama-cpp-python not installed")

    info.is_apple_silicon = (
        info.machine in ("arm64", "aarch64") and platform.system() == "Darwin"
    )
    info.has_metal = info.is_apple_silicon
    info.device    = "metal" if info.has_metal else "cpu"

    # CUDA check (non-Apple)
    try:
        import torch
        info.has_cuda = torch.cuda.is_available()
        if info.has_cuda:
            info.device = "cuda"
    except Exception:
        pass

    return info


if __name__ == "__main__":
    import json
    print(json.dumps(detect_host().as_dict(), indent=2))
