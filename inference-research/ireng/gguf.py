"""gguf.py — Minimal GGUF file reader.

Reads the header, metadata key-value pairs, and tensor info from a GGUF file
without loading any tensor data.  Used to discover the MoE architecture
(n_layers, n_experts, top_k, hidden_size, etc.) and to locate router weight
tensors so expert selection can be tracked.
"""
from __future__ import annotations

import struct
import os
from dataclasses import dataclass, field
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF"

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q4_K_M", 13: "Q5_K_M",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}

# GGUF metadata value type IDs
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int  # byte offset from data-section start

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"type_{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GGUFMeta:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    data_offset: int  # absolute byte offset where tensor data begins

    # ── Architecture helpers ──────────────────────────────────────────────────

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def n_layers(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.block_count", 0))

    @property
    def n_experts(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.expert_count", 0))

    @property
    def n_experts_used(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.expert_used_count", 0))

    @property
    def hidden_size(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.embedding_length", 0))

    @property
    def n_heads(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.attention.head_count", 0))

    @property
    def n_kv_heads(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.attention.head_count_kv", 0))

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("general.name", os.path.basename(self.path)))

    @property
    def quantization(self) -> str:
        """Best-effort quant level from tensor types."""
        for t in self.tensors.values():
            name = t.type_name
            if "K_M" in name or "K_S" in name:
                return name
        return "unknown"

    # ── Tensor name helpers ───────────────────────────────────────────────────

    def router_tensor_name(self, layer: int) -> str:
        return f"blk.{layer}.ffn_gate_inp.weight"

    def has_router(self, layer: int) -> bool:
        return self.router_tensor_name(layer) in self.tensors

    @property
    def moe_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.has_router(i)]

    def summary(self) -> dict:
        return {
            "path": self.path,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "n_experts_used": self.n_experts_used,
            "hidden_size": self.hidden_size,
            "is_moe": self.is_moe,
            "moe_layers": len(self.moe_layers),
            "quantization": self.quantization,
            "n_tensors": len(self.tensors),
            "file_size_gb": round(os.path.getsize(self.path) / 1024**3, 2),
        }


# ── Reader ────────────────────────────────────────────────────────────────────

def read_gguf_meta(path: str) -> GGUFMeta:
    """Read GGUF metadata and tensor layout without loading tensor data."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#010x}): {path}")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            metadata[key] = _read_value(f, vtype)

        tensors: dict[str, TensorInfo] = {}
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = TensorInfo(name=name, dims=list(dims),
                                       ggml_type=ggml_type, offset=offset)

        # Data section begins at the next 32-byte-aligned position
        pos = f.tell()
        data_offset = (pos + 31) & ~31

    return GGUFMeta(path=path, version=version, metadata=metadata,
                    tensors=tensors, data_offset=data_offset)


def find_gguf(directory: str, prefer: str = "Q4_K_M") -> str | None:
    """Return the path to the best GGUF file in *directory*."""
    if not os.path.isdir(directory):
        return None
    candidates = [f for f in os.listdir(directory) if f.endswith(".gguf")]
    if not candidates:
        return None
    # Prefer files whose name contains the preferred quant string
    preferred = [f for f in candidates if prefer.lower() in f.lower()]
    chosen = preferred[0] if preferred else candidates[0]
    return os.path.join(directory, chosen)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == _UINT8:   return struct.unpack("<B", f.read(1))[0]
    if vtype == _INT8:    return struct.unpack("<b", f.read(1))[0]
    if vtype == _UINT16:  return struct.unpack("<H", f.read(2))[0]
    if vtype == _INT16:   return struct.unpack("<h", f.read(2))[0]
    if vtype == _UINT32:  return struct.unpack("<I", f.read(4))[0]
    if vtype == _INT32:   return struct.unpack("<i", f.read(4))[0]
    if vtype == _FLOAT32: return struct.unpack("<f", f.read(4))[0]
    if vtype == _BOOL:    return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == _STRING:  return _read_string(f)
    if vtype == _UINT64:  return struct.unpack("<Q", f.read(8))[0]
    if vtype == _INT64:   return struct.unpack("<q", f.read(8))[0]
    if vtype == _FLOAT64: return struct.unpack("<d", f.read(8))[0]
    if vtype == _ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")
