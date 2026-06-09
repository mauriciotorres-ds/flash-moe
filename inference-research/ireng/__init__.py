"""ireng — MoE Inference Engine Research package (GGUF / llama-cpp-python)."""
from .config import EngineConfig, baseline_config, load_optimized_config, diff_configs
from .engine import LlamaMoEEngine, EngineError
from .metrics import GenerationMetrics, ExpertStats
from . import storage
