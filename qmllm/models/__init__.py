import importlib
import os
import sys

from loguru import logger

from qmllm.utils.registry import MODEL_REGISTRY


logger.remove()
logger.add(sys.stdout, level="WARNING")

_LAZY_MODEL_IMPORTS = {
    # model_name in MODEL_REGISTRY -> module path that registers it
    "internvl2": "qmllm.models.internvl2.internvl2",
    "llava_onevision": "qmllm.models.llava_onevision.llava_onevision",
    "qwen2_vl": "qmllm.models.qwen2_vl.qwen2_vl",
    "qwen2_5_vl": "qmllm.models.qwen2_vl.qwen2_vl",
}


def _ensure_model_loaded(model_name: str) -> None:
    if model_name in MODEL_REGISTRY:
        return
    if model_name not in _LAZY_MODEL_IMPORTS:
        raise ValueError(f"Model {model_name} not found in available models.")
    importlib.import_module(_LAZY_MODEL_IMPORTS[model_name])


def get_process_model(model_name):
    _ensure_model_loaded(model_name)
    return MODEL_REGISTRY[model_name]
