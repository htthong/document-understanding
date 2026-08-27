from typing import Dict, Type

from adapters.base import BaseAdapter
from adapters.qwen import QwenAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "qwen": QwenAdapter,
}


def get_adapter(model_type: str) -> BaseAdapter:
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Model '{model_type}' is not supported. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_type]()
