from typing import Dict, Type

from adapters.base import BaseAdapter
from adapters._qwen import QwenAdapter
from adapters._groq import GroqAdapter
from adapters._gemini import GeminiAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "qwen": QwenAdapter,
    "groq": GroqAdapter,
    "gemini": GeminiAdapter,
}


def get_adapter(model_type: str) -> BaseAdapter:
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Model '{model_type}' is not supported. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_type]()
