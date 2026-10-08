from typing import Any

from ..config import Config
from .base import BaseTranscriberEngine
from .gemini import GeminiEngine
from .local_whisper import LocalWhisperEngine

SUPPORTED_ENGINES = ("gemini", "local", "whisper")


def get_engine(
    engine_name: str | None = None,
    config: Config | None = None,
    **kwargs: Any,
) -> BaseTranscriberEngine:
    """
    Factory function to retrieve a transcriber engine.

    Args:
        engine_name: 'gemini' or 'local' (or 'whisper'). Defaults to config.default_engine.
        config: Application configuration.

    Returns:
        BaseTranscriberEngine instance.
    """
    cfg = config or Config()
    name = (engine_name or getattr(cfg, "default_engine", "gemini")).lower()

    if name in ("local", "whisper"):
        return LocalWhisperEngine(config=cfg)
    elif name == "gemini":
        return GeminiEngine(config=cfg)
    else:
        raise ValueError(
            f"Unsupported engine '{engine_name}'. Supported engines are: {list(SUPPORTED_ENGINES)}"
        )
