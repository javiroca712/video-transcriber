"""
Video Transcriber: Multimodal transcription using Google Gemini API or local faster-whisper.
"""

from .config import Config
from .transcriber import VideoTranscriber, TranscriptionError
from .formatters import TranscriptionResult
from .engines import (
    BaseTranscriberEngine,
    GeminiEngine,
    LocalWhisperEngine,
    get_engine,
    SUPPORTED_ENGINES,
)

__all__ = [
    "Config",
    "VideoTranscriber",
    "TranscriptionResult",
    "TranscriptionError",
    "BaseTranscriberEngine",
    "GeminiEngine",
    "LocalWhisperEngine",
    "get_engine",
    "SUPPORTED_ENGINES",
]
__version__ = "0.2.0"
