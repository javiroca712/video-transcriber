from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..formatters import TranscriptionResult


class BaseTranscriberEngine(ABC):
    """Abstract base class for transcription engines."""

    @abstractmethod
    def transcribe(
        self,
        file_path: str | Path,
        model: str | None = None,
        include_timestamps: bool = True,
        identify_speakers: bool = True,
        target_format: str = "txt",
        language: str | None = None,
        custom_instructions: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe a media file and return a normalized TranscriptionResult.
        """
        pass
