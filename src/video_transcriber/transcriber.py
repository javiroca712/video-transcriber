from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

from .config import Config
from .engines import get_engine, BaseTranscriberEngine
from .formatters import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when video transcription fails."""


class VideoTranscriber:
    """Unified entry point for video and audio transcription."""

    def __init__(
        self,
        config: Config | None = None,
        engine: str | None = None,
    ) -> None:
        self.config = config or Config()
        self.engine_name = engine or self.config.default_engine
        self.config.validate(engine=self.engine_name)
        self.engine: BaseTranscriberEngine = get_engine(self.engine_name, config=self.config)

    def transcribe(
        self,
        file_path: str | Path,
        model: str | None = None,
        include_timestamps: bool = True,
        identify_speakers: bool = True,
        target_format: str = "txt",
        language: str | None = None,
        custom_instructions: str | None = None,
        keep_remote_file: bool = False,
        progress_callback: Callable[[str], None] | None = None,
        engine: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe a media file using either the configured or per-call engine.
        """
        active_engine = self.engine
        if engine and engine.lower() != self.engine_name.lower():
            self.config.validate(engine=engine)
            active_engine = get_engine(engine, config=self.config)

        try:
            return active_engine.transcribe(
                file_path=file_path,
                model=model,
                include_timestamps=include_timestamps,
                identify_speakers=identify_speakers,
                target_format=target_format,
                language=language,
                custom_instructions=custom_instructions,
                progress_callback=progress_callback,
                keep_remote_file=keep_remote_file,
                **kwargs,
            )
        except Exception as err:
            if isinstance(err, (FileNotFoundError, ImportError)):
                raise
            raise TranscriptionError(str(err)) from err
