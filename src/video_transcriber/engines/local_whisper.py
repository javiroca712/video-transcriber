from collections.abc import Callable
import logging
from pathlib import Path
import time
from typing import Any

from ..config import Config
from ..formatters import TranscriptionResult
from .base import BaseTranscriberEngine

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}

# In-memory model cache to avoid reloading weights between batch files
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

# Compatibility patch for PyAV >= 14 with faster-whisper
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except ImportError:
    pass


def _format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format seconds into [HH:MM:SS] or SRT timestamp format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))

    if srt_format:
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

    if hrs > 0:
        return f"[{hrs:02d}:{mins:02d}:{secs:02d}]"
    return f"[{mins:02d}:{secs:02d}]"


class LocalWhisperEngine(BaseTranscriberEngine):
    """Local offline transcription engine using faster-whisper (CTranslate2)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def _get_model(
        self,
        model_size: str,
        device: str = "auto",
        compute_type: str = "auto",
        progress_callback: Callable[[str], None] | None = None,
    ) -> Any:
        try:
            from faster_whisper import WhisperModel
            import ctranslate2
        except ImportError as err:
            raise ImportError(
                "Local transcription requires 'faster-whisper' and 'av'. "
                "Install them with: pip install faster-whisper av (or pip install -e .[local])"
            ) from err

        if device == "auto":
            try:
                cuda_available = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                cuda_available = False
            device = "cuda" if cuda_available else "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        cache_key = (model_size, device, compute_type)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if progress_callback:
            progress_callback(
                f"Loading local Whisper model '{model_size}' (device: {device}, compute: {compute_type})..."
            )

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _MODEL_CACHE[cache_key] = model
        return model

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
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        **kwargs: Any,
    ) -> TranscriptionResult:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        model_size = model or getattr(self.config, "whisper_model", "small")
        whisper_model = self._get_model(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(f"Transcribing '{path.name}' locally with Whisper ({model_size})...")

        start_time = time.time()

        # Run transcription with Voice Activity Detection (VAD) filter to skip silence
        segments_generator, info = whisper_model.transcribe(
            str(path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = list(segments_generator)
        duration = round(time.time() - start_time, 2)

        if progress_callback:
            detected_lang = getattr(info, "language", language or "unknown")
            progress_callback(
                f"Transcription complete (detected language: '{detected_lang}', duration: {duration}s)."
            )

        lines: list[str] = []
        if target_format.lower() == "srt":
            for idx, seg in enumerate(segments, 1):
                start_str = _format_timestamp(seg.start, srt_format=True)
                end_str = _format_timestamp(seg.end, srt_format=True)
                text = seg.text.strip()
                lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            raw_text = "\n".join(lines).strip()
        else:
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if include_timestamps:
                    ts = _format_timestamp(seg.start)
                    lines.append(f"**{ts}** {text}")
                else:
                    lines.append(text)
            raw_text = "\n\n".join(lines).strip()

        return TranscriptionResult(
            raw_text=raw_text,
            source_file=path,
            model=f"whisper-{model_size}",
            duration_seconds=duration,
            timestamps_included=include_timestamps,
            speakers_identified=False,  # Whisper is acoustic ASR without diarization
        )
