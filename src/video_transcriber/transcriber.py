from collections.abc import Callable
import logging
from pathlib import Path
import time
from typing import Any

from google import genai
from google.genai import types

from .config import Config
from .formatters import TranscriptionResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}


class TranscriptionError(Exception):
    """Raised when video transcription fails."""


class VideoTranscriber:
    """Handles transcribing local media files using the Gemini API."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.validate()
        self.client = genai.Client(api_key=self.config.api_key)

    def _build_transcription_prompt(
        self,
        include_timestamps: bool = True,
        identify_speakers: bool = True,
        target_format: str = "txt",
        language: str | None = None,
        custom_instructions: str | None = None,
    ) -> str:
        """Construct the prompt sent to Gemini for accurate transcription."""
        if target_format.lower() == "srt":
            instructions = [
                "You are an expert audio/video transcriber.",
                "Generate precise, subtitle-ready transcription in standard SubRip (SRT) format.",
                "Follow standard SRT numbering, timestamp formatting (HH:MM:SS,mmm --> HH:MM:SS,mmm), and text lines.",
                "Capture every spoken word accurately without omissions or paraphrasing.",
                "Do not wrap the output in markdown codeblocks (e.g. ```srt); return ONLY raw valid SRT text.",
            ]
        else:
            instructions = [
                "You are an expert audio/video transcriber.",
                "Please provide an accurate, high-fidelity transcription of the spoken dialogue in this video.",
                "Capture spoken words verbatim, maintaining natural punctuation and sentence breaks.",
            ]

            if identify_speakers:
                instructions.append(
                    "Identify and differentiate different speakers if multiple people are speaking "
                    "(e.g., 'Speaker 1:', 'Speaker 2:', or names if introduced)."
                )

            if include_timestamps:
                instructions.append(
                    "Include concise timestamps at natural paragraph or speaker changes in [MM:SS] or [HH:MM:SS] format."
                )

        if language:
            instructions.append(f"The primary language of the video is: {language}.")
        else:
            instructions.append("Transcribe in the original language spoken in the video.")

        if custom_instructions:
            instructions.append(f"Additional user requirements: {custom_instructions}")

        return "\n".join(instructions)

    def _wait_for_file_active(
        self,
        file_obj: types.File,
        progress_callback: Callable[[str], None] | None = None,
        poll_interval: float = 3.0,
        max_retries: int = 60,
    ) -> types.File:
        """Poll until Gemini file processing changes from PROCESSING to ACTIVE."""
        file_name = file_obj.name
        for _ in range(max_retries):
            current = self.client.files.get(name=file_name)
            state = current.state
            state_name = getattr(state, "name", str(state))

            if state_name == "ACTIVE":
                if progress_callback:
                    progress_callback("File is ready for processing.")
                return current
            elif state_name == "FAILED":
                error_msg = getattr(current, "error", "Unknown error during file processing.")
                raise TranscriptionError(f"Gemini file processing failed: {error_msg}")
            else:
                if progress_callback:
                    progress_callback(f"Gemini is processing video on server (status: {state_name})...")
                time.sleep(poll_interval)

        raise TimeoutError(f"Timeout waiting for uploaded file {file_name} to become active.")

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
    ) -> TranscriptionResult:
        """
        Transcribe a local video file using Gemini API.

        Args:
            file_path: Path to the local video/audio file.
            model: Gemini model name (default: from Config or 'gemini-2.5-flash').
            include_timestamps: Include timestamp marks in output.
            identify_speakers: Diarize/identify speakers.
            target_format: Target format ('txt', 'srt', 'md', 'json').
            language: Optional language hint (e.g. 'English', 'Spanish').
            custom_instructions: Optional user-specified prompt requirements.
            keep_remote_file: Whether to keep the uploaded file in Gemini File API.
            progress_callback: Callback function for status updates.

        Returns:
            TranscriptionResult containing raw transcript and metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "File extension %s is not in standard list %s; attempting upload anyway.",
                path.suffix,
                SUPPORTED_EXTENSIONS,
            )

        model_name = model or self.config.default_model
        uploaded_file: Any = None

        try:
            if progress_callback:
                progress_callback(f"Uploading '{path.name}' ({path.stat().st_size / (1024 * 1024):.1f} MB) to Gemini...")

            uploaded_file = self.client.files.upload(file=path)

            if progress_callback:
                progress_callback("Verifying file processing state...")

            # Wait for video file to be ACTIVE (Gemini processes video before generating content)
            self._wait_for_file_active(uploaded_file, progress_callback=progress_callback)

            prompt = self._build_transcription_prompt(
                include_timestamps=include_timestamps,
                identify_speakers=identify_speakers,
                target_format=target_format,
                language=language,
                custom_instructions=custom_instructions,
            )

            prompt = self._build_transcription_prompt(
                include_timestamps=include_timestamps,
                identify_speakers=identify_speakers,
                target_format=target_format,
                language=language,
                custom_instructions=custom_instructions,
            )

            candidate_models = [model_name]
            for fallback in ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-latest"]:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

            max_retries = 3
            retry_delay = 4.0
            response = None
            last_err = None
            selected_model = model_name

            for candidate in candidate_models:
                selected_model = candidate
                model_succeeded = False
                for attempt in range(1, max_retries + 1):
                    try:
                        if progress_callback:
                            progress_callback(f"Requesting transcription with '{candidate}' (attempt {attempt}/{max_retries})...")

                        response = self.client.models.generate_content(
                            model=candidate,
                            contents=[uploaded_file, prompt],
                        )
                        if response and response.text:
                            model_succeeded = True
                            break
                    except Exception as err:
                        last_err = err
                        err_str = str(err)
                        is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "demand" in err_str
                        if is_transient and attempt < max_retries:
                            if progress_callback:
                                progress_callback(f"Temporary server demand spike (503/429). Retrying in {retry_delay:.0f}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                        else:
                            break

                if model_succeeded:
                    break
                elif progress_callback and len(candidate_models) > 1:
                    progress_callback(f"Model '{candidate}' unavailable. Trying alternative model...")

            if response is None or not response.text:
                raise TranscriptionError(f"All candidate models failed. Last error: {last_err}")

            transcript_text = response.text

            return TranscriptionResult(
                raw_text=transcript_text,
                source_file=path,
                model=selected_model,
                timestamps_included=include_timestamps,
                speakers_identified=identify_speakers,
            )

        finally:
            if uploaded_file and not keep_remote_file:
                try:
                    if progress_callback:
                        progress_callback("Cleaning up temporary file from Gemini server...")
                    self.client.files.delete(name=uploaded_file.name)
                except Exception as exc:
                    logger.warning("Failed to delete remote file %s: %s", uploaded_file.name, exc)
