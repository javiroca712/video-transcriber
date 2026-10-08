import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    api_key: str | None = None
    default_engine: str = "gemini"
    default_model: str = "gemini-3.5-flash"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    output_dir: Path = Path("output")

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")

        env_engine = os.getenv("TRANSCRIBER_ENGINE") or os.getenv("DEFAULT_ENGINE")
        if env_engine:
            self.default_engine = env_engine

        env_model = os.getenv("GEMINI_MODEL")
        if env_model:
            self.default_model = env_model

        env_whisper_model = os.getenv("WHISPER_MODEL")
        if env_whisper_model:
            self.whisper_model = env_whisper_model

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def validate(self, engine: str = "gemini") -> None:
        """Validate configuration settings for the active engine."""
        if engine.lower() == "gemini" and not self.api_key:
            raise ValueError(
                "Gemini API key is not set. Please set the GEMINI_API_KEY environment variable "
                "or define it in a .env file. You can obtain a key at https://aistudio.google.com/apikey"
            )
