from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TranscriptionResult:
    """Represents a completed transcription."""

    raw_text: str
    source_file: Path
    model: str
    duration_seconds: float | None = None
    timestamps_included: bool = False
    speakers_identified: bool = False

    def to_dict(self) -> dict:
        return {
            "source_file": str(self.source_file),
            "model": self.model,
            "duration_seconds": self.duration_seconds,
            "timestamps_included": self.timestamps_included,
            "speakers_identified": self.speakers_identified,
            "transcript": self.raw_text,
        }

    def save(self, output_path: Path, format_type: str = "txt") -> Path:
        """
        Save the transcription to disk.
        
        Args:
            output_path: Target file path or directory.
            format_type: One of 'txt', 'md', 'json', 'srt'.
        """
        # If output_path is an existing directory or has no file extension, treat as folder
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            stem = self.source_file.stem
            target = output_path / f"{stem}.{format_type}"
        else:
            target = output_path
            target.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        elif format_type == "md":
            content = f"# Transcription: {self.source_file.name}\n\n"
            content += f"- **Model**: `{self.model}`\n"
            content += f"- **Source**: `{self.source_file.resolve()}`\n\n"
            content += "---\n\n"
            content += self.raw_text.strip() + "\n"
        else:
            # txt, srt, or raw text
            content = self.raw_text.strip() + "\n"

        target.write_text(content, encoding="utf-8")
        return target
