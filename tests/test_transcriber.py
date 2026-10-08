import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_transcriber.config import Config
from video_transcriber.cli import create_parser, main
from video_transcriber.formatters import TranscriptionResult


class TestVideoTranscriber(unittest.TestCase):
    def test_config_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            config = Config(api_key=None)
            with self.assertRaises(ValueError):
                config.validate()

    def test_config_with_key(self):
        config = Config(api_key="test-api-key")
        config.validate()
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.default_model, "gemini-3.5-flash")

    def test_cli_parser(self):
        parser = create_parser()
        args = parser.parse_args(["sample.mp4", "-m", "gemini-2.5-pro", "-f", "srt"])
        self.assertEqual(args.files, [Path("sample.mp4")])
        self.assertEqual(args.model, "gemini-2.5-pro")
        self.assertEqual(args.format, "srt")

    def test_transcription_result_save_txt(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = TranscriptionResult(
                raw_text="Hello world transcription test",
                source_file=Path("test_video.mp4"),
                model="gemini-2.5-flash",
            )
            saved = result.save(out_dir, format_type="txt")
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_text(encoding="utf-8").strip(), "Hello world transcription test")

    def test_transcription_result_save_md(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = TranscriptionResult(
                raw_text="Markdown test transcript",
                source_file=Path("test_video.mp4"),
                model="gemini-2.5-pro",
            )
            saved = result.save(out_dir, format_type="md")
            self.assertTrue(saved.exists())
            content = saved.read_text(encoding="utf-8")
            self.assertIn("# Transcription: test_video.mp4", content)
            self.assertIn("Markdown test transcript", content)


if __name__ == "__main__":
    unittest.main()
