import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_transcriber.config import Config
from video_transcriber.cli import create_parser, main
from video_transcriber.formatters import TranscriptionResult
from video_transcriber.engines import get_engine, GeminiEngine, LocalWhisperEngine


class TestVideoTranscriber(unittest.TestCase):
    def test_config_missing_key_gemini(self):
        with patch.dict("os.environ", {}, clear=True):
            config = Config(api_key=None)
            with self.assertRaises(ValueError):
                config.validate(engine="gemini")

    def test_config_missing_key_local_allowed(self):
        with patch.dict("os.environ", {}, clear=True):
            config = Config(api_key=None)
            # Should not raise for local engine
            config.validate(engine="local")

    def test_config_with_key(self):
        config = Config(api_key="test-api-key")
        config.validate(engine="gemini")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.default_model, "gemini-3.5-flash")

    def test_cli_parser(self):
        parser = create_parser()
        args = parser.parse_args([
            "sample.mp4",
            "-e", "local",
            "-m", "small",
            "-f", "srt",
            "--device", "cpu",
            "--compute-type", "int8",
        ])
        self.assertEqual(args.files, [Path("sample.mp4")])
        self.assertEqual(args.engine, "local")
        self.assertEqual(args.model, "small")
        self.assertEqual(args.format, "srt")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.compute_type, "int8")

    def test_engine_factory(self):
        config = Config(api_key="fake-key")
        local_eng = get_engine("local", config=config)
        self.assertIsInstance(local_eng, LocalWhisperEngine)

        gemini_eng = get_engine("gemini", config=config)
        self.assertIsInstance(gemini_eng, GeminiEngine)

        with self.assertRaises(ValueError):
            get_engine("unsupported-engine", config=config)

    def test_transcription_result_save_txt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = TranscriptionResult(
                raw_text="Hello world transcription test",
                source_file=Path("test_video.mp4"),
                model="whisper-small",
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
                model="gemini-3.5-flash",
            )
            saved = result.save(out_dir, format_type="md")
            self.assertTrue(saved.exists())
            content = saved.read_text(encoding="utf-8")
            self.assertIn("# Transcription: test_video.mp4", content)
            self.assertIn("Markdown test transcript", content)

    def test_clean_text_and_wer_calculation(self):
        from scripts.run_whisper_benchmark import clean_text_for_comparison, calculate_wer

        gemini_sample = "# Transcription\n**[00:15]** Here is a transcription: Hello world this is a test"
        whisper_sample = "[00:15] Hello world this is another test"

        clean_g = clean_text_for_comparison(gemini_sample)
        clean_w = clean_text_for_comparison(whisper_sample)

        self.assertEqual(clean_g, "hello world this is a test")
        self.assertEqual(clean_w, "hello world this is another test")

        ref = clean_g.split()
        hyp = clean_w.split()
        wer = calculate_wer(ref, hyp)
        self.assertGreater(wer, 0.0)
        self.assertLess(wer, 1.0)
        self.assertEqual(calculate_wer(ref, ref), 0.0)


if __name__ == "__main__":
    unittest.main()

