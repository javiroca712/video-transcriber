import argparse
from pathlib import Path
import sys
import time

from .config import Config
from .transcriber import VideoTranscriber, TranscriptionError


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-transcriber",
        description="Transcribe local MP4 video/audio files using the Google Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Path to one or more video/audio files to transcribe (e.g., video.mp4)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Gemini model to use (default: GEMINI_MODEL env var or 'gemini-2.5-flash', or use 'gemini-2.5-pro')",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "srt", "md", "json", "all"],
        default="txt",
        help="Output format for the transcription",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where transcription files will be saved",
    )

    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Disable automatic timestamp markers in transcript",
    )

    parser.add_argument(
        "--no-speakers",
        action="store_true",
        help="Disable speaker identification/diarization",
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Hint for primary spoken language (e.g. 'English', 'Spanish')",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Additional custom instructions for the transcription prompt",
    )

    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Keep uploaded file in Gemini Files API instead of auto-deleting",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have existing transcripts in the output directory",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API Key (overrides GEMINI_API_KEY environment variable)",
    )

    return parser


def log_progress(msg: str) -> None:
    print(f"[*] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        config = Config(api_key=args.api_key, output_dir=args.output_dir)
        config.validate()
    except ValueError as err:
        print(f"[!] Error: {err}", file=sys.stderr)
        return 1

    try:
        transcriber = VideoTranscriber(config=config)
    except Exception as exc:
        print(f"[!] Initialization error: {exc}", file=sys.stderr)
        return 1

    formats_to_save = ["txt", "srt", "md", "json"] if args.format == "all" else [args.format]
    success_count = 0
    total_files = len(args.files)

    for idx, file_path in enumerate(args.files, 1):
        print(f"\n[{idx}/{total_files}] Processing: {file_path}")

        if args.skip_existing:
            stem = file_path.stem
            all_exist = all((config.output_dir / f"{stem}.{fmt}").exists() for fmt in formats_to_save)
            if all_exist:
                print(f"[i] Output already exists for '{file_path.name}'. Skipping.")
                success_count += 1
                continue

        start_time = time.time()

        try:
            result = transcriber.transcribe(
                file_path=file_path,
                model=args.model,
                include_timestamps=not args.no_timestamps,
                identify_speakers=not args.no_speakers,
                target_format=args.format if args.format != "all" else "txt",
                language=args.language,
                custom_instructions=args.prompt,
                keep_remote_file=args.keep_file,
                progress_callback=log_progress,
            )

            result.duration_seconds = round(time.time() - start_time, 2)

            for fmt in formats_to_save:
                saved_path = result.save(config.output_dir, format_type=fmt)
                print(f"[+] Saved {fmt.upper()} to: {saved_path}")

            success_count += 1
            print(f"[OK] Completed in {result.duration_seconds}s")

        except FileNotFoundError as err:
            print(f"[!] File error: {err}", file=sys.stderr)
        except TranscriptionError as err:
            print(f"[!] Transcription failed: {err}", file=sys.stderr)
        except Exception as err:
            print(f"[!] Unexpected error processing {file_path}: {err}", file=sys.stderr)

        if idx < total_files:
            time.sleep(3)

    print(f"\nFinished: {success_count}/{total_files} files transcribed successfully.")
    return 0 if success_count == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
