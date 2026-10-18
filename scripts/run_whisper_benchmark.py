import argparse
import difflib
import json
import os
from pathlib import Path
import re
import sys
import time

from video_transcriber.config import Config
from video_transcriber.engines import LocalWhisperEngine

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SUPPORTED_MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a"}


def clean_text_for_comparison(text: str) -> str:
    """Normalize text by stripping markdown headers, timestamps, and punctuation."""
    lines = []
    for line in text.splitlines():
        if line.startswith("# ") or line.startswith("- **") or line.strip() == "---":
            continue
        lines.append(line)
    clean = "\n".join(lines)
    # Remove timestamps like [00:15] or **[00:15]**
    clean = re.sub(r"\*?\*?\[\d{2}:\d{2}(?::\d{2})?\]\*?\*?", "", clean)
    # Remove common prefix phrases from Gemini
    clean = re.sub(r"Here is a [^\n]*transcription[^\n]*:\s*", "", clean, flags=re.IGNORECASE)
    # Normalize whitespace and lowercase
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    return clean


def calculate_wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Deletion
                    d[i][j - 1] + 1,      # Insertion
                    d[i - 1][j - 1] + 1,  # Substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Whisper transcription against Gemini transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_video_dir = os.getenv("BENCHMARK_VIDEO_DIR", "samples")
    parser.add_argument(
        "video_dir",
        nargs="?",
        type=Path,
        default=Path(default_video_dir),
        help="Directory containing video/audio files to benchmark",
    )
    parser.add_argument(
        "--whisper-output-dir",
        type=Path,
        default=Path("output/whisper"),
        help="Directory to save Whisper transcription outputs",
    )
    parser.add_argument(
        "--gemini-output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing reference Gemini transcription JSON files",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="small",
        help="Whisper model size to benchmark (e.g. 'tiny', 'base', 'small', 'medium', 'large-v3')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Local execution device ('cpu', 'cuda', 'auto')",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Quantization type ('int8', 'float16', 'float32', 'auto')",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dir: Path = args.video_dir
    whisper_output_dir: Path = args.whisper_output_dir
    gemini_output_dir: Path = args.gemini_output_dir

    if not video_dir.exists():
        print(f"[!] Video directory not found: '{video_dir}'\n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  python scripts/run_whisper_benchmark.py path/to/videos", file=sys.stderr)
        print("  Or set the BENCHMARK_VIDEO_DIR environment variable.", file=sys.stderr)
        return 1

    video_files = sorted([
        f for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ])

    if not video_files:
        print(f"[!] No supported media files ({', '.join(sorted(SUPPORTED_MEDIA_EXTENSIONS))}) found in '{video_dir}'.", file=sys.stderr)
        return 1

    whisper_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(video_files)} video(s) to transcribe and benchmark in '{video_dir}'.")
    print(f"Saving Whisper outputs to: {whisper_output_dir.resolve()}\n")

    config = Config(
        output_dir=whisper_output_dir,
        whisper_model=args.model,
        whisper_device=args.device,
        whisper_compute_type=args.compute_type,
    )
    engine = LocalWhisperEngine(config=config)

    benchmark_records = []

    for idx, video_path in enumerate(video_files, 1):
        stem = video_path.stem
        gemini_json_path = gemini_output_dir / f"{stem}.json"
        gemini_data = {}
        if gemini_json_path.exists():
            try:
                gemini_data = json.loads(gemini_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[!] Warning reading Gemini reference data for {stem}: {e}")

        print(f"[{idx}/{len(video_files)}] Processing with Whisper ({args.model}): {video_path.name}")
        file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
        start_time = time.time()

        try:
            whisper_result = engine.transcribe(
                file_path=video_path,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                include_timestamps=True,
                identify_speakers=False,
                target_format="txt",
                progress_callback=lambda msg: print(f"  [*] {msg}", flush=True),
            )
            elapsed_time = round(time.time() - start_time, 2)
            whisper_result.duration_seconds = elapsed_time

            # Save Whisper formats into whisper_output_dir
            for fmt in ["txt", "srt", "md", "json"]:
                whisper_result.save(whisper_output_dir, format_type=fmt)
            print(f"  [+] Saved outputs to {whisper_output_dir / stem}.[txt,srt,md,json]")

            # Compare against Gemini reference if available
            gemini_raw = gemini_data.get("transcript", "")
            gemini_time = gemini_data.get("duration_seconds", None)

            clean_gemini = clean_text_for_comparison(gemini_raw)
            clean_whisper = clean_text_for_comparison(whisper_result.raw_text)

            gemini_words = clean_gemini.split()
            whisper_words = clean_whisper.split()

            seq_matcher = difflib.SequenceMatcher(None, clean_gemini, clean_whisper)
            similarity = round(seq_matcher.ratio() * 100, 2)

            wer = round(calculate_wer(gemini_words, whisper_words) * 100, 2) if gemini_words else None

            record = {
                "filename": video_path.name,
                "stem": stem,
                "file_size_mb": file_size_mb,
                "whisper_duration_s": elapsed_time,
                "gemini_duration_s": gemini_time,
                "speedup_factor": round(gemini_time / elapsed_time, 2) if (gemini_time and elapsed_time > 0) else None,
                "gemini_word_count": len(gemini_words),
                "whisper_word_count": len(whisper_words),
                "text_similarity_percent": similarity if gemini_raw else None,
                "wer_percent": wer,
                "whisper_sample": whisper_result.raw_text[:200] + "...",
                "gemini_sample": gemini_raw[:200] + "..." if gemini_raw else "N/A",
            }
            benchmark_records.append(record)

            comparison_str = f"(Similarity to Gemini: {similarity}%, WER: {wer}%)" if gemini_raw else "(No Gemini reference found)"
            print(f"  [OK] Completed in {elapsed_time}s {comparison_str}\n")

        except Exception as exc:
            print(f"  [!] Failed {video_path.name}: {exc}\n", file=sys.stderr)

    benchmark_json_path = whisper_output_dir / "benchmark_results.json"
    benchmark_json_path.write_text(json.dumps(benchmark_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Benchmark dataset saved to: {benchmark_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
