import difflib
import json
from pathlib import Path
import re
import sys
import time

from video_transcriber.config import Config
from video_transcriber.engines import LocalWhisperEngine

VIDEO_DIR = Path("samples")
GEMINI_OUTPUT_DIR = Path("output")
WHISPER_OUTPUT_DIR = Path("output/whisper")


def clean_text_for_comparison(text: str) -> str:
    """Normalize text by stripping markdown headers, timestamps, and punctuation."""
    # Remove markdown headers and metadata lines
    lines = []
    for line in text.splitlines():
        if line.startswith("# ") or line.startswith("- **") or line.strip() == "---":
            continue
        lines.append(line)
    clean = "\n".join(lines)
    # Remove timestamps like [00:15] or **[00:15]**
    clean = re.sub(r"\*?\*?\[\d{2}:\d{2}(?::\d{2})?\]\*?\*?", "", clean)
    # Remove common prefix phrases from Gemini
    clean = re.sub(r"Here is a [^\n]+transcription[^\n]*:\s*", "", clean, flags=re.IGNORECASE)
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


def main():
    WHISPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_files = sorted(list(VIDEO_DIR.glob("*.mp4")))

    if not video_files:
        print(f"[!] No mp4 files found in {VIDEO_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(video_files)} videos to transcribe and benchmark.")
    print(f"Saving Whisper outputs to: {WHISPER_OUTPUT_DIR.resolve()}\n")

    config = Config(output_dir=WHISPER_OUTPUT_DIR, whisper_model="small")
    engine = LocalWhisperEngine(config=config)

    benchmark_records = []

    for idx, video_path in enumerate(video_files, 1):
        stem = video_path.stem
        gemini_json_path = GEMINI_OUTPUT_DIR / f"{stem}.json"
        gemini_data = {}
        if gemini_json_path.exists():
            try:
                gemini_data = json.loads(gemini_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[!] Warning reading Gemini data for {stem}: {e}")

        print(f"[{idx}/{len(video_files)}] Processing with Whisper (small): {video_path.name}")
        file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
        start_time = time.time()

        try:
            whisper_result = engine.transcribe(
                file_path=video_path,
                model="small",
                include_timestamps=True,
                identify_speakers=False,
                target_format="txt",
                progress_callback=lambda msg: print(f"  [*] {msg}", flush=True),
            )
            elapsed_time = round(time.time() - start_time, 2)
            whisper_result.duration_seconds = elapsed_time

            # Save Whisper formats into output/whisper/
            saved_paths = {}
            for fmt in ["txt", "srt", "md", "json"]:
                saved = whisper_result.save(WHISPER_OUTPUT_DIR, format_type=fmt)
                saved_paths[fmt] = str(saved)
            print(f"  [+] Saved outputs to {WHISPER_OUTPUT_DIR / stem}.[txt,srt,md,json]")

            # Compare against Gemini reference
            gemini_raw = gemini_data.get("transcript", "")
            gemini_time = gemini_data.get("duration_seconds", None)

            clean_gemini = clean_text_for_comparison(gemini_raw)
            clean_whisper = clean_text_for_comparison(whisper_result.raw_text)

            gemini_words = clean_gemini.split()
            whisper_words = clean_whisper.split()

            seq_matcher = difflib.SequenceMatcher(None, clean_gemini, clean_whisper)
            similarity = round(seq_matcher.ratio() * 100, 2)

            wer = round(calculate_wer(gemini_words, whisper_words) * 100, 2)

            record = {
                "filename": video_path.name,
                "stem": stem,
                "file_size_mb": file_size_mb,
                "whisper_duration_s": elapsed_time,
                "gemini_duration_s": gemini_time,
                "speedup_factor": round(gemini_time / elapsed_time, 2) if (gemini_time and elapsed_time > 0) else None,
                "gemini_word_count": len(gemini_words),
                "whisper_word_count": len(whisper_words),
                "text_similarity_percent": similarity,
                "wer_percent": wer,
                "whisper_sample": whisper_result.raw_text[:200] + "...",
                "gemini_sample": gemini_raw[:200] + "..." if gemini_raw else "N/A",
            }
            benchmark_records.append(record)
            print(f"  [OK] Completed in {elapsed_time}s (Similarity to Gemini: {similarity}%, WER: {wer}%)\n")

        except Exception as exc:
            print(f"  [!] Failed {video_path.name}: {exc}\n", file=sys.stderr)

    benchmark_json_path = WHISPER_OUTPUT_DIR / "benchmark_results.json"
    benchmark_json_path.write_text(json.dumps(benchmark_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Benchmark dataset saved to: {benchmark_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
