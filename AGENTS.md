# AGENTS.md — Agent & Project Guidelines

This file provides context, architectural patterns, environment quirks, and operational guidelines for AI coding assistants working in this repository.

---

## 1. Project Overview & Architecture

`video-transcriber` is a modular Python CLI and library designed to transcribe video/audio files using either:
1. **Cloud Multimodal AI**: Google Gemini (`gemini-2.5-flash` / `gemini-2.5-pro` via `google-genai` SDK) with native visual grounding, speaker diarization, and slide reading.
2. **Local Offline Engine**: `faster-whisper` (CTranslate2) running locally with zero API cost, zero cloud uploads, and low resource overhead.

### Key Directory Structure
```
video-transcriber/
├── src/video_transcriber/
│   ├── config.py              # Environment configuration & CLI defaults
│   ├── transcriber.py         # Unified facade (VideoTranscriber)
│   ├── formatters.py          # Exporters: md, json, srt, txt
│   ├── cli.py                 # CLI argument parsing and batch runner
│   └── engines/
│       ├── base.py            # BaseTranscriberEngine abstract base class
│       ├── factory.py         # Dynamic engine resolver (create_engine)
│       ├── gemini.py          # GeminiEngine implementation
│       └── local_whisper.py   # LocalWhisperEngine implementation (faster-whisper)
├── scripts/
│   └── run_whisper_benchmark.py # Benchmarking tool comparing Whisper vs Gemini transcripts
├── output/                    # Cloud Gemini transcriptions (gitignored)
│   └── whisper/               # Local Whisper transcriptions & benchmark_results.json (gitignored)
└── tests/
    └── test_transcriber.py    # Unit tests
```

---

## 2. Hardware Profile & Local Whisper Defaults

- **Host Environment**: Windows 11, Intel Iris Xe integrated graphics, multi-core Intel CPU.
- **Execution Target for Whisper**:
  - `device="cpu"`
  - `compute_type="int8"`
  - Default model: `"small"` (~480 MB download, ~500 MB RAM usage).
  - Speed: ~5.6x faster than cloud Gemini API round-trip, processing a 5-minute video in ~20–30 seconds on CPU.
- **Do not assume CUDA**: CUDA is not active by default on this host. Avoid forcing `device="cuda"`.

---

## 3. Known Gotchas & Environment Quirks

1. **PyAV >= 14 `metadata_errors` incompatibility**:
   - `faster-whisper` (v1.1.1) passes `metadata_errors="ignore"` to `av.open()`, which was removed in PyAV 14.x.
   - [src/video_transcriber/engines/local_whisper.py](src/video_transcriber/engines/local_whisper.py) patches `av.open` to strip unsupported kwargs before loading audio. Preserve this patch when modifying audio ingestion.
2. **Windows Console Encoding (`cp1252`)**:
   - When printing transcription text or diffs in CLI scripts, always configure stdout or open files with `encoding="utf-8"`:
     ```python
     if sys.platform == "win32":
         sys.stdout.reconfigure(encoding="utf-8")
     ```
3. **Data Safety for Transcriptions**:
   - Never overwrite or delete files in `output/` without explicit user permission.
   - Whisper runs should save under `output/whisper/` or respect `--output-dir`.

---

## 4. Benchmark & Qualitative Baseline

- **Benchmark Script**: `python scripts/run_whisper_benchmark.py`
- **Baseline Metrics (11 videos tested)**:
  - **Speed**: Whisper (`small`, CPU `int8`) took 252.6s (4.2 min) total vs. 1,427.7s (23.8 min) for Gemini (5.6x faster).
  - **Agreement**: Average Word Error Rate (WER) is ~13.6% against Gemini reference.
  - **Qualitative Trade-Off**:
    - Whisper is purely acoustic: mishears technical terms without visual context (*"bird metal"* for *"bare-metal"*, *"ROCAM"* for *"ROCm"*, *"Venn"* for *"Veeam"*).
    - Gemini is multimodal: reads presentation slides and understands technical context, but suffers from upload overhead and rate limits.
  - **Mitigation for Whisper**: Supply an `initial_prompt` with key domain vocabulary (e.g. `ROCm, vLLM, bare-metal, Veeam, Claude MCP, 2FA seeds`) to guide acoustic decoding.

---

## 5. Development & Testing Commands

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run unit tests
python -m unittest discover tests

# Run CLI with Gemini
python -m video_transcriber path/to/video.mp4 --engine gemini --format all

# Run CLI with Local Whisper
python -m video_transcriber path/to/video.mp4 --engine local --model small --format all

# Run comparative benchmark
python scripts/run_whisper_benchmark.py
```
