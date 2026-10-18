# Video Transcriber (Cloud & Local)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Engines: Gemini + Whisper](https://img.shields.io/badge/engines-Gemini%20%7C%20Whisper-green.svg)](#features)

A modern, modular Python CLI and library to transcribe local video and audio files using either **Google Gemini** (cloud multimodal AI) or **faster-whisper** (100% offline, local on CPU/GPU).

Built with a pluggable engine architecture, this tool lets you seamlessly switch between cloud processing (with Gemini's multimodal speaker diarization and visual grounding) and private local processing (with CTranslate2-quantized Whisper with zero API cost and zero cloud uploads).

---

## Features

- 🔄 **Dual Engine Architecture**: Switch effortlessly between cloud (`gemini`) and offline local (`local` via faster-whisper).
- 🎥 **Broad Media Container Support**: Ingests `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.wav`, `.mp3`, and `.m4a`.
- 🧠 **Gemini Multimodal (Cloud)**: Leverages Gemini 3.5 Flash / Pro with native speaker diarization and automated transient retry on server demand spikes.
- ⚡ **Local Whisper (Offline)**: Runs CTranslate2-accelerated Whisper with `int8` quantization on CPU or `float16` on CUDA with built-in Voice Activity Detection (VAD).
- ⏱️ **Timestamp Markers**: Formatted cleanly (`[00:15]`) for human review and downstream LLM prompting.
- 📝 **Multi-Format Exports**: Save to `.md` (Markdown, recommended for LLMs), `.json` (structured metadata), `.srt` (SubRip subtitles), and `.txt` (plain text), or `--format all`.
- 🛡️ **Resilient Batch Processing**: `--skip-existing` avoids re-transcribing completed files, with automatic retry and cooldown between batch jobs.
- 🔒 **Privacy First**: Local Whisper mode requires zero cloud uploads, zero telemetry, and zero API keys.

---

## Project Structure

```
video-transcriber/
├── .github/
│   └── workflows/
│       └── tests.yml               # Automated CI test suite
├── src/
│   └── video_transcriber/
│       ├── __init__.py             # Exports Config, VideoTranscriber, engines, formatters
│       ├── __main__.py             # python -m video_transcriber entry point
│       ├── cli.py                  # CLI argument parser & batch runner
│       ├── config.py               # Configuration and environment management
│       ├── formatters.py           # Exporters (txt, srt, md, json)
│       ├── transcriber.py          # Unified VideoTranscriber facade
│       └── engines/
│           ├── __init__.py         # Engine exports
│           ├── base.py             # BaseTranscriberEngine interface
│           ├── factory.py          # Dynamic engine factory
│           ├── gemini.py           # Cloud Gemini API engine
│           └── local_whisper.py    # Local faster-whisper engine
├── scripts/
│   └── run_whisper_benchmark.py    # Comparative benchmarking tool
├── tests/
│   ├── __init__.py
│   └── test_transcriber.py         # Unit tests
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules for media, venv, secrets, output
├── CONTRIBUTING.md                 # Guidelines for contributors
├── LICENSE                         # GNU General Public License v3.0
├── pyproject.toml                  # Modern Python packaging & dependencies
├── requirements.txt                # Pinned requirements
└── README.md                       # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12** installed.
- **Git** installed.
- *(Optional)* Free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/apikey) if you want to use the cloud Gemini engine. (Local Whisper requires **no** API key).

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/video-transcriber.git
cd video-transcriber
```

### 3. Create & Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 4. Install Dependencies

Choose the installation option that best fits your workflow:

```bash
# Option A: Full install (Cloud Gemini + Local Whisper) — Recommended
pip install -e .[all]

# Option B: Local Whisper only (Offline, zero API keys required)
pip install -e .[local]

# Option C: Cloud Gemini only
pip install -e .
```

### 5. Configure Environment Variables
Copy the template file to create your local `.env`:

```bash
cp .env.example .env
```

If you plan to use the cloud Gemini engine, edit `.env` and paste your API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If you are only using local Whisper, you can leave `.env` empty or set `TRANSCRIBER_ENGINE=local`.)*

---

## Usage

You can run `video-transcriber` either directly via the CLI or as a Python library.

### 1. CLI Usage

#### Local Offline Transcription (faster-whisper)
Zero cloud uploads, 100% private:

```bash
# Transcribe locally on CPU (default model: small)
video-transcriber path/to/video.mp4 --engine local

# Transcribe with ultra-fast tiny model to Markdown
video-transcriber path/to/video.mp4 --engine local --model tiny --format md

# Transcribe with higher-accuracy model on GPU (if CUDA available)
video-transcriber path/to/video.mp4 --engine local --model medium --device cuda
```

#### Cloud Multimodal Transcription (Gemini)
Requires `GEMINI_API_KEY` set in `.env` or `--api-key`:

```bash
# Basic transcription (saves to output/<name>.txt)
video-transcriber path/to/video.mp4

# Export all formats (.md, .json, .srt, .txt) using Gemini
video-transcriber path/to/video.mp4 --engine gemini --format all
```

#### Batch Processing with Auto-Skip
```bash
# Transcribe all mp4 videos in a directory, skipping files already transcribed
video-transcriber videos/*.mp4 --skip-existing --format all
```

> **Note**: You can also run the CLI via `python -m video_transcriber` instead of `video-transcriber`.

---

## CLI Options Reference

| Flag | Description | Default |
|---|---|---|
| `files` | One or more input media files | *(Required)* |
| `-e`, `--engine` | Engine choice: `gemini` (cloud) or `local` (whisper) | `gemini` |
| `-m`, `--model` | Model name (Gemini: `gemini-3.5-flash`; Local: `tiny`, `base`, `small`, `medium`, `large-v3`) | Engine default |
| `-f`, `--format` | Output format: `txt`, `srt`, `md`, `json`, `all` | `txt` |
| `-o`, `--output-dir` | Directory to save output files | `output` |
| `--device` | Local execution device: `auto`, `cpu`, `cuda` | `auto` |
| `--compute-type` | Local quantization: `auto`, `int8`, `float16`, `float32` | `auto` |
| `--skip-existing` | Skip files that already have transcripts saved in output directory | `False` |
| `--no-timestamps` | Turn off timestamp markers | `False` |
| `--no-speakers` | Turn off speaker identification (Gemini) | `False` |
| `-l`, `--language` | Language hint (e.g. `English`, `Spanish`) | Auto-detect |
| `-p`, `--prompt` | Extra prompt instructions for Gemini | `None` |
| `--keep-file` | Keep remote file on Gemini Files API | `False` (auto-deleted) |
| `--api-key` | Gemini API key override | Uses `GEMINI_API_KEY` |

---

## Python API Usage

You can integrate `video-transcriber` directly into your Python scripts:

```python
from video_transcriber import VideoTranscriber

# 1. Cloud Gemini
gemini_transcriber = VideoTranscriber(engine="gemini")
result = gemini_transcriber.transcribe("meeting.mp4", target_format="md")
result.save("output", format_type="md")

# 2. Local Whisper (Offline)
local_transcriber = VideoTranscriber(engine="local")
result_local = local_transcriber.transcribe(
    "meeting.mp4",
    model="small",
    device="cpu",
    target_format="srt",
)
result_local.save("output", format_type="srt")
```

---

## Benchmarking (Local vs. Cloud)

A benchmark script is included to compare execution speed, Word Error Rate (WER), and similarity between local Whisper and cloud Gemini reference transcripts:

```bash
# Run benchmark on a folder of video files
python scripts/run_whisper_benchmark.py path/to/videos
```

### Benchmark Baseline
Tested across 11 diverse presentation videos on CPU:
- **Speed**: Local Whisper (`small`, CPU `int8`) completed in **252.6s** (~4.2 min) total vs. **1,427.7s** (~23.8 min) for cloud Gemini Flash (a **5.6x speedup**).
- **Word Error Rate (WER)**: ~13.6% average acoustic variance against Gemini.
- **Multimodal Insight**:
  - **Whisper** is acoustic: fast, private, and exact on spoken dialogue, but may mishear technical terms without visual context (*"bird metal"* for *"bare-metal"*, *"ROCAM"* for *"ROCm"*).
  - **Gemini** is multimodal: reads presentation slides and understands visual context, but has upload overhead and rate limits.
  - **Tip for Whisper**: Pass domain terms or names via vocabulary prompts if needed.

---

## Troubleshooting & FAQ

### PyAV & Video Ingestion
- `video-transcriber` includes an automatic compatibility patch for `PyAV >= 14` with `faster-whisper`.
- If you encounter video decoding errors on Linux, ensure `ffmpeg` and `libavcodec-dev` are installed (`sudo apt install ffmpeg`).

### Windows Console Unicode (`cp1252`)
- If special characters or emojis in transcripts produce display errors on Windows PowerShell or CMD, the CLI automatically reconfigures stdout to UTF-8. You can also run `chcp 65001` in your terminal.

### GPU Acceleration
- If you have an NVIDIA GPU and CUDA installed, pass `--device cuda --compute-type float16` to accelerate Whisper processing.

---

## Running Tests

To run the unit test suite:

```bash
python -m unittest discover tests
```

---

## Contributing

Contributions, bug reports, and suggestions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up your environment and creating pull requests.

For AI assistant conventions and development notes, see [AGENTS.md](AGENTS.md).

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0-or-later). See the [LICENSE](LICENSE) file for the full text.
