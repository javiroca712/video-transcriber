# Video Transcriber (Cloud & Local)

A modern, modular Python tool to transcribe local MP4 (and other audio/video) files using either the **Google Gemini API** (cloud multimodal) or **faster-whisper** (100% offline, local).

Built with a pluggable engine architecture, this tool lets you seamlessly switch between cloud processing (with Gemini's multimodal speaker diarization and visual grounding) and private local processing (with quantized Whisper on CPU/GPU).

---

## Features

- 🔄 **Dual Engine Architecture**: Switch effortlessly between cloud (`gemini`) and offline local (`local` via faster-whisper).
- 🎥 **Native Video Container Support**: Ingests `.mp4`, `.mov`, `.mkv`, `.webm`, `.wav`, `.mp3`, and `.m4a`.
- 🧠 **Gemini Multimodal (Cloud)**: Leverages Gemini 3.5 Flash / Pro with native speaker diarization and automated transient retry on server demand spikes.
- ⚡ **Local Whisper (Offline)**: Runs CTranslate2-accelerated Whisper with `int8` quantization on CPU or `float16` on CUDA with built-in Voice Activity Detection (VAD).
- ⏱️ **Timestamps**: Timestamp markers (`[00:15]`) formatted cleanly for human review and downstream LLM prompting.
- 📝 **Multi-Format Exports**: Save to `.md` (Markdown, recommended for LLMs), `.json` (structured metadata), `.srt` (SubRip subtitles), and `.txt` (plain text), or `--format all`.
- 🛡️ **Resilient Batch Processing**: `--skip-existing` avoids re-transcribing completed files, with automatic retry and cooldown between batch jobs.

---

## Project Structure

```
video-transcriber/
├── .venv/                          # Virtual environment
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
├── tests/
│   ├── __init__.py
│   └── test_transcriber.py         # Unit tests
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore rules for media, venv, secrets, output
├── pyproject.toml                  # Modern Python packaging & dependencies
├── requirements.txt                # Pinned requirements
└── README.md                       # Documentation
```

---

## Installation & Setup

### 1. Virtual Environment Activation

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 2. Dependencies
- **Cloud Gemini only**: `pip install -e .`
- **Cloud + Local Whisper**: `pip install -e .[local]`

---

## Usage

### 1. Cloud Transcription (Gemini)
Requires `GEMINI_API_KEY` in `.env` or passed via `--api-key`:

```bash
# Basic transcription (saves to output/<name>.txt)
video-transcriber path/to/video.mp4

# Export all formats (.md, .json, .srt, .txt) using Gemini
video-transcriber path/to/video.mp4 --engine gemini --format all
```

### 2. Local Offline Transcription (faster-whisper)
Requires **zero** API keys or internet connection:

```bash
# Transcribe locally on CPU (default model: small)
video-transcriber path/to/video.mp4 --engine local

# Transcribe with ultra-fast tiny model
video-transcriber path/to/video.mp4 --engine local --model tiny --format md

# Transcribe with larger model on GPU (if CUDA available)
video-transcriber path/to/video.mp4 --engine local --model medium --device cuda
```

### 3. Batch Processing with Auto-Skip
```bash
# Transcribe all mp4 videos in a directory, skipping existing transcripts
video-transcriber videos/*.mp4 --skip-existing --format all
```

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
| `--skip-existing` | Skip files that already have transcripts saved | `False` |
| `--no-timestamps` | Turn off timestamp markers | `False` |
| `--no-speakers` | Turn off speaker identification (Gemini) | `False` |
| `-l`, `--language` | Language hint (e.g. `English`, `Spanish`) | Auto-detect |
| `-p`, `--prompt` | Extra prompt instructions for Gemini | `None` |
| `--keep-file` | Keep remote file on Gemini Files API | `False` (auto-deleted) |
| `--api-key` | Gemini API key override | Uses `GEMINI_API_KEY` |

---

## Python API Usage

```python
from video_transcriber import VideoTranscriber

# 1. Cloud Gemini
gemini_transcriber = VideoTranscriber(engine="gemini")
result = gemini_transcriber.transcribe("video.mp4", format_type="md")
result.save("output", format_type="md")

# 2. Local Whisper
local_transcriber = VideoTranscriber(engine="local")
result_local = local_transcriber.transcribe(
    "video.mp4",
    model="small",
    device="cpu",
    target_format="md",
)
result_local.save("output", format_type="md")
```

---

## Running Tests

```bash
python -m unittest discover tests
```
