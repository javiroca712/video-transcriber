# Video Transcriber (Gemini Pro API)

A Python tool to transcribe local MP4 (and other audio/video) files using the Google Gemini API.

Built with the official Google GenAI SDK (`google-genai`), this tool leverages Gemini's native multimodal understanding and File API to deliver high-accuracy transcriptions with speaker identification, timestamps, and multi-format exports.

---

## Features

- 🎥 **Native Video & Audio Support**: Handles `.mp4`, `.mov`, `.mkv`, `.webm`, `.wav`, `.mp3`, and `.m4a`.
- 🧠 **Gemini Models**: Works with `gemini-2.5-pro`, `gemini-2.5-flash`, and earlier models.
- ⏱️ **Timestamps & Speaker Diarization**: Automatically tags speakers (`Speaker 1:`, `Speaker 2:`) and marks timestamps (`[01:23]`).
- 📝 **Multiple Output Formats**: Export to `.txt`, `.srt` (SubRip subtitles), `.md` (Markdown), and `.json` (structured metadata), or `--format all`.
- 🧹 **Automatic Cleanup**: Uploads files using Gemini File API and automatically deletes them after processing to save storage quota and protect privacy.
- ⚡ **Batch Processing**: Transcribe multiple videos in a single command.

---

## Project Structure

```
video-transcriber/
├── .venv/                      # Virtual environment
├── src/
│   └── video_transcriber/
│       ├── __init__.py         # Package exports
│       ├── __main__.py         # python -m video_transcriber entry point
│       ├── cli.py              # CLI argument parsing and runner
│       ├── config.py           # Configuration and environment management
│       ├── formatters.py       # Exporters (txt, srt, md, json)
│       └── transcriber.py      # Core Gemini File API & generation engine
├── tests/
│   ├── __init__.py
│   └── test_transcriber.py     # Unit tests
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules for media, venv, secrets
├── pyproject.toml              # Modern Python packaging & dependencies
├── requirements.txt            # Pinned requirements
└── README.md                   # Documentation
```

---

## Quickstart

### 1. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

*(If execution policy blocks scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

---

### 2. Configure Your Gemini API Key

1. Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

3. Open `.env` and set your key:
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
```

*(You can also set the model to `gemini-2.5-pro` for higher reasoning or complex multi-speaker audio).*

---

### 3. Usage via CLI

#### Basic Transcription (saves to `./output/<filename>.txt`)
```bash
video-transcriber path/to/my_video.mp4
```

#### Generate SubRip Subtitles (`.srt`)
```bash
video-transcriber my_video.mp4 --format srt
```

#### Use Gemini 2.5 Pro with All Output Formats
```bash
video-transcriber my_video.mp4 --model gemini-2.5-pro --format all
```

#### Transcribe Multiple Files
```bash
video-transcriber video1.mp4 video2.mp4 -o transcripts/
```

#### Specify Language or Custom Prompt
```bash
video-transcriber lecture.mp4 -l "Spanish" -p "Include a bulleted summary of main topics at the top."
```

#### CLI Options Reference
| Flag | Description | Default |
|---|---|---|
| `files` | One or more input media files | *(Required)* |
| `-m`, `--model` | Model name (`gemini-2.5-pro`, `gemini-2.5-flash`) | From `.env` or `gemini-2.5-flash` |
| `-f`, `--format` | Output format: `txt`, `srt`, `md`, `json`, `all` | `txt` |
| `-o`, `--output-dir` | Directory to save output files | `output` |
| `--no-timestamps` | Turn off timestamp markers | `False` |
| `--no-speakers` | Turn off speaker identification | `False` |
| `-l`, `--language` | Language hint (e.g. `English`, `German`) | Auto-detect |
| `-p`, `--prompt` | Extra prompt instructions for Gemini | `None` |
| `--keep-file` | Keep remote file on Gemini Files API | `False` (auto-deleted) |
| `--api-key` | API key override | Uses `GEMINI_API_KEY` |

---

### 4. Python API Usage

You can also import and use `VideoTranscriber` directly in your own Python scripts:

```python
from video_transcriber import VideoTranscriber, Config

# Initialize transcriber (reads GEMINI_API_KEY from .env)
transcriber = VideoTranscriber()

# Transcribe video
result = transcriber.transcribe(
    file_path="presentation.mp4",
    model="gemini-2.5-pro",
    include_timestamps=True,
    identify_speakers=True,
    target_format="md",
)

# Access text directly
print(result.raw_text)

# Save to file
result.save("output", format_type="md")
```

---

## Running Tests

Run the test suite using unittest:

```bash
python -m unittest discover tests
```
