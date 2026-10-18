# Contributing to Video Transcriber

Thank you for your interest in contributing to `video-transcriber`! We welcome contributions, bug reports, feature requests, and pull requests from friends, colleagues, and the open-source community.

---

## 🛠️ Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- (Optional) Free Google Gemini API key if testing cloud multimodal features: [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Fork & Clone
```bash
git clone https://github.com/<your-username>/video-transcriber.git
cd video-transcriber
```

### 3. Create a Virtual Environment
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

### 4. Install Dependencies
To install the package in editable mode with development & local engine dependencies:
```bash
pip install -e .[all]
```
Or for cloud-only testing:
```bash
pip install -e .
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your `GEMINI_API_KEY` if testing the Gemini cloud engine. (Local Whisper requires zero API keys).

---

## 🧪 Running Tests

Always verify the unit test suite passes before submitting changes:

```bash
python -m unittest discover tests
```

---

## 📐 Project Structure & Guidelines

- `src/video_transcriber/`: Core package source code.
  - `config.py`: Environment and runtime configuration dataclass.
  - `transcriber.py`: High-level `VideoTranscriber` facade.
  - `formatters.py`: `TranscriptionResult` data structure and format exporters (`txt`, `md`, `srt`, `json`).
  - `cli.py`: Command-line interface and batch processing runner.
  - `engines/`: Modular transcription engines (`GeminiEngine`, `LocalWhisperEngine`, `factory.py`).
- `scripts/`: Benchmarks and utility scripts.
- `tests/`: Automated unit tests.

### Code Style & Conventions
- Adhere to **PEP 8** style guidelines.
- Use explicit type annotations where possible.
- When opening files or configuring console output, always specify `encoding="utf-8"` for cross-platform compatibility (especially Windows `cp1252`).
- Preserve the PyAV patch in `src/video_transcriber/engines/local_whisper.py` to maintain compatibility with PyAV >= 14.x.

---

## 🚀 Submitting Pull Requests

1. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make your changes** with clear, descriptive commit messages.
3. **Add or update unit tests** in `tests/` covering your changes.
4. **Run the test suite** to ensure all tests pass:
   ```bash
   python -m unittest discover tests
   ```
5. **Submit a Pull Request** to the `main` branch with a clear summary of your changes.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [GNU General Public License v3.0](LICENSE).
