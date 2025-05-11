# Audio Transcription Tool

A Python-based audio transcription tool using WhisperX to convert audio files to text with word-level timestamps and performance stats.

---

## 🚀 Quick Start (Recommended)

### ✅ Step 1: Run `install.bat`
This installs everything you need: Python deps, WhisperX, and sets up the workspace.

### ✅ Step 2: Run `transcribe.bat`
This watches the `1-audio/` folder and automatically transcribes anything you drop in.

> That's it — you're ready to go! Skip the manual setup below unless you're customizing things.

---

## 📁 Directory Structure

- `1-audio/`: Drop your `.mp3` or `.wav` files here
- `2-output/`: Where all transcripts and outputs land

```
2-output/
└── audio/                          Folder named after the audio file's base name
    ├── original/
    │   ├── audio.ai.txt            AI-cleaned transcript
    │   ├── audio.min.txt           Same as audio.txt in parent (cleaned)
    │   ├── audio.raw.txt           Raw transcript before cleaning
    │   └── audio.timestamps.txt    Transcript with timestamps
    ├── audio.mp3                   Original audio file 
    └── audio.txt                   Cleaned transcript, filler words removed
```

---

## ⚙️ Manual Setup (Optional)

If you want to dig deeper or customize things, here's the manual route.

### Prerequisites

- Python 3.12.x (must be < 3.13)
- [Poetry](https://python-poetry.org/)
- Hugging Face account + API token (for speaker diarization)

### Option A) Using Poetry (recommended)

```bash
# Install Poetry
pip install poetry
# or
pipx install poetry

# Clone and enter repo
git clone https://github.com/oneezy/transcribe.git
cd transcribe

# Install deps
poetry install

# Run
poetry run python transcribe.py
````

### Option B) Using pip

```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install deps
pip install -r requirements.txt

# Run
python transcribe.py
```

---

## 📦 Usage

### Basic

```bash
python transcribe.py
```

### Watch for files

```bash
poetry run python watcher.py
```

### Choose a model

```bash
python transcribe.py medium
```

### Advanced options

```bash
python transcribe.py --model large-v3-turbo --batch_size 4 --compute_type int8
```

---

## 📊 Output Includes

* Duration
* File size
* Token/char count
* Processing time
* Cleaned transcript
* Timestamps

---

## 🛠 WhisperX CLI (if needed)

```bash
whisperx 1-audio/sample.mp3 \
  --model large-v3-turbo \
  --output_dir 2-output/sample
```

---

## 💡 Tips

* Use `tiny` + `int8` for slow machines
* Use `.env` to store Hugging Face API key
* Auto fallback for OOM errors
* Everything’s bundled if you're using the `.exe` version (no Python needed)

---

Built with ❤️ by [oneezy.com](https://oneezy.com)