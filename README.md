# Audio Transcription Tool

A Python-based audio transcription tool that uses WhisperX to convert audio files to text with timestamps and provides detailed statistics.

## Overview

This tool transcribes audio files (MP3, WAV) into text documents with optional timestamps. It uses WhisperX, which provides fast and accurate transcription with word-level timestamp alignment.

## Directory Structure

- `audio/`: Place your audio files here for processing
- `audio_processed/`: Contains processed audio files
- `output/`: Contains transcription results
- `0-queu/`: Queue folder for files to be processed

## Requirements

- Python 3.x
- WhisperX
- PyTorch
- Tiktoken
- Tabulate
- Hugging Face account and API token (for diarization features)

## Setup

1. Clone the repository
2. Install required packages
3. Copy `.env-sample` to `.env` and add your Hugging Face API token:
   ```
   cp .env-sample .env
   ```
4. Edit the `.env` file and replace `your_hugging_face_token_here` with your actual Hugging Face API token
   
   > You can get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Usage

### Basic Usage

```bash
python transcribe.py
```

This will use the default model (large-v3-turbo) to transcribe all audio files in the `audio/` directory.

### Specifying a Model

```bash
python transcribe.py [model_name]
```

Where `[model_name]` can be one of:
- `tiny`: Fastest but least accurate
- `base`: Fast with basic accuracy
- `small`: Good balance for shorter files
- `medium`: Better accuracy, slower
- `large-v2`: High accuracy
- `large-v3`: Latest model with best accuracy
- `large-v3-turbo`: Latest model optimized for speed and accuracy (default)

### Advanced Options

```bash
python transcribe.py --model [model_name] --batch_size [size] --compute_type [type]
```

Options:
- `--model`: Specify the WhisperX model
- `--batch_size`: Control batch size for performance (default is based on model)
- `--compute_type`: Set computation precision (`float16`, `float32`, or `int8`)

## Output

For each audio file processed, the tool generates:

1. A plain text transcript: `output/[filename]/[filename].txt`
2. A timestamped transcript: `output/[filename]/[filename]-timestamps.txt`

The timestamps format is `MM:SS > transcript text`

## Performance Statistics

After processing, the tool displays a table with statistics for each file:
- File name
- Audio duration
- Output file size
- Character count
- Token count
- Processing time

## Example

```bash
# Process all files using the medium model
python transcribe.py medium

# Process with specific batch size and compute type
python transcribe.py --model large-v3-turbo --batch_size 4 --compute_type int8
```

## Using WhisperX CLI Commands

For direct WhisperX commands (as shown in `commands.txt`), make sure to use your environment variable:

```bash
whisperx audio/sample.mp3 \
  --model large-v3-turbo \
  --diarize \
  --hf_token ${HF_TOKEN} \
  --output_dir output/sample
```

## Tips

- For large files, use a smaller batch size to prevent memory issues
- If you encounter CUDA out-of-memory errors, the script will automatically try with a smaller batch size
- For faster processing on slower hardware, use the `tiny` or `base` models with `int8` compute type
- Keep your API tokens in the `.env` file and never commit them directly to version control