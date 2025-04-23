import os
import time
import whisperx
import torch
import tiktoken
from tabulate import tabulate
import argparse
import gc

# Create directories if they don't exist
os.makedirs("./3-output/", exist_ok=True)
os.makedirs("./2-audio_processed/", exist_ok=True)

# Timer start - use a more specific name to avoid collisions
script_start_time = time.time()
print(f"DEBUG - script_start_time type: {type(script_start_time)}, value: {script_start_time}")

# Argument parser
parser = argparse.ArgumentParser(description='Transcribe audio files using WhisperX')
parser.add_argument('model_pos', nargs='?', 
                    choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3', 'large-v3-turbo'],
                    help='WhisperX model to use (shorthand)',
                    default=None)
parser.add_argument('--model', 
                    choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3-turbo'],
                    help='WhisperX model to use',
                    default='large-v3-turbo')
parser.add_argument('--batch_size', type=int, help='Batch size for transcription', default=None)
parser.add_argument('--compute_type', choices=['float16', 'float32', 'int8'], help='Compute type', default='float16')
args = parser.parse_args()

# Process positional arguments
model_name = args.model_pos if args.model_pos is not None else args.model

# Determine appropriate batch size based on model
if args.batch_size is None:
    # Default batch sizes based on model size
    batch_sizes = {
        'tiny': 16,
        'base': 16,
        'small': 16,
        'medium': 8,
        'large-v2': 4,
        'large-v3': 2,
        'large-v3-turbo': 2
    }
    batch_size = batch_sizes.get(model_name, 4)
else:
    batch_size = args.batch_size

# Load WhisperX model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🤖 WhisperX Model: {model_name}")
print(f"🧠 Using batch size: {batch_size}")
print(f"💻 Device: {device} | Compute type: {args.compute_type}")

try:
    # Try to clear any existing models from GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    model = whisperx.load_model(model_name, device, compute_type=args.compute_type)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Try using a smaller model, reducing batch size, or using int8 compute type")
    exit(1)

# OpenAI tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")
def count_tokens(text):
    return len(encoding.encode(text))

audio_folder = "./1-audio"
output_folder = "./3-output/"
stats = []

total_size = total_chars = total_tokens = total_time = total_length = 0

for filename in os.listdir(audio_folder):
    if filename.endswith((".mp3", ".wav")):
        file_start_time = time.time()
        input_path = os.path.join(audio_folder, filename)
        print(f"\nProcessing {filename}...")

        try:
            # WhisperX transcription
            audio = whisperx.load_audio(input_path)
            
            # Use try/except with batch size reduction for OOM errors
            try:
                result = model.transcribe(audio, batch_size=batch_size)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and batch_size > 1:
                    print(f"🚨 CUDA out of memory. Reducing batch size to {batch_size//2}...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    result = model.transcribe(audio, batch_size=batch_size//2)
                else:
                    raise

            # Align word-level timestamps
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
            
            # Free up memory after alignment
            del model_a
            torch.cuda.empty_cache()
            gc.collect()

            # Process segments
            segments = result_aligned["segments"]
            
            # Format segments with MM:SS timestamps followed by > and then text
            transcript_segments = []
            
            # Format each segment with timestamps in MM:SS format
            for seg in segments:
                if not seg["text"].strip():
                    continue
                    
                # Format start time as MM:SS
                start_seconds = int(seg["start"])
                start_minutes = start_seconds // 60
                start_seconds_remainder = start_seconds % 60
                start_time_formatted = f"{start_minutes:02d}:{start_seconds_remainder:02d}"
                
                # Format with start timestamp followed by > and then the text
                transcript_segments.append(f"{start_time_formatted} > {seg['text'].strip()}")
            
            transcript = "\n".join(transcript_segments)

            # Save output
            base_filename = os.path.splitext(filename)[0]
            file_output_folder = os.path.join(output_folder, base_filename)
            os.makedirs(file_output_folder, exist_ok=True)

            # Save with timestamps
            timestamp_output_path = os.path.join(file_output_folder, f"{base_filename}-timestamps.txt")
            with open(timestamp_output_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Create a version without timestamps
            plain_text = []
            for seg in segments:
                if seg["text"].strip():
                    plain_text.append(seg["text"].strip())
            plain_transcript = " ".join(plain_text)

            # Save without timestamps
            plain_output_path = os.path.join(file_output_folder, f"{base_filename}.txt")
            with open(plain_output_path, "w", encoding="utf-8") as f:
                f.write(plain_transcript)

            # Calculate stats
            file_time = time.time() - file_start_time
            file_size = os.path.getsize(timestamp_output_path) / 1024
            char_count = len(transcript)
            token_count = count_tokens(transcript)

            audio_length = result["segments"][-1]["end"] / 60
            total_length += audio_length

            minutes = int(audio_length)
            seconds = int((audio_length - minutes) * 60)
            length_formatted = f"{minutes}:{seconds:02d}"

            total_size += file_size
            total_chars += char_count
            total_tokens += token_count
            total_time += file_time

            stats.append([
                f"✅ {os.path.splitext(filename)[0]}-{model_name}.txt",
                length_formatted,
                f"{file_size:.2f}",
                char_count,
                token_count,
                f"{file_time:.2f}"
            ])
            
            print(f"✅ Completed {filename} in {file_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            # Try to clear memory in case of error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

# Results
print("\nResults")
headers = ["✅ File", "Audio", "Size (KB)", "Characters", "Tokens", "Time (s)"]
print(tabulate(stats, headers=headers, tablefmt="grid"))

# Debug before calculating elapsed time
print(f"DEBUG - Before elapsed_time calculation:")
print(f"  - script_start_time type: {type(script_start_time)}, value: {script_start_time}")
print(f"  - current time type: {type(time.time())}, value: {time.time()}")

# Calculate elapsed time
elapsed_time = time.time() - script_start_time

total_minutes = int(total_length)
total_seconds = int((total_length - total_minutes) * 60)

print("\nTotals")
print(f"Time:         {elapsed_time:.2f} seconds")
print(f"Transcribed:  {total_minutes}:{total_seconds:02d} minutes")
print(f"Size:         {total_size:.2f} KB")
print(f"Characters:   {total_chars}")
print(f"Tokens:       {total_tokens}")


