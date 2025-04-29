import argparse
import gc
import multiprocessing
import os
import time

import tiktoken
import torch
import whisperx
from tabulate import tabulate


def main():
    # Create directories if they don't exist
    os.makedirs("./3-output/", exist_ok=True)
    os.makedirs("./2-audio_processed/", exist_ok=True)

    # Timer start
    script_start_time = time.time()
    print(
        f"DEBUG - script_start_time type: {type(script_start_time)}, value: {script_start_time}"
    )

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using WhisperX"
    )
    parser.add_argument(
        "model_pos",
        nargs="?",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ],
        help="WhisperX model to use (shorthand)",
        default=None,
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3-turbo"],
        help="WhisperX model to use",
        default="large-v3-turbo",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size",
        default=None,
    )
    parser.add_argument(
        "--compute_type",
        choices=["float16", "float32", "int8"],
        help="Override compute type",
        default=None,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code override (default: English)",
        default="en",
    )
    args = parser.parse_args()

    # Determine model name
    model_name = args.model_pos if args.model_pos is not None else args.model

    # Base batch sizes keyed by model
    base_batch_sizes = {
        "tiny": 16,
        "base": 16,
        "small": 16,
        "medium": 8,
        "large-v2": 4,
        "large-v3": 2,
        "large-v3-turbo": 2,
    }

    # --- Auto-detect hardware and pick optimal settings ---
    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        major = props.major
        vram_gb = props.total_memory / (1024**3)

        # Enable TF32 acceleration on Ampere+ GPUs
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Default to float16 on all CUDA GPUs
        auto_compute = "float16"

        # Base batch size and doubling for ample VRAM
        auto_batch = base_batch_sizes.get(model_name, 4)
        if vram_gb >= 16:
            auto_batch *= 2
    else:
        device = "cpu"
        auto_compute = "int8"
        cores = multiprocessing.cpu_count()
        auto_batch = min(max(1, cores // 2), 2)

    # Allow manual overrides
    compute_type = args.compute_type or auto_compute
    batch_size = args.batch_size or auto_batch

    # Force float32 on pre-Ampere if needed
    if device == "cuda" and torch.cuda.get_device_properties(0).major < 8:
        compute_type = "float32"

    print(f"\n🤖 WhisperX Model: {model_name}")
    print(f"🧠 Batch size:      {batch_size}")
    print(f"💻 Device:          {device}")
    print(f"⚙️  Compute type:    {compute_type}")
    print(f"🌐 Language:        {args.language}")
    # --- end auto-detect block ---

    # Load WhisperX model with fallback chain
    fallback_order = [compute_type]
    if "float16" not in fallback_order:
        fallback_order.insert(0, "float16")
    if "float32" not in fallback_order:
        fallback_order.append("float32")
    if "int8" not in fallback_order:
        fallback_order.append("int8")

    model = None
    last_error = None
    for ct in fallback_order:
        try:
            model = whisperx.load_model(model_name, device, compute_type=ct)
            compute_type = ct
            print(f"Loaded model with compute type: {ct}")
            break
        except Exception as err:
            last_error = err
            print(f"Failed load with {ct}, trying next...")
    else:
        print(f"Error loading model: {last_error}")
        exit(1)

    # OpenAI tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(text):
        return len(encoding.encode(text))

    audio_folder = "./1-audio"
    output_folder = "./3-output"
    stats = []

    total_size = total_chars = total_tokens = total_time = total_length = 0

    for filename in os.listdir(audio_folder):
        if filename.lower().endswith((".mp3", ".wav")):
            start_time = time.time()
            path = os.path.join(audio_folder, filename)
            print(f"\nProcessing {filename}...")
            try:
                audio = whisperx.load_audio(path)
                result = model.transcribe(
                    audio, batch_size=batch_size, language=args.language
                )

                # Handle possible OOM during transcription
                # (Older GPUs may need batch_size reduction)
            except RuntimeError as oom:
                if (
                    device == "cuda"
                    and "CUDA out of memory" in str(oom)
                    and batch_size > 1
                ):
                    print(f"🚨 OOM, reducing batch to {batch_size // 2}...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    result = model.transcribe(
                        audio, batch_size=batch_size // 2, language=args.language
                    )
                else:
                    raise

            # Word-level alignment
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            aligned = whisperx.align(
                result["segments"], model_a, metadata, audio, device
            )
            del model_a
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            segments = aligned["segments"]
            transcripts = []
            for seg in segments:
                text = seg["text"].strip()
                if text:
                    m, s = divmod(int(seg["start"]), 60)
                    transcripts.append(f"{m:02d}:{s:02d} > {text}")
            out_text = "\n".join(transcripts)

            # Save outputs
            base = os.path.splitext(filename)[0]
            out_dir = os.path.join(output_folder, base)
            os.makedirs(out_dir, exist_ok=True)
            with open(
                os.path.join(out_dir, f"{base}-timestamps.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(out_text)
            plain = " ".join(
                seg["text"].strip() for seg in segments if seg["text"].strip()
            )
            with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
                f.write(plain)

            # Stats
            duration = result["segments"][-1]["end"] / 60
            elapsed = time.time() - start_time
            size_kb = (
                os.path.getsize(os.path.join(out_dir, f"{base}-timestamps.txt")) / 1024
            )
            tcnt = count_tokens(out_text)
            cchar = len(out_text)
            total_length += duration
            total_time += elapsed
            total_size += size_kb
            total_tokens += tcnt
            total_chars += cchar
            stats.append(
                [
                    f"✅ {base}-{model_name}.txt",
                    f"{int(duration)}:{int((duration % 1) * 60):02d}",
                    f"{size_kb:.2f}",
                    cchar,
                    tcnt,
                    f"{elapsed:.2f}",
                ]
            )
            print(f"✅ Completed in {elapsed:.2f}s")

    # Report
    print("\nResults")
    print(
        tabulate(
            stats,
            headers=["File", "Audio", "Size (KB)", "Chars", "Tokens", "Time(s)"],
            tablefmt="grid",
        )
    )

    # Totals with multi-line formatting
    elapsed_total = time.time() - script_start_time
    m, s = divmod(int(total_length * 60), 60)

    print("\nTotals")
    print(f"Time:         {elapsed_total:.2f} seconds")
    print(f"Transcribed:  {m}:{s:02d} minutes")
    print(f"Size:         {total_size:.2f} KB")
    print(f"Characters:   {total_chars}")
    print(f"Tokens:       {total_tokens}")


if __name__ == "__main__":
    main()
