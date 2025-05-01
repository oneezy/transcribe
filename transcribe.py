import argparse
import gc
import multiprocessing
import os
import re
import time
import tiktoken
import torch
import whisperx
from tabulate import tabulate

def remove_filler_words(text):
    """Remove filler words, disfluencies, and nonverbal sounds from text cleanly."""
    filler_words = [
        "uh", "um", "er", "ah", "hmm", "mm-hmm", "uh-huh", "uh-uh", "mmm", "uh oh", "ugh",
        "like", "you know", "i mean", "well", "okay", "right", "yeah", "yep", "nope",
        "kinda", "sorta", "basically", "actually", "just", "really", "honestly", "literally",
        "you see", "you know what i mean", "you get me", "if that makes sense",
        "know what i'm saying", "you follow", "you feel me", "you get it",
        "so yeah", "anyway", "anyways", "alright so", "right so", "and yeah", "but yeah",
        "and uh", "and um", "or whatever", "or something", "or like", "i guess", "i suppose",
        "i dunno", "i don't know", "maybe", "probably", "possibly",
        "i i", "we we", "you you", "they they", "and and", "but but", "so so",
        "um um", "uh uh", "like like",
        "sorry but", "to be honest", "to be fair", "no offense", "don't take this the wrong way",
        "with all due respect",
        "i think", "i feel like", "i'm not sure", "i believe", "i'm just saying",
        "it's kind of like", "it's sort of like",
        "let me think", "let me see", "what was i saying", "what was i talking about",
        "what else", "hang on", "give me a sec", "hold on", "wait wait",
        "or whatnot", "you name it", "yada yada", "et cetera", "blah blah",
        "uh yeah so", "um yeah like", "oh okay", "ah right", "hm okay", "well i mean", "so um yeah",
        "anyhow", "incidentally", "essentially", "technically", "interestingly enough", "just saying",
    ]

    text = re.sub(r'\[.*?\]', '', text)
    pattern = r'\b(' + '|'.join(re.escape(word) for word in filler_words) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,?!])', r'\1', text)
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'([.,!?])\s*([.,!?])', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'(^|\.\s+)(so|and|but)\b\s*,?', r'\1', text, flags=re.IGNORECASE)

    def capitalize_sentence(m):
        return m.group(1) + m.group(2).upper()

    text = re.sub(r'([.!?]\s+)([a-z])', capitalize_sentence, text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    text = re.sub(r'\b(i)\b', 'I', text)
    return text.strip()

def create_ai_cleaned_text(text):
    """Produce a super-cleaned version of the text optimized for AI consumption."""
    text = remove_filler_words(text)
    text = re.sub(r'\b(so|and|but|then|well|okay)\b[, ]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(um|uh|ah|er|hmm|mmm)\b[, ]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,?!])', r'\1', text)
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\.{2,}', '.', text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    text = re.sub(r'\b(i)\b', 'I', text)
    return text.strip()

def main():
    os.makedirs("./3-output/", exist_ok=True)
    os.makedirs("./2-audio_processed/", exist_ok=True)

    script_start_time = time.time()

    parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX")
    parser.add_argument("model_pos", nargs="?", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"], default=None)
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3-turbo"], default="large-v3-turbo")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--compute_type", choices=["float16", "float32", "int8"], default=None)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    model_name = args.model_pos if args.model_pos else args.model

    base_batch_sizes = {"tiny": 16, "base": 16, "small": 16, "medium": 8, "large-v2": 4, "large-v3": 2, "large-v3-turbo": 2}

    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        major = props.major
        vram_gb = props.total_memory / (1024**3)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        auto_compute = "float16"
        auto_batch = base_batch_sizes.get(model_name, 4)
        if vram_gb >= 16:
            auto_batch *= 2
    else:
        device = "cpu"
        auto_compute = "int8"
        cores = multiprocessing.cpu_count()
        auto_batch = min(max(1, cores // 2), 2)

    compute_type = args.compute_type or auto_compute
    batch_size = args.batch_size or auto_batch

    if device == "cuda" and torch.cuda.get_device_properties(0).major < 8:
        compute_type = "float32"

    print(f"\n🤖 WhisperX Model: {model_name}")
    print(f"🧠 Batch size: {batch_size}")
    print(f"💻 Device: {device}")
    print(f"⚙️ Compute type: {compute_type}")
    print(f"🌐 Language: {args.language}")

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
                result = model.transcribe(audio, batch_size=batch_size, language=args.language)
            except RuntimeError as oom:
                if device == "cuda" and "CUDA out of memory" in str(oom) and batch_size > 1:
                    print(f"🚨 OOM, reducing batch to {batch_size // 2}...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    result = model.transcribe(audio, batch_size=batch_size // 2, language=args.language)
                else:
                    raise

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
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

            base = os.path.splitext(filename)[0]
            out_dir = os.path.join(output_folder, base)
            os.makedirs(out_dir, exist_ok=True)
            original_dir = os.path.join(out_dir, "original")
            os.makedirs(original_dir, exist_ok=True)

            timestamp_text = "\n".join(transcripts)
            raw_text = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())
            min_text = remove_filler_words(raw_text)

            with open(os.path.join(original_dir, f"{base}.timestamps.txt"), "w", encoding="utf-8") as f:
                f.write(timestamp_text)
            with open(os.path.join(original_dir, f"{base}.raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)
            with open(os.path.join(original_dir, f"{base}.min.txt"), "w", encoding="utf-8") as f:
                f.write(min_text)
            with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
                f.write(min_text)
            ai_text = create_ai_cleaned_text(raw_text)
            with open(os.path.join(original_dir, f"{base}.ai.txt"), "w", encoding="utf-8") as f:
                f.write(ai_text)

            duration = result["segments"][-1]["end"] / 60
            elapsed = time.time() - start_time
            size_kb = os.path.getsize(os.path.join(out_dir, f"{base}.txt")) / 1024
            tcnt = count_tokens(min_text)
            cchar = len(min_text)
            total_length += duration
            total_time += elapsed
            total_size += size_kb
            total_tokens += tcnt
            total_chars += cchar
            stats.append([f"✅ {base}-{model_name}.txt", f"{int(duration)}:{int((duration % 1) * 60):02d}", f"{size_kb:.2f}", cchar, tcnt, f"{elapsed:.2f}"])
            print(f"✅ Completed in {elapsed:.2f}s")

    print("\nResults")
    print(tabulate(stats, headers=["File", "Audio", "Size (KB)", "Chars", "Tokens", "Time(s)"], tablefmt="grid"))

    elapsed_total = time.time() - script_start_time
    m, s = divmod(int(total_length * 60), 60)
    print("\nTotals")
    print(f"Time: {elapsed_total:.2f} seconds")
    print(f"Transcribed: {m}:{s:02d} minutes")
    print(f"Size: {total_size:.2f} KB")
    print(f"Characters: {total_chars}")
    print(f"Tokens: {total_tokens}")

if __name__ == "__main__":
    main()
