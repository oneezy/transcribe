from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading
import time
import os
from colors import cyan, gray, green, red, yellow

def pretty_path(path):
    return path.replace("\\", "/")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_PATH = os.path.join(SCRIPT_DIR, "1-audio")
PRETTY_WATCH_PATH = pretty_path(os.path.relpath(WATCH_PATH, start=os.path.dirname(SCRIPT_DIR)))

class FileHandler(FileSystemEventHandler):
    def __init__(self, watch_path_pretty):
        self.watch_path_pretty = watch_path_pretty
        self.pending_files = []
        self.processing = False
        self.timer = None
        
    def process_files(self):
        if self.pending_files and not self.processing:
            self.processing = True
            
            # Add a newline before running the transcribe command for cleaner output
            print("")
            
            # Process all files that have been collected
            subprocess.run(["poetry", "run", "transcribe"])
            
            # Clear the list of pending files
            self.pending_files = []
            
            # Reset processing flag
            self.processing = False
            
            # Reprint the waiting message after processing is complete
            print("")
            print(f"\n🟢 Waiting to transcribe new audio files in {cyan(self.watch_path_pretty)} ...\n")
            
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.mp3', '.wav')):            
            file_name = os.path.basename(event.src_path)
            print(f"🎧 Audio File: {cyan(file_name)}")
            
            # Add file to pending list
            self.pending_files.append(file_name)
            
            # Cancel any existing timer
            if self.timer is not None:
                self.timer.cancel()
            
            # Set a new timer to process files after a delay
            self.timer = threading.Timer(3.0, self.process_files)
            self.timer.start()

def main():
    file_handler = FileHandler(PRETTY_WATCH_PATH)
    observer = Observer()
    observer.schedule(file_handler, path=WATCH_PATH, recursive=False)
    observer.start()

    print("")
    print("INSTRUCTIONS")
    print("")
    print(f" 1. Add audio files into {gray('1-audio/')}")
    print(f" 2. Files auto-process and move to {gray('2-audio_processed/')}")
    print(f" 3. Transcripts are saved to {gray('3-output/')}")

    print("\n" + "─" * 70 + "\n")

    print(f"🟢 Waiting to transcribe new audio files in {cyan(PRETTY_WATCH_PATH)} ...\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if file_handler.timer:
            file_handler.timer.cancel()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()