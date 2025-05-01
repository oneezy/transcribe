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
    def __init__(self, watch_path_pretty, watch_path):
        self.watch_path_pretty = watch_path_pretty
        self.watch_path = watch_path
        self.pending_files = []
        self.processing = False
        self.timer = None
        self.batch_timer = None
        self.batch_files = []
        self.last_file_time = 0

    def process_files(self):
        if self.pending_files and not self.processing:
            self.processing = True
            print("")  # newline before transcribing
            subprocess.run(["poetry", "run", "transcribe"])
            self.pending_files = []
            self.processing = False
            print(f"\n🟢 Waiting to transcribe new audio files in {cyan(self.watch_path_pretty)} ...\n")

    def display_batch_files(self):
        """Display consolidated message for multiple files detected in a batch"""
        if self.batch_files:
            # Add a short delay before showing the found files message
            time.sleep(2)
            print("\n")
            print(f"🔍 Found {len(self.batch_files)} Audio File{'s' if len(self.batch_files) > 1 else ''}!")
            print("─" * 40 + "\n")
            for file_name in self.batch_files:
                print(f"🎧 Audio File: {cyan(file_name)}")
            
            self.batch_files = []  # Clear the batch

    def check_existing_files(self):
        """Check for existing audio files in the watch directory and process them"""
        existing_files = [
            f for f in os.listdir(self.watch_path)
            if os.path.isfile(os.path.join(self.watch_path, f)) and f.lower().endswith(('.mp3', '.wav'))
        ]
        if existing_files:
            self.pending_files.extend(existing_files)
            if self.timer is not None:
                self.timer.cancel()
            self.timer = threading.Timer(2.0, self.process_files)
            self.timer.start()
            return existing_files
        return []

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.mp3', '.wav')):
            file_name = os.path.basename(event.src_path)
            self.pending_files.append(file_name)
            
            # Add to batch for consolidated display
            self.batch_files.append(file_name)
            current_time = time.time()
            self.last_file_time = current_time
            
            # Cancel existing timers
            if self.timer is not None:
                self.timer.cancel()
            if self.batch_timer is not None:
                self.batch_timer.cancel()
            
            # Set new timers
            self.batch_timer = threading.Timer(0.5, self.display_batch_files)  # Short delay to collect multiple files
            self.batch_timer.start()
            
            self.timer = threading.Timer(3.0, self.process_files)
            self.timer.start()

def main():
    file_handler = FileHandler(PRETTY_WATCH_PATH, WATCH_PATH)
    observer = Observer()
    observer.schedule(file_handler, path=WATCH_PATH, recursive=False)
    observer.start()

    print("\nINSTRUCTIONS\n")
    print(f" 1. Add audio files into {gray('1-audio/')}")
    print(f" 2. Files auto-process and move to {gray('2-audio_processed/')}")
    print(f" 3. Transcripts are saved to {gray('3-output/')}")
    print("\n" + "─" * 70 + "\n")

    print(f"🟢 Ready to transcribe new audio files in {cyan(PRETTY_WATCH_PATH)}...\n")

    existing_files = file_handler.check_existing_files()
    if existing_files:
        # Add a short delay before showing the found files message
        time.sleep(2)
        print("\n")
        print(f"🔍 Found {len(existing_files)} Audio File{'s' if len(existing_files) > 1 else ''}!")
        print("─" * 40 + "\n")
        for file_name in existing_files:
            print(f"🎧 Audio File: {cyan(file_name)}")

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
