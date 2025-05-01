from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time


WATCH_PATH = r"v:\10.python\transcribe\1-audio"

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.mp3', '.wav')):
            print(f"🎧 New file: {event.src_path}")
            time.sleep(3)  # Give it a sec to finish copying
            subprocess.run(["poetry", "run", "transcribe"])

def main():
    observer = Observer()
    observer.schedule(FileHandler(), path=WATCH_PATH, recursive=False)
    observer.start()
    print(f"🟢 Watching {WATCH_PATH} for new audio...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
