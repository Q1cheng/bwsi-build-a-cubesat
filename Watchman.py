import time
import os
from datetime import datetime, timedelta

from Inference import main


BASE_DIR = os.path.expanduser("~/Desktop/DataControl")
PENDING_FOLDER = os.path.join(BASE_DIR, "Pending")
LOGS_FOLDER = os.path.join(BASE_DIR, "Logs")

CHECK_INTERVAL_SECONDS = 6  # 20 times per hour
LOG_RETENTION_DAYS = 7


def pending_has_images():
    files = os.listdir(PENDING_FOLDER)
    png_files = [f for f in files if f.lower().endswith(".png")]
    return len(png_files) > 0


def delete_old_logs():
    if not os.path.exists(LOGS_FOLDER):
        return

    now = datetime.now()

    for file in os.listdir(LOGS_FOLDER):
        file_path = os.path.join(LOGS_FOLDER, file)

        if not file.endswith(".npy"):
            continue

        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

        if now - file_time > timedelta(days=LOG_RETENTION_DAYS):
            os.remove(file_path)


def watch():
    print("Watchman started. Monitoring Pending folder...")

    while True:

        try:
            delete_old_logs()

            if pending_has_images():
                print("Image detected. Starting processing.")
                main()
                print("Processing complete.")

            else:
                print("No images found.")

        except Exception as e:
            print("Error:", e)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    watch()