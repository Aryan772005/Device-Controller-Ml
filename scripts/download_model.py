"""
GestureFlow AI - MediaPipe Hand Landmarker Model Downloader
===========================================================
Downloads and validates the official Google MediaPipe Hand Landmarker model.
Saves model safely to models/hand_landmarker.task.
"""

import os
import sys
import urllib.request
import urllib.error

# Official Google Cloud Storage URL for MediaPipe Hand Landmarker float16 model bundle
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"
EXPECTED_MIN_BYTES = 7 * 1024 * 1024  # Expected to be ~7.46 MB (7,819,105 bytes)


def get_models_dir() -> str:
    # Project root is one level up from scripts/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def download_hand_landmarker_model() -> str:
    models_dir = get_models_dir()
    target_path = os.path.join(models_dir, MODEL_FILENAME)
    temp_path = target_path + ".tmp"

    print("=" * 60)
    print("MediaPipe Hand Landmarker Model Setup")
    print("=" * 60)
    print(f"Target location: {target_path}")

    # Check if a valid model file already exists
    if os.path.isfile(target_path):
        size = os.path.getsize(target_path)
        if size >= EXPECTED_MIN_BYTES:
            print(f"[INFO] Valid model file already exists ({size / (1024 * 1024):.2f} MB). Skipping download.")
            return target_path
        else:
            print(f"[WARN] Existing model file is undersized ({size} bytes). Re-downloading...")

    print(f"[INFO] Downloading official model from:\n       {MODEL_URL}")
    print("[INFO] Model: float16 bundle (Palm Detector + 21 3D Landmarks)")

    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "GestureFlow-AI-Setup/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 64 * 1024  # 64 KB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r[DOWNLOADING] {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({percent:.1f}%)", end="", flush=True)

            print()  # newline after progress bar

        # Validate downloaded size
        actual_size = os.path.getsize(temp_path)
        if actual_size < EXPECTED_MIN_BYTES:
            raise ValueError(f"Downloaded file size ({actual_size} bytes) is below minimum expected ({EXPECTED_MIN_BYTES} bytes).")

        # Atomic replacement: rename tmp file to target file
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

        print(f"[SUCCESS] Model downloaded and verified: {target_path} ({actual_size / (1024 * 1024):.2f} MB)")
        return target_path

    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network failure during download: {e.reason}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        sys.exit(1)


if __name__ == "__main__":
    download_hand_landmarker_model()
