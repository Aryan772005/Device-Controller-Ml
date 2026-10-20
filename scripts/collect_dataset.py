"""
GestureFlow AI - Custom Gesture Data Collector
==============================================
Collects 21 3D hand landmark data from your webcam to train your own
custom machine learning model for clicking, moving, or any custom posture.

Usage:
    python scripts/collect_dataset.py

Controls:
    [0] -> Label: MOVE (Pointing / cursor moving hand pose)
    [1] -> Label: CLICK (Pinch / click posture)
    [2] -> Label: DOUBLE_CLICK (Double click posture)
    [3] -> Label: RIGHT_CLICK (Fist / right click)
    [4] -> Label: IDLE (Rest / open hand)
    [C] -> Enter Custom Gesture Name in console
    [SPACE] -> Toggle recording ON / OFF
    [S] -> Save dataset to data/gestures_dataset.csv
    [Q] -> Save & Quit
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_tracker import HandTracker

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_FILE = DATA_DIR / "gestures_dataset.csv"

# Pre-configured gesture presets
PRESET_LABELS: Dict[str, str] = {
    "0": "MOVE",
    "1": "CLICK",
    "2": "DOUBLE_CLICK",
    "3": "RIGHT_CLICK",
    "4": "IDLE",
}


def extract_features_vector(landmarks) -> List[float]:
    """
    Extracts a scale-invariant, position-invariant 63-D feature vector from 21 landmarks.
    Wrist (landmark 0) is used as origin (0, 0, 0).
    Coordinates are normalized by palm size (dist between wrist and middle MCP).
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    palm_size = (dx * dx + dy * dy + dz * dz) ** 0.5
    if palm_size < 1e-4:
        palm_size = 1.0

    features = []
    for lm in landmarks:
        features.append((lm.x - wrist.x) / palm_size)
        features.append((lm.y - wrist.y) / palm_size)
        features.append((lm.z - wrist.z) / palm_size)
    return features


def init_csv():
    """Initializes CSV header if file doesn't exist."""
    if not CSV_FILE.exists():
        headers = ["label"]
        for i in range(21):
            headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def count_existing_samples() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if CSV_FILE.exists():
        with open(CSV_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    lbl = row[0]
                    counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def main():
    init_csv()
    sample_counts = count_existing_samples()

    print("=" * 65)
    print("   GESTUREFLOW AI - DATASET COLLECTOR FOR CUSTOM ML")
    print("=" * 65)
    print("Preset labels:")
    for key, name in PRESET_LABELS.items():
        print(f"  [{key}] : {name} (Current samples: {sample_counts.get(name, 0)})")
    print("\nControls:")
    print("  [SPACE] : Start / Stop recording frames for current label")
    print("  [0-4]   : Select preset label")
    print("  [C]     : Type a custom label name")
    print("  [Q]     : Quit")
    print(f"\nData will be appended to: {CSV_FILE}")
    print("=" * 65 + "\n")

    current_label = "MOVE"
    is_recording = False
    new_samples_recorded = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    tracker = HandTracker(num_hands=1, selfie_mode=True)

    try:
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                detected = tracker.process_frame(frame)

                hand_present = len(detected) > 0
                if hand_present:
                    hand = detected[0]
                    tracker.draw_skeleton(frame, hand)

                    if is_recording:
                        vec = extract_features_vector(hand.landmarks)
                        writer.writerow([current_label] + [f"{v:.5f}" for v in vec])
                        f.flush()
                        sample_counts[current_label] = sample_counts.get(current_label, 0) + 1
                        new_samples_recorded += 1

                # Draw UI HUD
                # Background banner
                cv2.rectangle(frame, (10, 10), (w - 10, 90), (20, 20, 20), -1)
                status_color = (0, 0, 255) if is_recording else (100, 100, 100)
                status_text = "● RECORDING" if is_recording else "PAUSED (Press SPACE)"

                cv2.putText(frame, f"Active Label: {current_label} ({sample_counts.get(current_label, 0)} samples)",
                            (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(frame, f"Status: {status_text} | Total New: {new_samples_recorded}",
                            (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

                # Show instructions at bottom
                cv2.putText(frame, "Keys: 0:MOVE | 1:CLICK | 2:DBL_CLICK | 3:RIGHT | SPACE:Record | Q:Quit",
                            (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("GestureFlow AI - Custom Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == 32:  # SPACE
                    is_recording = not is_recording
                    print(f"[{'RECORDING' if is_recording else 'PAUSED'}] Label: {current_label}")
                elif chr(key) in PRESET_LABELS:
                    current_label = PRESET_LABELS[chr(key)]
                    print(f"[LABEL SELECTED] -> {current_label}")
                elif key in (ord('c'), ord('C')):
                    is_recording = False
                    custom = input("\nEnter custom gesture label name: ").strip().upper()
                    if custom:
                        current_label = custom
                        print(f"[LABEL SELECTED] -> {current_label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print(f"\n[DONE] Finished! Dataset saved to {CSV_FILE}.")
        print("Summary of recorded samples:")
        for lbl, count in sample_counts.items():
            print(f"  {lbl}: {count} samples")
        print("\nNext step: Run 'python scripts/train_classifier.py' to train your model!")


if __name__ == "__main__":
    main()
