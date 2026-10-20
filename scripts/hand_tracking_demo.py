"""
GestureFlow AI - Hand Tracking Interactive Demo
===============================================
Demonstrates real-time hand detection and 21-landmark skeletal tracking
using the modular HandTracker class and OpenCV webcam stream.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_tracker import HandTracker


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture | None:
    """
    Attempts to open webcam with default backend first, then DirectShow on Windows.
    """
    print(f"[INFO] Opening camera index {camera_index} (default backend)...")
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Camera opened successfully with default backend.")
            return cap
        cap.release()

    print("[WARN] Default backend failed. Attempting DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Camera opened successfully with DirectShow backend.")
            return cap
        cap.release()

    return None


def run_hand_tracking_demo(camera_index: int = 0) -> None:
    print("=" * 65)
    print("GestureFlow AI - Milestone 2: Hand Tracking Demo")
    print("=" * 65)
    print(f"Python interpreter: {sys.executable}")
    print(f"OpenCV version:     {cv2.__version__}")
    print("-" * 65)

    # 1. Initialize HandTracker
    try:
        tracker = HandTracker(num_hands=2)
        print(f"[INFO] HandTracker initialized successfully.")
        print(f"[INFO] Loaded model: {tracker.model_path}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model asset not found:\n{e}\n")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize HandTracker:\n{e}\n")
        return

    # 2. Open Camera
    cap = open_webcam(camera_index)
    if cap is None:
        print("\n[ERROR] Could not open webcam. Check privacy settings and connections.\n")
        tracker.close()
        return

    print("\n[SUCCESS] Webcam stream running with MediaPipe Tasks HandLandmarker.")
    print("Controls:")
    print("  - Show one or both hands to the webcam.")
    print("  - Press 'Q' or ESC to exit cleanly.\n")

    window_name = "GestureFlow AI - Hand Tracking Demo (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[ERROR] Frame capture failed. Exiting loop.")
                break

            current_time = time.perf_counter()
            delta = current_time - prev_time
            prev_time = current_time

            # Compute smoothed FPS
            if delta > 0:
                current_fps = 1.0 / delta
                fps = current_fps if fps == 0.0 else (fps * 0.9 + current_fps * 0.1)

            frame_count += 1

            # Flip frame horizontally for natural selfie-mirror viewing
            frame = cv2.flip(frame, 1)

            # Process frame with HandTracker
            detected_hands = tracker.process_frame(frame)

            # Draw 21 landmarks and skeletal connections
            tracker.draw_landmarks(frame, detected_hands, draw_connections=True)

            # Overlay information HUD
            h, w, _ = frame.shape
            num_hands = len(detected_hands)

            # Semi-transparent top HUD banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 95), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Line 1: FPS and Hand count
            status_color = (0, 255, 0) if num_hands > 0 else (0, 200, 255)
            fps_text = f"FPS: {fps:.1f} | Hands Detected: {num_hands}"
            cv2.putText(frame, fps_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            # Line 2: Hand details or "No hand detected" message
            if num_hands == 0:
                cv2.putText(frame, "No hand detected. Show your hand to the camera.", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
            else:
                labels = [f"{h.handedness} ({h.score:.2f})" for h in detected_hands]
                detail_text = "Detected: " + ", ".join(labels)
                cv2.putText(frame, detail_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Line 3: Exit prompt
            cv2.putText(frame, "Press 'Q' or ESC to exit", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print(f"[INFO] Exit requested by user after {frame_count} frames.")
                break

    finally:
        print("[INFO] Releasing webcam hardware...")
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Closing HandTracker landmarker...")
        tracker.close()
        print("[INFO] Cleanup complete. Resources freed.")


if __name__ == "__main__":
    idx = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        idx = int(sys.argv[1])
    run_hand_tracking_demo(camera_index=idx)
