"""
GestureFlow AI - Camera Diagnostic Tool
=======================================
Verifies webcam hardware access, frame capture, and live display FPS.
Does not use MediaPipe or desktop automation.
"""

import sys
import time
import cv2


def open_camera(camera_index: int = 0):
    """
    Attempts to open the webcam using the default backend first.
    Falls back to DirectShow (CAP_DSHOW) on Windows if the default backend fails.
    """
    print(f"[INFO] Attempting to open camera at index {camera_index} (default backend)...")
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Camera opened successfully with default backend.")
            return cap
        cap.release()

    # Fallback for Windows: DirectShow backend often resolves MSMF conflicts
    print("[WARN] Default backend failed. Attempting DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Camera opened successfully with DirectShow backend.")
            return cap
        cap.release()

    return None


def run_camera_diagnostic(camera_index: int = 0):
    print("=" * 60)
    print("GestureFlow AI - Camera Diagnostic Test")
    print("=" * 60)
    print(f"Python interpreter: {sys.executable}")
    print(f"OpenCV version:     {cv2.__version__}")
    print("-" * 60)

    cap = open_camera(camera_index)

    if cap is None:
        print("\n" + "!" * 60)
        print("[ERROR] Could not open webcam!")
        print("Possible causes:")
        print("  1. Camera is already in use by another app (Zoom, Teams, OBS, browser).")
        print("  2. Windows Camera Privacy settings are blocking desktop app access.")
        print("     -> Open Settings > Privacy & Security > Camera > Allow desktop apps.")
        print(f"  3. External webcam is connected at a different index (try index 1 instead of 0).")
        print("  4. Running inside an environment without a connected physical display/camera.")
        print("!" * 60 + "\n")
        return False

    print("\n[SUCCESS] Webcam stream started.")
    print("Controls:")
    print("  - Press 'q' or 'Q' (or ESC) to close the camera and exit.\n")

    # Metrics for FPS computation
    prev_time = time.perf_counter()
    fps = 0.0
    frame_count = 0

    window_name = "GestureFlow AI - Camera Diagnostic (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[ERROR] Failed to read frame from camera stream. Exiting loop.")
                break

            current_time = time.perf_counter()
            delta = current_time - prev_time
            prev_time = current_time

            # Exponential moving average for smooth FPS display
            if delta > 0:
                current_fps = 1.0 / delta
                fps = current_fps if fps == 0.0 else (fps * 0.9 + current_fps * 0.1)

            frame_count += 1

            # Overlay information on the video frame
            h, w, _ = frame.shape
            fps_text = f"FPS: {fps:.1f} | Res: {w}x{h}"
            help_text = "Press 'Q' to Exit"

            # Draw semi-transparent background banner for readable text
            cv2.rectangle(frame, (10, 10), (320, 70), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, fps_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, help_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            # Wait 1ms for user keypress
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q', 'Q', or ESC
                print(f"[INFO] Exit key pressed by user after {frame_count} frames.")
                break

    finally:
        print("[INFO] Releasing camera hardware and destroying OpenCV windows...")
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera cleanup complete.")

    return True


if __name__ == "__main__":
    idx = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        idx = int(sys.argv[1])
    run_camera_diagnostic(camera_index=idx)
