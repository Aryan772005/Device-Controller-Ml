"""
GestureFlow AI - Real-Time Gesture Recognition & Diagnostics Demo
=================================================================
Demonstrates real-time hand tracking, rule-based gesture recognition,
temporal stabilization, and event-debouncing diagnostics.
Zero desktop automation or OS interactions are performed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_tracker import HandTracker
from src.gesture_recognizer import (
    GestureLabel,
    GestureRecognizer,
    GestureStabilizer,
    StabilizationResult,
)
from src.gesture_events import (
    GestureEvent,
    GestureEventManager,
    GestureEventType,
)


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture | None:
    """Attempts to open webcam with default backend first, then DirectShow fallback on Windows."""
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


def draw_gesture_card(
    frame: cv2.typing.MatLike,
    x: int,
    y: int,
    handedness: str,
    stab_res: StabilizationResult,
    trigger_count: int,
    recent_event: Optional[GestureEvent],
) -> None:
    """Renders a detailed diagnostic floating badge above each detected hand."""
    gesture = stab_res.gesture

    # Color-code by gesture type
    if gesture == GestureLabel.UNKNOWN:
        badge_color = (80, 80, 80)      # Gray
        text_color = (200, 200, 200)
    elif gesture in (GestureLabel.OPEN_PALM, GestureLabel.PEACE):
        badge_color = (0, 180, 0)       # Green
        text_color = (255, 255, 255)
    elif gesture == GestureLabel.FIST:
        badge_color = (0, 120, 220)     # Amber / Orange
        text_color = (255, 255, 255)
    elif gesture == GestureLabel.THUMBS_UP:
        badge_color = (200, 150, 0)     # Cyan
        text_color = (255, 255, 255)
    else:  # POINTING
        badge_color = (200, 0, 180)     # Magenta
        text_color = (255, 255, 255)

    # Flash white border if this frame triggered a new event
    border_color = (255, 255, 255) if (recent_event and recent_event.event_type == GestureEventType.TRIGGER) else badge_color

    card_x = max(10, min(x - 60, frame.shape[1] - 250))
    card_y = max(45, min(y - 45, frame.shape[0] - 80))

    # Badge background
    cv2.rectangle(frame, (card_x, card_y - 25), (card_x + 230, card_y + 35), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (card_x, card_y - 25), (card_x + 230, card_y + 35), border_color, 2)

    # Line 1: Handedness & Stable Gesture
    title_text = f"{handedness}: {gesture.value}"
    cv2.putText(frame, title_text, (card_x + 8, card_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)

    # Line 2: Raw gesture and Stability status
    status_str = "STABLE" if stab_res.is_stable else "UNSTABLE"
    status_color = (0, 255, 0) if stab_res.is_stable else (0, 165, 255)
    diag_text = f"Raw: {stab_res.raw_gesture.value[:4]} | {status_str} ({stab_res.agreement_ratio * 100:.0f}%)"
    cv2.putText(frame, diag_text, (card_x + 8, card_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

    # Line 3: Event state & Total Trigger Count
    event_str = recent_event.event_type.value if recent_event else "IDLE"
    event_text = f"State: {event_str} | Triggers: {trigger_count}"
    cv2.putText(frame, event_text, (card_x + 8, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def run_gesture_demo(camera_index: int = 0) -> None:
    print("=" * 68)
    print("GestureFlow AI - Milestone 4: Reliability, Diagnostics & Debouncing")
    print("=" * 68)
    print(f"Python interpreter: {sys.executable}")
    print(f"OpenCV version:     {cv2.__version__}")
    print("-" * 68)

    # 1. Initialize Hand Tracker, Gesture Engine, and Event Manager
    try:
        tracker = HandTracker(num_hands=2, selfie_mode=True)
        recognizer = GestureRecognizer()
        stabilizer = GestureStabilizer(window_size=5, agreement_ratio=0.60, max_missing_frames=5)
        event_manager = GestureEventManager(cooldown_seconds=1.0)
        print("[INFO] Computer vision and event pipeline initialized successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize pipeline: {e}\n")
        return

    # 2. Open Camera
    cap = open_webcam(camera_index)
    if cap is None:
        print("\n[ERROR] Could not open webcam. Check privacy settings and connections.\n")
        tracker.close()
        return

    print("\n[SUCCESS] Webcam stream running with Gesture Event Debouncing.")
    print("Diagnostic Controls:")
    print("  - Show hand: Gesture will stabilize (STABLE) and fire ONE TRIGGER event.")
    print("  - Hold hand: Gesture enters HOLD state (no accidental repeat triggers).")
    print("  - Relax hand: Gesture RELEASE is logged and state resets to IDLE.")
    print("  - Press 'Q' or ESC to exit cleanly (Zero OS automation performed).\n")

    window_name = "GestureFlow AI - Milestone 4 Diagnostics (Press Q to Exit)"
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

            if delta > 0:
                current_fps = 1.0 / delta
                fps = current_fps if fps == 0.0 else (fps * 0.9 + current_fps * 0.1)

            frame_count += 1

            # Flip frame horizontally for selfie-mirror view
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks
            detected_hands = tracker.process_frame(frame)

            seen_hands: Set[str] = set()
            active_summaries: List[str] = []

            for hand in detected_hands:
                seen_hands.add(hand.handedness)

                # 1. Raw Recognition
                prediction = recognizer.recognize(hand.pixel_landmarks)

                # 2. Temporal Stabilization
                stab_res = stabilizer.update_detailed(hand.handedness, prediction.label)

                # 3. Event Debouncing
                event = event_manager.process(
                    handedness=hand.handedness,
                    gesture=stab_res.gesture,
                    is_stable=stab_res.is_stable,
                    timestamp=current_time,
                )

                trigger_cnt = event_manager.get_trigger_count(hand.handedness)
                active_summaries.append(f"{hand.handedness}: {stab_res.gesture.value} (Triggers: {trigger_cnt})")

                # 4. Draw Floating Diagnostic Badge
                wrist_pt = hand.pixel_landmarks[0]
                draw_gesture_card(
                    frame=frame,
                    x=wrist_pt.x,
                    y=wrist_pt.y,
                    handedness=hand.handedness,
                    stab_res=stab_res,
                    trigger_count=trigger_cnt,
                    recent_event=event,
                )

            # 5. Handle Hands Leaving Camera Frame
            for hand_name in ("Left", "Right"):
                if hand_name not in seen_hands:
                    stabilizer.register_missing(hand_name)
                    event_manager.on_hand_lost(hand_name, timestamp=current_time)

            # Draw skeleton overlay
            tracker.draw_landmarks(frame, detected_hands, draw_connections=True)

            # HUD Display (Top Left)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (480, 100), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            num_hands = len(detected_hands)
            status_color = (0, 255, 0) if num_hands > 0 else (0, 200, 255)
            fps_text = f"FPS: {fps:.1f} | Hands: {num_hands}"
            cv2.putText(frame, fps_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            if num_hands == 0:
                cv2.putText(frame, "No hand detected. Show hand to camera.", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
            else:
                summary_line = " | ".join(active_summaries)
                cv2.putText(frame, summary_line, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, "Press 'Q' or ESC to exit (Diagnostics Mode - No Desktop Actions)", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

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
    run_gesture_demo(camera_index=idx)
