"""
GestureFlow AI - Real-Time Desktop Controller Interactive Demo
==============================================================
Demonstrates end-to-end hand gesture recognition connected to the safe
DesktopController for Windows media and volume control.

Safety Guarantee:
  - Starts in safe DRY_RUN mode by default (simulates actions without touching the OS).
  - Explicit keyboard command 'A' is required to arm ACTIVE mode.
  - Press 'D' at any time to return to DRY_RUN.
  - Press 'ESC' or 'E' for immediate EMERGENCY STOP.
  - Press 'Q' to cleanly release camera hardware and exit.
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
)
from src.gesture_events import (
    GestureEvent,
    GestureEventManager,
    GestureEventType,
)
from src.desktop_controller import (
    ActionType,
    ControllerState,
    DesktopController,
    WindowsMediaBackend,
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


def draw_hud(
    frame: cv2.typing.MatLike,
    state: ControllerState,
    fps: float,
    num_hands: int,
    last_action_msg: str,
) -> None:
    """Renders the top control banner and safety status overlay."""
    h, w, _ = frame.shape

    # 1. State-specific Banner Color & Label
    if state == ControllerState.ACTIVE:
        banner_bg = (0, 0, 180)       # Vivid Red
        state_label = "ACTIVE (LIVE CONTROL ENABLED)"
    elif state == ControllerState.DRY_RUN:
        banner_bg = (0, 140, 200)     # Amber / Yellow
        state_label = "DRY_RUN (SIMULATION MODE - SAFE)"
    elif state == ControllerState.EMERGENCY_STOP:
        banner_bg = (0, 0, 120)       # Dark Crimson
        state_label = "EMERGENCY STOP (ALL ACTIONS LOCKED)"
    else:  # DISABLED
        banner_bg = (40, 40, 40)      # Dark Gray
        state_label = "DISABLED (INERT)"

    # Draw top safety banner
    cv2.rectangle(frame, (0, 0), (w, 36), banner_bg, cv2.FILLED)
    cv2.putText(
        frame,
        f"STATE: {state_label}",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Telemetry & Action Bar (Left Side)
    cv2.rectangle(frame, (10, 44), (520, 130), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (10, 44), (520, 130), banner_bg, 2)

    # Line 1: FPS and Hands
    fps_color = (0, 255, 0) if num_hands > 0 else (0, 200, 255)
    cv2.putText(frame, f"FPS: {fps:.1f} | Hands: {num_hands}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    # Line 2: Last Action / Status
    action_color = (0, 255, 255) if state == ControllerState.ACTIVE else (200, 200, 200)
    cv2.putText(frame, f"Action: {last_action_msg}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.46, action_color, 1, cv2.LINE_AA)

    # Line 3: Keyboard Guide
    cv2.putText(frame, "[A] Arm Active | [D] Dry Run | [X] Disable | [ESC] E-Stop | [R] Reset | [Q] Quit", (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)


def run_desktop_demo(camera_index: int = 0) -> None:
    print("=" * 70)
    print("GestureFlow AI - Milestone 5: Safe Windows Desktop Controller Demo")
    print("=" * 70)
    print(f"Python interpreter: {sys.executable}")
    print(f"OpenCV version:     {cv2.__version__}")
    print("-" * 70)

    # 1. Initialize Pipeline & Controller
    try:
        tracker = HandTracker(num_hands=2, selfie_mode=True)
        recognizer = GestureRecognizer()
        stabilizer = GestureStabilizer(window_size=5, agreement_ratio=0.60, max_missing_frames=5)
        event_manager = GestureEventManager(cooldown_seconds=1.0)
        controller = DesktopController()
        print("[INFO] Computer vision pipeline and desktop controller ready.")
        print(f"[INFO] Initial Controller State: {controller.state.value} (Default Safe Mode)")
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}\n")
        return

    # 2. Open Camera
    cap = open_webcam(camera_index)
    if cap is None:
        print("\n[ERROR] Could not open webcam. Check privacy settings and connections.\n")
        tracker.close()
        return

    print("\n[CONFIGURED GESTURE MAPPINGS]:")
    print("  - FIST       -> Media Play / Pause")
    print("  - PEACE      -> Audio Mute Toggle")
    print("  - THUMBS_UP  -> Bounded Volume Up")
    print("  - POINTING   -> NO_OP (Reserved)")
    print("  - OPEN_PALM  -> NO_OP (Safety Rest Gesture)")
    print("\n[KEYBOARD CONTROLS]:")
    print("  - 'A' : ARM into ACTIVE mode (Live Windows control)")
    print("  - 'D' : Return to safe DRY_RUN mode (Simulation only)")
    print("  - 'X' : DISABLE all gesture controls")
    print("  - 'ESC' or 'E' : EMERGENCY STOP (Instantly lock all actions)")
    print("  - 'R' : Reset from EMERGENCY STOP back to DRY_RUN")
    print("  - 'Q' : Quit and release webcam cleanly\n")

    window_name = "GestureFlow AI - Safe Desktop Controller (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0
    frame_count = 0
    last_status_message = "Ready in DRY_RUN mode"

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

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks
            detected_hands = tracker.process_frame(frame)

            seen_hands: Set[str] = set()

            for hand in detected_hands:
                seen_hands.add(hand.handedness)

                # 1. Raw Recognition
                pred = recognizer.recognize(hand.pixel_landmarks)

                # 2. Temporal Stabilization
                stab = stabilizer.update_detailed(hand.handedness, pred.label)

                # 3. Gesture Event Debouncing
                event = event_manager.process(
                    handedness=hand.handedness,
                    gesture=stab.gesture,
                    is_stable=stab.is_stable,
                    timestamp=current_time,
                )

                # 4. Desktop Controller Dispatch (Handles state, allowlist, dry-run, cooldown)
                action_res = controller.handle_event(event, timestamp=current_time)
                if event and event.event_type == GestureEventType.TRIGGER:
                    last_status_message = action_res.message
                    print(f"[ACTION LOG] {action_res.message}")

                # Draw floating badge
                wrist = hand.pixel_landmarks[0]
                badge_color = (0, 0, 200) if controller.state == ControllerState.ACTIVE else (0, 160, 200)
                card_x = max(10, min(wrist.x - 50, frame.shape[1] - 220))
                card_y = max(45, min(wrist.y - 30, frame.shape[0] - 60))
                cv2.rectangle(frame, (card_x, card_y - 20), (card_x + 200, card_y + 20), (25, 25, 25), cv2.FILLED)
                cv2.rectangle(frame, (card_x, card_y - 20), (card_x + 200, card_y + 20), badge_color, 2)
                cv2.putText(frame, f"{hand.handedness}: {stab.gesture.value}", (card_x + 8, card_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Action: {action_res.action.value}", (card_x + 8, card_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            # Handle absent hands (tracking loss)
            for hand_name in ("Left", "Right"):
                if hand_name not in seen_hands:
                    stabilizer.register_missing(hand_name)
                    event_manager.on_hand_lost(hand_name, timestamp=current_time)

            # Draw skeletal overlay
            tracker.draw_landmarks(frame, detected_hands, draw_connections=True)

            # Draw Top HUD Banner
            draw_hud(
                frame=frame,
                state=controller.state,
                fps=fps,
                num_hands=len(detected_hands),
                last_action_msg=last_status_message,
            )

            cv2.imshow(window_name, frame)

            # Keyboard Command Handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print(f"[INFO] Exit requested by user after {frame_count} frames.")
                break
            elif key in (27, ord('e'), ord('E')):  # ESC or E
                controller.emergency_stop()
                last_status_message = "EMERGENCY STOP TRIGGERED (All Actions Locked)"
                print("\n[EMERGENCY STOP] All desktop actions immediately blocked!\n")
            elif key in (ord('a'), ord('A')):
                try:
                    controller.activate()
                    last_status_message = "ACTIVE MODE: Live Windows control is ON"
                    print("\n[WARNING] Controller ARMED in ACTIVE mode. Real Windows actions enabled.\n")
                except RuntimeError as e:
                    last_status_message = str(e)
                    print(f"[WARN] {e}")
            elif key in (ord('d'), ord('D')):
                try:
                    controller.enable_dry_run()
                    last_status_message = "Returned to safe DRY_RUN mode"
                    print("\n[INFO] Controller set to safe DRY_RUN mode (Simulation only).\n")
                except RuntimeError as e:
                    last_status_message = str(e)
                    print(f"[WARN] {e}")
            elif key in (ord('x'), ord('X')):
                controller.disable()
                last_status_message = "Controller DISABLED"
                print("\n[INFO] Controller DISABLED.\n")
            elif key in (ord('r'), ord('R')):
                if controller.state == ControllerState.EMERGENCY_STOP:
                    controller.reset_emergency_stop()
                    last_status_message = "Emergency stop reset -> Ready in DRY_RUN"
                    print("\n[INFO] Emergency stop reset. Controller is in safe DRY_RUN mode.\n")

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
    run_desktop_demo(camera_index=idx)
