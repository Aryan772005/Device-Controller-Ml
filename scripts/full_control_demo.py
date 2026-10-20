from __future__ import annotations
import ctypes, sys, time
from pathlib import Path
from typing import Optional, Set
from collections import deque
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureLabel, GestureRecognizer, GestureStabilizer
from src.gesture_events import GestureEventManager, GestureEventType
from src.system_controller import (
    SystemController, ControllerState, ControlMode,
    DESKTOP_TRIGGER_MAP, MEDIA_TRIGGER_MAP, GAMING_HOLD_MAP, BROWSER_TRIGGER_MAP,
)

# ---------- Windows focus control ------------------------------------------
user32 = ctypes.windll.user32
HWND_TOPMOST  = -1
SWP_NOMOVE    = 0x0002
SWP_NOSIZE    = 0x0001
SWP_NOACTIVATE = 0x0010   # keeps window on top WITHOUT stealing focus from browser/game


def make_topmost_nofocus(title: str) -> None:
    hwnd = user32.FindWindowW(None, title)
    if hwnd:
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


# ---------- Virtual key codes ----------------------------------------------
VK_LEFT  = 0x25
VK_UP    = 0x26
VK_RIGHT = 0x27
VK_DOWN  = 0x28
VK_LALT  = 0xA4
VK_TAB   = 0x09

SWIPE_VK = {
    "SWIPE_LEFT":  VK_LEFT,
    "SWIPE_RIGHT": VK_RIGHT,
    "SWIPE_UP":    VK_UP,
    "SWIPE_DOWN":  VK_DOWN,
}
SWIPE_LABEL = {
    "SWIPE_LEFT":  "LEFT arrow - dodge left",
    "SWIPE_RIGHT": "RIGHT arrow - dodge right",
    "SWIPE_UP":    "UP arrow - jump",
    "SWIPE_DOWN":  "DOWN arrow - slide/duck",
}


def tap_key(vk: int) -> None:
    """Press and release a key. Goes to whatever window has focus (browser/game)."""
    user32.keybd_event(vk, 0, 0x0000, 0)
    user32.keybd_event(vk, 0, 0x0002, 0)


# ---------- Swipe detector -------------------------------------------------
class SwipeDetector:
    """Detects hand swipe direction from wrist position over N frames."""

    def __init__(self, frames: int = 8, threshold: float = 0.07, cooldown: float = 0.5):
        self.threshold = threshold
        self.cooldown  = cooldown
        self._pos      = deque(maxlen=frames)
        self._last_ts  = 0.0

    def update(self, x: float, y: float, ts: float) -> Optional[str]:
        self._pos.append((x, y))
        if len(self._pos) < 4:
            return None
        if ts - self._last_ts < self.cooldown:
            return None
        o = self._pos[0]
        n = self._pos[-1]
        dx = n[0] - o[0]
        dy = n[1] - o[1]
        if abs(dx) < self.threshold and abs(dy) < self.threshold:
            return None
        direction = ("SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT") if abs(dx) > abs(dy) \
                    else ("SWIPE_DOWN" if dy > 0 else "SWIPE_UP")
        self._last_ts = ts
        self._pos.clear()
        return direction

    def reset(self) -> None:
        self._pos.clear()


# ---------- HUD colors & labels -------------------------------------------
STATE_COLOR = {
    ControllerState.ACTIVE:         (0,  200,  80),
    ControllerState.DRY_RUN:        (0,  160, 230),
    ControllerState.EMERGENCY_STOP: (0,    0, 200),
    ControllerState.DISABLED:       (60,  60,  60),
}
MODE_COLOR = {
    ControlMode.DESKTOP: (30,  150, 230),
    ControlMode.MEDIA:   (160,   0, 200),
    ControlMode.GAMING:  (0,   200,  50),
    ControlMode.BROWSER: (0,   180, 160),
}
HINTS = {
    ControlMode.DESKTOP: {
        GestureLabel.POINTING:    "MOUSE MOVE",
        GestureLabel.OPEN_PALM:   "MOUSE MOVE",
        GestureLabel.PINCH:       "LEFT CLICK",
        GestureLabel.FIST:        "RIGHT CLICK",
        GestureLabel.PEACE:       "ALT+TAB",
        GestureLabel.THUMBS_UP:   "VOL UP",
        GestureLabel.THUMBS_DOWN: "VOL DOWN",
        GestureLabel.ROCK:        "SCREENSHOT",
        GestureLabel.CALL_ME:     "SHOW DESKTOP",
        GestureLabel.THREE:       "NEXT MODE",
    },
    ControlMode.MEDIA: {
        GestureLabel.FIST:        "PLAY/PAUSE",
        GestureLabel.PEACE:       "MUTE",
        GestureLabel.THUMBS_UP:   "VOL UP",
        GestureLabel.THUMBS_DOWN: "VOL DOWN",
        GestureLabel.PINCH:       "NEXT TRACK",
        GestureLabel.ROCK:        "PREV TRACK",
        GestureLabel.THREE:       "NEXT MODE",
    },
    ControlMode.GAMING: {
        GestureLabel.POINTING:    "W (forward)",
        GestureLabel.OPEN_PALM:   "S (backward)",
        GestureLabel.THUMBS_UP:   "D (strafe right)",
        GestureLabel.THUMBS_DOWN: "A (strafe left)",
        GestureLabel.FIST:        "SHIFT (sprint)",
        GestureLabel.PEACE:       "SPACE (jump)",
        GestureLabel.ROCK:        "R (reload)",
        GestureLabel.PINCH:       "F (interact)",
    },
    ControlMode.BROWSER: {
        GestureLabel.OPEN_PALM:   "SCROLL UP",
        GestureLabel.FIST:        "SCROLL DOWN",
        GestureLabel.PEACE:       "BACK",
        GestureLabel.POINTING:    "FORWARD",
        GestureLabel.THUMBS_UP:   "NEW TAB",
        GestureLabel.THUMBS_DOWN: "CLOSE TAB",
        GestureLabel.PINCH:       "CLICK",
        GestureLabel.THREE:       "NEXT MODE",
    },
}


def draw_overlay(frame, state, mode, fps, gesture, last_msg, game_mode):
    h, w = frame.shape[:2]
    sc = STATE_COLOR.get(state, (100, 100, 100))
    mc = (0, 120, 255) if game_mode else MODE_COLOR.get(mode, (150, 150, 150))

    # Top state banner
    cv2.rectangle(frame, (0, 0), (w, 44), sc, cv2.FILLED)
    stxt = {
        ControllerState.ACTIVE:         "ACTIVE - LIVE CONTROL",
        ControllerState.DRY_RUN:        "DRY RUN  (press A to arm)",
        ControllerState.EMERGENCY_STOP: "EMERGENCY STOP",
        ControllerState.DISABLED:       "DISABLED",
    }.get(state, state.value)
    cv2.putText(frame, stxt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    mlbl = "BROWSER GAME - SWIPE HAND!" if game_mode else mode.value
    cv2.putText(frame, mlbl, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (220, 220, 220), 1, cv2.LINE_AA)

    # Info box
    cv2.rectangle(frame, (8, 48), (w - 8, 132), (10, 10, 10), cv2.FILLED)
    cv2.rectangle(frame, (8, 48), (w - 8, 132), mc, 2)

    fps_col = (0, 255, 100) if fps > 20 else (0, 100, 255)
    cv2.putText(frame, "FPS %.1f" % fps, (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_col, 2, cv2.LINE_AA)

    hint = HINTS.get(mode, {}).get(gesture, "")
    g_txt = "%s  ->  %s" % (gesture.value, hint)
    cv2.putText(frame, g_txt, (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 100), 2, cv2.LINE_AA)

    cv2.putText(frame, last_msg[:58], (16, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)

    guide = "[A]Arm  [D]Safe  [E]Stop  [M]Mode  [G]=SubwaySurfers  [Q]Quit"
    cv2.putText(frame, guide, (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1, cv2.LINE_AA)


def open_webcam(idx: int = 0) -> Optional[cv2.VideoCapture]:
    for args in [(idx,), (idx, cv2.CAP_DSHOW)]:
        cap = cv2.VideoCapture(*args)
        if cap.isOpened():
            ret, f = cap.read()
            if ret and f is not None:
                return cap
            cap.release()
    return None


# ---------- Main -----------------------------------------------------------
def run(camera_index: int = 0, start_active: bool = False) -> None:
    print("=" * 60)
    print("GestureFlow AI - Full System Controller v2")
    print("=" * 60)
    print("Fixes: mouse tracks per-frame, overlay never steals focus")
    print("Press G to enter BROWSER GAME mode (Subway Surfers etc.)")
    print()

    tracker     = HandTracker(num_hands=1, selfie_mode=True)
    recognizer  = GestureRecognizer()
    stabilizer  = GestureStabilizer(window_size=5, agreement_ratio=0.60,
                                     max_missing_frames=5)
    evt_mgr     = GestureEventManager(cooldown_seconds=0.6,
                                      gesture_cooldowns={GestureLabel.PINCH: 0.18})
    controller  = SystemController(smooth_alpha=0.25, action_cooldown=0.4)
    if start_active:
        controller.activate()
        last_msg = "ACTIVE - Real control ON!"
        print("[ACTIVE] Real Windows control enabled directly!")
    else:
        last_msg = "Ready - press A to arm"
    swiper      = SwipeDetector()
    game_mode   = False

    cap = open_webcam(camera_index)
    if cap is None:
        print("[ERROR] Cannot open webcam.")
        tracker.close()
        return

    WIN = "GestureFlow AI  [Q=Quit  A=Arm  M=Mode  G=SubwaySurfers]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 420, 340)
    screen_w = user32.GetSystemMetrics(0)
    cv2.moveWindow(WIN, screen_w - 435, 10)   # top-right corner

    prev     = time.perf_counter()
    start    = prev
    fps      = 0.0
    cur_g    = GestureLabel.UNKNOWN
    focus_ok = False

    if start_active:
        print("[READY] ACTIVE MODE - Hands will control Windows now! (Press ESC for emergency stop, Q to quit)\n")
    else:
        print("[READY] A=Arm  G=Subway Surfers  M=cycle mode  Q=quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now   = time.perf_counter()
            dt    = now - prev
            prev  = now
            fps   = 1.0 / dt if fps == 0.0 else fps * 0.9 + (1.0 / dt) * 0.1

            frame = cv2.flip(frame, 1)
            hf, wf = frame.shape[:2]
            detected = tracker.process_frame(frame)
            seen: Set[str] = set()

            for hand in detected:
                seen.add(hand.handedness)
                lms = hand.pixel_landmarks

                # Normalized positions (0.0 - 1.0)
                wrist_x = lms[0].x / wf
                wrist_y = lms[0].y / hf
                idx_x   = lms[8].x / wf    # index finger tip
                idx_y   = lms[8].y / hf

                pred = recognizer.recognize(lms)
                stab = stabilizer.update_detailed(hand.handedness, pred.label)
                cur_g = stab.gesture

                # ── BROWSER GAME MODE (Subway Surfers) ──────────────────────
                if game_mode:
                    sw_dir = swiper.update(wrist_x, wrist_y, now)
                    if sw_dir:
                        vk = SWIPE_VK.get(sw_dir)
                        if vk:
                            if controller.state == ControllerState.ACTIVE:
                                tap_key(vk)
                            last_msg = SWIPE_LABEL.get(sw_dir, sw_dir)
                            print("[SWIPE]", last_msg)

                    # Static gesture overrides for jump / slide
                    event = evt_mgr.process(hand.handedness, stab.gesture,
                                            stab.is_stable, now)
                    if event and event.event_type == GestureEventType.TRIGGER:
                        if stab.gesture == GestureLabel.PEACE:
                            if controller.state == ControllerState.ACTIVE:
                                tap_key(VK_UP)
                            last_msg = "JUMP! (Peace sign)"
                            print("[GAME] JUMP")
                        elif stab.gesture == GestureLabel.FIST:
                            if controller.state == ControllerState.ACTIVE:
                                tap_key(VK_DOWN)
                            last_msg = "SLIDE! (Fist)"
                            print("[GAME] SLIDE")
                        elif stab.gesture == GestureLabel.THUMBS_UP:
                            if controller.state == ControllerState.ACTIVE:
                                tap_key(VK_RIGHT)
                            last_msg = "DODGE RIGHT (Thumbs up)"
                        elif stab.gesture == GestureLabel.THUMBS_DOWN:
                            if controller.state == ControllerState.ACTIVE:
                                tap_key(VK_LEFT)
                            last_msg = "DODGE LEFT (Thumbs down)"

                # ── NORMAL MODES ─────────────────────────────────────────────
                else:
                    # MOUSE TRACKING: every frame, no cooldown
                    if (controller.mode == ControlMode.DESKTOP
                            and controller.state == ControllerState.ACTIVE
                            and stab.gesture in (GestureLabel.POINTING,
                                                  GestureLabel.OPEN_PALM)):
                        controller._io.mouse_move_absolute(idx_x, idx_y)
                        last_msg = "Mouse (%.3f, %.3f)" % (idx_x, idx_y)

                    # Discrete events (clicks, shortcuts, gaming keys, scroll)
                    event = evt_mgr.process(hand.handedness, stab.gesture,
                                            stab.is_stable, now)
                    result = controller.handle_event(event,
                                                     index_tip_norm=(idx_x, idx_y),
                                                     timestamp=now)
                    if event and event.event_type == GestureEventType.TRIGGER:
                        if result.action_name != "MOUSE_MOVE":
                            last_msg = result.message
                        print("[%s] %s" % (result.mode.value, result.message))

                # Yellow dot on index finger tip
                cv2.circle(frame, (int(lms[8].x), int(lms[8].y)), 10,
                           (0, 255, 255), -1)

            for hn in ("Left", "Right"):
                if hn not in seen:
                    stabilizer.register_missing(hn)
                    evt_mgr.on_hand_lost(hn, timestamp=now)
                    swiper.reset()

            tracker.draw_landmarks(frame, detected, draw_connections=True)

            # Resize to compact overlay and draw HUD
            small = cv2.resize(frame, (420, 340))
            draw_overlay(small, controller.state, controller.mode,
                         fps, cur_g, last_msg, game_mode)
            cv2.imshow(WIN, small)

            # Set always-on-top WITHOUT stealing focus (do once after 1 second)
            if not focus_ok and (now - start) > 1.0:
                make_topmost_nofocus(WIN)
                focus_ok = True

            # ── Keyboard controls (processed even when cv window not focused)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (27, ord('e'), ord('E')):
                controller.emergency_stop()
                last_msg = "EMERGENCY STOP"
                print("[STOP] All actions blocked")
            elif key in (ord('a'), ord('A')):
                try:
                    controller.activate()
                    last_msg = "ACTIVE - Live control ON!"
                    print("[ACTIVE] Real Windows control enabled")
                except RuntimeError as ex:
                    last_msg = str(ex)
            elif key in (ord('d'), ord('D')):
                try:
                    controller.enable_dry_run()
                    last_msg = "DRY RUN - safe simulation"
                except RuntimeError as ex:
                    last_msg = str(ex)
            elif key in (ord('x'), ord('X')):
                controller.disable()
                last_msg = "DISABLED"
            elif key in (ord('r'), ord('R')):
                if controller.state == ControllerState.EMERGENCY_STOP:
                    controller.reset_emergency_stop()
                    last_msg = "Reset -> DRY RUN"
            elif key in (ord('m'), ord('M')):
                nm = controller.cycle_mode()
                game_mode = False
                last_msg = "Mode: " + nm.value
                print("[MODE]", nm.value)
            elif key in (ord('g'), ord('G')):
                game_mode = not game_mode
                swiper.reset()
                if game_mode:
                    last_msg = "BROWSER GAME ON! Swipe your hand left/right/up/down!"
                    print("[GAME MODE] ON - Swipe to play Subway Surfers!")
                    print("  SWIPE LEFT/RIGHT = dodge,  SWIPE UP = jump,  SWIPE DOWN = slide")
                    print("  PEACE sign = JUMP,  FIST = SLIDE,  THUMBS_UP = right,  THUMBS_DOWN = left")
                else:
                    last_msg = "Normal mode"
                    print("[GAME MODE] OFF")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("\n[INFO] Cleanup done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GestureFlow AI Full System Controller")
    parser.add_argument("camera_index", nargs="?", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--active", "-a", action="store_true", help="Start directly in ACTIVE mode (live Windows control)")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Camera device index")
    args = parser.parse_args()
    cam = args.camera if args.camera is not None else args.camera_index
    run(camera_index=cam, start_active=args.active)
