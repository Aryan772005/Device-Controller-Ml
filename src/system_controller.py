"""
GestureFlow AI - Full Windows System Controller
================================================
Translates debounced gesture events + live hand position into complete Windows
system control: mouse movement, clicks, scrolling, keyboard shortcuts,
media playback, volume, and gaming inputs.

Architecture:
  - MouseController  : Maps normalized hand position to absolute screen coordinates.
  - KeyboardController: Sends virtual-key events via user32 (no shell commands).
  - SystemController : Main dispatcher -- handles TRIGGER, HOLD, and RELEASE events.

Safety:
  - Starts in DRY_RUN mode; arm with activate().
  - Emergency stop locks all output instantly.
  - All key codes are in a strict allowlist.
  - Mouse movement is bounded to screen dimensions.
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from src.gesture_events import GestureEvent, GestureEventType
from src.gesture_recognizer import GestureLabel


# ---------------------------------------------------------------------------
# Windows API constants
# ---------------------------------------------------------------------------
class _VK:
    MEDIA_PLAY_PAUSE = 0xB3
    VOLUME_MUTE      = 0xAD
    VOLUME_DOWN      = 0xAE
    VOLUME_UP        = 0xAF
    MEDIA_NEXT       = 0xB0
    MEDIA_PREV       = 0xB1
    ESCAPE   = 0x1B
    TAB      = 0x09
    LWIN     = 0x5B
    LALT     = 0xA4
    LCTRL    = 0xA2
    LSHIFT   = 0xA0
    SPACE    = 0x20
    RETURN   = 0x0D
    LEFT  = 0x25
    UP    = 0x26
    RIGHT = 0x27
    DOWN  = 0x28
    PGUP  = 0x21
    PGDN  = 0x22
    SNAPSHOT = 0x2C
    W = 0x57; A = 0x41; S = 0x53; D = 0x44
    Q = 0x51; E = 0x45; R = 0x52; F = 0x46; G = 0x47
    MOUSEEVENTF_LEFTDOWN   = 0x0002
    MOUSEEVENTF_LEFTUP     = 0x0004
    MOUSEEVENTF_RIGHTDOWN  = 0x0008
    MOUSEEVENTF_RIGHTUP    = 0x0010
    MOUSEEVENTF_WHEEL      = 0x0800

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP   = 0x0002


# ---------------------------------------------------------------------------
# State and mode enums
# ---------------------------------------------------------------------------
class ControllerState(str, Enum):
    DISABLED       = "DISABLED"
    DRY_RUN        = "DRY_RUN"
    ACTIVE         = "ACTIVE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class ControlMode(str, Enum):
    DESKTOP  = "DESKTOP"
    MEDIA    = "MEDIA"
    GAMING   = "GAMING"
    BROWSER  = "BROWSER"


@dataclass(frozen=True)
class SystemActionResult:
    gesture: GestureLabel
    handedness: str
    event_type: str
    action_name: str
    state: ControllerState
    mode: ControlMode
    executed: bool
    message: str
    timestamp: float


# ---------------------------------------------------------------------------
# Low-level Windows I/O
# ---------------------------------------------------------------------------
class _WindowsIO:
    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self._screen_w: int = self._user32.GetSystemMetrics(0)
        self._screen_h: int = self._user32.GetSystemMetrics(1)

    def key_tap(self, vk: int) -> bool:
        try:
            self._user32.keybd_event(vk, 0, KEYEVENTF_KEYDOWN, 0)
            self._user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def key_down(self, vk: int) -> bool:
        try:
            self._user32.keybd_event(vk, 0, KEYEVENTF_KEYDOWN, 0)
            return True
        except Exception:
            return False

    def key_up(self, vk: int) -> bool:
        try:
            self._user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def combo(self, *vk_codes: int) -> bool:
        try:
            for vk in vk_codes:
                self._user32.keybd_event(vk, 0, KEYEVENTF_KEYDOWN, 0)
            for vk in reversed(vk_codes):
                self._user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def mouse_move_absolute(self, x_norm: float, y_norm: float) -> bool:
        try:
            px = int(max(0.0, min(1.0, x_norm)) * self._screen_w)
            py = int(max(0.0, min(1.0, y_norm)) * self._screen_h)
            self._user32.SetCursorPos(px, py)
            return True
        except Exception:
            return False

    def mouse_click_left(self) -> bool:
        try:
            self._user32.mouse_event(_VK.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self._user32.mouse_event(_VK.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        except Exception:
            return False

    def mouse_click_right(self) -> bool:
        try:
            self._user32.mouse_event(_VK.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self._user32.mouse_event(_VK.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            return True
        except Exception:
            return False

    def mouse_double_click(self) -> bool:
        self.mouse_click_left()
        time.sleep(0.08)
        self.mouse_click_left()
        return True

    def mouse_scroll(self, delta: int) -> bool:
        try:
            self._user32.mouse_event(_VK.MOUSEEVENTF_WHEEL, 0, 0, delta, 0)
            return True
        except Exception:
            return False

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_w, self._screen_h


# ---------------------------------------------------------------------------
# Mouse smoother
# ---------------------------------------------------------------------------
class _MouseSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = max(0.05, min(1.0, alpha))
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
            return x, y
        self._x = self._x + self.alpha * (x - self._x)
        self._y = self._y + self.alpha * (y - self._y)
        return self._x, self._y

    def reset(self) -> None:
        self._x = None
        self._y = None


# ---------------------------------------------------------------------------
# Gesture -> Action mappings
# ---------------------------------------------------------------------------
DESKTOP_TRIGGER_MAP: Dict[GestureLabel, str] = {
    GestureLabel.PINCH:       "LEFT_CLICK",
    GestureLabel.FIST:        "RIGHT_CLICK",
    GestureLabel.PEACE:       "ALT_TAB",
    GestureLabel.THUMBS_UP:   "VOLUME_UP",
    GestureLabel.THUMBS_DOWN: "VOLUME_DOWN",
    GestureLabel.ROCK:        "SCREENSHOT",
    GestureLabel.OPEN_PALM:   "NO_OP",
    GestureLabel.POINTING:    "NO_OP",
    GestureLabel.CALL_ME:     "WIN_D",
    GestureLabel.THREE:       "SWITCH_MODE",
}

MEDIA_TRIGGER_MAP: Dict[GestureLabel, str] = {
    GestureLabel.FIST:        "PLAY_PAUSE",
    GestureLabel.PEACE:       "MUTE_TOGGLE",
    GestureLabel.THUMBS_UP:   "VOLUME_UP",
    GestureLabel.THUMBS_DOWN: "VOLUME_DOWN",
    GestureLabel.PINCH:       "MEDIA_NEXT",
    GestureLabel.ROCK:        "MEDIA_PREV",
    GestureLabel.CALL_ME:     "FULL_SCREEN",
    GestureLabel.THREE:       "SWITCH_MODE",
    GestureLabel.OPEN_PALM:   "NO_OP",
    GestureLabel.POINTING:    "NO_OP",
}

GAMING_HOLD_MAP: Dict[GestureLabel, int] = {
    GestureLabel.POINTING:    _VK.W,
    GestureLabel.FIST:        _VK.LSHIFT,
    GestureLabel.PEACE:       _VK.SPACE,
    GestureLabel.OPEN_PALM:   _VK.S,
    GestureLabel.THUMBS_UP:   _VK.D,
    GestureLabel.THUMBS_DOWN: _VK.A,
    GestureLabel.ROCK:        _VK.R,
    GestureLabel.PINCH:       _VK.F,
    GestureLabel.CALL_ME:     _VK.G,
    GestureLabel.THREE:       _VK.Q,
}

BROWSER_TRIGGER_MAP: Dict[GestureLabel, str] = {
    GestureLabel.OPEN_PALM:   "SCROLL_UP",
    GestureLabel.FIST:        "SCROLL_DOWN",
    GestureLabel.PEACE:       "BROWSER_BACK",
    GestureLabel.POINTING:    "BROWSER_FORWARD",
    GestureLabel.THUMBS_UP:   "NEW_TAB",
    GestureLabel.THUMBS_DOWN: "CLOSE_TAB",
    GestureLabel.PINCH:       "LEFT_CLICK",
    GestureLabel.ROCK:        "REFRESH",
    GestureLabel.CALL_ME:     "BROWSER_HOME",
    GestureLabel.THREE:       "SWITCH_MODE",
}

MODE_CYCLE = [ControlMode.DESKTOP, ControlMode.MEDIA, ControlMode.GAMING, ControlMode.BROWSER]


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------
class SystemController:
    def __init__(
        self,
        smooth_alpha: float = 0.35,
        action_cooldown: float = 0.5,
    ) -> None:
        self._io = _WindowsIO()
        self._smoother = _MouseSmoother(alpha=smooth_alpha)
        self._state = ControllerState.DRY_RUN
        self._mode = ControlMode.DESKTOP
        self._action_cooldown = action_cooldown
        self._last_action_times: Dict[str, float] = {}
        self._held_keys: Set[int] = set()
        self._log: List[SystemActionResult] = []

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def mode(self) -> ControlMode:
        return self._mode

    def activate(self) -> None:
        if self._state == ControllerState.EMERGENCY_STOP:
            raise RuntimeError("Cannot activate while in EMERGENCY_STOP.")
        self._state = ControllerState.ACTIVE

    def enable_dry_run(self) -> None:
        if self._state == ControllerState.EMERGENCY_STOP:
            raise RuntimeError("Cannot set DRY_RUN while in EMERGENCY_STOP.")
        self._release_all_held_keys()
        self._state = ControllerState.DRY_RUN

    def disable(self) -> None:
        self._release_all_held_keys()
        self._state = ControllerState.DISABLED

    def emergency_stop(self) -> None:
        self._release_all_held_keys()
        self._state = ControllerState.EMERGENCY_STOP

    def reset_emergency_stop(self) -> None:
        self._state = ControllerState.DRY_RUN
        self._last_action_times.clear()

    def set_mode(self, mode: ControlMode) -> None:
        self._release_all_held_keys()
        self._mode = mode

    def cycle_mode(self) -> ControlMode:
        idx = MODE_CYCLE.index(self._mode) if self._mode in MODE_CYCLE else 0
        self._mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
        self._release_all_held_keys()
        return self._mode

    def handle_event(
        self,
        event: Optional[GestureEvent],
        index_tip_norm: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
    ) -> SystemActionResult:
        ts = time.perf_counter() if timestamp is None else float(timestamp)

        if event is None:
            return self._result(GestureLabel.UNKNOWN, "None", "NONE", "NO_OP", False, "No event", ts)

        if self._state == ControllerState.DISABLED:
            return self._result(event.gesture, event.handedness, event.event_type.value,
                                "NO_OP", False, "DISABLED", ts)

        if self._state == ControllerState.EMERGENCY_STOP:
            return self._result(event.gesture, event.handedness, event.event_type.value,
                                "NO_OP", False, "EMERGENCY_STOP", ts)

        if self._mode == ControlMode.GAMING:
            return self._handle_gaming(event, ts)

        if (self._mode == ControlMode.DESKTOP and index_tip_norm is not None
                and event.gesture in (GestureLabel.OPEN_PALM, GestureLabel.POINTING)
                and event.event_type in (GestureEventType.HOLD, GestureEventType.TRIGGER)):
            return self._handle_mouse_move(event, index_tip_norm, ts)

        if event.event_type != GestureEventType.TRIGGER:
            return self._result(event.gesture, event.handedness, event.event_type.value,
                                "NO_OP", False, f"Ignored {event.event_type.value}", ts)

        return self._dispatch_trigger(event, ts)

    def _handle_gaming(self, event: GestureEvent, ts: float) -> SystemActionResult:
        vk = GAMING_HOLD_MAP.get(event.gesture)
        if vk is None:
            return self._result(event.gesture, event.handedness, event.event_type.value,
                                "NO_OP", False, "No gaming key mapped", ts)
        action_name = f"KEY_{hex(vk)}"
        if event.event_type == GestureEventType.TRIGGER:
            if self._state == ControllerState.DRY_RUN:
                return self._result(event.gesture, event.handedness, "TRIGGER",
                                    action_name, False, f"[DRY_RUN] Would press {hex(vk)}", ts)
            self._io.key_down(vk)
            self._held_keys.add(vk)
            return self._result(event.gesture, event.handedness, "TRIGGER",
                                action_name, True, f"Key DOWN {hex(vk)}", ts)
        elif event.event_type == GestureEventType.HOLD:
            return self._result(event.gesture, event.handedness, "HOLD",
                                action_name, True, f"Key HELD {hex(vk)}", ts)
        elif event.event_type == GestureEventType.RELEASE:
            if vk in self._held_keys:
                if self._state == ControllerState.ACTIVE:
                    self._io.key_up(vk)
                self._held_keys.discard(vk)
            return self._result(event.gesture, event.handedness, "RELEASE",
                                action_name, True, f"Key UP {hex(vk)}", ts)
        return self._result(event.gesture, event.handedness, event.event_type.value,
                            "NO_OP", False, "Unhandled gaming event", ts)

    def _handle_mouse_move(self, event: GestureEvent,
                           index_tip_norm: Tuple[float, float], ts: float) -> SystemActionResult:
        sx, sy = self._smoother.update(*index_tip_norm)
        if self._state == ControllerState.DRY_RUN:
            return self._result(event.gesture, event.handedness, event.event_type.value,
                                "MOUSE_MOVE", False, f"[DRY_RUN] ({sx:.3f}, {sy:.3f})", ts)
        success = self._io.mouse_move_absolute(sx, sy)
        return self._result(event.gesture, event.handedness, event.event_type.value,
                            "MOUSE_MOVE", success, f"Mouse({sx:.3f},{sy:.3f})", ts)

    def _dispatch_trigger(self, event: GestureEvent, ts: float) -> SystemActionResult:
        if self._mode == ControlMode.MEDIA:
            action_name = MEDIA_TRIGGER_MAP.get(event.gesture, "NO_OP")
        elif self._mode == ControlMode.BROWSER:
            action_name = BROWSER_TRIGGER_MAP.get(event.gesture, "NO_OP")
        else:
            action_name = DESKTOP_TRIGGER_MAP.get(event.gesture, "NO_OP")

        if action_name == "NO_OP":
            return self._result(event.gesture, event.handedness, "TRIGGER",
                                "NO_OP", False,
                                f"No action for {event.gesture.value} in {self._mode.value}", ts)

        if action_name == "SWITCH_MODE":
            new_mode = self.cycle_mode()
            return self._result(event.gesture, event.handedness, "TRIGGER",
                                "SWITCH_MODE", True, f"Mode -> {new_mode.value}", ts)

        last = self._last_action_times.get(action_name, -9999.0)
        cooldown = 0.18 if action_name in ("LEFT_CLICK", "RIGHT_CLICK") else self._action_cooldown
        if ts - last < cooldown:
            return self._result(event.gesture, event.handedness, "TRIGGER",
                                action_name, False, f"Cooldown ({ts-last:.2f}s)", ts)

        if self._state == ControllerState.DRY_RUN:
            self._last_action_times[action_name] = ts
            return self._result(event.gesture, event.handedness, "TRIGGER",
                                action_name, False, f"[DRY_RUN] {action_name}", ts)

        success = self._execute_action(action_name)
        self._last_action_times[action_name] = ts
        return self._result(event.gesture, event.handedness, "TRIGGER",
                            action_name, success, f"{'OK' if success else 'FAIL'}: {action_name}", ts)

    def _execute_action(self, action_name: str) -> bool:
        io = self._io
        actions = {
            "PLAY_PAUSE":      lambda: io.key_tap(_VK.MEDIA_PLAY_PAUSE),
            "MUTE_TOGGLE":     lambda: io.key_tap(_VK.VOLUME_MUTE),
            "MEDIA_NEXT":      lambda: io.key_tap(_VK.MEDIA_NEXT),
            "MEDIA_PREV":      lambda: io.key_tap(_VK.MEDIA_PREV),
            "FULL_SCREEN":     lambda: io.key_tap(0x7A),
            "VOLUME_UP":       lambda: all(io.key_tap(_VK.VOLUME_UP) or True for _ in range(3)),
            "VOLUME_DOWN":     lambda: all(io.key_tap(_VK.VOLUME_DOWN) or True for _ in range(3)),
            "LEFT_CLICK":      io.mouse_click_left,
            "RIGHT_CLICK":     io.mouse_click_right,
            "DOUBLE_CLICK":    io.mouse_double_click,
            "SCROLL_UP":       lambda: io.mouse_scroll(360),
            "SCROLL_DOWN":     lambda: io.mouse_scroll(-360),
            "ALT_TAB":         lambda: io.combo(_VK.LALT, _VK.TAB),
            "WIN_D":           lambda: io.combo(_VK.LWIN, _VK.D),
            "WIN_TAB":         lambda: io.combo(_VK.LWIN, _VK.TAB),
            "SCREENSHOT":      lambda: io.key_tap(_VK.SNAPSHOT),
            "COPY":            lambda: io.combo(_VK.LCTRL, 0x43),
            "PASTE":           lambda: io.combo(_VK.LCTRL, 0x56),
            "UNDO":            lambda: io.combo(_VK.LCTRL, 0x5A),
            "BROWSER_BACK":    lambda: io.combo(_VK.LALT, _VK.LEFT),
            "BROWSER_FORWARD": lambda: io.combo(_VK.LALT, _VK.RIGHT),
            "NEW_TAB":         lambda: io.combo(_VK.LCTRL, 0x54),
            "CLOSE_TAB":       lambda: io.combo(_VK.LCTRL, 0x57),
            "REFRESH":         lambda: io.key_tap(0x74),
            "BROWSER_HOME":    lambda: io.key_tap(0x24),
            "ARROW_UP":        lambda: io.key_tap(_VK.UP),
            "ARROW_DOWN":      lambda: io.key_tap(_VK.DOWN),
            "ARROW_LEFT":      lambda: io.key_tap(_VK.LEFT),
            "ARROW_RIGHT":     lambda: io.key_tap(_VK.RIGHT),
            "PAGE_UP":         lambda: io.key_tap(_VK.PGUP),
            "PAGE_DOWN":       lambda: io.key_tap(_VK.PGDN),
        }
        fn = actions.get(action_name)
        if fn is None:
            return False
        try:
            result = fn()
            return bool(result) if result is not None else True
        except Exception:
            return False

    def _release_all_held_keys(self) -> None:
        for vk in list(self._held_keys):
            try:
                self._io.key_up(vk)
            except Exception:
                pass
        self._held_keys.clear()
        self._smoother.reset()

    def _result(self, gesture, handedness, event_type, action_name,
                executed, message, ts) -> SystemActionResult:
        res = SystemActionResult(
            gesture=gesture, handedness=handedness, event_type=event_type,
            action_name=action_name, state=self._state, mode=self._mode,
            executed=executed, message=message, timestamp=ts,
        )
        self._log.append(res)
        if len(self._log) > 100:
            self._log.pop(0)
        return res

    def get_recent_logs(self, count: int = 5) -> List[SystemActionResult]:
        return self._log[-count:]

    def get_mode_description(self) -> str:
        lines = [f"=== {self._mode.value} MODE ==="]
        if self._mode == ControlMode.GAMING:
            for g, vk in GAMING_HOLD_MAP.items():
                lines.append(f"  {g.value:<16} -> HOLD key {hex(vk)}")
        elif self._mode == ControlMode.MEDIA:
            for g, a in MEDIA_TRIGGER_MAP.items():
                if a != "NO_OP":
                    lines.append(f"  {g.value:<16} -> {a}")
        elif self._mode == ControlMode.BROWSER:
            for g, a in BROWSER_TRIGGER_MAP.items():
                if a != "NO_OP":
                    lines.append(f"  {g.value:<16} -> {a}")
        else:
            lines.append("  OPEN_PALM / POINTING -> MOUSE MOVE (live tracking)")
            for g, a in DESKTOP_TRIGGER_MAP.items():
                if a != "NO_OP":
                    lines.append(f"  {g.value:<16} -> {a}")
        return "\n".join(lines)
