"""
GestureFlow AI - Safe Windows Desktop Controller
================================================
Translates discrete, debounced gesture events into well-scoped, safe Windows actions
(media play/pause, mute toggle, bounded volume adjustments).

Safety Principles:
  1. Default State: Always starts in DRY_RUN, never ACTIVE.
  2. Single-Trigger Binding: Binds strictly to TRIGGER events, ignoring continuous HOLDs.
  3. Strict Allowlist: Only permitted, pre-defined actions may execute.
  4. Emergency Stop: Immediately locks down execution; cannot automatically re-arm.
  5. Zero Shell/Filesystem Execution: Communicates only via Windows user32 virtual key events.
  6. Decoupled Hardware: Built with mockable backend protocols for safe unit testing.
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Set

from src.gesture_events import GestureEvent, GestureEventType
from src.gesture_recognizer import GestureLabel


class ControllerState(str, Enum):
    """Safety state machine for desktop automation."""
    DISABLED = "DISABLED"              # Completely inert; ignores all gestures
    DRY_RUN = "DRY_RUN"                # Default mode; logs and simulates actions without touching the OS
    ACTIVE = "ACTIVE"                  # Live control; sends genuine Windows media/volume key events
    EMERGENCY_STOP = "EMERGENCY_STOP"  # Emergency lock; all actions blocked until manual reset


class ActionType(str, Enum):
    """Strict allowlist of permitted desktop actions."""
    MEDIA_PLAY_PAUSE = "MEDIA_PLAY_PAUSE"
    MUTE_TOGGLE = "MUTE_TOGGLE"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    NO_OP = "NO_OP"


# Allowlist validation set
ALLOWED_ACTIONS: Set[ActionType] = {
    ActionType.MEDIA_PLAY_PAUSE,
    ActionType.MUTE_TOGGLE,
    ActionType.VOLUME_UP,
    ActionType.VOLUME_DOWN,
    ActionType.NO_OP,
}

# Centralized, single configuration mapping gestures to permitted actions
DEFAULT_GESTURE_ACTION_MAPPING: Dict[GestureLabel, ActionType] = {
    GestureLabel.FIST: ActionType.MEDIA_PLAY_PAUSE,
    GestureLabel.PEACE: ActionType.MUTE_TOGGLE,
    GestureLabel.THUMBS_UP: ActionType.VOLUME_UP,
    GestureLabel.POINTING: ActionType.NO_OP,       # Reserved for future pointer navigation
    GestureLabel.OPEN_PALM: ActionType.NO_OP,      # Reserved as safety rest gesture
    GestureLabel.UNKNOWN: ActionType.NO_OP,        # Ambiguous postures produce no action
}


@dataclass(frozen=True)
class ActionResult:
    """
    Detailed audit trail of an action dispatch attempt.

    Attributes:
        action: The ActionType requested.
        gesture: The triggering GestureLabel.
        handedness: 'Left' or 'Right'.
        state: Controller state when the action was evaluated.
        executed: True ONLY if a real Windows action was executed in ACTIVE mode.
        message: Human-readable status description.
        timestamp: Time of dispatch in seconds.
    """
    action: ActionType
    gesture: GestureLabel
    handedness: str
    state: ControllerState
    executed: bool
    message: str
    timestamp: float


class ActionBackend(Protocol):
    """Protocol for abstracting action execution, enabling mock testing without touching OS."""
    def play_pause(self) -> bool: ...
    def toggle_mute(self) -> bool: ...
    def volume_up(self, steps: int = 1) -> bool: ...
    def volume_down(self, steps: int = 1) -> bool: ...


class WindowsMediaBackend:
    """
    Native Windows implementation using user32.dll virtual-key events.
    Zero third-party dependencies, zero shell commands, zero filesystem access.
    """
    # Windows Virtual Key Codes
    VK_VOLUME_MUTE = 0xAD        # 173
    VK_VOLUME_DOWN = 0xAE        # 174
    VK_VOLUME_UP = 0xAF          # 175
    VK_MEDIA_PLAY_PAUSE = 0xB3   # 179
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self) -> None:
        try:
            self._user32 = ctypes.windll.user32
        except Exception as e:
            raise RuntimeError(f"Unable to access Windows user32.dll: {e}") from e

    def _send_key(self, vk_code: int) -> bool:
        """Sends key-down followed by key-up to user32."""
        try:
            self._user32.keybd_event(vk_code, 0, 0, 0)
            self._user32.keybd_event(vk_code, 0, self.KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def play_pause(self) -> bool:
        return self._send_key(self.VK_MEDIA_PLAY_PAUSE)

    def toggle_mute(self) -> bool:
        return self._send_key(self.VK_VOLUME_MUTE)

    def volume_up(self, steps: int = 1) -> bool:
        success = True
        for _ in range(max(1, min(steps, 5))):
            if not self._send_key(self.VK_VOLUME_UP):
                success = False
            time.sleep(0.02)
        return success

    def volume_down(self, steps: int = 1) -> bool:
        success = True
        for _ in range(max(1, min(steps, 5))):
            if not self._send_key(self.VK_VOLUME_DOWN):
                success = False
            time.sleep(0.02)
        return success


class MockActionBackend:
    """Mock backend for unit testing and headless environments."""
    def __init__(self) -> None:
        self.call_history: List[str] = []

    def play_pause(self) -> bool:
        self.call_history.append("play_pause")
        return True

    def toggle_mute(self) -> bool:
        self.call_history.append("toggle_mute")
        return True

    def volume_up(self, steps: int = 1) -> bool:
        self.call_history.append(f"volume_up({steps})")
        return True

    def volume_down(self, steps: int = 1) -> bool:
        self.call_history.append(f"volume_down({steps})")
        return True


class DesktopController:
    """
    Centralized, safety-first desktop automation controller.

    Safety Guarantees:
      - Defaults to DRY_RUN state.
      - Dispatches actions only for stable TRIGGER events.
      - Enforces per-action cooldowns.
      - Emergency stop requires explicit reset back to DRY_RUN before re-arming.
    """

    def __init__(
        self,
        backend: Optional[ActionBackend] = None,
        gesture_mapping: Optional[Dict[GestureLabel, ActionType]] = None,
        action_cooldowns: Optional[Dict[ActionType, float]] = None,
        max_volume_steps: int = 1,
    ) -> None:
        """
        Initializes the controller in DRY_RUN mode.

        Args:
            backend: Action execution implementation (defaults to native WindowsMediaBackend).
            gesture_mapping: Custom gesture-to-action dictionary (defaults to DEFAULT_GESTURE_ACTION_MAPPING).
            action_cooldowns: Per-action minimum interval in seconds.
            max_volume_steps: Maximum volume increments allowed per trigger (bounded safety).
        """
        self._state = ControllerState.DRY_RUN
        self._backend: ActionBackend = backend if backend is not None else WindowsMediaBackend()
        self._gesture_mapping: Dict[GestureLabel, ActionType] = (
            gesture_mapping if gesture_mapping is not None else DEFAULT_GESTURE_ACTION_MAPPING.copy()
        )
        self.max_volume_steps = max(1, min(max_volume_steps, 5))

        # Action cooldowns in seconds
        self._action_cooldowns: Dict[ActionType, float] = {
            ActionType.MEDIA_PLAY_PAUSE: 1.0,
            ActionType.MUTE_TOGGLE: 1.0,
            ActionType.VOLUME_UP: 0.3,
            ActionType.VOLUME_DOWN: 0.3,
            ActionType.NO_OP: 0.0,
        }
        if action_cooldowns:
            self._action_cooldowns.update(action_cooldowns)

        self._last_action_times: Dict[ActionType, float] = {}
        self._log: Deque[ActionResult] = deque(maxlen=50)

    @property
    def state(self) -> ControllerState:
        """Current operating state of the controller."""
        return self._state

    def enable_dry_run(self) -> None:
        """Switches controller to safe DRY_RUN mode (logs actions without calling OS)."""
        if self._state == ControllerState.EMERGENCY_STOP:
            # Cannot switch directly from Emergency Stop to Dry Run without reset
            raise RuntimeError("Cannot switch to DRY_RUN while in EMERGENCY_STOP. Call reset_emergency_stop() first.")
        self._state = ControllerState.DRY_RUN

    def activate(self) -> None:
        """
        Arms the controller into ACTIVE mode.
        Incoming stable gestures will execute genuine Windows actions.
        """
        if self._state == ControllerState.EMERGENCY_STOP:
            raise RuntimeError("Cannot activate while in EMERGENCY_STOP. Call reset_emergency_stop() first.")
        self._state = ControllerState.ACTIVE

    def disable(self) -> None:
        """Disables the controller completely (ignores all gestures)."""
        self._state = ControllerState.DISABLED

    def emergency_stop(self) -> None:
        """
        Immediately locks down the controller.
        Halts all real actions and enters EMERGENCY_STOP state.
        Cannot automatically reactivate without explicit user reset.
        """
        self._state = ControllerState.EMERGENCY_STOP

    def reset_emergency_stop(self) -> None:
        """
        Clears the emergency stop and returns safely to DRY_RUN mode.
        Does NOT return to ACTIVE mode.
        """
        self._state = ControllerState.DRY_RUN
        self._last_action_times.clear()

    def handle_event(
        self,
        event: Optional[GestureEvent],
        timestamp: Optional[float] = None,
    ) -> ActionResult:
        """
        Centralized event evaluation and dispatch method.

        Args:
            event: GestureEvent produced by GestureEventManager.
            timestamp: Time of evaluation (defaults to time.perf_counter()).

        Returns:
            ActionResult documenting the decision, execution status, and audit message.
        """
        ts = time.perf_counter() if timestamp is None else float(timestamp)

        # 1. Null check / Tracking loss
        if event is None:
            return ActionResult(
                action=ActionType.NO_OP,
                gesture=GestureLabel.UNKNOWN,
                handedness="None",
                state=self._state,
                executed=False,
                message="Tracking lost or event suppressed",
                timestamp=ts,
            )

        # 2. Continuous HOLD or RELEASE events do not execute actions
        if event.event_type != GestureEventType.TRIGGER:
            return ActionResult(
                action=ActionType.NO_OP,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"Ignored non-trigger event type ({event.event_type.value})",
                timestamp=ts,
            )

        # 3. Resolve configured action
        action = self._gesture_mapping.get(event.gesture, ActionType.NO_OP)

        # 4. Strict Allowlist check
        if action not in ALLOWED_ACTIONS:
            res = ActionResult(
                action=ActionType.NO_OP,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"Action '{action}' is not in the permitted allowlist",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 5. NO_OP or UNKNOWN gesture
        if action == ActionType.NO_OP or event.gesture == GestureLabel.UNKNOWN:
            res = ActionResult(
                action=ActionType.NO_OP,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"No action mapped for gesture '{event.gesture.value}'",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 6. Safety State Gate: DISABLED
        if self._state == ControllerState.DISABLED:
            res = ActionResult(
                action=action,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"Action '{action.value}' suppressed: controller is DISABLED",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 7. Safety State Gate: EMERGENCY_STOP
        if self._state == ControllerState.EMERGENCY_STOP:
            res = ActionResult(
                action=action,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"Action '{action.value}' BLOCKED: EMERGENCY_STOP is active",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 8. Per-Action Cooldown Gate
        min_cooldown = self._action_cooldowns.get(action, 0.5)
        last_time = self._last_action_times.get(action, -999.0)
        elapsed = ts - last_time
        if elapsed < min_cooldown:
            res = ActionResult(
                action=action,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"Action '{action.value}' suppressed by cooldown ({elapsed:.2f}s < {min_cooldown:.2f}s)",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 9. Safety State Gate: DRY_RUN mode
        if self._state == ControllerState.DRY_RUN:
            self._last_action_times[action] = ts
            res = ActionResult(
                action=action,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=False,
                message=f"[DRY_RUN SIMULATION] Would execute '{action.value}' on Windows",
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # 10. ACTIVE mode execution
        if self._state == ControllerState.ACTIVE:
            self._last_action_times[action] = ts
            success = False
            err_msg = ""
            try:
                if action == ActionType.MEDIA_PLAY_PAUSE:
                    success = self._backend.play_pause()
                elif action == ActionType.MUTE_TOGGLE:
                    success = self._backend.toggle_mute()
                elif action == ActionType.VOLUME_UP:
                    success = self._backend.volume_up(steps=self.max_volume_steps)
                elif action == ActionType.VOLUME_DOWN:
                    success = self._backend.volume_down(steps=self.max_volume_steps)
            except Exception as e:
                success = False
                err_msg = f" (Backend Exception: {e})"

            status_msg = f"Executed '{action.value}' on Windows{err_msg}" if success else f"Failed to execute '{action.value}'{err_msg}"
            res = ActionResult(
                action=action,
                gesture=event.gesture,
                handedness=event.handedness,
                state=self._state,
                executed=success,
                message=status_msg,
                timestamp=ts,
            )
            self._log.append(res)
            return res

        # Fallthrough safety
        return ActionResult(
            action=ActionType.NO_OP,
            gesture=event.gesture,
            handedness=event.handedness,
            state=self._state,
            executed=False,
            message="Unhandled state fallthrough",
            timestamp=ts,
        )

    def get_recent_logs(self, count: int = 5) -> List[ActionResult]:
        """Returns the most recent n action dispatch results."""
        return list(self._log)[-count:]
