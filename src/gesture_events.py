"""
GestureFlow AI - Gesture Event Detection & Debouncing
=====================================================
Transforms continuous stabilized hand gestures into discrete software events.
Enforces debouncing, cooldowns, hold tracking, and single-trigger semantics.
Contains ZERO operating system or desktop automation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.gesture_recognizer import GestureLabel


class GestureEventType(str, Enum):
    """Lifecycle stages of a recognized gesture."""
    TRIGGER = "TRIGGER"  # Fired once when a stable gesture is newly activated
    HOLD = "HOLD"        # Fired while the gesture remains continuously stable
    RELEASE = "RELEASE"  # Fired when the gesture ends or transitions


@dataclass(frozen=True)
class GestureEvent:
    """
    Structured software event emitted when gesture states change.

    Attributes:
        event_type: TRIGGER, HOLD, or RELEASE.
        gesture: The GestureLabel associated with this event.
        handedness: 'Left' or 'Right'.
        timestamp: Time of event in seconds.
        hold_duration: Total seconds the gesture has been held continuously.
        frame_count: Number of consecutive frames the gesture has been held.
    """
    event_type: GestureEventType
    gesture: GestureLabel
    handedness: str
    timestamp: float
    hold_duration: float = 0.0
    frame_count: int = 1


class HandEventState:
    """Internal state tracking for a single hand."""

    def __init__(self) -> None:
        self.active_gesture: Optional[GestureLabel] = None
        self.gesture_start_time: float = 0.0
        self.gesture_frame_count: int = 0
        self.last_trigger_times: Dict[GestureLabel, float] = {}
        self.total_triggers: int = 0

    def reset(self) -> None:
        self.active_gesture = None
        self.gesture_start_time = 0.0
        self.gesture_frame_count = 0
        self.last_trigger_times.clear()
        self.total_triggers = 0


class GestureEventManager:
    """
    Event management and debouncing system.

    Key Features:
      - Single-Trigger Semantics: An extended gesture generates exactly ONE TRIGGER event.
        Subsequent frames emit HOLD events without re-triggering actions.
      - Cooldown Enforcement: A gesture must be released and wait for the cooldown
        period before it can trigger again.
      - Clean Release: Transitions or tracking loss cleanly emit a RELEASE event.
      - Independent Hands: Left and Right hands have completely isolated state machines.
    """

    def __init__(
        self,
        cooldown_seconds: float = 1.0,
        gesture_cooldowns: Optional[Dict[GestureLabel, float]] = None,
    ) -> None:
        """
        Args:
            cooldown_seconds: Minimum seconds required between repeated triggers of the same gesture.
            gesture_cooldowns: Optional per-gesture cooldown overrides (e.g. fast PINCH for clicking).
        """
        self.cooldown_seconds = cooldown_seconds
        self.gesture_cooldowns = gesture_cooldowns or {}
        self._states: Dict[str, HandEventState] = {
            "Left": HandEventState(),
            "Right": HandEventState(),
        }

    def _get_state(self, handedness: str) -> HandEventState:
        if handedness not in self._states:
            self._states[handedness] = HandEventState()
        return self._states[handedness]

    def process(
        self,
        handedness: str,
        gesture: GestureLabel,
        is_stable: bool,
        timestamp: Optional[float] = None,
    ) -> Optional[GestureEvent]:
        """
        Evaluates a frame's stabilized gesture prediction and produces a GestureEvent if appropriate.

        Args:
            handedness: 'Left' or 'Right'.
            gesture: Stabilized GestureLabel.
            is_stable: Whether the prediction reached stabilization confidence.
            timestamp: Optional timestamp in seconds (defaults to time.perf_counter()).

        Returns:
            Optional[GestureEvent]: TRIGGER, HOLD, or RELEASE event, or None if debounced.
        """
        ts = time.perf_counter() if timestamp is None else float(timestamp)
        state = self._get_state(handedness)

        # Case 1: Unstable, transitional, or UNKNOWN posture
        if not is_stable or gesture == GestureLabel.UNKNOWN:
            if state.active_gesture is not None:
                # Active gesture was released
                prev_gesture = state.active_gesture
                hold_dur = ts - state.gesture_start_time
                frames = state.gesture_frame_count

                state.active_gesture = None
                state.gesture_frame_count = 0
                return GestureEvent(
                    event_type=GestureEventType.RELEASE,
                    gesture=prev_gesture,
                    handedness=handedness,
                    timestamp=ts,
                    hold_duration=hold_dur,
                    frame_count=frames,
                )
            return None

        # Case 2: Gesture is stable and recognized
        if state.active_gesture == gesture:
            # Continuously held gesture: update duration, do NOT re-trigger
            state.gesture_frame_count += 1
            hold_dur = ts - state.gesture_start_time
            return GestureEvent(
                event_type=GestureEventType.HOLD,
                gesture=gesture,
                handedness=handedness,
                timestamp=ts,
                hold_duration=hold_dur,
                frame_count=state.gesture_frame_count,
            )

        # Case 3: Gesture changed to a new recognized gesture
        release_event: Optional[GestureEvent] = None
        if state.active_gesture is not None and state.active_gesture != gesture:
            # Release previous gesture first
            prev_gesture = state.active_gesture
            hold_dur = ts - state.gesture_start_time
            frames = state.gesture_frame_count
            release_event = GestureEvent(
                event_type=GestureEventType.RELEASE,
                gesture=prev_gesture,
                handedness=handedness,
                timestamp=ts,
                hold_duration=hold_dur,
                frame_count=frames,
            )
            state.active_gesture = None
            state.gesture_frame_count = 0

        # Check cooldown for the new gesture
        last_triggered = state.last_trigger_times.get(gesture, -999.0)
        cd = self.gesture_cooldowns.get(gesture, self.cooldown_seconds)
        if (ts - last_triggered) >= cd:
            state.active_gesture = gesture
            state.gesture_start_time = ts
            state.gesture_frame_count = 1
            state.last_trigger_times[gesture] = ts
            state.total_triggers += 1

            return GestureEvent(
                event_type=GestureEventType.TRIGGER,
                gesture=gesture,
                handedness=handedness,
                timestamp=ts,
                hold_duration=0.0,
                frame_count=1,
            )

        # If in cooldown, return any release event from prior gesture, or None
        return release_event

    def on_hand_lost(
        self,
        handedness: str,
        timestamp: Optional[float] = None,
    ) -> Optional[GestureEvent]:
        """
        Called when a hand leaves the camera frame or tracking is lost.
        Immediately releases any active gesture.
        """
        ts = time.perf_counter() if timestamp is None else float(timestamp)
        state = self._get_state(handedness)

        if state.active_gesture is not None:
            prev_gesture = state.active_gesture
            hold_dur = ts - state.gesture_start_time
            frames = state.gesture_frame_count

            state.active_gesture = None
            state.gesture_frame_count = 0
            return GestureEvent(
                event_type=GestureEventType.RELEASE,
                gesture=prev_gesture,
                handedness=handedness,
                timestamp=ts,
                hold_duration=hold_dur,
                frame_count=frames,
            )

        return None

    def get_active_gesture(self, handedness: str) -> Optional[GestureLabel]:
        """Returns the currently held gesture for the specified hand, if any."""
        return self._get_state(handedness).active_gesture

    def get_trigger_count(self, handedness: str) -> int:
        """Returns the total number of TRIGGER events emitted for the specified hand."""
        return self._get_state(handedness).total_triggers

    def reset(self, handedness: Optional[str] = None) -> None:
        """Resets tracking state for a specific hand or all hands."""
        if handedness is not None:
            self._get_state(handedness).reset()
        else:
            for s in self._states.values():
                s.reset()
