"""
GestureFlow AI - Unit Tests for Gesture Events and Debouncing
============================================================
Tests the GestureEventManager:
  - Distinction between TRIGGER, HOLD, and RELEASE events.
  - Prevention of accidental repeat triggers on held gestures.
  - Cooldown timing enforcement.
  - Safe reset and hand disappearance handling.
  - Independent state isolation between Left and Right hands.
"""

import time
import unittest

from src.gesture_events import (
    GestureEvent,
    GestureEventManager,
    GestureEventType,
)
from src.gesture_recognizer import (
    GestureLabel,
    GestureStabilizer,
    StabilizationResult,
)


class TestGestureEventsAndDebouncing(unittest.TestCase):
    """Tests the gesture event lifecycle and debouncing rules."""

    def setUp(self):
        # 0.5s cooldown for fast test execution
        self.manager = GestureEventManager(cooldown_seconds=0.5)

    def test_single_trigger_for_held_gesture(self):
        """Verifies that a continuously held gesture emits exactly ONE TRIGGER, followed by HOLDs."""
        t0 = 1000.0

        # Frame 1: Gesture becomes stable -> TRIGGER
        ev1 = self.manager.process("Right", GestureLabel.OPEN_PALM, is_stable=True, timestamp=t0)
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.event_type, GestureEventType.TRIGGER)
        self.assertEqual(ev1.gesture, GestureLabel.OPEN_PALM)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)

        # Frame 2: Same gesture held -> HOLD (no duplicate TRIGGER)
        ev2 = self.manager.process("Right", GestureLabel.OPEN_PALM, is_stable=True, timestamp=t0 + 0.033)
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2.event_type, GestureEventType.HOLD)
        self.assertEqual(ev2.frame_count, 2)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)  # Count remains 1!

        # Frame 3: Same gesture still held -> HOLD
        ev3 = self.manager.process("Right", GestureLabel.OPEN_PALM, is_stable=True, timestamp=t0 + 0.066)
        self.assertIsNotNone(ev3)
        self.assertEqual(ev3.event_type, GestureEventType.HOLD)
        self.assertEqual(ev3.frame_count, 3)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)

    def test_release_on_unknown_or_unstable(self):
        """Verifies that transitioning to UNKNOWN or unstable emits a RELEASE event."""
        t0 = 1000.0
        self.manager.process("Right", GestureLabel.PEACE, is_stable=True, timestamp=t0)

        # Hand relaxes into UNKNOWN
        ev_rel = self.manager.process("Right", GestureLabel.UNKNOWN, is_stable=True, timestamp=t0 + 0.20)
        self.assertIsNotNone(ev_rel)
        self.assertEqual(ev_rel.event_type, GestureEventType.RELEASE)
        self.assertEqual(ev_rel.gesture, GestureLabel.PEACE)
        self.assertAlmostEqual(ev_rel.hold_duration, 0.20, places=2)

        # Hand remains in UNKNOWN -> returns None (IDLE)
        ev_idle = self.manager.process("Right", GestureLabel.UNKNOWN, is_stable=True, timestamp=t0 + 0.25)
        self.assertIsNone(ev_idle)

    def test_release_on_hand_lost(self):
        """Verifies that tracking loss immediately emits a RELEASE event."""
        t0 = 1000.0
        self.manager.process("Left", GestureLabel.FIST, is_stable=True, timestamp=t0)

        # Hand leaves frame
        ev_lost = self.manager.on_hand_lost("Left", timestamp=t0 + 0.50)
        self.assertIsNotNone(ev_lost)
        self.assertEqual(ev_lost.event_type, GestureEventType.RELEASE)
        self.assertEqual(ev_lost.gesture, GestureLabel.FIST)
        self.assertAlmostEqual(ev_lost.hold_duration, 0.50, places=2)
        self.assertIsNone(self.manager.get_active_gesture("Left"))

    def test_cooldown_enforcement(self):
        """Verifies that a gesture cannot be re-triggered before cooldown expires."""
        t0 = 1000.0

        # 1. Trigger gesture
        ev1 = self.manager.process("Right", GestureLabel.THUMBS_UP, is_stable=True, timestamp=t0)
        self.assertEqual(ev1.event_type, GestureEventType.TRIGGER)

        # 2. Release gesture at t0 + 0.1s
        self.manager.process("Right", GestureLabel.UNKNOWN, is_stable=True, timestamp=t0 + 0.1)

        # 3. Attempt re-trigger at t0 + 0.2s (< 0.5s cooldown) -> suppressed!
        ev_suppressed = self.manager.process("Right", GestureLabel.THUMBS_UP, is_stable=True, timestamp=t0 + 0.2)
        self.assertIsNone(ev_suppressed)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)

        # 4. Attempt re-trigger at t0 + 0.6s (>= 0.5s cooldown) -> allowed!
        ev_allowed = self.manager.process("Right", GestureLabel.THUMBS_UP, is_stable=True, timestamp=t0 + 0.6)
        self.assertIsNotNone(ev_allowed)
        self.assertEqual(ev_allowed.event_type, GestureEventType.TRIGGER)
        self.assertEqual(self.manager.get_trigger_count("Right"), 2)

    def test_gesture_transition_release_and_trigger(self):
        """Verifies that transitioning from one gesture directly to another releases the old and triggers the new."""
        t0 = 1000.0

        # Start with POINTING
        ev1 = self.manager.process("Right", GestureLabel.POINTING, is_stable=True, timestamp=t0)
        self.assertEqual(ev1.event_type, GestureEventType.TRIGGER)

        # Change directly to FIST after 0.6s
        ev2 = self.manager.process("Right", GestureLabel.FIST, is_stable=True, timestamp=t0 + 0.6)
        self.assertEqual(ev2.event_type, GestureEventType.TRIGGER)
        self.assertEqual(ev2.gesture, GestureLabel.FIST)

    def test_independent_hands_isolation(self):
        """Verifies that Left and Right hands do not interfere with each other's event states."""
        t0 = 1000.0

        # Left triggers OPEN_PALM
        ev_left = self.manager.process("Left", GestureLabel.OPEN_PALM, is_stable=True, timestamp=t0)
        self.assertEqual(ev_left.event_type, GestureEventType.TRIGGER)
        self.assertEqual(self.manager.get_trigger_count("Left"), 1)
        self.assertEqual(self.manager.get_trigger_count("Right"), 0)

        # Right triggers FIST
        ev_right = self.manager.process("Right", GestureLabel.FIST, is_stable=True, timestamp=t0 + 0.1)
        self.assertEqual(ev_right.event_type, GestureEventType.TRIGGER)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)
        self.assertEqual(self.manager.get_trigger_count("Left"), 1)

        # Left releases, Right remains held
        self.manager.process("Left", GestureLabel.UNKNOWN, is_stable=True, timestamp=t0 + 0.2)
        self.assertIsNone(self.manager.get_active_gesture("Left"))
        self.assertEqual(self.manager.get_active_gesture("Right"), GestureLabel.FIST)

    def test_reset_behavior(self):
        """Verifies that reset() clears all counters and active states."""
        t0 = 1000.0
        self.manager.process("Right", GestureLabel.OPEN_PALM, is_stable=True, timestamp=t0)
        self.assertEqual(self.manager.get_trigger_count("Right"), 1)

        self.manager.reset("Right")
        self.assertIsNone(self.manager.get_active_gesture("Right"))
        self.assertEqual(self.manager.get_trigger_count("Right"), 0)


class TestStabilizerDiagnostics(unittest.TestCase):
    """Tests the detailed diagnostic capabilities of GestureStabilizer."""

    def setUp(self):
        # Require 85% agreement (5/5) so that a single differing frame triggers the ambiguity hold logic
        self.stabilizer = GestureStabilizer(window_size=5, agreement_ratio=0.85, hold_ambiguous=True, max_hold_frames=2)

    def test_stabilization_result_fields(self):
        """Verifies that update_detailed returns a rich StabilizationResult."""
        res = self.stabilizer.update_detailed("Right", GestureLabel.OPEN_PALM)
        self.assertIsInstance(res, StabilizationResult)
        self.assertEqual(res.raw_gesture, GestureLabel.OPEN_PALM)

    def test_hold_ambiguous_and_expiration(self):
        """Verifies that hold_ambiguous bridges transient glitches and expires cleanly."""
        # 5 consecutive OPEN_PALM frames -> becomes stable
        for _ in range(5):
            res = self.stabilizer.update_detailed("Right", GestureLabel.OPEN_PALM)
        self.assertTrue(res.is_stable)
        self.assertEqual(res.gesture, GestureLabel.OPEN_PALM)

        # 1 frame glitch of UNKNOWN -> held as previous stable gesture
        glitch_1 = self.stabilizer.update_detailed("Right", GestureLabel.UNKNOWN)
        self.assertFalse(glitch_1.is_stable)
        self.assertEqual(glitch_1.gesture, GestureLabel.OPEN_PALM)

        # 2nd frame glitch of UNKNOWN -> held
        glitch_2 = self.stabilizer.update_detailed("Right", GestureLabel.UNKNOWN)
        self.assertFalse(glitch_2.is_stable)
        self.assertEqual(glitch_2.gesture, GestureLabel.OPEN_PALM)

        # 3rd frame glitch -> max_hold_frames (2) exceeded, falls back to UNKNOWN!
        glitch_3 = self.stabilizer.update_detailed("Right", GestureLabel.UNKNOWN)
        self.assertFalse(glitch_3.is_stable)
        self.assertEqual(glitch_3.gesture, GestureLabel.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
