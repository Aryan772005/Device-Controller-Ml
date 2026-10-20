"""
GestureFlow AI - Unit Tests for Safe Desktop Controller
======================================================
Tests safety state transitions, action allowlist, cooldown enforcement,
emergency stop locks, and mock backend isolation.
NEVER triggers real Windows keystrokes or media actions.
"""

import time
import unittest

from src.desktop_controller import (
    ActionType,
    ControllerState,
    DesktopController,
    MockActionBackend,
)
from src.gesture_events import GestureEvent, GestureEventType
from src.gesture_recognizer import GestureLabel


class BrokenBackend:
    """Mock backend that simulates system hardware exceptions."""
    def play_pause(self) -> bool:
        raise OSError("Simulated Windows user32 driver crash")

    def toggle_mute(self) -> bool:
        raise OSError("Simulated audio endpoint error")

    def volume_up(self, steps: int = 1) -> bool:
        raise OSError("Simulated volume bus error")

    def volume_down(self, steps: int = 1) -> bool:
        raise OSError("Simulated volume bus error")


class TestDesktopControllerSafety(unittest.TestCase):
    """Verifies all safety invariants of the DesktopController."""

    def setUp(self):
        self.mock_backend = MockActionBackend()
        self.controller = DesktopController(
            backend=self.mock_backend,
            action_cooldowns={
                ActionType.MEDIA_PLAY_PAUSE: 0.5,
                ActionType.MUTE_TOGGLE: 0.5,
                ActionType.VOLUME_UP: 0.2,
                ActionType.VOLUME_DOWN: 0.2,
            },
        )

    def test_default_state_is_dry_run(self):
        """Verifies that the controller always starts in safe DRY_RUN mode, never ACTIVE."""
        self.assertEqual(self.controller.state, ControllerState.DRY_RUN)

    def test_dry_run_simulation_without_execution(self):
        """Verifies that DRY_RUN simulates and logs actions without touching the backend."""
        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.FIST,
            handedness="Right",
            timestamp=1000.0,
        )
        res = self.controller.handle_event(event, timestamp=1000.0)

        self.assertEqual(res.action, ActionType.MEDIA_PLAY_PAUSE)
        self.assertFalse(res.executed)  # NOT executed on hardware!
        self.assertIn("DRY_RUN SIMULATION", res.message)
        self.assertEqual(len(self.mock_backend.call_history), 0)  # Backend never called

    def test_disabled_state_suppression(self):
        """Verifies that when DISABLED, all actions are completely suppressed."""
        self.controller.disable()
        self.assertEqual(self.controller.state, ControllerState.DISABLED)

        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.PEACE,
            handedness="Right",
            timestamp=1000.0,
        )
        res = self.controller.handle_event(event, timestamp=1000.0)

        self.assertEqual(res.action, ActionType.MUTE_TOGGLE)
        self.assertFalse(res.executed)
        self.assertIn("suppressed: controller is DISABLED", res.message)
        self.assertEqual(len(self.mock_backend.call_history), 0)

    def test_active_mode_executes_backend(self):
        """Verifies that ACTIVE mode calls the backend and logs execution."""
        self.controller.activate()
        self.assertEqual(self.controller.state, ControllerState.ACTIVE)

        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.FIST,
            handedness="Right",
            timestamp=1000.0,
        )
        res = self.controller.handle_event(event, timestamp=1000.0)

        self.assertEqual(res.action, ActionType.MEDIA_PLAY_PAUSE)
        self.assertTrue(res.executed)
        self.assertIn("play_pause", self.mock_backend.call_history)

    def test_emergency_stop_locks_execution(self):
        """Verifies that emergency_stop() immediately blocks all actions."""
        self.controller.activate()
        self.controller.emergency_stop()
        self.assertEqual(self.controller.state, ControllerState.EMERGENCY_STOP)

        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.THUMBS_UP,
            handedness="Right",
            timestamp=1000.0,
        )
        res = self.controller.handle_event(event, timestamp=1000.0)

        self.assertFalse(res.executed)
        self.assertIn("EMERGENCY_STOP is active", res.message)
        self.assertEqual(len(self.mock_backend.call_history), 0)

    def test_no_automatic_reactivation_after_emergency_stop(self):
        """Verifies that controller CANNOT be activated directly from EMERGENCY_STOP."""
        self.controller.emergency_stop()

        with self.assertRaises(RuntimeError):
            self.controller.activate()

        with self.assertRaises(RuntimeError):
            self.controller.enable_dry_run()

        # Must explicitly reset emergency stop first
        self.controller.reset_emergency_stop()
        # After reset, returns to safe DRY_RUN, NOT ACTIVE!
        self.assertEqual(self.controller.state, ControllerState.DRY_RUN)

    def test_action_cooldown_enforcement(self):
        """Verifies that repeated triggers within the cooldown window are suppressed."""
        self.controller.activate()
        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.PEACE,
            handedness="Right",
            timestamp=1000.0,
        )

        # 1st call -> executed
        res1 = self.controller.handle_event(event, timestamp=1000.0)
        self.assertTrue(res1.executed)

        # 2nd call at 1000.2s (< 0.5s cooldown) -> suppressed by cooldown
        res2 = self.controller.handle_event(event, timestamp=1000.2)
        self.assertFalse(res2.executed)
        self.assertIn("suppressed by cooldown", res2.message)
        self.assertEqual(len(self.mock_backend.call_history), 1)

        # 3rd call at 1000.6s (>= 0.5s cooldown) -> executed
        res3 = self.controller.handle_event(event, timestamp=1000.6)
        self.assertTrue(res3.executed)
        self.assertEqual(len(self.mock_backend.call_history), 2)

    def test_hold_events_do_not_trigger_actions(self):
        """Verifies that continuous HOLD events are strictly ignored by the controller."""
        self.controller.activate()
        hold_event = GestureEvent(
            event_type=GestureEventType.HOLD,
            gesture=GestureLabel.FIST,
            handedness="Right",
            timestamp=1000.0,
            hold_duration=2.5,
            frame_count=75,
        )
        res = self.controller.handle_event(hold_event, timestamp=1000.0)

        self.assertFalse(res.executed)
        self.assertIn("Ignored non-trigger event type", res.message)
        self.assertEqual(len(self.mock_backend.call_history), 0)

    def test_unknown_gesture_suppression(self):
        """Verifies that UNKNOWN gestures map to NO_OP and are never executed."""
        self.controller.activate()
        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.UNKNOWN,
            handedness="Right",
            timestamp=1000.0,
        )
        res = self.controller.handle_event(event, timestamp=1000.0)

        self.assertEqual(res.action, ActionType.NO_OP)
        self.assertFalse(res.executed)
        self.assertEqual(len(self.mock_backend.call_history), 0)

    def test_tracking_loss_suppression(self):
        """Verifies that None event (tracking lost) safely returns NO_OP."""
        self.controller.activate()
        res = self.controller.handle_event(None, timestamp=1000.0)

        self.assertEqual(res.action, ActionType.NO_OP)
        self.assertFalse(res.executed)
        self.assertIn("Tracking lost", res.message)

    def test_backend_exception_handling(self):
        """Verifies that backend driver errors are caught gracefully without crashing the application."""
        broken_controller = DesktopController(backend=BrokenBackend())
        broken_controller.activate()

        event = GestureEvent(
            event_type=GestureEventType.TRIGGER,
            gesture=GestureLabel.FIST,
            handedness="Right",
            timestamp=1000.0,
        )
        # Must not raise an unhandled exception!
        res = broken_controller.handle_event(event, timestamp=1000.0)

        self.assertFalse(res.executed)
        self.assertIn("Backend Exception", res.message)


if __name__ == "__main__":
    unittest.main()
