"""
GestureFlow AI - Core Source Package
"""

from src.hand_tracker import HandTracker, HAND_CONNECTIONS, HandData, PixelLandmark
from src.feature_extractor import extract_hand_features, HandFeatures, FingerState
from src.gesture_recognizer import (
    GestureRecognizer,
    GestureStabilizer,
    GestureLabel,
    GesturePrediction,
    StabilizationResult,
)
from src.gesture_events import (
    GestureEventManager,
    GestureEvent,
    GestureEventType,
)
from src.desktop_controller import (
    DesktopController,
    ControllerState,
    ActionType,
    ActionResult,
    DEFAULT_GESTURE_ACTION_MAPPING,
    MockActionBackend,
    WindowsMediaBackend,
)

__all__ = [
    "HandTracker",
    "HAND_CONNECTIONS",
    "HandData",
    "PixelLandmark",
    "extract_hand_features",
    "HandFeatures",
    "FingerState",
    "GestureRecognizer",
    "GestureStabilizer",
    "GestureLabel",
    "GesturePrediction",
    "StabilizationResult",
    "GestureEventManager",
    "GestureEvent",
    "GestureEventType",
    "DesktopController",
    "ControllerState",
    "ActionType",
    "ActionResult",
    "DEFAULT_GESTURE_ACTION_MAPPING",
    "MockActionBackend",
    "WindowsMediaBackend",
]
