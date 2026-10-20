"""
GestureFlow AI - Rule-Based Gesture Recognizer & Stabilizer
===========================================================
Classifies hand postures into discrete semantic gesture labels using scale-invariant
and rotation-robust geometric rules extracted from 21 MediaPipe hand landmarks.
Provides temporal stabilization to eliminate frame-to-frame prediction flickering.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.feature_extractor import (
    HandFeatures,
    extract_hand_features,
    validate_landmarks,
)


class GestureLabel(str, Enum):
    """Supported discrete gesture labels."""
    OPEN_PALM    = "OPEN_PALM"
    FIST         = "FIST"
    POINTING     = "POINTING"
    PEACE        = "PEACE"
    THUMBS_UP    = "THUMBS_UP"
    THUMBS_DOWN  = "THUMBS_DOWN"   # Thumb extended downward; 4 fingers curled
    PINCH        = "PINCH"         # Thumb tip touching index tip
    ROCK         = "ROCK"          # Index + pinky extended; middle + ring curled (devil horns)
    CALL_ME      = "CALL_ME"       # Thumb + pinky extended; middle + ring + index curled
    THREE        = "THREE"         # Index + middle + ring extended; pinky + thumb curled
    UNKNOWN      = "UNKNOWN"


@dataclass(frozen=True)
class GesturePrediction:
    """
    Result of gesture recognition for a single hand:
      - label: Recognized GestureLabel
      - confidence: Rule confidence score or temporal agreement [0.0, 1.0]
      - features: Extracted HandFeatures used for the decision
    """
    label: GestureLabel
    confidence: float
    features: HandFeatures


class GestureRecognizer:
    """
    Rule-based gesture recognizer using transparent geometric relationships.

    Recognizes:
      - OPEN_PALM: All 5 fingers extended outward.
      - FIST: All fingers curled tightly into the palm.
      - POINTING: Index finger extended; middle, ring, pinky curled.
      - PEACE: Index and middle fingers extended in a V-shape; ring and pinky curled.
      - THUMBS_UP: Thumb extended vertically upward; other 4 fingers curled.
      - UNKNOWN: Any pose not unambiguously matching the above.
    """

    def __init__(
        self,
        extension_ratio_threshold: float = 1.10,
        thumb_extension_threshold: float = 1.15,
        thumbs_up_y_offset_threshold: float = -0.30,
        peace_tip_separation_threshold: float = 0.20,
    ) -> None:
        """
        Configures classification thresholds.

        Args:
            extension_ratio_threshold: Minimum dist(wrist, tip) / dist(wrist, pip) for extension.
            thumb_extension_threshold: Minimum dist(wrist, thumb_tip) / dist(wrist, thumb_ip).
            thumbs_up_y_offset_threshold: Thumb tip must be at least this fraction of palm_size
                                         above the wrist (negative means higher in image).
            peace_tip_separation_threshold: Minimum distance between index tip and middle tip
                                           normalized by palm_size for peace sign.
        """
        self.extension_ratio_threshold = extension_ratio_threshold
        self.thumb_extension_threshold = thumb_extension_threshold
        self.thumbs_up_y_offset_threshold = thumbs_up_y_offset_threshold
        self.peace_tip_separation_threshold = peace_tip_separation_threshold

    def recognize(self, landmarks: Sequence[Any]) -> GesturePrediction:
        """
        Classifies a single hand's 21 landmarks into a gesture.

        Args:
            landmarks: Sequence of 21 landmark objects (from HandData or list of coordinates).

        Returns:
            GesturePrediction: The predicted gesture label, confidence, and features.
        """
        features = extract_hand_features(
            landmarks,
            extension_ratio_threshold=self.extension_ratio_threshold,
            thumb_extension_threshold=self.thumb_extension_threshold,
        )

        f_state = features.finger_states
        thumb = f_state.thumb
        index = f_state.index
        middle = f_state.middle
        ring = f_state.ring
        pinky = f_state.pinky

        # Rule 1: OPEN_PALM
        # Assumption: All five fingers extended away from the palm.
        if thumb and index and middle and ring and pinky:
            return GesturePrediction(
                label=GestureLabel.OPEN_PALM,
                confidence=0.95,
                features=features,
            )

        # Rule 2: FIST
        # Assumption: All 4 main fingers are curled. Thumb is either curled across palm or tucked.
        if (not index) and (not middle) and (not ring) and (not pinky) and (not thumb):
            return GesturePrediction(
                label=GestureLabel.FIST,
                confidence=0.95,
                features=features,
            )

        # Rule 3: POINTING
        # Assumption: Index finger is extended. Middle, ring, and pinky are folded.
        # Thumb is usually folded across the other fingers or slightly relaxed.
        if index and (not middle) and (not ring) and (not pinky):
            # If thumb is also pointing straight up like a gun, it is not a pure pointing gesture
            return GesturePrediction(
                label=GestureLabel.POINTING,
                confidence=0.90,
                features=features,
            )

        # Rule 4: PEACE (Victory Sign)
        # Assumption: Index and middle fingers extended. Ring and pinky folded.
        if index and middle and (not ring) and (not pinky):
            # Verify V-shape separation between index and middle tips
            pts = validate_landmarks(landmarks)
            index_tip = pts[8]
            middle_tip = pts[12]
            tip_dist = ((index_tip[0] - middle_tip[0]) ** 2 + (index_tip[1] - middle_tip[1]) ** 2) ** 0.5
            norm_separation = tip_dist / features.palm_size

            if norm_separation >= self.peace_tip_separation_threshold:
                return GesturePrediction(
                    label=GestureLabel.PEACE,
                    confidence=0.90,
                    features=features,
                )

        # Rule 5: THUMBS_UP
        # Assumption: Thumb extended and pointing upwards relative to wrist; other 4 fingers folded into fist.
        if thumb and (not index) and (not middle) and (not ring) and (not pinky):
            # Verify vertical alignment: thumb tip is higher than wrist by at least threshold
            if features.thumb_tip_y_offset <= self.thumbs_up_y_offset_threshold:
                return GesturePrediction(
                    label=GestureLabel.THUMBS_UP,
                    confidence=0.90,
                    features=features,
                )

        # Rule 6: THUMBS_DOWN
        # Thumb extended and pointing downward; 4 fingers curled.
        if thumb and (not index) and (not middle) and (not ring) and (not pinky):
            # Thumb tip is BELOW wrist (positive y_offset in image coords)
            if features.thumb_tip_y_offset >= 0.30:
                return GesturePrediction(
                    label=GestureLabel.THUMBS_DOWN,
                    confidence=0.88,
                    features=features,
                )

        # Rule 7: ROCK (devil horns / rock on)
        # Index + pinky extended; middle + ring curled.
        if index and (not middle) and (not ring) and pinky:
            return GesturePrediction(
                label=GestureLabel.ROCK,
                confidence=0.88,
                features=features,
            )

        # Rule 8: CALL_ME (hang-loose / shaka)
        # Thumb + pinky extended; index + middle + ring curled.
        if thumb and (not index) and (not middle) and (not ring) and pinky:
            return GesturePrediction(
                label=GestureLabel.CALL_ME,
                confidence=0.85,
                features=features,
            )

        # Rule 9: THREE fingers
        # Index + middle + ring extended; pinky + thumb curled.
        if (not thumb) and index and middle and ring and (not pinky):
            return GesturePrediction(
                label=GestureLabel.THREE,
                confidence=0.85,
                features=features,
            )

        # Rule 10: PINCH (thumb tip close to index tip)
        # All other fingers can be in any state.
        pts = validate_landmarks(landmarks)
        thumb_tip = pts[4]
        index_tip = pts[8]
        pinch_dist = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
        norm_pinch = pinch_dist / features.palm_size
        if norm_pinch < 0.25 and (not index):   # index curled because tip is near thumb
            return GesturePrediction(
                label=GestureLabel.PINCH,
                confidence=0.87,
                features=features,
            )

        # Fallback: UNKNOWN
        # Assumption: Hand posture is transitional, partial, or does not match predefined gestures.
        return GesturePrediction(
            label=GestureLabel.UNKNOWN,
            confidence=0.50,
            features=features,
        )


@dataclass(frozen=True)
class StabilizationResult:
    """
    Detailed output of the temporal stabilization filter.

    Attributes:
        gesture: Stabilized GestureLabel (or UNKNOWN if unstable/ambiguous).
        agreement_ratio: Fraction of window frames agreeing on the most common gesture.
        is_stable: True if agreement_ratio >= agreement_threshold and window is sufficiently full.
        raw_gesture: The instantaneous, unfiltered prediction from the current frame.
    """
    gesture: GestureLabel
    agreement_ratio: float
    is_stable: bool
    raw_gesture: GestureLabel


class GestureStabilizer:
    """
    Temporal filter that eliminates prediction flickering across consecutive video frames.

    Maintains an independent sliding window of recent predictions for each hand (Left/Right).
    Emits a gesture only when a supermajority of recent frames agree. Automatically
    resets history when a hand disappears from the frame for more than max_missing_frames.

    Responsiveness vs Stability Trade-Off:
      - Window size = 5 (at 30 FPS ~166ms latency): Very responsive, eliminates 1-2 frame noise.
      - Window size = 10 (at 30 FPS ~333ms latency): Maximum stability, noticeable transition delay.
    """

    def __init__(
        self,
        window_size: int = 5,
        agreement_ratio: float = 0.60,
        max_missing_frames: int = 5,
        hold_ambiguous: bool = True,
        max_hold_frames: int = 3,
    ) -> None:
        """
        Args:
            window_size: Number of recent frames to consider in the rolling buffer.
            agreement_ratio: Fraction of window frames that must predict the same gesture (default: 60%).
            max_missing_frames: Number of consecutive absent frames before resetting a hand's buffer.
            hold_ambiguous: Whether to briefly retain the previous stable gesture during transient noise.
            max_hold_frames: Maximum number of ambiguous frames to hold the previous stable gesture.
        """
        self.window_size = window_size
        self.agreement_ratio = agreement_ratio
        self.max_missing_frames = max_missing_frames
        self.hold_ambiguous = hold_ambiguous
        self.max_hold_frames = max_hold_frames

        # Separate deques and states for Left and Right hands
        self._buffers: Dict[str, deque[GestureLabel]] = {
            "Left": deque(maxlen=window_size),
            "Right": deque(maxlen=window_size),
        }
        self._missing_counts: Dict[str, int] = {"Left": 0, "Right": 0}
        self._hold_counts: Dict[str, int] = {"Left": 0, "Right": 0}
        self._last_stable: Dict[str, GestureLabel] = {
            "Left": GestureLabel.UNKNOWN,
            "Right": GestureLabel.UNKNOWN,
        }

    def update_detailed(
        self,
        handedness: str,
        current_prediction: GestureLabel,
    ) -> StabilizationResult:
        """
        Updates the stabilization buffer and returns a rich StabilizationResult.

        Args:
            handedness: 'Left' or 'Right' (matches HandData.handedness).
            current_prediction: Raw GestureLabel from GestureRecognizer.

        Returns:
            StabilizationResult containing stabilized label, ratio, is_stable, and raw_gesture.
        """
        if handedness not in self._buffers:
            self._buffers[handedness] = deque(maxlen=self.window_size)
            self._missing_counts[handedness] = 0
            self._hold_counts[handedness] = 0
            self._last_stable[handedness] = GestureLabel.UNKNOWN

        # Reset missing counter since hand was observed in this frame
        self._missing_counts[handedness] = 0

        # Append new prediction to rolling window
        buf = self._buffers[handedness]
        buf.append(current_prediction)

        # Count frequencies in current window
        counts = Counter(buf)
        most_common_label, count = counts.most_common(1)[0]
        ratio = count / len(buf)

        min_required = int(math.ceil(self.window_size * self.agreement_ratio))
        is_stable = (count >= min_required) and (len(buf) >= min_required)

        if is_stable:
            self._last_stable[handedness] = most_common_label
            self._hold_counts[handedness] = 0
            return StabilizationResult(
                gesture=most_common_label,
                agreement_ratio=ratio,
                is_stable=True,
                raw_gesture=current_prediction,
            )

        # Ambiguous / fluctuating state
        if self.hold_ambiguous and self._last_stable[handedness] != GestureLabel.UNKNOWN:
            self._hold_counts[handedness] += 1
            if self._hold_counts[handedness] <= self.max_hold_frames:
                # Retain previous stable gesture briefly to bridge 1-2 frame tracking glitches
                return StabilizationResult(
                    gesture=self._last_stable[handedness],
                    agreement_ratio=ratio,
                    is_stable=False,
                    raw_gesture=current_prediction,
                )

        # Revert to UNKNOWN when ambiguous or hold frames exceeded
        self._last_stable[handedness] = GestureLabel.UNKNOWN
        return StabilizationResult(
            gesture=GestureLabel.UNKNOWN,
            agreement_ratio=ratio,
            is_stable=False,
            raw_gesture=current_prediction,
        )

    def update(
        self,
        handedness: str,
        current_prediction: GestureLabel,
    ) -> tuple:
        """
        Backward-compatible tuple interface returning (gesture, agreement_ratio).
        """
        res = self.update_detailed(handedness, current_prediction)
        return res.gesture, res.agreement_ratio

    def register_missing(self, handedness: str) -> None:
        """
        Notifies the stabilizer that a hand was not detected in the current frame.
        Clears history if missing for longer than max_missing_frames.
        """
        if handedness in self._missing_counts:
            self._missing_counts[handedness] += 1
            if self._missing_counts[handedness] >= self.max_missing_frames:
                self.reset(handedness)

    def is_stable(self, handedness: str) -> bool:
        """Returns True if the current prediction for the hand is stable."""
        buf = self._buffers.get(handedness)
        if not buf:
            return False
        counts = Counter(buf)
        _, count = counts.most_common(1)[0]
        min_required = int(math.ceil(self.window_size * self.agreement_ratio))
        return (count >= min_required) and (len(buf) >= min_required)

    def reset(self, handedness: Optional[str] = None) -> None:
        """Resets the history for a specific hand or all hands."""
        if handedness is not None:
            if handedness in self._buffers:
                self._buffers[handedness].clear()
                self._missing_counts[handedness] = 0
                self._hold_counts[handedness] = 0
                self._last_stable[handedness] = GestureLabel.UNKNOWN
        else:
            for h in list(self._buffers.keys()):
                self.reset(h)
