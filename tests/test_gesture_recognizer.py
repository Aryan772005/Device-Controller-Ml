"""
GestureFlow AI - Unit Tests for Feature Extractor and Gesture Recognizer
========================================================================
Tests input validation, scale-invariant geometric feature extraction,
rule-based gesture recognition with synthetic landmark geometries,
and temporal stabilization mechanics.
"""

import math
import unittest
from typing import List, Tuple

from src.feature_extractor import (
    HandFeatures,
    extract_hand_features,
    validate_landmarks,
)
from src.gesture_recognizer import (
    GestureLabel,
    GestureRecognizer,
    GestureStabilizer,
)


def build_synthetic_hand(
    thumb_extended: bool = True,
    index_extended: bool = True,
    middle_extended: bool = True,
    ring_extended: bool = True,
    pinky_extended: bool = True,
    thumb_up: bool = False,
    v_separation: float = 40.0,
) -> List[Tuple[float, float, float]]:
    """
    Constructs a geometrically valid set of 21 (x, y, z) landmarks with controlled finger states.
    Wrist is anchored at (200, 400, 0).
    Palm reference length is ~130 pixels (Wrist to Middle MCP).
    """
    pts: List[Tuple[float, float, float]] = [(200.0, 400.0, 0.0)]  # 0: Wrist

    # 1-4: Thumb
    if thumb_up:
        # Extended straight upward
        pts.extend([
            (180.0, 360.0, 0.0),  # 1: CMC
            (170.0, 310.0, 0.0),  # 2: MCP
            (165.0, 250.0, 0.0),  # 3: IP
            (160.0, 190.0, 0.0),  # 4: TIP (significantly higher than wrist and MCP)
        ])
    elif thumb_extended:
        # Extended outward sideways
        pts.extend([
            (180.0, 370.0, 0.0),  # 1: CMC
            (150.0, 340.0, 0.0),  # 2: MCP
            (120.0, 310.0, 0.0),  # 3: IP
            (90.0, 280.0, 0.0),   # 4: TIP (far from index MCP)
        ])
    else:
        # Folded across palm
        pts.extend([
            (190.0, 380.0, 0.0),  # 1: CMC
            (185.0, 350.0, 0.0),  # 2: MCP
            (185.0, 330.0, 0.0),  # 3: IP
            (185.0, 310.0, 0.0),  # 4: TIP (tucked against base of index finger)
        ])

    # 4 Finger Base X Coordinates
    finger_bases = [
        (5, 175.0, 280.0, index_extended),
        (9, 200.0, 270.0, middle_extended),
        (13, 225.0, 280.0, ring_extended),
        (17, 245.0, 290.0, pinky_extended),
    ]

    for mcp_idx, base_x, base_y, is_ext in finger_bases:
        pts.append((base_x, base_y, 0.0))  # MCP

        if is_ext:
            # Special adjustment for index & middle if demonstrating peace sign V-spread
            adj_x = 0.0
            if mcp_idx == 5 and index_extended and middle_extended:
                adj_x = -v_separation / 2
            elif mcp_idx == 9 and index_extended and middle_extended:
                adj_x = v_separation / 2

            # Straight extension outward
            pts.append((base_x + adj_x * 0.3, base_y - 45.0, 0.0))   # PIP
            pts.append((base_x + adj_x * 0.7, base_y - 80.0, 0.0))   # DIP
            pts.append((base_x + adj_x, base_y - 120.0, 0.0))        # TIP
        else:
            # Folded into palm (curls down toward palm)
            pts.append((base_x, base_y - 25.0, 0.0))   # PIP
            pts.append((base_x + 3.0, base_y - 8.0, 0.0))  # DIP
            pts.append((base_x, base_y + 12.0, 0.0))       # TIP

    assert len(pts) == 21, f"Expected 21 synthetic landmarks, got {len(pts)}"
    return pts


class TestFeatureExtractor(unittest.TestCase):
    """Tests input validation and scale-invariant geometric feature extraction."""

    def test_invalid_landmark_counts(self):
        """Verifies that sequences with other than 21 landmarks raise ValueError."""
        with self.assertRaises(ValueError):
            validate_landmarks([(0, 0, 0)] * 20)

        with self.assertRaises(ValueError):
            validate_landmarks([(0, 0, 0)] * 22)

        with self.assertRaises(ValueError):
            validate_landmarks([])

        with self.assertRaises(TypeError):
            validate_landmarks(None)

    def test_non_finite_coordinates(self):
        """Verifies that NaN or Infinite values in landmarks raise ValueError."""
        nan_pts = build_synthetic_hand()
        nan_pts[8] = (float("nan"), 100.0, 0.0)
        with self.assertRaises(ValueError):
            validate_landmarks(nan_pts)

        inf_pts = build_synthetic_hand()
        inf_pts[4] = (100.0, float("inf"), 0.0)
        with self.assertRaises(ValueError):
            validate_landmarks(inf_pts)

    def test_feature_extraction_types_and_ranges(self):
        """Verifies that HandFeatures produces valid, non-zero scale metrics."""
        pts = build_synthetic_hand()
        features = extract_hand_features(pts)

        self.assertIsInstance(features, HandFeatures)
        self.assertGreater(features.palm_size, 50.0)
        self.assertEqual(len(features.wrist_to_tip_ratios), 4)
        self.assertIn("index", features.wrist_to_tip_ratios)
        self.assertGreater(features.thumb_extension_ratio, 0.0)
        self.assertEqual(features.extended_count, 5)


class TestGestureRecognizer(unittest.TestCase):
    """Tests rule-based classification against synthetic postures."""

    def setUp(self):
        self.recognizer = GestureRecognizer()

    def test_open_palm_recognition(self):
        """Verifies that OPEN_PALM is recognized when all 5 fingers are extended."""
        hand = build_synthetic_hand(thumb_extended=True, index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)
        pred = self.recognizer.recognize(hand)
        self.assertEqual(pred.label, GestureLabel.OPEN_PALM)
        self.assertGreaterEqual(pred.confidence, 0.90)

    def test_fist_recognition(self):
        """Verifies that FIST is recognized when all 5 fingers are curled."""
        hand = build_synthetic_hand(thumb_extended=False, index_extended=False, middle_extended=False, ring_extended=False, pinky_extended=False)
        pred = self.recognizer.recognize(hand)
        self.assertEqual(pred.label, GestureLabel.FIST)
        self.assertGreaterEqual(pred.confidence, 0.90)

    def test_pointing_recognition(self):
        """Verifies that POINTING is recognized when only index finger is extended."""
        hand = build_synthetic_hand(thumb_extended=False, index_extended=True, middle_extended=False, ring_extended=False, pinky_extended=False)
        pred = self.recognizer.recognize(hand)
        self.assertEqual(pred.label, GestureLabel.POINTING)

    def test_peace_sign_recognition(self):
        """Verifies that PEACE is recognized when index and middle fingers are extended with V-separation."""
        hand = build_synthetic_hand(thumb_extended=False, index_extended=True, middle_extended=True, ring_extended=False, pinky_extended=False, v_separation=50.0)
        pred = self.recognizer.recognize(hand)
        self.assertEqual(pred.label, GestureLabel.PEACE)

    def test_thumbs_up_recognition(self):
        """Verifies that THUMBS_UP is recognized when thumb points up and other fingers are folded."""
        hand = build_synthetic_hand(thumb_extended=True, index_extended=False, middle_extended=False, ring_extended=False, pinky_extended=False, thumb_up=True)
        pred = self.recognizer.recognize(hand)
        self.assertEqual(pred.label, GestureLabel.THUMBS_UP)

    def test_unknown_for_ambiguous_pose(self):
        """Verifies that UNKNOWN is returned for unmapped poses (e.g. index and ring only)."""
        ambiguous = build_synthetic_hand(thumb_extended=False, index_extended=True, middle_extended=False, ring_extended=True, pinky_extended=False)
        pred = self.recognizer.recognize(ambiguous)
        self.assertEqual(pred.label, GestureLabel.UNKNOWN)

    def test_three_recognition(self):
        """Verifies that THREE is recognized when index, middle, and ring are extended."""
        three_fingers = build_synthetic_hand(thumb_extended=False, index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=False)
        pred = self.recognizer.recognize(three_fingers)
        self.assertEqual(pred.label, GestureLabel.THREE)


class TestGestureStabilizer(unittest.TestCase):
    """Tests temporal smoothing, threshold agreement, and reset behavior."""

    def setUp(self):
        self.stabilizer = GestureStabilizer(window_size=5, agreement_ratio=0.60, max_missing_frames=3)

    def test_repeated_prediction_stabilization(self):
        """Verifies that consistent repeated predictions yield stable output."""
        for _ in range(5):
            label, ratio = self.stabilizer.update("Right", GestureLabel.OPEN_PALM)

        self.assertEqual(label, GestureLabel.OPEN_PALM)
        self.assertEqual(ratio, 1.0)

    def test_flickering_rejection(self):
        """Verifies that rapid single-frame noise is rejected before reaching threshold agreement."""
        # 4 consecutive FISTs
        for _ in range(4):
            self.stabilizer.update("Right", GestureLabel.FIST)

        # 1 glitch frame of OPEN_PALM
        label, ratio = self.stabilizer.update("Right", GestureLabel.OPEN_PALM)
        # Stable prediction should remain FIST because FIST is still the supermajority (80%)
        self.assertEqual(label, GestureLabel.FIST)

    def test_independent_hands(self):
        """Verifies that Left and Right hands maintain strictly separated history buffers."""
        for _ in range(5):
            self.stabilizer.update("Right", GestureLabel.OPEN_PALM)
            self.stabilizer.update("Left", GestureLabel.FIST)

        right_label, _ = self.stabilizer.update("Right", GestureLabel.OPEN_PALM)
        left_label, _ = self.stabilizer.update("Left", GestureLabel.FIST)

        self.assertEqual(right_label, GestureLabel.OPEN_PALM)
        self.assertEqual(left_label, GestureLabel.FIST)

    def test_missing_hand_reset(self):
        """Verifies that history buffer is cleared when hand is absent for max_missing_frames."""
        for _ in range(5):
            self.stabilizer.update("Right", GestureLabel.PEACE)

        # Simulate hand missing for 3 consecutive frames
        for _ in range(3):
            self.stabilizer.register_missing("Right")

        # After reset, buffer should be empty, so single prediction does not immediately lock
        label, ratio = self.stabilizer.update("Right", GestureLabel.POINTING)
        # First frame after reset has len=1 < min_required (3), so remains initial UNKNOWN
        self.assertEqual(label, GestureLabel.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
