"""
GestureFlow AI - Hand Feature Extractor
=======================================
Extracts scale-invariant and rotation-robust geometric features from 21 MediaPipe hand landmarks.
Provides normalized joint distances, relative extension ratios, and boolean finger extension states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

# Landmark Indices Constants (MediaPipe 21 Hand Landmarks)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_INDICES = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


@dataclass(frozen=True)
class FingerState:
    """Boolean extension flags for all five fingers."""
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def extended_count(self) -> int:
        """Total number of extended fingers (0 to 5)."""
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])


@dataclass(frozen=True)
class HandFeatures:
    """
    Structured geometric features extracted from a single hand's 21 landmarks.

    Attributes:
        palm_size: Euclidean distance between Wrist (0) and Middle MCP (9), used as normalizer.
        wrist_to_tip_ratios: Ratio of dist(wrist, tip) to dist(wrist, pip) for the 4 fingers.
                             Values > 1.05 typically indicate extended fingers regardless of hand angle.
        mcp_to_tip_ratios: Ratio of dist(mcp, tip) to dist(mcp, pip).
        thumb_extension_ratio: Ratio of dist(wrist, thumb_tip) to dist(wrist, thumb_ip).
        thumb_to_index_distance: Distance between thumb tip and index MCP normalized by palm size.
        thumb_tip_y_offset: Vertical distance of thumb tip relative to wrist, normalized by palm size.
        finger_states: Boolean extension flags for each finger.
        extended_count: Total number of extended fingers (0 to 5).
    """
    palm_size: float
    wrist_to_tip_ratios: Dict[str, float]
    mcp_to_tip_ratios: Dict[str, float]
    thumb_extension_ratio: float
    thumb_to_index_distance: float
    thumb_tip_y_offset: float
    finger_states: FingerState
    extended_count: int


def _euclidean_dist_2d(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Computes 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _extract_coords(landmark: Any) -> Tuple[float, float, float]:
    """Extracts (x, y, z) from either a NormalizedLandmark, PixelLandmark, or tuple/list."""
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        z = getattr(landmark, "z", 0.0)
        return float(landmark.x), float(landmark.y), float(z)
    elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
        z = float(landmark[2]) if len(landmark) > 2 else 0.0
        return float(landmark[0]), float(landmark[1]), z
    raise TypeError(f"Unsupported landmark format: {type(landmark)}")


def validate_landmarks(landmarks: Sequence[Any]) -> List[Tuple[float, float, float]]:
    """
    Validates that landmarks sequence contains exactly 21 valid, finite coordinate points.

    Args:
        landmarks: Sequence of 21 landmark items.

    Returns:
        List of 21 (x, y, z) tuples.

    Raises:
        ValueError: If landmarks length != 21 or if any coordinate is NaN or Infinite.
        TypeError: If landmarks is not a sequence.
    """
    if landmarks is None or not isinstance(landmarks, (list, tuple)):
        raise TypeError(f"Landmarks must be a list or tuple, got {type(landmarks)}")

    if len(landmarks) != 21:
        raise ValueError(f"Expected exactly 21 landmarks, received {len(landmarks)}")

    parsed: List[Tuple[float, float, float]] = []
    for idx, lm in enumerate(landmarks):
        try:
            x, y, z = _extract_coords(lm)
        except Exception as e:
            raise ValueError(f"Error parsing landmark at index {idx}: {e}") from e

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise ValueError(f"Landmark at index {idx} contains non-finite values: ({x}, {y}, {z})")

        parsed.append((x, y, z))

    return parsed


def extract_hand_features(
    landmarks: Sequence[Any],
    extension_ratio_threshold: float = 1.10,
    thumb_extension_threshold: float = 1.15,
    thumb_to_index_threshold: float = 0.55,
) -> HandFeatures:
    """
    Extracts geometric scale-normalized features and determines finger extension states.

    Finger Extension Logic (Rotation and Scale Invariant):
      For the 4 fingers (Index, Middle, Ring, Pinky):
        - A finger is extended if its fingertip is significantly further from the wrist
          than its PIP joint: dist(wrist, tip) / dist(wrist, pip) > extension_ratio_threshold
          AND dist(mcp, tip) / dist(mcp, pip) > 1.30.
        - When curled into the palm, the fingertip is closer to the wrist and MCP than the PIP joint.
        - Because this relies on Euclidean distance ratios from anatomical roots, it works whether
          the hand is vertical, horizontal, tilted, or upside down.

      For the Thumb:
        - The thumb moves via a saddle joint across the palm.
        - A thumb is extended if:
          1. dist(wrist, thumb_tip) / dist(wrist, thumb_ip) > thumb_extension_threshold
          2. dist(thumb_tip, index_mcp) / palm_size > thumb_to_index_threshold (not tucked across palm)

    Args:
        landmarks: Sequence of 21 landmarks from HandData (normalized or pixel coordinates).
        extension_ratio_threshold: Minimum wrist-to-tip vs wrist-to-pip ratio for finger extension.
        thumb_extension_threshold: Minimum wrist-to-tip vs wrist-to-ip ratio for thumb extension.
        thumb_to_index_threshold: Minimum normalized distance between thumb tip and index MCP.

    Returns:
        HandFeatures dataclass containing scale metrics, extension ratios, and finger states.
    """
    pts = validate_landmarks(landmarks)

    wrist = pts[WRIST]
    middle_mcp = pts[MIDDLE_MCP]

    # Hand Scale Reference: Palm length from Wrist (0) to Middle MCP (9)
    # Metacarpal bones are rigid; palm length is invariant to finger extension
    palm_size = _euclidean_dist_2d(wrist, middle_mcp)
    if palm_size < 1e-6:
        # Avoid division by zero if hand is degraded or single point
        palm_size = 1e-6

    # 1. Four Fingers (Index, Middle, Ring, Pinky) Extension Features
    wrist_to_tip_ratios: Dict[str, float] = {}
    mcp_to_tip_ratios: Dict[str, float] = {}
    extended_flags: Dict[str, bool] = {}

    for name, (mcp_idx, pip_idx, dip_idx, tip_idx) in FINGER_INDICES.items():
        mcp = pts[mcp_idx]
        pip = pts[pip_idx]
        tip = pts[tip_idx]

        dist_wrist_tip = _euclidean_dist_2d(wrist, tip)
        dist_wrist_pip = _euclidean_dist_2d(wrist, pip)
        dist_mcp_tip = _euclidean_dist_2d(mcp, tip)
        dist_mcp_pip = _euclidean_dist_2d(mcp, pip)

        ratio_wrist = dist_wrist_tip / (dist_wrist_pip + 1e-6)
        ratio_mcp = dist_mcp_tip / (dist_mcp_pip + 1e-6)

        wrist_to_tip_ratios[name] = ratio_wrist
        mcp_to_tip_ratios[name] = ratio_mcp

        # Finger is extended if tip is further from both wrist and MCP than PIP
        is_extended = (ratio_wrist >= extension_ratio_threshold) and (ratio_mcp >= 1.25)
        extended_flags[name] = is_extended

    # 2. Thumb Extension Features
    thumb_tip = pts[THUMB_TIP]
    thumb_ip = pts[THUMB_IP]
    thumb_mcp = pts[THUMB_MCP]
    index_mcp = pts[INDEX_MCP]

    dist_wrist_thumb_tip = _euclidean_dist_2d(wrist, thumb_tip)
    dist_wrist_thumb_ip = _euclidean_dist_2d(wrist, thumb_ip)
    dist_thumb_to_index = _euclidean_dist_2d(thumb_tip, index_mcp) / palm_size

    thumb_ratio = dist_wrist_thumb_tip / (dist_wrist_thumb_ip + 1e-6)

    # Vertical offset of thumb tip relative to wrist (negative y means higher in image)
    thumb_tip_y_offset = (thumb_tip[1] - wrist[1]) / palm_size

    # Thumb is extended if it projects away from the wrist and is not tucked into the palm
    is_thumb_extended = (thumb_ratio >= thumb_extension_threshold) and (dist_thumb_to_index >= thumb_to_index_threshold)

    finger_states = FingerState(
        thumb=is_thumb_extended,
        index=extended_flags["index"],
        middle=extended_flags["middle"],
        ring=extended_flags["ring"],
        pinky=extended_flags["pinky"],
    )

    return HandFeatures(
        palm_size=palm_size,
        wrist_to_tip_ratios=wrist_to_tip_ratios,
        mcp_to_tip_ratios=mcp_to_tip_ratios,
        thumb_extension_ratio=thumb_ratio,
        thumb_to_index_distance=dist_thumb_to_index,
        thumb_tip_y_offset=thumb_tip_y_offset,
        finger_states=finger_states,
        extended_count=finger_states.extended_count,
    )
