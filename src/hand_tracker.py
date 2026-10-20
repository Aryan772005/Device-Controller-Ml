"""
GestureFlow AI - Hand Tracker Module
====================================
Modular real-time hand-tracking pipeline built on the official MediaPipe Tasks API.
Processes OpenCV BGR video frames in RunningMode.VIDEO and extracts 21 3D landmarks per hand.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Standard MediaPipe 21-hand landmark skeletal connections
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index finger (MCP, PIP, DIP, TIP)
# 9-12: Middle finger (MCP, PIP, DIP, TIP)
# 13-16: Ring finger (MCP, PIP, DIP, TIP)
# 17-20: Pinky finger (MCP, PIP, DIP, TIP)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm base & MCP arches
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20),
)


class PixelLandmark(NamedTuple):
    """
    Landmark representation in pixel image coordinates:
      - x: Horizontal pixel coordinate [0, width - 1]
      - y: Vertical pixel coordinate [0, height - 1]
      - z: Relative depth coordinate (smaller value means closer to camera)
    """
    x: int
    y: int
    z: float


class HandData(NamedTuple):
    """
    Complete data for a single detected hand:
      - handedness: 'Left' or 'Right'
      - score: Detection confidence score [0.0, 1.0]
      - normalized_landmarks: List of 21 MediaPipe NormalizedLandmark objects (x in [0,1], y in [0,1], z)
      - pixel_landmarks: List of 21 PixelLandmark objects scaled to image dimensions
    """
    handedness: str
    score: float
    normalized_landmarks: list
    pixel_landmarks: List[PixelLandmark]


class HandTracker:
    """
    Real-time Hand Tracking pipeline using the MediaPipe Tasks Vision API.

    Attributes:
        model_path (Path): Resolved absolute path to the .task model file.
        num_hands (int): Maximum number of hands to detect simultaneously.
        min_detection_confidence (float): Minimum confidence score for hand detection.
        min_presence_confidence (float): Minimum confidence score for hand presence.
        min_tracking_confidence (float): Minimum confidence score for landmark tracking.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        selfie_mode: bool = True,
    ) -> None:
        """
        Initializes the HandTracker with the MediaPipe Tasks HandLandmarker.

        Args:
            model_path: Optional path to hand_landmarker.task. If None, resolves
                        automatically to <project_root>/models/hand_landmarker.task.
            num_hands: Maximum number of hands to detect (default: 2).
            min_detection_confidence: Threshold for palm detection (default: 0.5).
            min_presence_confidence: Threshold for hand presence score (default: 0.5).
            min_tracking_confidence: Threshold for tracking landmark continuity (default: 0.5).
            selfie_mode: Invert handedness labels if the video frame is horizontally flipped
                         for natural selfie-mirror viewing (default: True).

        Raises:
            FileNotFoundError: If the model file cannot be found at the resolved path.
            RuntimeError: If MediaPipe fails to create the HandLandmarker detector.
        """
        self.model_path = self._resolve_model_path(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.selfie_mode = selfie_mode

        self._last_timestamp_ms: int = -1
        self._landmarker: Optional[HandLandmarker] = None

        self._initialize_detector()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str | Path]) -> Path:
        """
        Resolves model path robustly, independent of the terminal's current working directory.
        """
        if model_path is not None:
            resolved = Path(model_path).resolve()
        else:
            # src/hand_tracker.py -> parent is src/ -> parent is project root
            project_root = Path(__file__).resolve().parent.parent
            resolved = project_root / "models" / "hand_landmarker.task"

        if not resolved.is_file():
            raise FileNotFoundError(
                f"MediaPipe model asset not found at: '{resolved}'.\n"
                f"Please verify the model exists or run 'python scripts/download_model.py'."
            )

        if resolved.stat().st_size == 0:
            raise FileNotFoundError(f"Model file at '{resolved}' is empty (0 bytes).")

        return resolved

    def _initialize_detector(self) -> None:
        """Configures and builds the MediaPipe HandLandmarker in VIDEO mode."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        try:
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MediaPipe HandLandmarker with model '{self.model_path}': {e}"
            ) from e

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> List[HandData]:
        """
        Processes a single BGR OpenCV frame and returns detected hands with their landmarks.

        Args:
            frame (np.ndarray): BGR image matrix from cv2.VideoCapture (shape: H x W x 3).
            timestamp_ms (int, optional): Frame timestamp in milliseconds. If None,
                                         calculated automatically from time.perf_counter().

        Returns:
            List[HandData]: A list of HandData objects (one per detected hand, up to num_hands).
                           Each HandData contains handedness, confidence score, 21 normalized
                           landmarks, and 21 scaled pixel coordinates. Returns empty list if no hand found.

        Raises:
            ValueError: If the input frame is invalid or not 3-channel.
            RuntimeError: If called on a closed HandTracker.
        """
        if self._landmarker is None:
            raise RuntimeError("HandTracker is closed. Re-instantiate to process frames.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Input frame must be a valid 3-channel numpy array (OpenCV BGR).")

        height, width, _ = frame.shape

        # 1. Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Wrap RGB buffer into MediaPipe Image container
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Ensure strictly monotonically increasing timestamp for RunningMode.VIDEO
        if timestamp_ms is None:
            current_ms = int(time.perf_counter() * 1000)
        else:
            current_ms = int(timestamp_ms)

        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        # 4. Perform hand landmark detection
        raw_result = self._landmarker.detect_for_video(mp_image, current_ms)

        # 5. Parse and structure the output data
        detected_hands: List[HandData] = []

        if raw_result and raw_result.hand_landmarks:
            num_detected = len(raw_result.hand_landmarks)

            for i in range(num_detected):
                norm_landmarks = raw_result.hand_landmarks[i]

                # Extract handedness and score if available
                handedness_label = "Unknown"
                confidence_score = 0.0
                if raw_result.handedness and i < len(raw_result.handedness):
                    category = raw_result.handedness[i][0]
                    raw_label = category.category_name
                    confidence_score = category.score

                    # In selfie/mirror mode (when frame is horizontally flipped for natural viewing),
                    # horizontal reflection inverts hand geometry: a physical Right hand looks like
                    # a Left hand to the neural network. We map it back to physical reality.
                    if self.selfie_mode:
                        if raw_label == "Left":
                            handedness_label = "Right"
                        elif raw_label == "Right":
                            handedness_label = "Left"
                        else:
                            handedness_label = raw_label
                    else:
                        handedness_label = raw_label

                # Compute pixel coordinates for all 21 landmarks
                # x and y are normalized to [0, 1]; z is relative depth
                pixel_landmarks: List[PixelLandmark] = []
                for lm in norm_landmarks:
                    px = min(int(lm.x * width), width - 1)
                    py = min(int(lm.y * height), height - 1)
                    pixel_landmarks.append(PixelLandmark(x=px, y=py, z=lm.z))

                detected_hands.append(
                    HandData(
                        handedness=handedness_label,
                        score=confidence_score,
                        normalized_landmarks=norm_landmarks,
                        pixel_landmarks=pixel_landmarks,
                    )
                )

        return detected_hands

    @staticmethod
    def draw_landmarks(
        frame: np.ndarray,
        hands: List[HandData],
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Renders hand skeletal connections and 21 landmark keypoints directly onto the frame.

        Args:
            frame: OpenCV BGR image matrix to draw on (modified in-place).
            hands: List of HandData returned by process_frame().
            draw_connections: Whether to draw lines between skeletal joints (default: True).

        Returns:
            np.ndarray: The modified frame with visual tracking overlays.
        """
        for hand in hands:
            pts = [(lm.x, lm.y) for lm in hand.pixel_landmarks]

            # 1. Draw skeletal connection lines
            if draw_connections:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(pts) and end_idx < len(pts):
                        pt1 = pts[start_idx]
                        pt2 = pts[end_idx]
                        # Cyan skeleton lines with anti-aliasing
                        cv2.line(frame, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

            # 2. Draw keypoint landmark circles
            for idx, (px, py) in enumerate(pts):
                # Highlight fingertips (landmarks 4, 8, 12, 16, 20) with distinct color
                if idx in (4, 8, 12, 16, 20):
                    # Fingertip: Magenta outer ring, white center
                    cv2.circle(frame, (px, py), 6, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 2, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                elif idx == 0:
                    # Wrist: Bright Yellow
                    cv2.circle(frame, (px, py), 7, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
                else:
                    # Joint landmarks: Green dot with thin dark border
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 4, (0, 80, 0), 1, cv2.LINE_AA)

            # 3. Label Handedness above wrist
            if pts:
                wrist_x, wrist_y = pts[0]
                label = f"{hand.handedness} ({hand.score:.2f})"
                text_y = max(wrist_y + 24, 20)
                cv2.putText(
                    frame,
                    label,
                    (wrist_x - 30, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def close(self) -> None:
        """Closes the MediaPipe HandLandmarker detector and frees native resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
