"""
GestureFlow AI - ML Gesture Recognizer
======================================
Loads a trained machine learning model from models/custom_gesture_model.joblib
and classifies real-time hand landmarks with confidence scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "custom_gesture_model.joblib"


def extract_features(landmarks) -> List[float]:
    """Extracts 63-D normalized landmark vector matching collect_dataset.py."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    palm_size = (dx * dx + dy * dy + dz * dz) ** 0.5
    if palm_size < 1e-4:
        palm_size = 1.0

    features = []
    for lm in landmarks:
        features.append((lm.x - wrist.x) / palm_size)
        features.append((lm.y - wrist.y) / palm_size)
        features.append((lm.z - wrist.z) / palm_size)
    return features


class MLGestureRecognizer:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.classes = []
        self._load()

    def _load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.classes = bundle["classes"]
            return True
        except Exception as e:
            print(f"[WARN] Failed to load custom ML model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, landmarks) -> Tuple[str, float]:
        """
        Returns (predicted_label, confidence_score [0.0 - 1.0]).
        If model not loaded, returns ('UNKNOWN', 0.0).
        """
        if self.model is None or len(landmarks) != 21:
            return "UNKNOWN", 0.0

        vec = np.array(extract_features(landmarks), dtype=np.float32).reshape(1, -1)
        probs = self.model.predict_proba(vec)[0]
        best_idx = int(np.argmax(probs))
        label = str(self.classes[best_idx])
        confidence = float(probs[best_idx])

        return label, confidence
