"""
GestureFlow AI - Custom Gesture Model Trainer
=============================================
Trains a machine learning classifier on your recorded gesture dataset
(data/gestures_dataset.csv) and saves the trained model to models/custom_gesture_model.joblib.

Usage:
    python scripts/train_classifier.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "gestures_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "custom_gesture_model.joblib"


def load_dataset():
    if not CSV_FILE.exists():
        print(f"[ERROR] Dataset not found at: {CSV_FILE}")
        print("Please run 'python scripts/collect_dataset.py' first to record your gesture samples!")
        sys.exit(1)

    labels = []
    features = []

    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if not row or len(row) < 2:
                continue
            labels.append(row[0])
            features.append([float(val) for val in row[1:]])

    if len(features) < 10:
        print(f"[ERROR] Dataset has only {len(features)} samples. Please record at least 50-100 samples per gesture.")
        sys.exit(1)

    return np.array(features, dtype=np.float32), np.array(labels)


def train():
    print("=" * 65)
    print("   GESTUREFLOW AI - TRAINING CUSTOM ML CLASSIFIER")
    print("=" * 65)

    X, y = load_dataset()
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\nLoaded {len(X)} total samples across {len(unique_classes)} gesture classes:")
    for cls_name, count in zip(unique_classes, counts):
        print(f"  * {cls_name}: {count} samples")

    if len(unique_classes) < 2:
        print("\n[WARNING] You only recorded 1 gesture class! You need at least 2 distinct classes (e.g. MOVE and CLICK) to train a classifier.")
        print("Run 'python scripts/collect_dataset.py' to record more classes.")
        sys.exit(1)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")
    print("Training Random Forest Classifier (100 estimators)...")

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100.0
    test_acc = clf.score(X_test, y_test) * 100.0

    print(f"\n[RESULTS]")
    print(f"  Training Accuracy : {train_acc:.2f}%")
    print(f"  Testing Accuracy  : {test_acc:.2f}%")

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and class metadata
    model_bundle = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_dim": X.shape[1],
    }

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model_bundle, MODEL_FILE)
    print(f"\n[SUCCESS] Model saved cleanly to: {MODEL_FILE}")
    print("You can now run live custom ML gesture control!")


if __name__ == "__main__":
    train()
