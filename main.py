"""
GestureFlow AI - Real Windows Gesture Controller
=================================================
Main entry point for real-time touchless device control on Windows.
Runs directly in ACTIVE mode with live mouse, keyboard, media, and browser control.

Usage:
    python main.py                     # Runs real Windows control immediately
    python main.py --dry-run           # Runs in safe simulation mode
    python main.py --camera 1          # Use external USB camera
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.full_control_demo import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GestureFlow AI - Real-time Windows Touchless Gesture Controller"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Start in safe DRY_RUN simulation mode instead of live ACTIVE control",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Use custom trained ML model (trained via scripts/train_classifier.py)",
    )
    args = parser.parse_args()

    start_active = not args.dry_run

    print("\n" + "=" * 65)
    print("   GESTUREFLOW AI - REAL WINDOWS DEVICE CONTROLLER")
    print("=" * 65)
    if args.ml:
        print(" [MODEL] CUSTOM ML MODEL ACTIVE (--ml)")
    if start_active:
        print(" [STATUS] ACTIVE MODE (REAL CONTROL)")
        print("  * Point with index finger to MOVE MOUSE CURSOR")
        print("  * Pinch thumb + index to LEFT CLICK")
        print("  * Fist to RIGHT CLICK")
        print("  * Thumbs Up / Down to change VOLUME")
        print("  * Peace sign (V) for ALT+TAB / JUMP")
        print("  * Press 'G' to play Subway Surfers / browser games with hand swipes!")
        print("  * Press 'M' to switch modes (DESKTOP -> MEDIA -> GAMING -> BROWSER)")
        print("  * Press 'ESC' or 'E' anytime for EMERGENCY STOP")
        print("  * Press 'Q' to quit")
    else:
        print(" [STATUS] DRY RUN MODE (SIMULATION ONLY - NO OS ACTIONS)")
        print("  * Press 'A' to arm real Windows control")
    print("=" * 65 + "\n")

    run(camera_index=args.camera, start_active=start_active, use_ml=args.ml)


if __name__ == "__main__":
    main()
