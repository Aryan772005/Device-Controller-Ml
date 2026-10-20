# GestureFlow 

GestureFlow AI is a real-time hand gesture recognition system designed for Windows, enabling intuitive, touchless control of desktop and media functions through computer vision and machine learning.

---

## 1. Project Architecture & Setup

### Environment Details
- **Python Version:** `3.13.9` (64-bit)
- **Virtual Environment:** Project-local `.venv`
- **Core Libraries:**
  - `opencv-contrib-python` / `opencv-python` (`5.0.0` / `4.13.0`)
  - `numpy` (`2.4.6`)
  - `mediapipe` (`0.10.35` with modern **Tasks Vision API**)

---

## 2. Virtual Environment Activation

Open PowerShell or Command Prompt in the project root directory:

### PowerShell (Windows)
```powershell
# If execution policy restricts scripts, allow local process execution:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Command Prompt (cmd.exe)
```cmd
.venv\Scripts\activate.bat
```

When activated, you will see `(.venv)` prefixed in your terminal prompt.

To verify your active Python interpreter:
```powershell
python -c "import sys; print(sys.executable)"
```
*(Should output `... \Device controller ml\.venv\Scripts\python.exe`)*

---

## 3. MediaPipe Model Asset

GestureFlow AI uses the modern **MediaPipe Tasks Vision API** (`HandLandmarker`). Unlike deprecated legacy APIs, the Tasks API requires an official model bundle (`.task` file).

- **Model Location:** [`models/hand_landmarker.task`](models/hand_landmarker.task) (~7.46 MB)
- **Model Bundle Details:** Official Google MediaPipe Float16 bundle containing the Palm Detector and 21 3D Hand Landmarks detector.
- **Git Status:** Ignored by [`.gitignore`](.gitignore) to keep the repository lightweight.

To verify or re-download the model at any time:
```powershell
python scripts/download_model.py
```
*(The script is idempotent: it will verify the local copy and skip downloading if valid).*

---

## 4. Webcam Diagnostic Test

Before running gesture recognition, verify that OpenCV can capture live frames from your physical webcam and measure frame rates.

### Run Diagnostic:
```powershell
python scripts/camera_test.py
```

### Controls:
- **`Q` or `ESC`:** Exit diagnostic and cleanly release camera hardware.
- **Select External Camera:** If you have an external USB webcam, supply the camera index:
  ```powershell
  python scripts/camera_test.py 1
  ```

---

## 5. Webcam Troubleshooting Guide (Windows 11)

If the camera test reports `[ERROR] Could not open webcam!`:

1. **Windows 11 Privacy Settings:**
   - Open **Windows Settings** (`Win + I`) -> **Privacy & Security** -> **Camera**.
   - Ensure **Camera access** is toggled **ON**.
   - Ensure **Let desktop apps access your camera** is toggled **ON**.
2. **Camera In Use:**
   - Check if another application is actively locking the camera stream (e.g., Microsoft Teams, Zoom, OBS Studio, Discord, or a browser tab).
   - Close competing camera applications and re-run `python scripts/camera_test.py`.
3. **Backend Fallback:**
   - `scripts/camera_test.py` automatically attempts the Windows DirectShow backend (`cv2.CAP_DSHOW`) if the default Media Foundation backend fails.
4. **Camera Index:**
   - On laptops with built-in IR cameras (Windows Hello) or external USB webcams, the RGB camera may be mapped to index `1` or `2` instead of `0`. Test with:
     ```powershell
     python scripts/camera_test.py 1
     ```

---

## 6. How to Verify Setup

Run this single-line verification check from your activated `.venv`:
```powershell
python -c "import cv2, numpy, mediapipe; from mediapipe.tasks.python import vision; vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=mediapipe.tasks.python.BaseOptions(model_asset_path='models/hand_landmarker.task'))); print('SUCCESS: All dependencies and HandLandmarker model loaded cleanly!')"
```

---

## 7. Milestone 2: Real-Time Hand Tracking Pipeline

Milestone 2 introduces the modular hand tracking engine and visual verification demo.

> [!IMPORTANT]
> **Desktop Safety Boundary:**
> This milestone strictly tracks hand landmarks and displays them visually. It **does NOT** move the mouse cursor, trigger mouse clicks, send keystrokes, adjust volume, or perform any desktop automation. Gesture classification and desktop control will be introduced in subsequent milestones.

### Components Added:
- [`src/hand_tracker.py`](src/hand_tracker.py): A reusable, self-contained `HandTracker` class wrapping Google's MediaPipe Tasks API (`RunningMode.VIDEO`). It processes BGR frames, enforces monotonic timestamps, extracts 21 3D landmarks (x, y, z) per hand, and renders skeletal visualizations.
- [`scripts/hand_tracking_demo.py`](scripts/hand_tracking_demo.py): An interactive visual verification demo with a real-time HUD showing FPS, hand count, and handedness.

### Running the Hand Tracking Demo:
```powershell
# In your activated (.venv) terminal:
python scripts/hand_tracking_demo.py
```

### Expected Visual Output:
1. **Live Mirrored Stream:** The video frame is flipped horizontally for intuitive selfie-style interaction.
2. **21 Landmark Skeleton:**
   - **Fingertips (Landmarks 4, 8, 12, 16, 20):** Highlighted with magenta outer rings and white centers.
   - **Wrist (Landmark 0):** Marked with a bright yellow circle.
   - **Joints:** Marked with bright green dots.
   - **Skeletal Bones:** Rendered as cyan connecting lines.
3. **Handedness & Confidence Tag:** Displays whether the detected hand is Left or Right with detection confidence (e.g., `Right (0.95)`).
4. **Information HUD (Top Left):** Displays real-time smoothed FPS, number of hands detected, and status messages.

### Landmark Coordinates Explained:
- **`x` and `y` (Normalized):** Range from `0.0` to `1.0` relative to image width and height.
- **`z` (Relative Depth):** Represents landmark depth relative to the wrist. Smaller/negative values indicate the point is closer to the webcam.
- **`pixel_landmarks`:** Scaled integer coordinates `(px, py)` mapped directly to your webcam frame resolution for easy drawing and geometric distance calculations.

---

## 8. Milestone 3: Rule-Based Gesture Recognition & Temporal Stabilization

Milestone 3 adds transparent, scale-invariant geometric feature extraction, discrete gesture classification, and temporal filtering.

> [!IMPORTANT]
> **Desktop Safety Boundary:**
> This milestone strictly classifies postures into semantic labels and visual badges. It **does NOT** move the mouse, generate clicks, control volume, or interact with Windows OS functions.

### Architecture Pipeline
```
Webcam Frame -> HandTracker -> 21 3D Landmarks -> FeatureExtractor -> GestureRecognizer -> GestureStabilizer -> Visual HUD
```

### Supported Gestures:
| Gesture Label | Visual Badge | Physical Definition & Rules |
| :--- | :--- | :--- |
| `OPEN_PALM` | Green | All five fingers (Thumb, Index, Middle, Ring, Pinky) fully extended away from palm. |
| `FIST` | Amber | All fingers curled into the palm; thumb folded across fingers. |
| `POINTING` | Magenta | Index finger extended outward; middle, ring, pinky curled into palm. |
| `PEACE` | Green | Index and middle fingers extended with non-zero V-separation; ring and pinky curled. |
| `THUMBS_UP` | Cyan | Thumb extended vertically upward ($y_{\text{tip}} < y_{\text{wrist}} - 0.3 \times \text{palm}$); other 4 fingers curled. |
| `UNKNOWN` | Gray | Transitional states, ambiguous postures, or unsupported hand shapes. |

### How Geometric Feature Extraction Works:
1. **Palm Scale Normalization:** Uses the distance between Wrist (`Landmark 0`) and Middle MCP (`Landmark 9`) as an anatomical scale baseline ($\text{palm\_size}$). Hand distances scale naturally whether close to or far from the camera.
2. **Rotation-Invariant Extension Ratios:** Rather than testing vulnerable vertical $y$-coordinates ($y_{\text{tip}} < y_{\text{pip}}$), each finger tests:
   $$\text{ratio}_{\text{wrist}} = \frac{\text{dist}(\text{wrist}, \text{tip})}{\text{dist}(\text{wrist}, \text{pip})} \ge 1.10 \quad \text{and} \quad \text{ratio}_{\text{mcp}} = \frac{\text{dist}(\text{mcp}, \text{tip})}{\text{dist}(\text{mcp}, \text{pip})} \ge 1.25$$
   When extended, the fingertip is always further from the wrist than its PIP joint, regardless of whether the hand is tilted sideways, pointing diagonally, or upside down.
3. **Thumb Saddle Joint Dynamics:** Thumb extension accounts for abduction away from the index finger MCP base.

### How Temporal Stabilization Works:
- Maintains a rolling history of predictions (window size: 5 frames) per hand (`Left` and `Right` are tracked independently).
- Emits a gesture only when at least 60% of recent frames agree on the same label, eliminating single-frame tracking flicker.
- When a hand disappears from view for 5+ frames, its buffer automatically resets.

### Running the Gesture Recognition Demo:
```powershell
# In your activated (.venv) terminal:
python scripts/gesture_recognition_demo.py
```

### Running Automated Gesture Tests:
```powershell
python -m unittest tests/test_gesture_recognizer.py -v
```

### Known Limitations:
1. **Extreme Fore-shortening:** If a finger points directly into the camera lens along the z-axis, 2D perspective projection collapses its length, which can lead to `UNKNOWN`.
2. **Partial Occlusions:** If the thumb is hidden behind the palm, it may not register as extended.
3. **Motion Blur:** Extremely fast hand waving degrades landmark confidence; the stabilizer gracefully outputs `UNKNOWN` until tracking recovers.

---

## 9. Milestone 4: Gesture Reliability, Temporal Filtering & Event Debouncing

Milestone 4 prepares the gesture recognition engine for downstream automation by introducing **lifecycle event debouncing**, **cooldown enforcement**, and **diagnostic telemetry**.

> [!IMPORTANT]
> **Safety Guarantee:**
> This milestone strictly emits in-memory software events (`GestureEvent`). It **does NOT** move the mouse cursor, simulate keystrokes, adjust volume, or interact with the operating system.

### The Problem Milestone 4 Solves:
In video streams at 30 FPS, holding an `OPEN_PALM` gesture for 3 seconds causes the recognizer to emit `OPEN_PALM` 90 consecutive times!
If directly linked to a desktop action (e.g., "mute volume" or "switch window"), it would trigger 90 times per second, crashing the workflow.

### Architecture & Lifecycle States:
```
Stabilized Gesture -> GestureEventManager -> Single TRIGGER Event -> Continuous HOLD Events -> RELEASE Event
```

1. **`TRIGGER`:** Fired exactly **once** when a stable gesture is first recognized.
2. **`HOLD`:** Emitted on subsequent frames while the gesture remains continuously held. Continuous holding **never** repeats action triggers.
3. **`RELEASE`:** Fired when the hand relaxes, changes to another gesture, or leaves the camera frame.
4. **Configurable Cooldown:** After a gesture triggers and releases, a cooldown timer (default: `1.0s`) prevents accidental immediate re-triggers.
5. **Hand Isolation:** Left and Right hands have completely separated state machines, timers, and event counters.

### Hand Association Limitations Across Frames:
- MediaPipe classifies hand chirality (`Left` vs `Right`) using local hand appearance.
- It does not perform full-body skeletal tracking or spatial trajectory ID tracking.
- If the user crosses wrists or brings both hands close together, severe optical occlusion can cause MediaPipe to temporarily swap or drop hand labels.
- When hands re-enter the camera view, state is restored strictly based on the reported `handedness` label.

### Running the Diagnostic Demo:
```powershell
# In your activated (.venv) terminal:
python scripts/gesture_recognition_demo.py
```

#### Diagnostic Overlays:
- **Floating Badge:** Displays `Hand: StableGesture`, raw prediction, `STABLE` / `UNSTABLE` agreement percentage, current event state (`TRIGGER` / `HOLD` / `RELEASE`), and total trigger count.
- **Visual Event Flash:** When a new `TRIGGER` event occurs, the floating badge outline flashes white for that frame.

### Running the Full Test Suite:
```powershell
python -m unittest discover tests -v
```
*(Runs all 33 automated tests covering geometric validation, gesture rules, temporal filtering, hold expiration, debouncing, safety states, and desktop automation dispatch).*

---

## 10. Milestone 5: Safe Windows Desktop Controller

Milestone 5 connects the vision and event pipeline to a **safe, modular Windows desktop controller**.

> [!CAUTION]
> **Active Mode Safety Warning:**
> When the controller is in **`ACTIVE`** mode, detected gestures send genuine virtual key commands to Windows (e.g. toggling audio mute, pausing YouTube/Spotify, or increasing volume).
> - The application **always starts in safe `DRY_RUN` mode**.
> - It will **never** execute real actions without your explicit keyboard arming command (`A`).
> - Press **`ESC`** or **`E`** at any moment for an immediate **Emergency Stop**.

### System Architecture Pipeline
```
Webcam Frame
    │
    ▼
MediaPipe HandTracker (RunningMode.VIDEO)
    │
    ▼
21 3D Landmarks
    │
    ▼
FeatureExtractor (Scale & Rotation Invariant)
    │
    ▼
GestureRecognizer (OPEN_PALM, FIST, POINTING, PEACE, THUMBS_UP)
    │
    ▼
GestureStabilizer (5-Frame Supermajority Window)
    │
    ▼
GestureEventManager (Single-Trigger, Cooldown, Hold Tracking)
    │
    ▼
DesktopController (Safety State Machine & Virtual Key Dispatch)
```

### Configured Gesture-to-Action Mappings:
| Gesture | Action | Windows Key Code | Behavior |
| :--- | :--- | :--- | :--- |
| `FIST` | `MEDIA_PLAY_PAUSE` | `VK_MEDIA_PLAY_PAUSE` (0xB3) | Toggles play/pause on active media (Spotify, YouTube, VLC). |
| `PEACE` | `MUTE_TOGGLE` | `VK_VOLUME_MUTE` (0xAD) | Mutes or unmutes system audio. |
| `THUMBS_UP` | `VOLUME_UP` | `VK_VOLUME_UP` (0xAF) | Increases volume by a bounded single step (2 volume notches). |
| `POINTING` | `NO_OP` | N/A | Reserved for future mouse cursor navigation. |
| `OPEN_PALM` | `NO_OP` | N/A | Reserved as safety rest gesture (no action performed). |
| `UNKNOWN` | `NO_OP` | N/A | Suppressed; never triggers actions. |

### Safety State Machine & Transitions:
```
               ┌──────────────┐
               │   DISABLED   │
               └──────┬───────┘
                      │ (Press 'D')
                      ▼
               ┌──────────────┐   (Press 'A')   ┌──────────────┐
  (Reset 'R')  │   DRY_RUN    │ ──────────────> │    ACTIVE    │
    ┌────────> │  (Default)   │ <────────────── │(Live Windows)│
    │          └──────┬───────┘   (Press 'D')   └──────┬───────┘
    │                 │                                │
    │                 │ (Press 'ESC' or 'E')           │ (Press 'ESC' or 'E')
    │                 ▼                                ▼
    │          ┌───────────────────────────────────────────────┐
    └───────── │                EMERGENCY_STOP                 │
               │ (Locks all actions; cannot auto-rearm to ACT) │
               └───────────────────────────────────────────────┘
```

1. **`DRY_RUN` (Default):** Incoming gesture triggers are simulated and logged on screen without touching the operating system.
2. **`ACTIVE`:** Requires pressing **`A`** to arm. Dispatches real Windows virtual key events.
3. **`DISABLED`:** Pressing **`X`** silences all gesture processing.
4. **`EMERGENCY_STOP`:** Pressing **`ESC`** or **`E`** locks all actions immediately. The controller cannot automatically re-arm to `ACTIVE`. You must press **`R`** to return to safe `DRY_RUN`.

### Interactive Desktop Controller Demo:
```powershell
# In your activated (.venv) terminal:
python scripts/desktop_controller_demo.py
```

#### Keyboard Controls:
- **`A`**: **Arm / Activate** live Windows control (`ACTIVE` mode).
- **`D`**: Return to **DRY_RUN** simulation mode.
- **`X`**: **Disable** controller.
- **`ESC`** or **`E`**: **Emergency Stop** (instantly halts and blocks all actions).
- **`R`**: **Reset** Emergency Stop back to `DRY_RUN`.
- **`Q`**: **Quit** demo and cleanly release webcam hardware.

### Windows Permissions & Native Implementation:
- Uses Windows `user32.dll` via Python's standard library `ctypes.windll.user32.keybd_event`.
- **Zero shell commands:** Does not call `subprocess`, PowerShell, or Command Prompt.
- **Zero third-party driver dependencies:** Operates with standard Windows user permissions without requiring administrator escalation.
- **Mock Backends in Tests:** All 33 unit tests use dependency-injected mocks (`MockActionBackend`), guaranteeing that running automated tests never sends keystrokes or changes your computer volume.
