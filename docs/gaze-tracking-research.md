# Gaze Tracking Research

Date: 2026-02-22

## 1. Overview

Hand gesture + eye gaze tracking fusion for precise mouse/cursor control.
Goal: eye provides coarse cursor positioning, hand provides commands (click, scroll, drag) and fine adjustment.

## 2. Gaze Estimation Models Comparison

### Calibration-Free, Pretrained Available

| Model | MPIIGaze Error | Gaze360 Error | Size | Code | Weights | ONNX |
|-------|---------------|---------------|------|------|---------|------|
| **L2CS-Net** | **3.92°** | 10.41° | 91MB (ResNet50) | [GitHub](https://github.com/Ahmednull/L2CS-Net) | Yes | No (PyTorch) |
| **MobileGaze** | - | 12.58° (MobileOne S0) | **4.8MB** | [GitHub](https://github.com/yakhyo/gaze-estimation) | Yes | **Yes** |
| **MobileGaze** (ResNet18) | - | 12.84° | 43MB | same | Yes | **Yes** |
| **MobileGaze** (MobileNetV2) | - | 13.07° | 9.59MB | same | Yes | **Yes** |
| **RAGE-net** | **3.96°** | - | lightweight | [GitHub](https://github.com/ragenetresearch/democratizing-eye-tracking-rage-net) | Yes | No |
| **3DGazeNet** (ECCV 2024) | - | - | - | [GitHub](https://github.com/eververas/3DGazeNet) | Yes | No |

### Paper-Only (No Code/Weights)

| Model | MPIIFaceGaze | Gaze360 | Note |
|-------|-------------|---------|------|
| GazeCapsNet | 4.06° | 5.10° | No GitHub, no weights. CC BY 4.0 paper only |
| GazeTR-Hybrid | - | - | [GitHub](https://github.com/yihuacheng/GazeTR), pretrained on ETH-XGaze |

### Landmark-Based (No ML Model Needed)

| Approach | Accuracy | Note |
|----------|----------|------|
| MediaPipe Iris | Low (~direction only) | Already in MediaPipe dependency. Iris position only, not gaze angle |

## 3. Recommended Model: MobileGaze (L2CS-Net Based)

**Why:**
- ONNX export supported → Rust deployment via `ort` crate
- MobileOne S0 = 4.8MB (smaller than hand_landmarker.task at 7.8MB)
- Calibration-free: pretrained weights output yaw/pitch directly
- Built on L2CS-Net (3.92° on MPIIGaze, well-validated)

**Output:** yaw (horizontal angle), pitch (vertical angle) in degrees

**Screen Mapping (no calibration):**
```
YAW_RANGE = 35.0°
PITCH_RANGE = 25.0°
screen_x = 0.5 + (yaw / YAW_RANGE) * 0.5    → 0.0~1.0
screen_y = 0.5 - (pitch / PITCH_RANGE) * 0.5  → 0.0~1.0
```

## 4. Hand + Gaze Fusion Strategy

### Coarse-Fine Separation (IEEE Validated)

Reference: [Combining eye gaze and hand tracking for pointer control](https://ieeexplore.ieee.org/document/6504305/)

```
Eye (fast, imprecise ~100-300px) → Coarse cursor position
Hand (slow, precise ~10-30px)    → Commands + fine adjustment
```

### AOI (Area of Interest) Method

```
Full Screen (1920x1080)
  ├── Eye gaze → AOI center (e.g., 400x400 region)
  └── Hand finger position → mapped within AOI for precision
```

### Gaze Warping Method

```
1. Gaze shifts to new area → cursor jumps (warp)
2. Hand gesture → fine-tune position relative to gaze point
```

### Interaction Matrix

| Scenario | Eye | Hand | Result |
|----------|-----|------|--------|
| Browse | Look at link | Pinch | Click at gaze point |
| Read | Look down | Thumbs up | Scroll down |
| Select | Look at area | Open hand + move | Fine cursor adjust |
| Drag | Look at start | Pinch hold | Drag from gaze point |

## 5. Pipeline Architecture

### Python Prototype (Phase 1)

```
webcam frame (shared)
  ├── HandTracker.detect()   → [HandData]  (existing)
  ├── GazeTracker.detect()   → (yaw, pitch) (new)
  └── FusionController       → combined control (new)
       ├── Gaze → cursor position (replaces move_pointer)
       ├── Hand gesture → click/scroll/drag (existing logic)
       └── Hand → fine adjustment (optional)
```

### Rust Deployment (Phase 2)

```
Single binary + ONNX model files (~15-25MB total)
  ├── ort crate        — ONNX inference (DirectML GPU)
  ├── nokhwa/opencv    — webcam capture
  ├── enigo            — mouse/keyboard control (or KVM project integration)
  └── rusty_scrfd      — face detection (SCRFD ONNX)
```

## 6. Rust Ecosystem for Deployment

| Crate | Purpose | Note |
|-------|---------|------|
| [ort](https://github.com/pykeio/ort) v2.0.0-rc.11 | ONNX Runtime (DirectML, CUDA, TensorRT) | 2k stars, mature |
| [rusty_scrfd](https://lib.rs/crates/rusty_scrfd) | Face detection (SCRFD ONNX) | Bounding box + keypoints |
| [nokhwa](https://crates.io/crates/nokhwa) | Webcam capture | Cross-platform |
| [opencv](https://crates.io/crates/opencv) | Image processing | Rust bindings |
| [enigo](https://crates.io/crates/enigo) | Mouse/keyboard control | pyautogui equivalent |

### ONNX Models Needed for Rust

| Model | Purpose | Source |
|-------|---------|--------|
| `mobilegaze_s0.onnx` | Gaze estimation (yaw, pitch) | MobileGaze repo export |
| `scrfd_500m.onnx` | Face detection | SCRFD pretrained |
| `hand_landmark.onnx` | Hand pose (21 keypoints) | MediaPipe TFLite → ONNX convert |
| `palm_detection.onnx` | Hand detection | MediaPipe TFLite → ONNX convert |

### Binary Size Estimate

```
Rust exe:          ~5MB
ONNX Runtime DLL:  ~20MB
ONNX models:       ~15MB (gaze 4.8MB + face 2MB + hand 8MB)
Total:             ~40MB (vs PyInstaller 150-300MB)
```

## 7. Key Considerations

### Calibration-Free Mapping Limitations
- Linear yaw/pitch → screen mapping works for coarse pointing
- Range constants (YAW_RANGE, PITCH_RANGE) need one-time tuning
- Head movement compensation needed (head pose from face landmarks)
- Not suitable for pixel-precise pointing alone → hand fusion solves this

### Midas Touch Problem
"Looking at something" ≠ "wanting to interact with it"
→ Solved by requiring hand gesture activation (hand = intent signal)

### Performance Budget (per frame)
```
Face detection (SCRFD):    ~3ms
Gaze estimation (MobileGaze): ~3ms
Hand detection + landmark:  ~10ms
Fusion logic:              ~1ms
Total:                     ~17ms → 58 FPS achievable
```

## 8. References

- [L2CS-Net Paper](https://arxiv.org/abs/2203.03339) — Fine-Grained Gaze Estimation in Unconstrained Environments
- [MobileGaze](https://github.com/yakhyo/gaze-estimation) — Pre-trained mobile nets for gaze estimation
- [RAGE-net](https://github.com/ragenetresearch/democratizing-eye-tracking-rage-net) — Calibration-free gaze estimation
- [3DGazeNet](https://github.com/eververas/3DGazeNet) — ECCV 2024, generalizing without adaptation
- [GazeCapsNet Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11860563/) — Lightweight capsule network (no code)
- [gaze-tracking-pipeline](https://github.com/pperle/gaze-tracking-pipeline) — Full camera-to-screen pipeline reference
- [Eye+Hand Fusion (IEEE)](https://ieeexplore.ieee.org/document/6504305/) — Combining gaze and hand for pointer control
- [Gaze+Gesture](https://www.semanticscholar.org/paper/Gaze%2BGesture:-Expressive,-Precise-and-Targeted-Chatterjee-Xiao/97ca8a44db7ed7d89049e76287583b2cedac30e8) — Expressive, Precise and Targeted Free-Space Interactions
- [MediaPipe Iris](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/) — Real-time iris tracking
- [ort crate](https://github.com/pykeio/ort) — Rust ONNX Runtime bindings
- [PINTO0309 hand-gesture ONNX](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) — MediaPipe → ONNX converted models
