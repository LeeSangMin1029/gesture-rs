# Hand Detection Pipeline

## Architecture
Palm Detection (192x192) → ROI Extraction → Hand Landmark (224x224) → Gesture Classification

## Models
- `palm_detection.onnx`: SSD-based, input `[1,192,192,3]` HWC, normalize `[-1,1]` (/127.5-1.0)
  - Output: `Identity [1,2016,18]` bboxes (y,x,h,w order), `Identity_1 [1,2016,1]` scores
  - 2016 anchors: strides [8,16,16,16], 2 per grid cell
- `hand_landmark.onnx`: input `[1,224,224,3]` HWC, normalize `[0,1]` (/255.0)
  - Output: `Identity [1,63]` keypoints (21x3), `Identity_1 [1,1]` confidence, `Identity_2 [1,1]` handedness

## Key Constants
```
SCORE_THRESH = 0.2 (palm detection)
NMS_THRESH = 0.3
PALM_ROI_SCALE = 2.0, PALM_ROI_SHIFT_Y = -0.25
TRACK_ROI_SCALE = 1.5, TRACK_ROI_SHIFT_Y = -0.05
```

## Multi-Hand Tracking (up to 2 hands)
`TrackedHand` struct: per-hand `RotatedRoi` (pixel space) + `OneEuroFilter` (21 landmarks x 2)

detect() 3-step pipeline:
1. **Update tracked hands**: run landmark model on existing ROIs, update filters/ROI
2. **Palm detection**: find new palms, skip if overlap with tracked (roi_distance < 0.5*size)
3. **Scan fallback**: cycle through 4 candidate ROIs to catch back-of-hand / far-distance

## Rotation-Aware Affine Transform
- `RotatedRoi`: cx_px, cy_px, size_px, rotation (all pixel space)
- `roi_from_palm()`: rotation from keypoints[0]→keypoints[2], screen Y negation
- `roi_from_landmarks_rotated()`: rotation from wrist(0)→mean(PIP 6,10,14)
- `affine_crop()`: 2x3 inverse affine matrix for landmark→frame mapping

## Gestures (classify_gesture)
Fist (curled>=3), Pinch, TwoFingers, ThumbsUp, ThumbsDown, OpenHand, Pointer

## Known Issues
- Fast hand shaking loses tracking (scan fallback latency)
- Back-of-hand detection relies on scan fallback (no dedicated palm model for back)

## Files
- `src/hand.rs`: HandDetector, TrackedHand, RotatedRoi, affine transform, gesture classification
- `src/filter.rs`: OneEuroFilter for landmark smoothing
- `src/debug_view.rs`: skeleton rendering (green=1st hand, blue=2nd hand)
- `src/controller.rs`: GestureController state machine
- `src/gesture.rs`: Gesture/InputEvent enums
