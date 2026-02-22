// Hand detection pipeline: Palm Detection → Hand Landmark → Gesture Classification
//
// Palm Detection: 192x192 HWC → [2016, 18] bboxes + [2016, 1] scores
// Hand Landmark:  224x224 HWC → [63] keypoints + [1] confidence + [1] handedness

use crate::gesture::Gesture;
use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

const PALM_SIZE: usize = 192;
const LANDMARK_SIZE: usize = 224;
const SCORE_THRESH: f32 = 0.5;
const NMS_THRESH: f32 = 0.3;
const ROI_EXPAND: f32 = 2.9;
const ROI_TRACK_EXPAND: f32 = 2.0;
const PINCH_RATIO_THRESH: f32 = 0.25; // pinch_dist / hand_size ratio

// --- Types ---

struct Anchor {
    cx: f32,
    cy: f32,
}

#[derive(Debug, Clone)]
struct PalmDetection {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    score: f32,
    _keypoints: [(f32, f32); 7],
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HandResult {
    /// 21 keypoints (x, y, z) normalized to original frame [0..1]
    pub points: [(f32, f32, f32); 21],
    pub label: HandLabel,
    pub gesture: Gesture,
    pub pinch_dist: f32,
    pub middle_pinch_dist: f32,
    pub finger_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandLabel {
    Left,
    Right,
}

impl std::fmt::Display for HandLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandLabel::Left => write!(f, "Left"),
            HandLabel::Right => write!(f, "Right"),
        }
    }
}

// --- Hand Detector ---

pub struct HandDetector {
    palm_session: Session,
    landmark_session: Session,
    anchors: Vec<Anchor>,
    prev_roi: Option<(f32, f32, f32)>, // (cx, cy, side) for frame-to-frame tracking
    frame_count: u64,
}

impl HandDetector {
    pub fn new(palm_model: &str, landmark_model: &str) -> Result<Self> {
        let palm_session = Session::builder()?
            .commit_from_file(palm_model)
            .context("Failed to load palm detection model")?;
        let landmark_session = Session::builder()?
            .commit_from_file(landmark_model)
            .context("Failed to load hand landmark model")?;
        let anchors = generate_anchors();
        tracing::info!("HandDetector loaded ({} anchors)", anchors.len());
        Ok(Self {
            palm_session,
            landmark_session,
            anchors,
            prev_roi: None,
            frame_count: 0,
        })
    }

    pub fn is_tracking(&self) -> bool {
        self.prev_roi.is_some()
    }

    /// Detect hands in an RGB frame (HWC, u8, 3 channels)
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<HandResult>> {
        self.frame_count += 1;
        // Tracking: reuse previous ROI to skip palm detection
        if let Some((cx, cy, side)) = self.prev_roi {
            // Reset if ROI drifted off-screen
            if cx < -0.1 || cx > 1.1 || cy < -0.1 || cy > 1.1 {
                self.prev_roi = None;
            } else if let Some(hand) = self.detect_landmarks_roi(rgb, w, h, cx, cy, side)? {
                let new_roi = roi_from_landmarks(&hand.points);
                if new_roi.0 > -0.1 && new_roi.0 < 1.1 && new_roi.1 > -0.1 && new_roi.1 < 1.1 {
                    self.prev_roi = Some(new_roi);
                } else {
                    self.prev_roi = None;
                }
                return Ok(vec![hand]);
            } else {
                self.prev_roi = None;
            }
        }

        let palms = self.detect_palms(rgb, w, h)?;
        let mut results = Vec::new();
        for palm in &palms {
            let side = palm.w.max(palm.h) * ROI_EXPAND;
            if let Some(hand) = self.detect_landmarks_roi(rgb, w, h, palm.cx, palm.cy, side)? {
                self.prev_roi = Some(roi_from_landmarks(&hand.points));
                results.push(hand);
            }
        }
        Ok(results)
    }

    fn detect_palms(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<PalmDetection>> {
        let input = preprocess_image(rgb, w, h, PALM_SIZE);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .palm_session
            .run(ort::inputs!["input_1" => tensor])?;

        // Identity = [1, 2016, 18], Identity_1 = [1, 2016, 1]
        let (_, bbox_data) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, score_data) = outputs["Identity_1"].try_extract_tensor::<f32>()?;

        let scale = PALM_SIZE as f32;
        let mut detections = Vec::new();

        for i in 0..self.anchors.len() {
            let score = sigmoid(score_data[i]);
            if score < SCORE_THRESH {
                continue;
            }
            let a = &self.anchors[i];

            // MediaPipe SSD output order: [y_center, x_center, h, w, ky0, kx0, ...]
            let cy = bbox_data[i * 18] / scale + a.cy;
            let cx = bbox_data[i * 18 + 1] / scale + a.cx;
            let bh = bbox_data[i * 18 + 2] / scale;
            let bw = bbox_data[i * 18 + 3] / scale;

            let mut keypoints = [(0.0f32, 0.0f32); 7];
            for k in 0..7 {
                keypoints[k] = (
                    bbox_data[i * 18 + 4 + k * 2 + 1] / scale + a.cx, // kx
                    bbox_data[i * 18 + 4 + k * 2] / scale + a.cy,     // ky
                );
            }
            detections.push(PalmDetection {
                cx,
                cy,
                w: bw.abs(),
                h: bh.abs(),
                score,
                _keypoints: keypoints,
            });
        }

        Ok(nms(detections, NMS_THRESH))
    }

    fn detect_landmarks_roi(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        roi_cx: f32,
        roi_cy: f32,
        side: f32,
    ) -> Result<Option<HandResult>> {
        let input = crop_and_resize(rgb, w, h, roi_cx, roi_cy, side, LANDMARK_SIZE);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .landmark_session
            .run(ort::inputs!["input_1" => tensor])?;

        // Identity = [1, 63] keypoints, Identity_1 = [1, 1] confidence
        let (_, kp_data) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, conf_data) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        let confidence = sigmoid(conf_data[0]);

        if confidence < 0.5 {
            return Ok(None);
        }

        // Parse 21 keypoints and map back to original frame coordinates
        let mut points = [(0.0f32, 0.0f32, 0.0f32); 21];
        let roi_x1 = roi_cx - side / 2.0;
        let roi_y1 = roi_cy - side / 2.0;

        for i in 0..21 {
            // Landmark output is in pixel space [0..224] within the crop
            let lx = kp_data[i * 3] / LANDMARK_SIZE as f32;
            let ly = kp_data[i * 3 + 1] / LANDMARK_SIZE as f32;
            let lz = kp_data[i * 3 + 2] / LANDMARK_SIZE as f32;
            // Map back to original frame [0..1]
            points[i] = (roi_x1 + lx * side, roi_y1 + ly * side, lz);
        }

        // Determine handedness from wrist-to-middle-finger direction
        let label = estimate_handedness(&points);

        // Classify gesture
        let (gesture, pinch_dist, mid_pinch_dist, fingers) =
            classify_gesture(&points, label);

        Ok(Some(HandResult {
            points,
            label,
            gesture,
            pinch_dist,
            middle_pinch_dist: mid_pinch_dist,
            finger_count: fingers,
        }))
    }
}

// --- Anchor Generation ---
// SSD anchors: strides=[8,16,16,16], 2 per position, fixed_size=True

fn generate_anchors() -> Vec<Anchor> {
    let strides = [8, 16, 16, 16];
    let mut anchors = Vec::with_capacity(2016);
    for &stride in &strides {
        let grid = (PALM_SIZE as f32 / stride as f32).ceil() as usize;
        for y in 0..grid {
            for x in 0..grid {
                let cx = (x as f32 + 0.5) / grid as f32;
                let cy = (y as f32 + 0.5) / grid as f32;
                anchors.push(Anchor { cx, cy });
                anchors.push(Anchor { cx, cy });
            }
        }
    }
    anchors
}

// --- Image Preprocessing ---

/// Resize RGB image to target×target (stretch), normalize to [0..1]
fn preprocess_image(rgb: &[u8], w: u32, h: u32, target: usize) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, target, target, 3));
    let buf = out.as_slice_mut().unwrap();
    let scale_x = w as f32 / target as f32;
    let scale_y = h as f32 / target as f32;

    for ty in 0..target {
        for tx in 0..target {
            let sx = ((tx as f32 + 0.5) * scale_x) as u32;
            let sy = ((ty as f32 + 0.5) * scale_y) as u32;
            let sx = sx.min(w - 1);
            let sy = sy.min(h - 1);
            let src = ((sy * w + sx) * 3) as usize;
            let dst = (ty * target + tx) * 3;
            buf[dst] = rgb[src] as f32 / 255.0;
            buf[dst + 1] = rgb[src + 1] as f32 / 255.0;
            buf[dst + 2] = rgb[src + 2] as f32 / 255.0;
        }
    }
    out
}

/// Crop a square ROI and resize to target_size, normalize to [0..1], HWC (1D slice access)
fn crop_and_resize(
    rgb: &[u8],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    side: f32,
    target: usize,
) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, target, target, 3));
    let buf = out.as_slice_mut().unwrap();
    let x1 = (cx - side / 2.0) * w as f32;
    let y1 = (cy - side / 2.0) * h as f32;
    let crop_w = side * w as f32;
    let crop_h = side * h as f32;

    for ty in 0..target {
        for tx in 0..target {
            let sx = x1 + (tx as f32 + 0.5) / target as f32 * crop_w;
            let sy = y1 + (ty as f32 + 0.5) / target as f32 * crop_h;
            let sx = sx as i32;
            let sy = sy as i32;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let src = ((sy as u32 * w + sx as u32) * 3) as usize;
                let dst = (ty * target + tx) * 3;
                buf[dst] = rgb[src] as f32 / 255.0;
                buf[dst + 1] = rgb[src + 1] as f32 / 255.0;
                buf[dst + 2] = rgb[src + 2] as f32 / 255.0;
            }
        }
    }
    out
}

// --- NMS ---

fn iou(a: &PalmDetection, b: &PalmDetection) -> f32 {
    let ax1 = a.cx - a.w / 2.0;
    let ay1 = a.cy - a.h / 2.0;
    let ax2 = a.cx + a.w / 2.0;
    let ay2 = a.cy + a.h / 2.0;
    let bx1 = b.cx - b.w / 2.0;
    let by1 = b.cy - b.h / 2.0;
    let bx2 = b.cx + b.w / 2.0;
    let by2 = b.cy + b.h / 2.0;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

fn nms(mut dets: Vec<PalmDetection>, thresh: f32) -> Vec<PalmDetection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..dets.len() {
            if !suppressed[j] && iou(&dets[i], &dets[j]) > thresh {
                suppressed[j] = true;
            }
        }
        keep.push(dets[i].clone());
    }
    keep
}

// --- Utilities ---

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// 3D distance for rotation/scale-invariant geometry
fn dist3d(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    dist3d_sq(a, b).sqrt()
}

fn dist3d_sq(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)
}

fn vec3_sub(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn vec3_dot(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn vec3_mag_sq(v: (f32, f32, f32)) -> f32 {
    v.0 * v.0 + v.1 * v.1 + v.2 * v.2
}

/// Angle between two 3D vectors in degrees
fn vec3_angle_deg(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let d = vec3_dot(a, b);
    let m = (vec3_mag_sq(a) * vec3_mag_sq(b)).sqrt() + 1e-14;
    (d / m).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Reference hand size: wrist(0) → middle MCP(9) in 3D
fn hand_ref_size(pts: &[(f32, f32, f32); 21]) -> f32 {
    dist3d(pts[0], pts[9]) + 1e-7
}

fn roi_from_landmarks(pts: &[(f32, f32, f32); 21]) -> (f32, f32, f32) {
    // Bbox center (stable) + clamped side (prevents death spiral & overshoot)
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for p in pts {
        min_x = min_x.min(p.0);
        max_x = max_x.max(p.0);
        min_y = min_y.min(p.1);
        max_y = max_y.max(p.1);
    }
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let bbox_side = (max_x - min_x).max(max_y - min_y);
    let side = (bbox_side * ROI_TRACK_EXPAND).clamp(0.2, 0.9);
    (cx, cy, side)
}

fn estimate_handedness(pts: &[(f32, f32, f32); 21]) -> HandLabel {
    // 2D cross product of wrist→index_mcp × wrist→pinky_mcp
    // Sign determines chirality independent of hand rotation
    let v1 = vec3_sub(pts[5], pts[0]);  // wrist → index MCP
    let v2 = vec3_sub(pts[17], pts[0]); // wrist → pinky MCP
    let cross_z = v1.0 * v2.1 - v1.1 * v2.0;
    if cross_z > 0.0 {
        HandLabel::Right
    } else {
        HandLabel::Left
    }
}

// --- Gesture Classification (rotation/scale/depth invariant) ---

/// Finger extension via PIP joint angle (2D, rotation/scale invariant).
/// Straight finger ≈ 180°, bent finger ≈ 30-90°.
fn finger_extended(pts: &[(f32, f32, f32); 21], tip: usize, pip: usize) -> bool {
    let mcp = pip - 1;
    // Angle at PIP: vectors PIP→MCP and PIP→TIP
    let v1 = (pts[mcp].0 - pts[pip].0, pts[mcp].1 - pts[pip].1);
    let v2 = (pts[tip].0 - pts[pip].0, pts[tip].1 - pts[pip].1);
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let mag = ((v1.0 * v1.0 + v1.1 * v1.1) * (v2.0 * v2.0 + v2.1 * v2.1)).sqrt() + 1e-7;
    let angle = (dot / mag).clamp(-1.0, 1.0).acos().to_degrees();
    angle > 130.0
}

fn is_thumb_extended(pts: &[(f32, f32, f32); 21], _label: HandLabel) -> bool {
    let hs = hand_ref_size(pts);

    // Vote 1: Thumb tip protrudes from palm center (3D ratio)
    let palm_center = (
        (pts[0].0 + pts[5].0 + pts[17].0) / 3.0,
        (pts[0].1 + pts[5].1 + pts[17].1) / 3.0,
        (pts[0].2 + pts[5].2 + pts[17].2) / 3.0,
    );
    let is_protruding = dist3d(pts[4], palm_center) / hs > 0.45;

    // Vote 2: Thumb straight (angle CMC→MCP→tip < 50°, 3D vectors)
    let is_straight = vec3_angle_deg(
        vec3_sub(pts[2], pts[1]),
        vec3_sub(pts[4], pts[2]),
    ) < 50.0;

    // Vote 3: Thumb tip farther from pinky MCP than thumb CMC (3D)
    // — measures outward extension independent of hand rotation
    let is_extended = dist3d(pts[4], pts[17]) > dist3d(pts[1], pts[17]);

    let votes = is_protruding as u8 + is_straight as u8 + is_extended as u8;
    votes >= 2
}

fn is_thumbs_up(pts: &[(f32, f32, f32); 21], _label: HandLabel) -> bool {
    let hs = hand_ref_size(pts);

    // 1: Thumb 3D-far from curled finger tips
    let fist_center = (
        (pts[8].0 + pts[12].0 + pts[16].0 + pts[20].0) / 4.0,
        (pts[8].1 + pts[12].1 + pts[16].1 + pts[20].1) / 4.0,
        (pts[8].2 + pts[12].2 + pts[16].2 + pts[20].2) / 4.0,
    );
    if dist3d(pts[4], fist_center) / hs < 0.50 {
        return false;
    }

    // 2: Thumb 3D-far from index tip (not wrapping around fist)
    if dist3d(pts[4], pts[8]) / hs < 0.30 {
        return false;
    }

    // 3: Thumb straight (3D angle < 60°)
    if vec3_angle_deg(vec3_sub(pts[2], pts[1]), vec3_sub(pts[4], pts[2])) >= 60.0 {
        return false;
    }

    // 4: Thumb projects beyond MCP center along palm axis
    // palm_axis: wrist(0) → MCP center (finger direction)
    let mcp_center = (
        (pts[5].0 + pts[9].0 + pts[13].0 + pts[17].0) / 4.0,
        (pts[5].1 + pts[9].1 + pts[13].1 + pts[17].1) / 4.0,
        (pts[5].2 + pts[9].2 + pts[13].2 + pts[17].2) / 4.0,
    );
    let palm_axis = vec3_sub(mcp_center, pts[0]);
    let thumb_proj = vec3_dot(vec3_sub(pts[4], pts[0]), palm_axis);
    let mcp_proj = vec3_dot(vec3_sub(mcp_center, pts[0]), palm_axis);
    if thumb_proj < mcp_proj * 0.8 {
        return false;
    }

    // 5: Thumb direction has upward screen component (negative Y)
    // — "up" vs "down" inherently requires absolute reference
    let thumb_dir = vec3_sub(pts[4], pts[1]);
    let thumb_len = (vec3_mag_sq(thumb_dir)).sqrt() + 1e-7;
    thumb_dir.1 / thumb_len < -0.25
}

fn is_thumbs_down(pts: &[(f32, f32, f32); 21], label: HandLabel) -> bool {
    if !is_thumb_extended(pts, label) {
        return false;
    }
    let hs = hand_ref_size(pts);

    // Thumb far from other finger tips (3D ratio)
    if dist3d(pts[4], pts[8]) / hs < 0.30 {
        return false;
    }

    // Thumb straight (3D)
    if vec3_angle_deg(vec3_sub(pts[2], pts[1]), vec3_sub(pts[4], pts[2])) >= 60.0 {
        return false;
    }

    // Thumb direction has downward screen component (positive Y)
    let thumb_dir = vec3_sub(pts[4], pts[1]);
    let thumb_len = (vec3_mag_sq(thumb_dir)).sqrt() + 1e-7;
    thumb_dir.1 / thumb_len > 0.25
}

fn is_two_fingers(pts: &[(f32, f32, f32); 21]) -> bool {
    finger_extended(pts, 8, 6)
        && finger_extended(pts, 12, 10)
        && !finger_extended(pts, 16, 14)
        && !finger_extended(pts, 20, 18)
}

fn count_fingers(pts: &[(f32, f32, f32); 21], label: HandLabel) -> u8 {
    let mut count = 0u8;
    if is_thumb_extended(pts, label) {
        count += 1;
    }
    for &(tip, pip) in &[(8, 6), (12, 10), (16, 14), (20, 18)] {
        if finger_extended(pts, tip, pip) {
            count += 1;
        }
    }
    count
}

fn classify_gesture(
    pts: &[(f32, f32, f32); 21],
    label: HandLabel,
) -> (Gesture, f32, f32, u8) {
    let fingers = count_fingers(pts, label);
    let hs = hand_ref_size(pts);

    // 3D distances for controller (absolute, used by click/drag logic)
    let pinch_dist = dist3d(pts[4], pts[8]);
    let mid_pinch = dist3d(pts[4], pts[12]);

    // Scale-invariant ratios for gesture classification
    let pinch_ratio = pinch_dist / hs;

    // Step 1: All 4 non-thumb fingers curled?
    let all_curled = [(8, 6), (12, 10), (16, 14), (20, 18)]
        .iter()
        .all(|&(t, p)| !finger_extended(pts, t, p));

    if all_curled {
        if is_thumbs_up(pts, label) {
            return (Gesture::ThumbsUp, pinch_dist, mid_pinch, fingers);
        }
        if is_thumbs_down(pts, label) {
            return (Gesture::ThumbsDown, pinch_dist, mid_pinch, fingers);
        }
        return (Gesture::Fist, pinch_dist, mid_pinch, fingers);
    }

    // Step 2: Fingers open
    if is_two_fingers(pts) {
        return (Gesture::TwoFingers, pinch_dist, mid_pinch, fingers);
    }
    if pinch_ratio < PINCH_RATIO_THRESH {
        return (Gesture::Pinch, pinch_dist, mid_pinch, fingers);
    }
    if fingers >= 4 {
        return (Gesture::OpenHand, pinch_dist, mid_pinch, fingers);
    }
    (Gesture::Pointer, pinch_dist, mid_pinch, fingers)
}
