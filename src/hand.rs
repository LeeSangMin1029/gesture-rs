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
const PINCH_THRESHOLD: f32 = 0.08; // normalized distance

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
        })
    }

    /// Detect hands in an RGB frame (HWC, u8, 3 channels)
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<HandResult>> {
        let palms = self.detect_palms(rgb, w, h)?;
        let mut results = Vec::new();
        for palm in &palms {
            if let Some(hand) = self.detect_landmarks(rgb, w, h, palm)? {
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
            let score = sigmoid(score_data[i]); // shape [1, 2016, 1] → flat index i
            if score < SCORE_THRESH {
                continue;
            }
            let a = &self.anchors[i];
            // shape [1, 2016, 18] → flat index i * 18 + j
            let cx = bbox_data[i * 18] / scale + a.cx;
            let cy = bbox_data[i * 18 + 1] / scale + a.cy;
            let bw = bbox_data[i * 18 + 2] / scale;
            let bh = bbox_data[i * 18 + 3] / scale;

            let mut keypoints = [(0.0f32, 0.0f32); 7];
            for k in 0..7 {
                keypoints[k] = (
                    bbox_data[i * 18 + 4 + k * 2] / scale + a.cx,
                    bbox_data[i * 18 + 5 + k * 2] / scale + a.cy,
                );
            }
            detections.push(PalmDetection {
                cx,
                cy,
                w: bw,
                h: bh,
                score,
                _keypoints: keypoints,
            });
        }

        Ok(nms(detections, NMS_THRESH))
    }

    fn detect_landmarks(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        palm: &PalmDetection,
    ) -> Result<Option<HandResult>> {
        // Compute square ROI from palm, expanded by ROI_EXPAND
        let side = palm.w.max(palm.h) * ROI_EXPAND;
        let roi_cx = palm.cx;
        let roi_cy = palm.cy;

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
            // Landmark output is normalized to [0..224] within the crop
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

/// Resize RGB image to target_size × target_size, normalize to [0..1], HWC layout
fn preprocess_image(rgb: &[u8], w: u32, h: u32, target: usize) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, target, target, 3));
    let scale_x = w as f32 / target as f32;
    let scale_y = h as f32 / target as f32;

    for ty in 0..target {
        for tx in 0..target {
            let sx = ((tx as f32 + 0.5) * scale_x) as u32;
            let sy = ((ty as f32 + 0.5) * scale_y) as u32;
            let sx = sx.min(w - 1);
            let sy = sy.min(h - 1);
            let idx = ((sy * w + sx) * 3) as usize;
            out[[0, ty, tx, 0]] = rgb[idx] as f32 / 255.0;
            out[[0, ty, tx, 1]] = rgb[idx + 1] as f32 / 255.0;
            out[[0, ty, tx, 2]] = rgb[idx + 2] as f32 / 255.0;
        }
    }
    out
}

/// Crop a square ROI (center + side in normalized coords) and resize to target_size
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
                let idx = ((sy as u32 * w + sx as u32) * 3) as usize;
                out[[0, ty, tx, 0]] = rgb[idx] as f32 / 255.0;
                out[[0, ty, tx, 1]] = rgb[idx + 1] as f32 / 255.0;
                out[[0, ty, tx, 2]] = rgb[idx + 2] as f32 / 255.0;
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

fn dist2d(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn estimate_handedness(pts: &[(f32, f32, f32); 21]) -> HandLabel {
    // In mirrored view: compare index MCP (5) and pinky MCP (17) x-positions
    // If index MCP is left of pinky → right hand (palm facing camera)
    if pts[5].0 < pts[17].0 {
        HandLabel::Right
    } else {
        HandLabel::Left
    }
}

// --- Gesture Classification (ported from Python hand_tracker.py) ---

fn finger_extended(pts: &[(f32, f32, f32); 21], tip: usize, pip: usize) -> bool {
    pts[tip].1 < pts[pip].1
}

fn is_thumb_extended(pts: &[(f32, f32, f32); 21], label: HandLabel) -> bool {
    let tip = pts[4];
    let ip = pts[3];
    let mcp = pts[2];
    let cmc = pts[1];
    let wrist = pts[0];

    // Vote 1: X-direction
    let x_ext = match label {
        HandLabel::Right => tip.0 > ip.0,
        HandLabel::Left => tip.0 < ip.0,
    };

    // Vote 2: Thumb straightness (angle at IP joint)
    let v1x = mcp.0 - cmc.0;
    let v1y = mcp.1 - cmc.1;
    let v2x = tip.0 - mcp.0;
    let v2y = tip.1 - mcp.1;
    let dot = v1x * v2x + v1y * v2y;
    let mag1 = (v1x * v1x + v1y * v1y).sqrt() + 1e-7;
    let mag2 = (v2x * v2x + v2y * v2y).sqrt() + 1e-7;
    let cos_a = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    let angle = cos_a.acos().to_degrees();
    let is_straight = angle < 50.0;

    // Vote 3: Protrusion from palm center
    let palm_cx = (wrist.0 + pts[5].0 + pts[17].0) / 3.0;
    let palm_cy = (wrist.1 + pts[5].1 + pts[17].1) / 3.0;
    let hand_size = dist2d((wrist.0, wrist.1), (pts[9].0, pts[9].1)) + 1e-7;
    let tip_dist = dist2d((tip.0, tip.1), (palm_cx, palm_cy));
    let is_protruding = tip_dist / hand_size > 0.45;

    let votes = x_ext as u8 + is_straight as u8 + is_protruding as u8;
    votes >= 2
}

fn is_thumbs_up(pts: &[(f32, f32, f32); 21], _label: HandLabel) -> bool {
    // All 4 non-thumb fingers must be folded (checked by caller)
    let tip = pts[4];
    let mcp = pts[2];
    let cmc = pts[1];
    let wrist = pts[0];
    let index_tip = pts[8];

    let hand_size = dist2d((wrist.0, wrist.1), (pts[9].0, pts[9].1)) + 1e-7;

    // Criterion 1: Thumb far from fist cluster
    let fist_cx = (pts[8].0 + pts[12].0 + pts[16].0 + pts[20].0) / 4.0;
    let fist_cy = (pts[8].1 + pts[12].1 + pts[16].1 + pts[20].1) / 4.0;
    let thumb_to_fist = dist2d((tip.0, tip.1), (fist_cx, fist_cy));
    if thumb_to_fist / hand_size < 0.50 {
        return false;
    }

    // Criterion 2: Thumb far from index (not wrapping)
    let thumb_to_index = dist2d((tip.0, tip.1), (index_tip.0, index_tip.1));
    if thumb_to_index / hand_size < 0.30 {
        return false;
    }

    // Criterion 3: Thumb straight (angle < 60°)
    let v1x = mcp.0 - cmc.0;
    let v1y = mcp.1 - cmc.1;
    let v2x = tip.0 - mcp.0;
    let v2y = tip.1 - mcp.1;
    let dot = v1x * v2x + v1y * v2y;
    let mag1 = (v1x * v1x + v1y * v1y).sqrt() + 1e-7;
    let mag2 = (v2x * v2x + v2y * v2y).sqrt() + 1e-7;
    let cos_a = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    let angle = cos_a.acos().to_degrees();
    angle < 60.0
}

fn is_thumbs_down(pts: &[(f32, f32, f32); 21], label: HandLabel) -> bool {
    if !is_thumb_extended(pts, label) {
        return false;
    }
    // All other fingers must be folded (checked by caller)
    let tip = pts[4];
    let cmc = pts[1];
    let dx = tip.0 - cmc.0;
    let dy = tip.1 - cmc.1; // positive = downward
    let angle = dy.atan2(dx.abs()).to_degrees();
    angle > 30.0
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
    let pinch_dist = dist2d((pts[4].0, pts[4].1), (pts[8].0, pts[8].1));
    let mid_pinch = dist2d((pts[4].0, pts[4].1), (pts[12].0, pts[12].1));

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
    if pinch_dist < PINCH_THRESHOLD {
        return (Gesture::Pinch, pinch_dist, mid_pinch, fingers);
    }
    if fingers >= 4 {
        return (Gesture::OpenHand, pinch_dist, mid_pinch, fingers);
    }
    (Gesture::Pointer, pinch_dist, mid_pinch, fingers)
}
