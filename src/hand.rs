// Hand detection pipeline: Palm Detection → Hand Landmark → Gesture Classification
//
// Palm Detection: 192x192 HWC → [2016, 18] bboxes + [2016, 1] scores
// Hand Landmark:  224x224 HWC → [63] keypoints + [1] confidence + [1] handedness

use crate::gesture::Gesture;
use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

use crate::filter::OneEuroFilter;

const PALM_SIZE: usize = 192;
const LANDMARK_SIZE: usize = 224;
const SCORE_THRESH: f32 = 0.2;
const NMS_THRESH: f32 = 0.3;
const PINCH_RATIO_THRESH: f32 = 0.25; // pinch_dist / hand_size ratio

// ROI parameters (tighter than MediaPipe defaults to avoid forearm inclusion)
const PALM_ROI_SCALE: f32 = 2.0;
const PALM_ROI_SHIFT_Y: f32 = -0.25;
const TRACK_ROI_SCALE: f32 = 1.5;
const TRACK_ROI_SHIFT_Y: f32 = -0.05;

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
    keypoints: [(f32, f32); 7],
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

// --- Rotated ROI (pixel space) ---

#[derive(Debug, Clone, Copy)]
struct RotatedRoi {
    cx_px: f32,    // center x in pixels
    cy_px: f32,    // center y in pixels
    size_px: f32,  // square side length in pixels
    rotation: f32, // radians
}

fn normalize_radians(angle: f32) -> f32 {
    let mut a = angle % (2.0 * std::f32::consts::PI);
    if a > std::f32::consts::PI {
        a -= 2.0 * std::f32::consts::PI;
    } else if a < -std::f32::consts::PI {
        a += 2.0 * std::f32::consts::PI;
    }
    a
}

/// Build ROI from palm detection using keypoints[0] (wrist) and keypoints[2] (middle finger MCP)
fn roi_from_palm(palm: &PalmDetection, w: u32, h: u32) -> RotatedRoi {
    let wf = w as f32;
    let hf = h as f32;

    // Convert keypoints to pixel space for rotation
    let wrist_px = (palm.keypoints[0].0 * wf, palm.keypoints[0].1 * hf);
    let mid_px = (palm.keypoints[2].0 * wf, palm.keypoints[2].1 * hf);

    // Rotation: negate Y for screen→math coordinate conversion
    let angle = normalize_radians(
        std::f32::consts::FRAC_PI_2 - (wrist_px.1 - mid_px.1).atan2(mid_px.0 - wrist_px.0),
    );

    // Compute palm bbox in pixels, then scale
    let bbox_w_px = palm.w * wf;
    let bbox_h_px = palm.h * hf;
    let size_px = bbox_w_px.max(bbox_h_px) * PALM_ROI_SCALE;

    // Center in pixels with shift along rotation direction
    let cx_px = palm.cx * wf;
    let cy_px = palm.cy * hf;
    let shift = size_px * PALM_ROI_SHIFT_Y;
    let cx_px = cx_px - shift * angle.sin();
    let cy_px = cy_px + shift * angle.cos();

    RotatedRoi { cx_px, cy_px, size_px, rotation: angle }
}

/// Build ROI from tracked landmarks using wrist(0) → mean(PIP 6,10,14) for rotation
fn roi_from_landmarks_rotated(pts: &[(f32, f32, f32); 21], w: u32, h: u32) -> RotatedRoi {
    let wf = w as f32;
    let hf = h as f32;

    // Pixel-space rotation reference
    let wrist_px = (pts[0].0 * wf, pts[0].1 * hf);
    let target_x_px = (pts[6].0 + pts[10].0 + pts[14].0) / 3.0 * wf;
    let target_y_px = (pts[6].1 + pts[10].1 + pts[14].1) / 3.0 * hf;
    let angle = normalize_radians(
        std::f32::consts::FRAC_PI_2 - (wrist_px.1 - target_y_px).atan2(target_x_px - wrist_px.0),
    );

    // Bounding box in pixel space
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for p in pts {
        let px = p.0 * wf;
        let py = p.1 * hf;
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }
    let bbox_size_px = (max_x - min_x).max(max_y - min_y);
    let max_dim = wf.max(hf);
    let size_px = (bbox_size_px * TRACK_ROI_SCALE).clamp(0.2 * max_dim, 0.9 * max_dim);

    let cx_px = (min_x + max_x) / 2.0;
    let cy_px = (min_y + max_y) / 2.0;

    // Shift along rotation (MediaPipe convention)
    let shift = size_px * TRACK_ROI_SHIFT_Y;
    let cx_px = cx_px - shift * angle.sin();
    let cy_px = cy_px + shift * angle.cos();

    RotatedRoi { cx_px, cy_px, size_px, rotation: angle }
}

// --- Per-hand tracking state ---

struct TrackedHand {
    roi: RotatedRoi,
    filters: Vec<(OneEuroFilter, OneEuroFilter)>,
}

impl TrackedHand {
    fn new(roi: RotatedRoi) -> Self {
        Self {
            roi,
            filters: (0..21)
                .map(|_| (OneEuroFilter::new(3.0, 0.15), OneEuroFilter::new(3.0, 0.15)))
                .collect(),
        }
    }

    fn smooth(&mut self, points: &mut [(f32, f32, f32); 21], t: f32) {
        for i in 0..21 {
            points[i].0 = self.filters[i].0.filter(t, points[i].0);
            points[i].1 = self.filters[i].1.filter(t, points[i].1);
        }
    }
}

fn roi_distance(a: &RotatedRoi, b: &RotatedRoi) -> f32 {
    ((a.cx_px - b.cx_px).powi(2) + (a.cy_px - b.cy_px).powi(2)).sqrt()
}

// --- Hand Detector ---

pub struct HandDetector {
    palm_session: Session,
    landmark_session: Session,
    anchors: Vec<Anchor>,
    tracked: Vec<TrackedHand>,
    frame_count: u64,
    start_time: Instant,
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
            tracked: Vec::new(),
            frame_count: 0,
            start_time: Instant::now(),
        })
    }

    pub fn is_tracking(&self) -> bool {
        !self.tracked.is_empty()
    }

    /// Detect up to 2 hands in an RGB frame (HWC, u8, 3 channels)
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<HandResult>> {
        self.frame_count += 1;
        let wf = w as f32;
        let hf = h as f32;
        let t = self.start_time.elapsed().as_secs_f32();
        let mut results = Vec::new();

        // Take tracked hands out to avoid borrow conflicts with session methods
        let mut prev = std::mem::take(&mut self.tracked);
        let mut new_tracked: Vec<TrackedHand> = Vec::new();

        // Step 1: Update existing tracked hands
        for mut track in prev.drain(..) {
            if track.roi.cx_px < 0.0 || track.roi.cx_px > wf
                || track.roi.cy_px < 0.0 || track.roi.cy_px > hf
            {
                continue;
            }
            if let Some(mut hand) = self.detect_landmarks_roi(rgb, w, h, &track.roi)? {
                track.smooth(&mut hand.points, t);
                if !landmarks_in_frame(&hand.points) {
                    continue;
                }
                let (gesture, pinch, mid_pinch, fingers) =
                    classify_gesture(&hand.points, hand.label);
                hand.gesture = gesture;
                hand.pinch_dist = pinch;
                hand.middle_pinch_dist = mid_pinch;
                hand.finger_count = fingers;

                let new_roi = roi_from_landmarks_rotated(&hand.points, w, h);
                if new_roi.cx_px >= 0.0 && new_roi.cx_px <= wf
                    && new_roi.cy_px >= 0.0 && new_roi.cy_px <= hf
                {
                    track.roi = new_roi;
                    new_tracked.push(track);
                    results.push(hand);
                }
            }
        }

        // Step 2: Find new hands via palm detection (up to 2 total)
        if new_tracked.len() < 2 {
            let palms = self.detect_palms(rgb, w, h)?;
            for palm in &palms {
                if new_tracked.len() >= 2 {
                    break;
                }
                let roi = roi_from_palm(palm, w, h);
                if new_tracked
                    .iter()
                    .any(|tr| roi_distance(&roi, &tr.roi) < tr.roi.size_px * 0.5)
                {
                    continue;
                }
                if let Some(mut hand) = self.detect_landmarks_roi(rgb, w, h, &roi)? {
                    let lm_roi = roi_from_landmarks_rotated(&hand.points, w, h);
                    let mut track = TrackedHand::new(lm_roi);
                    track.smooth(&mut hand.points, t);
                    new_tracked.push(track);
                    results.push(hand);
                }
            }
        }

        // Step 3: Scan fallback — catches back-of-hand and far-distance cases
        if new_tracked.len() < 2 {
            let scan_rois = [
                RotatedRoi { cx_px: wf * 0.5, cy_px: hf * 0.5, size_px: hf * 0.7, rotation: 0.0 },
                RotatedRoi { cx_px: wf * 0.3, cy_px: hf * 0.5, size_px: hf * 0.5, rotation: 0.0 },
                RotatedRoi { cx_px: wf * 0.7, cy_px: hf * 0.5, size_px: hf * 0.5, rotation: 0.0 },
                RotatedRoi { cx_px: wf * 0.5, cy_px: hf * 0.5, size_px: hf * 0.35, rotation: 0.0 },
            ];
            let idx = self.frame_count as usize % scan_rois.len();
            let scan_roi = &scan_rois[idx];
            let far_enough = new_tracked
                .iter()
                .all(|tr| roi_distance(scan_roi, &tr.roi) >= tr.roi.size_px * 0.5);
            if far_enough {
                if let Some(mut hand) = self.detect_landmarks_roi(rgb, w, h, scan_roi)? {
                    let lm_roi = roi_from_landmarks_rotated(&hand.points, w, h);
                    let mut track = TrackedHand::new(lm_roi);
                    track.smooth(&mut hand.points, t);
                    new_tracked.push(track);
                    results.push(hand);
                }
            }
        }

        self.tracked = new_tracked;
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

        // Debug: log max score every 30 frames
        if self.frame_count % 30 == 0 {
            let max_score = (0..self.anchors.len())
                .map(|i| sigmoid(score_data[i]))
                .fold(0.0f32, f32::max);
            tracing::debug!("Palm max_score={:.4}", max_score);
        }

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
                keypoints,
            });
        }

        Ok(nms(detections, NMS_THRESH))
    }

    fn detect_landmarks_roi(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        roi: &RotatedRoi,
    ) -> Result<Option<HandResult>> {
        let (input, inv_affine) = affine_crop(rgb, w, h, roi, LANDMARK_SIZE);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .landmark_session
            .run(ort::inputs!["input_1" => tensor])?;

        // Identity = [1, 63] keypoints, Identity_1 = [1, 1] confidence
        let (_, kp_data) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, conf_data) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        let confidence = sigmoid(conf_data[0]);

        if confidence < 0.65 {
            return Ok(None);
        }

        // Map landmarks back to original frame via inverse affine
        let points = transform_landmarks(&kp_data, &inv_affine, w, h, LANDMARK_SIZE);

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

/// Resize RGB image to target×target (stretch), normalize to [-1..1]
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
            buf[dst] = rgb[src] as f32 / 127.5 - 1.0;
            buf[dst + 1] = rgb[src + 1] as f32 / 127.5 - 1.0;
            buf[dst + 2] = rgb[src + 2] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// Rotation-aware affine crop: extract rotated square ROI into target×target
/// Returns (tensor, inverse_affine_matrix) for landmark coordinate back-projection
fn affine_crop(
    rgb: &[u8],
    w: u32,
    h: u32,
    roi: &RotatedRoi,
    target: usize,
) -> (Array4<f32>, [f32; 6]) {
    let mut out = Array4::<f32>::zeros((1, target, target, 3));
    let buf = out.as_slice_mut().unwrap();

    // Compute square ROI corners directly in pixel space
    let half = roi.size_px / 2.0;
    let (sin_r, cos_r) = roi.rotation.sin_cos();
    let offsets = [(-half, -half), (half, -half), (half, half), (-half, half)];
    let mut corners = [(0.0f32, 0.0f32); 4];
    for (i, &(dx, dy)) in offsets.iter().enumerate() {
        corners[i] = (
            roi.cx_px + dx * cos_r - dy * sin_r,
            roi.cy_px + dx * sin_r + dy * cos_r,
        );
    }

    let c0 = corners[0]; // top-left
    let c1 = corners[1]; // top-right
    let c3 = corners[3]; // bottom-left

    // Affine basis vectors: how to step through the source image
    let t = target as f32;
    let ax = (c1.0 - c0.0) / t;
    let ay = (c1.1 - c0.1) / t;
    let bx = (c3.0 - c0.0) / t;
    let by = (c3.1 - c0.1) / t;

    for ty in 0..target {
        for tx in 0..target {
            let px = (tx as f32 + 0.5) * ax + (ty as f32 + 0.5) * bx + c0.0;
            let py = (tx as f32 + 0.5) * ay + (ty as f32 + 0.5) * by + c0.1;

            let sx = px as i32;
            let sy = py as i32;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let src = ((sy as u32 * w + sx as u32) * 3) as usize;
                let dst = (ty * target + tx) * 3;
                buf[dst] = rgb[src] as f32 / 255.0;
                buf[dst + 1] = rgb[src + 1] as f32 / 255.0;
                buf[dst + 2] = rgb[src + 2] as f32 / 255.0;
            }
        }
    }

    // Forward affine: maps model coords [0..target] → source image pixels
    let inv = [ax, bx, c0.0, ay, by, c0.1];

    (out, inv)
}

/// Map landmark coordinates from model output space back to normalized frame coords
fn transform_landmarks(
    kp_data: &[f32],
    inv_affine: &[f32; 6],
    w: u32,
    h: u32,
    target: usize,
) -> [(f32, f32, f32); 21] {
    let mut points = [(0.0f32, 0.0f32, 0.0f32); 21];
    let t = target as f32;
    for i in 0..21 {
        let lx = kp_data[i * 3];
        let ly = kp_data[i * 3 + 1];
        let lz = kp_data[i * 3 + 2] / t;

        // Apply inverse affine to get source image pixel coords
        let src_x = inv_affine[0] * lx + inv_affine[1] * ly + inv_affine[2];
        let src_y = inv_affine[3] * lx + inv_affine[4] * ly + inv_affine[5];

        // Normalize to [0..1]
        points[i] = (src_x / w as f32, src_y / h as f32, lz);
    }
    points
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

/// Reject detection if fewer than 16 of 21 landmarks fall within [0..1]
fn landmarks_in_frame(pts: &[(f32, f32, f32); 21]) -> bool {
    let count = pts
        .iter()
        .filter(|p| p.0 >= 0.0 && p.0 <= 1.0 && p.1 >= 0.0 && p.1 <= 1.0)
        .count();
    count >= 16
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

    // Step 1: Mostly curled? (3+ of 4 non-thumb fingers curled = fist family)
    let curled_count = [(8, 6), (12, 10), (16, 14), (20, 18)]
        .iter()
        .filter(|&&(t, p)| !finger_extended(pts, t, p))
        .count();

    if curled_count >= 3 {
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
