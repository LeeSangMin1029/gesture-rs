// Gaze tracking: SCRFD face detection → Iris Landmark (MediaPipe)
//
// Key insight: measure iris position RELATIVE to eye contour center,
// not relative to crop center. This makes gaze HEAD-INVARIANT:
//   Head moves → iris AND eye contour shift together → offset unchanged
//   Eyes move  → iris shifts within eye contour     → offset changes
//
// Sign conventions:
//   Yaw:  negative = LEFT,  positive = RIGHT
//   Pitch: positive = UP,   negative = DOWN

use crate::filter::OneEuroFilter;
use crate::gesture::GazeDirection;
use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

const FACE_INPUT_SIZE: usize = 640;
const IRIS_INPUT_SIZE: usize = 64;
const FACE_SCORE_THRESH: f32 = 0.3;
const NMS_THRESH: f32 = 0.4;
const STRIDES: [usize; 3] = [8, 16, 32];
const NUM_ANCHORS: usize = 2;
const CALIB_FRAMES: usize = 15;

const EYE_CROP_RATIO: f32 = 0.35;

// Iris offset is measured relative to eye contour centroid,
// normalized by eye width/height. Typical range: ±0.15 to ±0.25
const IRIS_YAW_GAIN: f32 = 75.0;
const IRIS_PITCH_GAIN: f32 = 70.0;

// Number of eye contour landmarks to use (skip eyebrow points)
const EYE_CONTOUR_POINTS: usize = 16;

// Dead zone: ignore relative offsets smaller than this (noise suppression)
const DEAD_ZONE: f32 = 0.02;

// EMA smoothing for SCRFD keypoints (lower = smoother)
const KP_SMOOTH_ALPHA: f32 = 0.15;

#[derive(Debug, Clone)]
pub struct GazeResult {
    pub direction: GazeDirection,
    pub yaw: f32,
    pub pitch: f32,
    pub face_box: [f32; 4],
}

struct FaceDetection {
    bbox: [f32; 4],
    left_eye: (f32, f32),
    right_eye: (f32, f32),
}

#[derive(Clone)]
struct SmoothedKeypoints {
    left_eye: (f32, f32),
    right_eye: (f32, f32),
}

pub struct GazeTracker {
    face_session: Session,
    iris_session: Session,
    face_output_names: Vec<String>,
    yaw_filter: OneEuroFilter,
    pitch_filter: OneEuroFilter,
    start_time: Instant,
    calib_samples: Vec<(f32, f32)>,
    calib_face_widths: Vec<f32>,
    calib_yaw_offset: f32,
    calib_pitch_offset: f32,
    ref_face_width: f32,
    calibrated: bool,
    prev_kps: Option<SmoothedKeypoints>,
}

impl GazeTracker {
    pub fn new(face_model: &str, iris_model: &str) -> Result<Self> {
        anyhow::ensure!(
            std::path::Path::new(face_model).exists(),
            "Face model not found: {face_model}"
        );
        anyhow::ensure!(
            std::path::Path::new(iris_model).exists(),
            "Iris model not found: {iris_model}"
        );

        let face_session = Session::builder()?
            .commit_from_file(face_model)
            .context("Failed to load face detection model")?;
        let iris_session = Session::builder()?
            .commit_from_file(iris_model)
            .context("Failed to load iris landmark model")?;

        let face_output_names: Vec<String> = face_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        tracing::info!("GazeTracker loaded (SCRFD + Iris Landmark)");
        Ok(Self {
            face_session,
            iris_session,
            face_output_names,
            yaw_filter: OneEuroFilter::new(0.02, 0.003),
            pitch_filter: OneEuroFilter::new(0.02, 0.003),
            start_time: Instant::now(),
            calib_samples: Vec::with_capacity(CALIB_FRAMES),
            calib_face_widths: Vec::with_capacity(CALIB_FRAMES),
            calib_yaw_offset: 0.0,
            calib_pitch_offset: 0.0,
            ref_face_width: 0.0,
            calibrated: false,
            prev_kps: None,
        })
    }

    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<GazeResult>> {
        let face = match self.detect_face(rgb, w, h)? {
            Some(f) => f,
            None => return Ok(None),
        };

        let skp = self.smooth_keypoints(&face);
        let face_w = face.bbox[2] - face.bbox[0]; // normalized face width
        let (raw_yaw, raw_pitch) = self.estimate_gaze(rgb, w, h, &face.bbox, &skp)?;

        if !self.calibrated {
            self.calib_samples.push((raw_yaw, raw_pitch));
            self.calib_face_widths.push(face_w);
            if self.calib_samples.len() >= CALIB_FRAMES {
                let n = self.calib_samples.len() as f32;
                self.calib_yaw_offset =
                    self.calib_samples.iter().map(|s| s.0).sum::<f32>() / n;
                self.calib_pitch_offset =
                    self.calib_samples.iter().map(|s| s.1).sum::<f32>() / n;
                self.ref_face_width =
                    self.calib_face_widths.iter().sum::<f32>() / n;
                self.calibrated = true;
                tracing::info!(
                    "Auto-calibrated: yaw_offset={:.1} pitch_offset={:.1} ref_face_w={:.3}",
                    self.calib_yaw_offset, self.calib_pitch_offset, self.ref_face_width
                );
            }
        }

        // Distance-based gain scaling:
        // far (face small) → scale > 1 → higher sensitivity
        // close (face big) → scale < 1 → lower sensitivity
        let dist_scale = if self.ref_face_width > 0.0 && face_w > 0.0 {
            (self.ref_face_width / face_w).clamp(0.5, 2.0)
        } else {
            1.0
        };

        let centered_yaw = (raw_yaw - self.calib_yaw_offset) * dist_scale;
        let centered_pitch = (raw_pitch - self.calib_pitch_offset) * dist_scale;
        let t = self.start_time.elapsed().as_secs_f32();
        let yaw = self.yaw_filter.filter(t, centered_yaw);
        let pitch = self.pitch_filter.filter(t, centered_pitch);
        tracing::debug!(
            "gaze raw=({:.1},{:.1}) smooth=({:.1},{:.1})",
            raw_yaw, raw_pitch, yaw, pitch
        );

        let direction = if yaw < -15.0 {
            GazeDirection::Left
        } else if yaw > 15.0 {
            GazeDirection::Right
        } else {
            GazeDirection::Center
        };

        Ok(Some(GazeResult {
            direction,
            yaw,
            pitch,
            face_box: face.bbox,
        }))
    }

    fn smooth_keypoints(&mut self, face: &FaceDetection) -> SmoothedKeypoints {
        let raw = SmoothedKeypoints {
            left_eye: face.left_eye,
            right_eye: face.right_eye,
        };

        let smoothed = match &self.prev_kps {
            Some(prev) => {
                let a = KP_SMOOTH_ALPHA;
                let b = 1.0 - a;
                SmoothedKeypoints {
                    left_eye: (
                        a * raw.left_eye.0 + b * prev.left_eye.0,
                        a * raw.left_eye.1 + b * prev.left_eye.1,
                    ),
                    right_eye: (
                        a * raw.right_eye.0 + b * prev.right_eye.0,
                        a * raw.right_eye.1 + b * prev.right_eye.1,
                    ),
                }
            }
            None => raw,
        };

        self.prev_kps = Some(smoothed.clone());
        smoothed
    }

    fn detect_face(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<FaceDetection>> {
        let input = preprocess_scrfd(rgb, w, h);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .face_session
            .run(ort::inputs!["input.1" => tensor])?;

        let mut all_boxes: Vec<[f32; 4]> = Vec::new();
        let mut all_scores: Vec<f32> = Vec::new();
        let mut all_kps: Vec<[(f32, f32); 5]> = Vec::new();

        for (stride_idx, &stride) in STRIDES.iter().enumerate() {
            let (_, score_data) = outputs[self.face_output_names[stride_idx].as_str()]
                .try_extract_tensor::<f32>()?;
            let (_, bbox_data) = outputs[self.face_output_names[stride_idx + 3].as_str()]
                .try_extract_tensor::<f32>()?;
            let (_, kps_data) = outputs[self.face_output_names[stride_idx + 6].as_str()]
                .try_extract_tensor::<f32>()?;

            let grid = FACE_INPUT_SIZE / stride;
            let num_cells = grid * grid * NUM_ANCHORS;

            for i in 0..num_cells {
                let score = score_data[i];
                if score < FACE_SCORE_THRESH {
                    continue;
                }

                let cell_idx = i / NUM_ANCHORS;
                let row = cell_idx / grid;
                let col = cell_idx % grid;
                let ax = (col as f32) * stride as f32;
                let ay = (row as f32) * stride as f32;
                let s = stride as f32;

                let x1 = (ax - bbox_data[i * 4] * s) / FACE_INPUT_SIZE as f32;
                let y1 = (ay - bbox_data[i * 4 + 1] * s) / FACE_INPUT_SIZE as f32;
                let x2 = (ax + bbox_data[i * 4 + 2] * s) / FACE_INPUT_SIZE as f32;
                let y2 = (ay + bbox_data[i * 4 + 3] * s) / FACE_INPUT_SIZE as f32;

                all_boxes.push([x1.max(0.0), y1.max(0.0), x2.min(1.0), y2.min(1.0)]);
                all_scores.push(score);

                let mut kps = [(0.0f32, 0.0f32); 5];
                for k in 0..5 {
                    kps[k] = (
                        (ax + kps_data[i * 10 + k * 2] * s) / FACE_INPUT_SIZE as f32,
                        (ay + kps_data[i * 10 + k * 2 + 1] * s) / FACE_INPUT_SIZE as f32,
                    );
                }
                all_kps.push(kps);
            }
        }

        if all_boxes.is_empty() {
            return Ok(None);
        }

        let keep = nms(&all_boxes, &all_scores, NMS_THRESH);
        if keep.is_empty() {
            return Ok(None);
        }

        let best = keep[0];
        let kps = all_kps[best];

        Ok(Some(FaceDetection {
            bbox: all_boxes[best],
            left_eye: kps[0],
            right_eye: kps[1],
        }))
    }

    fn estimate_gaze(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        bbox: &[f32; 4],
        kp: &SmoothedKeypoints,
    ) -> Result<(f32, f32)> {
        let face_w = (bbox[2] - bbox[0]) * w as f32;
        let crop_px = face_w * EYE_CROP_RATIO;

        // Run iris model: returns iris offset RELATIVE to eye contour center
        let left = self.run_iris(rgb, w, h, kp.left_eye, crop_px, false)?;
        let right = self.run_iris(rgb, w, h, kp.right_eye, crop_px, true)?;

        let avg_rx = (left.0 + right.0) / 2.0;
        let avg_ry = (left.1 + right.1) / 2.0;

        // rel_x positive → iris RIGHT of eye center → looking right → yaw > 0
        let iris_yaw = avg_rx * IRIS_YAW_GAIN;
        // rel_y positive → iris BELOW eye center → looking down → pitch < 0
        let iris_pitch = -avg_ry * IRIS_PITCH_GAIN;

        tracing::debug!(
            "iris_rel L=({:.3},{:.3}) R=({:.3},{:.3}) avg=({:.3},{:.3}) yaw={:.1} pitch={:.1}",
            left.0, left.1, right.0, right.1, avg_rx, avg_ry, iris_yaw, iris_pitch
        );

        Ok((iris_yaw, iris_pitch))
    }

    /// Run iris model on one eye crop.
    /// Returns iris position RELATIVE to eye contour center,
    /// normalized by eye dimensions. Range: roughly [-0.3, 0.3]
    fn run_iris(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        eye_center: (f32, f32),
        crop_px: f32,
        flip_h: bool,
    ) -> Result<(f32, f32)> {
        let input = crop_eye(rgb, w, h, eye_center, crop_px, flip_h);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .iris_session
            .run(ort::inputs!["input_1" => tensor])?;

        let (_, iris_data) = outputs["output_iris"].try_extract_tensor::<f32>()?;
        let (_, contour_data) =
            outputs["output_eyes_contours_and_brows"].try_extract_tensor::<f32>()?;

        // Eye contour: use centroid (average) for stable center,
        // bounding box for normalization size
        let n = EYE_CONTOUR_POINTS.min(contour_data.len() / 3);
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for i in 0..n {
            let x = contour_data[i * 3];
            let y = contour_data[i * 3 + 1];
            sum_x += x;
            sum_y += y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        // Centroid is more stable than bbox center (robust to single outlier)
        let eye_cx = sum_x / n as f32;
        let eye_cy = sum_y / n as f32;
        let eye_w = (max_x - min_x).max(1.0);
        let eye_h = (max_y - min_y).max(1.0);

        // Iris center (first point of output_iris) in 64×64 space
        let iris_x = iris_data[0];
        let iris_y = iris_data[1];

        // Relative offset: how far iris is from eye centroid, normalized by eye size
        let mut rel_x = (iris_x - eye_cx) / eye_w;
        let mut rel_y = (iris_y - eye_cy) / eye_h;

        // Undo horizontal flip for right eye
        if flip_h {
            rel_x = -rel_x;
        }

        // Dead zone: suppress noise when iris is near center
        if rel_x.abs() < DEAD_ZONE {
            rel_x = 0.0;
        }
        if rel_y.abs() < DEAD_ZONE {
            rel_y = 0.0;
        }

        Ok((rel_x, rel_y))
    }
}

// --- Preprocessing ---

fn preprocess_scrfd(rgb: &[u8], w: u32, h: u32) -> Array4<f32> {
    let target = FACE_INPUT_SIZE;
    let mut out = Array4::<f32>::zeros((1, 3, target, target));
    let buf = out.as_slice_mut().unwrap();
    let sx = w as f32 / target as f32;
    let sy = h as f32 / target as f32;
    let plane = target * target;

    for ty in 0..target {
        for tx in 0..target {
            let src_x = ((tx as f32 + 0.5) * sx) as u32;
            let src_y = ((ty as f32 + 0.5) * sy) as u32;
            let src_x = src_x.min(w - 1);
            let src_y = src_y.min(h - 1);
            let idx = ((src_y * w + src_x) * 3) as usize;
            let pos = ty * target + tx;
            buf[pos] = (rgb[idx] as f32 - 127.5) / 128.0;
            buf[plane + pos] = (rgb[idx + 1] as f32 - 127.5) / 128.0;
            buf[2 * plane + pos] = (rgb[idx + 2] as f32 - 127.5) / 128.0;
        }
    }
    out
}

fn crop_eye(
    rgb: &[u8],
    w: u32,
    h: u32,
    eye_center: (f32, f32),
    crop_px: f32,
    flip_h: bool,
) -> Array4<f32> {
    let target = IRIS_INPUT_SIZE;
    let mut out = Array4::<f32>::zeros((1, 3, target, target));
    let buf = out.as_slice_mut().unwrap();
    let plane = target * target;

    let cx = eye_center.0 * w as f32;
    let cy = eye_center.1 * h as f32;
    let half = crop_px / 2.0;
    let crop_x = cx - half;
    let crop_y = cy - half;

    let wf = w as f32;
    let hf = h as f32;

    for ty in 0..target {
        for tx in 0..target {
            let ratio_x = (tx as f32 + 0.5) / target as f32;
            let ratio_y = (ty as f32 + 0.5) / target as f32;
            let ratio_x = if flip_h { 1.0 - ratio_x } else { ratio_x };

            let src_xf = crop_x + ratio_x * crop_px;
            let src_yf = crop_y + ratio_y * crop_px;

            let sx = src_xf.clamp(0.0, wf - 1.001);
            let sy = src_yf.clamp(0.0, hf - 1.001);

            let x0 = sx as u32;
            let y0 = sy as u32;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let pos = ty * target + tx;
            for c in 0..3usize {
                let v00 = rgb[((y0 * w + x0) * 3 + c as u32) as usize] as f32;
                let v10 = rgb[((y0 * w + x1) * 3 + c as u32) as usize] as f32;
                let v01 = rgb[((y1 * w + x0) * 3 + c as u32) as usize] as f32;
                let v11 = rgb[((y1 * w + x1) * 3 + c as u32) as usize] as f32;
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                buf[c * plane + pos] = v / 127.5 - 1.0;
            }
        }
    }
    out
}

// --- NMS ---

fn nms(boxes: &[[f32; 4]], scores: &[f32], thresh: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; indices.len()];

    for i in 0..indices.len() {
        if suppressed[i] {
            continue;
        }
        let idx_i = indices[i];
        keep.push(idx_i);

        for j in (i + 1)..indices.len() {
            if suppressed[j] {
                continue;
            }
            let idx_j = indices[j];
            if iou(&boxes[idx_i], &boxes[idx_j]) > thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter + 1e-6)
}
