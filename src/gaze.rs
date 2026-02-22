// Gaze tracking: Face Detection (YuNet) → Gaze Estimation (MobileGaze)
//
// YuNet:     640x640 CHW → multi-scale face bboxes + 5 keypoints
// MobileGaze: 448x448 CHW → yaw[90] + pitch[90] (bin classification)

use crate::gesture::GazeDirection;
use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

const FACE_SIZE: usize = 640;
const GAZE_SIZE: usize = 448;
const FACE_SCORE_THRESH: f32 = 0.6;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GazeResult {
    pub direction: GazeDirection,
    pub yaw: f32,   // degrees: negative=left, positive=right
    pub pitch: f32,  // degrees: negative=up, positive=down
    pub face_box: [f32; 4], // x1, y1, x2, y2 normalized
}

pub struct GazeTracker {
    face_session: Session,
    gaze_session: Session,
}

impl GazeTracker {
    pub fn new(face_model: &str, gaze_model: &str) -> Result<Self> {
        let face_session = Session::builder()?
            .commit_from_file(face_model)
            .context("Failed to load face detection model")?;
        let gaze_session = Session::builder()?
            .commit_from_file(gaze_model)
            .context("Failed to load gaze estimation model")?;
        tracing::info!("GazeTracker loaded");
        Ok(Self {
            face_session,
            gaze_session,
        })
    }

    /// Detect gaze direction from RGB frame (HWC, u8)
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<GazeResult>> {
        // Step 1: Face detection
        let face = self.detect_face(rgb, w, h)?;
        let face = match face {
            Some(f) => f,
            None => return Ok(None),
        };

        // Step 2: Crop face region, run gaze estimation
        let (yaw, pitch) = self.estimate_gaze(rgb, w, h, &face)?;

        // Step 3: Map yaw to direction
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
            face_box: face,
        }))
    }

    fn detect_face(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Option<[f32; 4]>> {
        // YuNet input: [1, 3, 640, 640] CHW, normalized [0..1]
        let input = preprocess_chw(rgb, w, h, FACE_SIZE);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .face_session
            .run(ort::inputs!["input" => tensor])?;

        // Multi-scale outputs: cls, obj, bbox, kps at strides 8, 16, 32
        // Parse stride-8 (highest resolution) first for best face
        // Each returns (&Shape, &[f32])
        let (_, cls_data) = outputs["cls_8"].try_extract_tensor::<f32>()?;
        let (_, obj_data) = outputs["obj_8"].try_extract_tensor::<f32>()?;
        let (_, bbox_data) = outputs["bbox_8"].try_extract_tensor::<f32>()?;

        let grid = FACE_SIZE / 8; // 80

        let mut best_score = 0.0f32;
        let mut best_box = [0.0f32; 4];

        // cls/obj shape: [1, grid*grid, 1] → flat index = i
        // bbox shape: [1, grid*grid, 4] → flat index = i * 4 + j
        for i in 0..grid * grid {
            let score = sigmoid(cls_data[i]) * sigmoid(obj_data[i]);
            if score > best_score && score > FACE_SCORE_THRESH {
                best_score = score;
                let row = i / grid;
                let col = i % grid;
                let stride = 8.0;
                // Decode bbox (center-based)
                let cx = (col as f32 + 0.5 + bbox_data[i * 4]) * stride / FACE_SIZE as f32;
                let cy = (row as f32 + 0.5 + bbox_data[i * 4 + 1]) * stride / FACE_SIZE as f32;
                let bw = (bbox_data[i * 4 + 2].exp()) * stride / FACE_SIZE as f32;
                let bh = (bbox_data[i * 4 + 3].exp()) * stride / FACE_SIZE as f32;
                best_box = [
                    cx - bw / 2.0,
                    cy - bh / 2.0,
                    cx + bw / 2.0,
                    cy + bh / 2.0,
                ];
            }
        }

        // Also check stride-16 and stride-32 for larger faces
        check_scale(&outputs, "cls_16", "obj_16", "bbox_16", 16,
                    &mut best_score, &mut best_box)?;
        check_scale(&outputs, "cls_32", "obj_32", "bbox_32", 32,
                    &mut best_score, &mut best_box)?;

        if best_score > FACE_SCORE_THRESH {
            Ok(Some(best_box))
        } else {
            Ok(None)
        }
    }

    fn estimate_gaze(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        face_box: &[f32; 4],
    ) -> Result<(f32, f32)> {
        // Crop face and resize to 448x448 CHW
        let input = crop_and_resize_chw(rgb, w, h, face_box, GAZE_SIZE);
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .gaze_session
            .run(ort::inputs!["input" => tensor])?;

        // Output: yaw[1,90], pitch[1,90] — 90-bin classification
        let (_, yaw_data) = outputs["yaw"].try_extract_tensor::<f32>()?;
        let (_, pitch_data) = outputs["pitch"].try_extract_tensor::<f32>()?;

        // Softmax + expectation to get continuous angle
        let yaw = bins_to_angle(&yaw_data[..90]);
        let pitch = bins_to_angle(&pitch_data[..90]);

        Ok((yaw, pitch))
    }
}

fn check_scale(
    outputs: &ort::session::SessionOutputs,
    cls_name: &str,
    obj_name: &str,
    bbox_name: &str,
    stride: usize,
    best_score: &mut f32,
    best_box: &mut [f32; 4],
) -> Result<()> {
    let (_, cls_data) = outputs[cls_name].try_extract_tensor::<f32>()?;
    let (_, obj_data) = outputs[obj_name].try_extract_tensor::<f32>()?;
    let (_, bbox_data) = outputs[bbox_name].try_extract_tensor::<f32>()?;
    let grid = FACE_SIZE / stride;

    for i in 0..grid * grid {
        let score = sigmoid(cls_data[i]) * sigmoid(obj_data[i]);
        if score > *best_score && score > FACE_SCORE_THRESH {
            *best_score = score;
            let row = i / grid;
            let col = i % grid;
            let s = stride as f32;
            let cx = (col as f32 + 0.5 + bbox_data[i * 4]) * s / FACE_SIZE as f32;
            let cy = (row as f32 + 0.5 + bbox_data[i * 4 + 1]) * s / FACE_SIZE as f32;
            let bw = bbox_data[i * 4 + 2].exp() * s / FACE_SIZE as f32;
            let bh = bbox_data[i * 4 + 3].exp() * s / FACE_SIZE as f32;
            *best_box = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0];
        }
    }
    Ok(())
}

// --- Preprocessing ---

/// Resize RGB HWC to target CHW, normalize [0..1] (1D slice access)
fn preprocess_chw(rgb: &[u8], w: u32, h: u32, target: usize) -> Array4<f32> {
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
            buf[pos] = rgb[idx] as f32 / 255.0;
            buf[plane + pos] = rgb[idx + 1] as f32 / 255.0;
            buf[2 * plane + pos] = rgb[idx + 2] as f32 / 255.0;
        }
    }
    out
}

/// Crop face box from RGB HWC and resize to CHW target (1D slice access)
fn crop_and_resize_chw(
    rgb: &[u8],
    w: u32,
    h: u32,
    face_box: &[f32; 4],
    target: usize,
) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, 3, target, target));
    let buf = out.as_slice_mut().unwrap();
    let fx1 = face_box[0] * w as f32;
    let fy1 = face_box[1] * h as f32;
    let fw = (face_box[2] - face_box[0]) * w as f32;
    let fh = (face_box[3] - face_box[1]) * h as f32;
    let plane = target * target;

    for ty in 0..target {
        for tx in 0..target {
            let sx = fx1 + (tx as f32 + 0.5) / target as f32 * fw;
            let sy = fy1 + (ty as f32 + 0.5) / target as f32 * fh;
            let sx = sx as i32;
            let sy = sy as i32;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let idx = ((sy as u32 * w + sx as u32) * 3) as usize;
                let pos = ty * target + tx;
                buf[pos] = rgb[idx] as f32 / 255.0;
                buf[plane + pos] = rgb[idx + 1] as f32 / 255.0;
                buf[2 * plane + pos] = rgb[idx + 2] as f32 / 255.0;
            }
        }
    }
    out
}

// --- Utilities ---

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert 90-bin classification output to continuous angle via softmax + expectation
fn bins_to_angle(logits: &[f32]) -> f32 {
    // Bins represent angles from -180 to 180 (or -90 to 90 depending on model)
    // MobileGaze uses -99..99 degrees mapped to 90 bins
    let bin_width = 198.0 / 90.0; // ~2.2 degrees per bin
    let offset = -99.0;

    // Softmax
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();

    // Expectation
    let mut angle = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        angle += p * (offset + (i as f32 + 0.5) * bin_width);
    }
    angle
}
