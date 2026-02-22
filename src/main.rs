mod camera;
mod gaze;
mod gesture;
mod hand;
mod kvm;

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

use crate::gaze::GazeResult;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("gesture_control=info".parse()?))
        .init();

    tracing::info!("Gesture Control starting...");

    // Load models
    let mut hand_detector = hand::HandDetector::new(
        "models/palm_detection.onnx",
        "models/hand_landmark.onnx",
    )?;

    let mut gaze_tracker = gaze::GazeTracker::new(
        "models/face_detection.onnx",
        "models/gaze_mobileone_s0.onnx",
    )?;

    tracing::info!("All models loaded. Starting camera...");

    let mut cam = camera::CameraCapture::new(0)?;
    cam.open()?;

    // Warm up with first frame
    let (rgb, w, h) = cam.frame_rgb()?;
    tracing::info!("Frame: {}x{}", w, h);
    let _ = hand_detector.detect(&rgb, w, h)?;
    tracing::info!("Warm-up done. Entering main loop (Ctrl+C to stop)");

    let mut frame_count = 0u64;
    let mut last_gaze: Option<GazeResult> = None;
    let mut fps_timer = Instant::now();
    let mut fps_frame_count = 0u32;

    loop {
        let frame_start = Instant::now();

        let (rgb, w, h) = cam.frame_rgb()?;

        // Hand detection every frame
        let hand_start = Instant::now();
        let hands = hand_detector.detect(&rgb, w, h)?;
        let hand_ms = hand_start.elapsed().as_secs_f32() * 1000.0;

        // Gaze detection every 5 frames
        let gaze_ms;
        if frame_count % 5 == 0 {
            let gaze_start = Instant::now();
            if let Some(g) = gaze_tracker.detect(&rgb, w, h)? {
                last_gaze = Some(g);
            }
            gaze_ms = gaze_start.elapsed().as_secs_f32() * 1000.0;
        } else {
            gaze_ms = 0.0;
        }

        // Log results periodically (every 30 frames)
        if frame_count % 30 == 0 {
            let hand_info = if let Some(h) = hands.first() {
                format!("{} - {} [{} fingers]", h.label, h.gesture, h.finger_count)
            } else {
                "none".to_string()
            };
            let gaze_info = match &last_gaze {
                Some(g) => format!("{} (yaw={:.1}°)", g.direction, g.yaw),
                None => "no face".to_string(),
            };
            tracing::info!(
                "hand={} | gaze={} | hand:{:.1}ms gaze:{:.1}ms total:{:.1}ms",
                hand_info, gaze_info, hand_ms, gaze_ms,
                frame_start.elapsed().as_secs_f32() * 1000.0
            );
        }

        // FPS counter
        fps_frame_count += 1;
        if fps_timer.elapsed().as_secs() >= 5 {
            let fps = fps_frame_count as f32 / fps_timer.elapsed().as_secs_f32();
            tracing::info!("FPS: {:.1}", fps);
            fps_timer = Instant::now();
            fps_frame_count = 0;
        }

        frame_count += 1;
    }
}
