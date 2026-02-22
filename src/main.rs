mod camera;
mod controller;
mod debug_view;
mod filter;
mod frame_channel;
mod gaze;
mod gesture;
mod hand;
mod kvm;

use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

use crate::controller::GestureController;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("gesture_control=debug".parse()?),
        )
        .init();

    tracing::info!("Gesture Control starting...");

    let mut hand_detector = hand::HandDetector::new(
        "models/palm_detection.onnx",
        "models/hand_landmark.onnx",
    )?;

    tracing::info!("All models loaded.");

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_cam = shutdown.clone();

    let (tx, rx) = frame_channel::channel();

    let camera_thread = std::thread::spawn(move || -> Result<()> {
        let mut cam = camera::CameraCapture::new(0)?;
        cam.open()?;
        tracing::info!("Camera thread started");
        loop {
            if shutdown_cam.load(Ordering::Relaxed) {
                break;
            }
            let (rgb, w, h) = cam.frame_rgb()?;
            tx.send(rgb, w, h);
        }
        Ok(())
    });

    let (first_rgb, w, h) = rx.recv().ok_or_else(|| anyhow::anyhow!("Camera failed"))?;
    tracing::info!("Frame: {}x{}", w, h);

    let _ = hand_detector.detect(&first_rgb, w, h)?;
    tracing::info!("Warm-up done.");

    // Debug preview window
    let mut view = debug_view::DebugView::new(w, h)?;

    let mut ctrl = GestureController::new();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut fps_frame_count = 0u32;
    let mut last_fps = 0.0f32;

    tracing::info!("Entering main loop (close window or Ctrl+C to stop)");

    while view.is_open() {
        let Some((rgb, w, h)) = rx.recv() else {
            tracing::error!("Camera disconnected");
            break;
        };

        let frame_start = Instant::now();

        let hands = hand_detector.detect(&rgb, w, h)?;

        let inference_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        let primary_hand = hands.first();
        let _events = ctrl.update(primary_hand, None);

        // Render debug preview
        view.render(&rgb, &hands, None)?;

        // Debug log every 10 frames
        if frame_count % 10 == 0 {
            let track = if hand_detector.is_tracking() { "TRK" } else { "DET" };
            let hand_info = if let Some(h) = hands.first() {
                format!(
                    "{} {} [{} fingers] pinch={:.3} mid={:.3}",
                    h.label, h.gesture, h.finger_count, h.pinch_dist, h.middle_pinch_dist
                )
            } else {
                "none".to_string()
            };
            tracing::info!(
                "[{}] {:.1}ms | fps={:.0} | hand={} | ctrl={}",
                track, inference_ms, last_fps, hand_info, ctrl.current_gesture(),
            );
        }

        fps_frame_count += 1;
        if fps_timer.elapsed().as_secs() >= 5 {
            last_fps = fps_frame_count as f32 / fps_timer.elapsed().as_secs_f32();
            tracing::info!("FPS: {:.1}", last_fps);
            fps_timer = Instant::now();
            fps_frame_count = 0;
        }

        frame_count += 1;
    }

    // Signal camera thread to stop
    shutdown.store(true, Ordering::Relaxed);

    if let Err(e) = camera_thread.join() {
        tracing::error!("Camera thread error: {:?}", e);
    }

    tracing::info!("Shutdown complete.");
    Ok(())
}
