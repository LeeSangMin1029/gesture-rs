mod calibration;
mod camera;
mod controller;
mod debug_view;
mod filter;
mod frame_channel;
mod gaze;
mod gesture;
mod hand;
mod kvm;

use anyhow::{anyhow, Result};
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

    // Hand detection (disabled — switching to gaze tracking)
    // let mut hand_detector = hand::HandDetector::new(
    //     "models/palm_detection.onnx",
    //     "models/hand_landmark.onnx",
    // )?;

    let mut gaze_tracker = gaze::GazeTracker::new(
        "models/det_2.5g.onnx",
        "models/iris_landmark.onnx",
    )?;

    let calib = calibration::CalibrationConfig::load("calibration.json").unwrap_or_else(|_| {
        let c = calibration::CalibrationConfig::default();
        if let Err(e) = c.save("calibration.json") {
            tracing::warn!("Failed to save default calibration: {}", e);
        }
        c
    });
    tracing::info!("Calibration: yaw_center={}, yaw_range={}", calib.yaw_center, calib.yaw_range);

    tracing::info!("All models loaded.");

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_cam = shutdown.clone();
    let shutdown_sig = shutdown.clone();
    ctrlc::set_handler(move || {
        tracing::info!("Ctrl+C received, shutting down...");
        shutdown_sig.store(true, Ordering::Relaxed);
    })?;

    let (tx, rx) = frame_channel::channel();

    let camera_thread = std::thread::spawn(move || -> Result<()> {
        let mut cam = camera::CameraCapture::new(0)?;
        cam.open()?;
        tracing::info!("Camera thread started");
        let mut consecutive_errors = 0u32;
        loop {
            if shutdown_cam.load(Ordering::Relaxed) {
                break;
            }
            match cam.frame_rgb() {
                Ok((rgb, w, h)) => {
                    consecutive_errors = 0;
                    tx.send(rgb, w, h);
                }
                Err(e) => {
                    consecutive_errors += 1;
                    if consecutive_errors <= 30 {
                        tracing::warn!("Frame capture error ({}): {}", consecutive_errors, e);
                        std::thread::sleep(std::time::Duration::from_millis(33));
                        continue;
                    }
                    tracing::error!("Too many consecutive camera errors, stopping");
                    return Err(e);
                }
            }
        }
        Ok(())
    });

    let (first_rgb, w, h) = rx.recv().ok_or_else(|| anyhow!("Camera failed"))?;
    tracing::info!("Frame: {}x{}", w, h);

    // let _ = hand_detector.detect(&first_rgb, w, h)?;
    let _ = gaze_tracker.detect(&first_rgb, w, h)?;
    tracing::info!("Warm-up done.");

    // Debug preview window
    let mut view = debug_view::DebugView::new(w, h)?;

    let mut ctrl = GestureController::new();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut fps_frame_count = 0u32;
    let mut last_fps = 0.0f32;

    tracing::info!("Entering main loop (close window or Ctrl+C to stop)");

    while view.is_open() && !shutdown.load(Ordering::Relaxed) {
        let Some((rgb, w, h)) = rx.recv() else {
            tracing::error!("Camera disconnected");
            break;
        };

        let frame_start = Instant::now();

        // Hand detection (disabled)
        // let hands = hand_detector.detect(&rgb, w, h)?;
        let hands = Vec::new();

        // Gaze tracking
        let gaze_result = gaze_tracker.detect(&rgb, w, h)?;

        let inference_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        let primary_hand = hands.first();
        let _events = ctrl.update(primary_hand, gaze_result.as_ref());

        // Render debug preview
        view.render(&rgb, &hands, gaze_result.as_ref(), &calib)?;

        // Debug log every 10 frames
        if frame_count % 10 == 0 {
            let gaze_info = if let Some(g) = &gaze_result {
                format!("{} yaw={:.1} pitch={:.1}", g.direction, g.yaw, g.pitch)
            } else {
                "no face".to_string()
            };
            tracing::info!(
                "{:.1}ms | fps={:.0} | gaze={}",
                inference_ms, last_fps, gaze_info,
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
