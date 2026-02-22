mod camera;
mod controller;
mod filter;
mod frame_channel;
mod gaze;
mod gesture;
mod hand;
mod kvm;

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

use crate::controller::GestureController;
use crate::gaze::GazeResult;

const KVM_ADDR: &str = "127.0.0.1:9877";

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("gesture_control=info".parse()?),
        )
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

    tracing::info!("All models loaded.");

    // --- Optimization 1: Camera capture in separate thread ---
    // Latest-frame-only channel: old frames dropped automatically
    let (tx, rx) = frame_channel::channel();

    let camera_thread = std::thread::spawn(move || -> Result<()> {
        let mut cam = camera::CameraCapture::new(0)?;
        cam.open()?;
        tracing::info!("Camera thread started");
        loop {
            let (rgb, w, h) = cam.frame_rgb()?;
            tx.send(rgb, w, h);
        }
    });

    // Wait for first frame to confirm camera works
    let (first_rgb, w, h) = rx.recv().ok_or_else(|| anyhow::anyhow!("Camera failed"))?;
    tracing::info!("Frame: {}x{}", w, h);

    // Warm up models
    let _ = hand_detector.detect(&first_rgb, w, h)?;
    tracing::info!("Warm-up done.");

    // KVM connection (optional)
    let mut kvm = match kvm::KvmClient::connect(KVM_ADDR) {
        Ok(client) => Some(client),
        Err(e) => {
            tracing::warn!("KVM not available ({}), standalone mode", e);
            None
        }
    };

    let mut ctrl = GestureController::new();
    let mut frame_count = 0u64;
    let mut last_gaze: Option<GazeResult> = None;
    let mut fps_timer = Instant::now();
    let mut fps_frame_count = 0u32;

    tracing::info!("Entering main loop (Ctrl+C to stop)");

    loop {
        // Get latest frame (blocks until camera produces one)
        let Some((rgb, w, h)) = rx.recv() else {
            tracing::error!("Camera disconnected");
            break;
        };

        let frame_start = Instant::now();

        // --- Optimization 2 & 4: Parallel hand + gaze inference ---
        // Arc<Vec<u8>> enables zero-copy sharing between threads
        let run_gaze = frame_count % 5 == 0;
        let rgb_ref: &[u8] = &rgb;

        let (hands_result, gaze_result) = if run_gaze {
            // Parallel: hand + gaze on same frame using scoped threads
            let rgb_clone = Arc::clone(&rgb);
            std::thread::scope(|s| {
                let hand_handle = s.spawn(|| hand_detector.detect(rgb_ref, w, h));
                let gaze_handle = s.spawn(|| gaze_tracker.detect(&rgb_clone, w, h));

                let hands = hand_handle.join().expect("hand thread panicked");
                let gaze = gaze_handle.join().expect("gaze thread panicked");
                (hands, gaze)
            })
        } else {
            // Hand only — no thread overhead
            (hand_detector.detect(rgb_ref, w, h), Ok(None))
        };

        let hands = hands_result?;
        if let Ok(Some(g)) = gaze_result {
            last_gaze = Some(g);
        }

        let inference_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        // --- Optimization 3: Controller with OneEuroFilter + deadzone ---
        // Only meaningful events pass through (no jitter)
        let primary_hand = hands.first();
        let events = ctrl.update(primary_hand, last_gaze.as_ref());

        // Send events to KVM (batched, TCP_NODELAY)
        if !events.is_empty() {
            if let Some(ref mut client) = kvm {
                if let Err(e) = client.send_batch(&events) {
                    tracing::warn!("KVM send failed: {}, disconnecting", e);
                    kvm = None;
                }
            }
        }

        // Log periodically (every 30 frames)
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
                "hand={} | gaze={} | ctrl={} | events={} | infer:{:.1}ms",
                hand_info,
                gaze_info,
                ctrl.current_gesture(),
                events.len(),
                inference_ms
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

    // Wait for camera thread (won't normally reach here)
    if let Err(e) = camera_thread.join() {
        tracing::error!("Camera thread error: {:?}", e);
    }

    Ok(())
}
