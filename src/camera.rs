// Camera capture module using nokhwa (Windows MSMF backend)

use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::Camera;

pub const FRAME_W: u32 = 1280;
pub const FRAME_H: u32 = 720;

pub struct CameraCapture {
    camera: Camera,
}

impl CameraCapture {
    pub fn new(index: u32) -> Result<Self> {
        let fmt = CameraFormat::new(
            Resolution::new(FRAME_W, FRAME_H),
            FrameFormat::MJPEG,
            30,
        );
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(fmt));
        let camera = Camera::new(CameraIndex::Index(index), requested)
            .context("Failed to open camera")?;
        Ok(Self { camera })
    }

    pub fn open(&mut self) -> Result<()> {
        self.camera.open_stream().context("Failed to open camera stream")
    }

    /// Capture a frame as RGB bytes (H x W x 3), horizontally flipped (mirror mode)
    pub fn frame_rgb(&mut self) -> Result<(Vec<u8>, u32, u32)> {
        let frame = self.camera.frame().context("Failed to capture frame")?;
        let decoded = frame.decode_image::<RgbFormat>().context("Failed to decode frame")?;
        let (w, h) = (decoded.width(), decoded.height());
        let mut raw = decoded.into_raw();

        // Horizontal flip (mirror) — matches MediaPipe Python cv2.flip(frame, 1)
        for y in 0..h as usize {
            let row_start = y * w as usize * 3;
            for x in 0..w as usize / 2 {
                let left = row_start + x * 3;
                let right = row_start + (w as usize - 1 - x) * 3;
                for c in 0..3 {
                    raw.swap(left + c, right + c);
                }
            }
        }

        Ok((raw, w, h))
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        let _ = self.camera.stop_stream();
    }
}

/// Quick test: open camera, capture 10 frames, print info
#[allow(dead_code)]
pub fn run_camera_test() -> Result<()> {
    tracing::info!("Camera test: opening device 0...");
    let mut cam = CameraCapture::new(0)?;
    cam.open()?;

    for i in 0..10 {
        let (rgb, w, h) = cam.frame_rgb()?;
        if i == 0 {
            tracing::info!("Frame: {}x{}, {} bytes", w, h, rgb.len());
        }
    }

    tracing::info!("Camera test passed!");
    Ok(())
}
