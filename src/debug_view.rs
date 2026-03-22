// Debug preview window + fullscreen gaze overlay
//
// Camera preview: shows camera feed with hand skeleton + face box
// Gaze overlay: fullscreen black window with green gaze dot on monitor coordinates

use crate::calibration::CalibrationConfig;
use crate::gaze::GazeResult;
use crate::hand::HandResult;
use anyhow::Result;
use minifb::{Window, WindowOptions};

const HAND_CONNECTIONS: [(usize, usize); 23] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
];

// Background color used as transparency key
const OVERLAY_BG: u32 = 0x010101;

#[allow(unsafe_code)] // WHY: read-only Windows API calls for screen size + window transparency
mod win32 {
    #[link(name = "user32")]
    unsafe extern "system" {
        pub fn GetSystemMetrics(nIndex: i32) -> i32;
        pub fn FindWindowA(lpClassName: *const u8, lpWindowName: *const u8) -> isize;
        pub fn GetWindowLongA(hWnd: isize, nIndex: i32) -> i32;
        pub fn SetWindowLongA(hWnd: isize, nIndex: i32, dwNewLong: i32) -> i32;
        pub fn SetLayeredWindowAttributes(hWnd: isize, crKey: u32, bAlpha: u8, dwFlags: u32) -> i32;
    }

    pub const GWL_EXSTYLE: i32 = -20;
    pub const WS_EX_LAYERED: i32 = 0x80000;
    pub const WS_EX_TRANSPARENT: i32 = 0x20;
    pub const LWA_COLORKEY: u32 = 0x1;
}

fn screen_size() -> (usize, usize) {
    #[allow(unsafe_code)]
    unsafe {
        let w = win32::GetSystemMetrics(0);
        let h = win32::GetSystemMetrics(1);
        (w as usize, h as usize)
    }
}

/// Make the overlay window transparent (color-key) and click-through
fn make_overlay_transparent() {
    #[allow(unsafe_code)]
    unsafe {
        let title = b"Gaze Overlay\0";
        let hwnd = win32::FindWindowA(std::ptr::null(), title.as_ptr());
        if hwnd == 0 {
            tracing::warn!("Could not find overlay window for transparency");
            return;
        }
        let ex = win32::GetWindowLongA(hwnd, win32::GWL_EXSTYLE);
        win32::SetWindowLongA(
            hwnd,
            win32::GWL_EXSTYLE,
            ex | win32::WS_EX_LAYERED | win32::WS_EX_TRANSPARENT,
        );
        // COLORREF = 0x00BBGGRR; OVERLAY_BG 0x010101 → R=1,G=1,B=1
        win32::SetLayeredWindowAttributes(hwnd, OVERLAY_BG, 0, win32::LWA_COLORKEY);
        tracing::info!("Overlay: transparent + click-through enabled");
    }
}

pub struct DebugView {
    window: Window,
    buf: Vec<u32>,
    w: usize,
    h: usize,
    // Fullscreen gaze overlay
    overlay: Window,
    overlay_buf: Vec<u32>,
    screen_w: usize,
    screen_h: usize,
}

impl DebugView {
    pub fn new(w: u32, h: u32) -> Result<Self> {
        let window = Window::new(
            "Gesture Debug",
            w as usize,
            h as usize,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )?;

        let (screen_w, screen_h) = screen_size();
        tracing::info!("Screen: {}x{}", screen_w, screen_h);

        let overlay = Window::new(
            "Gaze Overlay",
            screen_w,
            screen_h,
            WindowOptions {
                borderless: true,
                topmost: true,
                ..WindowOptions::default()
            },
        )?;

        make_overlay_transparent();

        Ok(Self {
            window,
            buf: vec![0u32; (w * h) as usize],
            w: w as usize,
            h: h as usize,
            overlay,
            overlay_buf: vec![0u32; screen_w * screen_h],
            screen_w,
            screen_h,
        })
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open() && self.overlay.is_open()
    }

    pub fn render(
        &mut self,
        rgb: &[u8],
        hands: &[HandResult],
        gaze: Option<&GazeResult>,
        calib: &CalibrationConfig,
    ) -> Result<()> {
        // --- Camera preview window ---
        for i in 0..self.w * self.h {
            let r = rgb[i * 3] as u32;
            let g = rgb[i * 3 + 1] as u32;
            let b = rgb[i * 3 + 2] as u32;
            self.buf[i] = (r << 16) | (g << 8) | b;
        }

        // Hand skeletons
        let hand_colors: [(u32, u32); 2] = [
            (0x00FF00, 0xFF0000),
            (0x00AAFF, 0xFF8800),
        ];
        for (hi, hand) in hands.iter().enumerate() {
            let (line_col, tip_col) = hand_colors[hi.min(1)];
            let (w, h) = (self.w as f32, self.h as f32);
            for &(a, b) in &HAND_CONNECTIONS {
                self.draw_line_cam(
                    (hand.points[a].0 * w) as i32, (hand.points[a].1 * h) as i32,
                    (hand.points[b].0 * w) as i32, (hand.points[b].1 * h) as i32,
                    line_col,
                );
            }
            for (i, pt) in hand.points.iter().enumerate() {
                let color = if i == 4 || i == 8 { tip_col } else { line_col };
                self.draw_dot_cam((pt.0 * w) as i32, (pt.1 * h) as i32, 3, color);
            }
        }

        // Face box on camera preview
        if let Some(g) = gaze {
            let (w, h) = (self.w as f32, self.h as f32);
            let (x1, y1) = ((g.face_box[0] * w) as i32, (g.face_box[1] * h) as i32);
            let (x2, y2) = ((g.face_box[2] * w) as i32, (g.face_box[3] * h) as i32);
            self.draw_rect_cam(x1, y1, x2, y2, 0x00FFFF);
        }

        // Window title
        if let Some(g) = gaze {
            self.window.set_title(&format!(
                "Gaze: {} yaw={:.1} pitch={:.1}",
                g.direction, g.yaw, g.pitch
            ));
        } else {
            self.window.set_title("Gaze: no face");
        }

        self.window.update_with_buffer(&self.buf, self.w, self.h)?;

        // --- Fullscreen gaze overlay ---
        // Clear to dark background
        self.overlay_buf.fill(OVERLAY_BG);

        // Draw corner markers for reference
        let sw = self.screen_w as i32;
        let sh = self.screen_h as i32;
        let margin = 40;
        let mark_len = 30;
        let mark_col = 0x333333;
        // Top-left
        self.draw_line_ovl(margin, margin, margin + mark_len, margin, mark_col);
        self.draw_line_ovl(margin, margin, margin, margin + mark_len, mark_col);
        // Top-right
        self.draw_line_ovl(sw - margin, margin, sw - margin - mark_len, margin, mark_col);
        self.draw_line_ovl(sw - margin, margin, sw - margin, margin + mark_len, mark_col);
        // Bottom-left
        self.draw_line_ovl(margin, sh - margin, margin + mark_len, sh - margin, mark_col);
        self.draw_line_ovl(margin, sh - margin, margin, sh - margin - mark_len, mark_col);
        // Bottom-right
        self.draw_line_ovl(sw - margin, sh - margin, sw - margin - mark_len, sh - margin, mark_col);
        self.draw_line_ovl(sw - margin, sh - margin, sw - margin, sh - margin - mark_len, mark_col);

        // Gaze point on screen
        if let Some(g) = gaze {
            let gx = (calib.yaw_to_screen_x(g.yaw) * self.screen_w as f32) as i32;
            let gy = (calib.pitch_to_screen_y(g.pitch) * self.screen_h as f32) as i32;
            self.draw_dot_ovl(gx, gy, 10, 0x00FF00);
            self.draw_line_ovl(gx - 20, gy, gx + 20, gy, 0x00FF00);
            self.draw_line_ovl(gx, gy - 20, gx, gy + 20, 0x00FF00);
        }

        self.overlay.update_with_buffer(&self.overlay_buf, self.screen_w, self.screen_h)?;

        Ok(())
    }

    // --- Camera preview drawing ---

    fn draw_dot_cam(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        let (w, h) = (self.w as i32, self.h as i32);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let (x, y) = (cx + dx, cy + dy);
                    if x >= 0 && x < w && y >= 0 && y < h {
                        self.buf[y as usize * self.w + x as usize] = color;
                    }
                }
            }
        }
    }

    fn draw_line_cam(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let (w, h) = (self.w as i32, self.h as i32);
        bresenham(&mut self.buf, w, h, x0, y0, x1, y1, color);
    }

    fn draw_rect_cam(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, color: u32) {
        self.draw_line_cam(x1, y1, x2, y1, color);
        self.draw_line_cam(x2, y1, x2, y2, color);
        self.draw_line_cam(x2, y2, x1, y2, color);
        self.draw_line_cam(x1, y2, x1, y1, color);
    }

    // --- Overlay drawing ---

    fn draw_dot_ovl(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        let (w, h) = (self.screen_w as i32, self.screen_h as i32);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let (x, y) = (cx + dx, cy + dy);
                    if x >= 0 && x < w && y >= 0 && y < h {
                        self.overlay_buf[y as usize * self.screen_w + x as usize] = color;
                    }
                }
            }
        }
    }

    fn draw_line_ovl(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let (w, h) = (self.screen_w as i32, self.screen_h as i32);
        bresenham(&mut self.overlay_buf, w, h, x0, y0, x1, y1, color);
    }
}

// Shared Bresenham line drawing
fn bresenham(buf: &mut [u32], w: i32, h: i32, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let (mut err, mut x, mut y) = (dx + dy, x0, y0);
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            buf[y as usize * w as usize + x as usize] = color;
        }
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}
