// Debug preview window — shows camera feed with hand skeleton + face box overlay

use crate::gaze::GazeResult;
use crate::hand::HandResult;
use anyhow::Result;
use minifb::{Window, WindowOptions};

const HAND_CONNECTIONS: [(usize, usize); 21] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9),
];

pub struct DebugView {
    window: Window,
    buf: Vec<u32>,
    w: usize,
    h: usize,
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
        Ok(Self {
            window,
            buf: vec![0u32; (w * h) as usize],
            w: w as usize,
            h: h as usize,
        })
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn render(
        &mut self,
        rgb: &[u8],
        hands: &[HandResult],
        gaze: Option<&GazeResult>,
    ) -> Result<()> {
        // Camera feed → ARGB
        for i in 0..self.w * self.h {
            let r = rgb[i * 3] as u32;
            let g = rgb[i * 3 + 1] as u32;
            let b = rgb[i * 3 + 2] as u32;
            self.buf[i] = (r << 16) | (g << 8) | b;
        }

        // Hand skeleton
        if let Some(hand) = hands.first() {
            let (w, h) = (self.w as f32, self.h as f32);
            for &(a, b) in &HAND_CONNECTIONS {
                self.line(
                    (hand.points[a].0 * w) as i32, (hand.points[a].1 * h) as i32,
                    (hand.points[b].0 * w) as i32, (hand.points[b].1 * h) as i32,
                    0x00FF00,
                );
            }
            for (i, pt) in hand.points.iter().enumerate() {
                let color = if i == 4 || i == 8 { 0xFF0000 } else { 0x00FF00 };
                self.dot((pt.0 * w) as i32, (pt.1 * h) as i32, 3, color);
            }
        }

        // Face box
        if let Some(g) = gaze {
            let (w, h) = (self.w as f32, self.h as f32);
            let (x1, y1) = ((g.face_box[0] * w) as i32, (g.face_box[1] * h) as i32);
            let (x2, y2) = ((g.face_box[2] * w) as i32, (g.face_box[3] * h) as i32);
            self.rect(x1, y1, x2, y2, 0x00FFFF);
        }

        self.window.update_with_buffer(&self.buf, self.w, self.h)?;
        Ok(())
    }

    fn dot(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
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

    fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let (w, h) = (self.w as i32, self.h as i32);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let (mut err, mut x, mut y) = (dx + dy, x0, y0);
        loop {
            if x >= 0 && x < w && y >= 0 && y < h {
                self.buf[y as usize * self.w + x as usize] = color;
            }
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    }

    fn rect(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, color: u32) {
        self.line(x1, y1, x2, y1, color);
        self.line(x2, y1, x2, y2, color);
        self.line(x2, y2, x1, y2, color);
        self.line(x1, y2, x1, y1, color);
    }
}
