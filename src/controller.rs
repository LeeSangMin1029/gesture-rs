// Gesture Controller — state machine that converts raw detections into KVM InputEvents
//
// Responsibilities:
// - Smooth pointer coordinates (OneEuroFilter)
// - Debounce gestures (majority vote over N frames)
// - Click/drag detection (pinch timing)
// - Scroll handling (thumbs-up lock-in)
// - Gaze switch events (direction change with hysteresis)

use std::collections::VecDeque;
use std::time::Instant;

use crate::filter::OneEuroFilter;
use crate::gaze::GazeResult;
use crate::gesture::{GazeDirection, Gesture, InputEvent, MouseButton};
use crate::hand::HandResult;

// --- Tunable constants (ported from Python mouse_controller.py) ---
const POINTER_GAIN: f32 = 2.5;
const POINTER_DEADZONE: f32 = 2.0;
const SCROLL_SENS: f32 = 135.0;
const SCROLL_DEADZONE: f32 = 3.0;
const PINCH_CLICK_THRESHOLD: f32 = 0.06; // normalized distance for click
const MIDDLE_PINCH_THRESHOLD: f32 = 0.06;
const DRAG_HOLD_SECS: f32 = 0.4;
const SCROLL_LOCK_SECS: f32 = 0.5;
const GESTURE_BUF_SIZE: usize = 5;

pub struct GestureController {
    // Pointer smoothing
    filter_x: OneEuroFilter,
    filter_y: OneEuroFilter,
    prev_smooth: Option<(f32, f32)>,
    start_time: Instant,

    // Gesture debouncing
    gesture_buf: VecDeque<Gesture>,
    current_gesture: Gesture,

    // Pinch/click state
    pinch_active: bool,
    pinch_start: Option<Instant>,
    drag_active: bool,
    middle_pinch_active: bool,

    // Scroll state (thumbs-up)
    scroll_locked: bool,
    scroll_lock_start: Option<Instant>,
    scroll_origin_y: Option<f32>,
    scroll_filter: OneEuroFilter,

    // Two-finger scroll
    two_finger_origin_y: Option<f32>,

    // Gaze state
    last_gaze_dir: GazeDirection,
    gaze_stable_count: u32,
}

impl GestureController {
    pub fn new() -> Self {
        Self {
            filter_x: OneEuroFilter::new(0.8, 0.01),
            filter_y: OneEuroFilter::new(0.8, 0.01),
            prev_smooth: None,
            start_time: Instant::now(),

            gesture_buf: VecDeque::with_capacity(GESTURE_BUF_SIZE),
            current_gesture: Gesture::Fist,

            pinch_active: false,
            pinch_start: None,
            drag_active: false,
            middle_pinch_active: false,

            scroll_locked: false,
            scroll_lock_start: None,
            scroll_origin_y: None,
            scroll_filter: OneEuroFilter::new(0.3, 0.005),

            two_finger_origin_y: None,

            last_gaze_dir: GazeDirection::Center,
            gaze_stable_count: 0,
        }
    }

    /// Process one frame of detections, return events to send to KVM
    pub fn update(
        &mut self,
        hand: Option<&HandResult>,
        gaze: Option<&GazeResult>,
    ) -> Vec<InputEvent> {
        let mut events = Vec::new();

        // Gaze events
        if let Some(g) = gaze {
            if let Some(evt) = self.process_gaze(g) {
                events.push(evt);
            }
        }

        // Hand events
        let Some(hand) = hand else {
            self.reset_hand_state();
            return events;
        };

        let gesture = self.debounce_gesture(hand.gesture);
        self.current_gesture = gesture;

        match gesture {
            Gesture::Fist => {
                self.reset_pointer();
                self.reset_scroll();
                if self.drag_active {
                    self.drag_active = false;
                    events.push(InputEvent::DragEnd);
                }
            }
            Gesture::OpenHand | Gesture::Pointer => {
                self.reset_scroll();
                // Pointer movement
                if let Some(evt) = self.process_pointer(hand) {
                    events.push(evt);
                }
                // Pinch check (click/drag)
                events.extend(self.process_pinch(hand));
            }
            Gesture::Pinch => {
                // Pinch detected by gesture classifier — handle click/drag
                if let Some(evt) = self.process_pointer(hand) {
                    events.push(evt);
                }
                events.extend(self.process_pinch(hand));
            }
            Gesture::TwoFingers => {
                self.reset_pointer();
                if let Some(evt) = self.process_two_finger_scroll(hand) {
                    events.push(evt);
                }
            }
            Gesture::ThumbsUp => {
                self.reset_pointer();
                if let Some(evt) = self.process_thumbs_up_scroll(hand) {
                    events.push(evt);
                }
            }
            Gesture::ThumbsDown => {
                self.reset_pointer();
                self.reset_scroll();
                // ThumbsDown event can be handled by KVM as minimize
            }
        }

        events
    }

    pub fn current_gesture(&self) -> Gesture {
        self.current_gesture
    }

    // --- Pointer ---

    fn process_pointer(&mut self, hand: &HandResult) -> Option<InputEvent> {
        let t = self.start_time.elapsed().as_secs_f32();
        // Track index MCP (landmark 5) for stable pointer
        let raw_x = hand.points[5].0;
        let raw_y = hand.points[5].1;

        let smooth_x = self.filter_x.filter(t, raw_x);
        let smooth_y = self.filter_y.filter(t, raw_y);

        let result = if let Some((px, py)) = self.prev_smooth {
            // Convert normalized delta to pixel movement
            // Assuming ~640px frame width, gain converts to screen pixels
            let dx = (smooth_x - px) * 640.0 * POINTER_GAIN;
            let dy = (smooth_y - py) * 480.0 * POINTER_GAIN;

            if dx.abs() >= POINTER_DEADZONE || dy.abs() >= POINTER_DEADZONE {
                let dx = if dx.abs() < POINTER_DEADZONE { 0.0 } else { dx };
                let dy = if dy.abs() < POINTER_DEADZONE { 0.0 } else { dy };
                Some(InputEvent::MouseMove {
                    dx: dx as i32,
                    dy: dy as i32,
                })
            } else {
                None
            }
        } else {
            None
        };

        self.prev_smooth = Some((smooth_x, smooth_y));
        result
    }

    fn reset_pointer(&mut self) {
        self.filter_x.reset();
        self.filter_y.reset();
        self.prev_smooth = None;
    }

    // --- Pinch / Click / Drag ---

    fn process_pinch(&mut self, hand: &HandResult) -> Vec<InputEvent> {
        let mut events = Vec::new();

        // Double-click: middle finger + thumb
        if hand.middle_pinch_dist < MIDDLE_PINCH_THRESHOLD {
            if !self.middle_pinch_active {
                self.middle_pinch_active = true;
                if self.drag_active {
                    self.drag_active = false;
                    events.push(InputEvent::DragEnd);
                }
                events.push(InputEvent::DoubleClick {
                    button: MouseButton::Left,
                });
            }
            return events;
        } else {
            self.middle_pinch_active = false;
        }

        // Single click / drag: index + thumb
        if hand.pinch_dist < PINCH_CLICK_THRESHOLD {
            if !self.pinch_active {
                self.pinch_active = true;
                self.pinch_start = Some(Instant::now());
            } else if let Some(start) = self.pinch_start {
                if !self.drag_active && start.elapsed().as_secs_f32() >= DRAG_HOLD_SECS {
                    self.drag_active = true;
                    events.push(InputEvent::DragStart);
                }
            }
        } else if self.pinch_active {
            // Pinch released
            if self.drag_active {
                self.drag_active = false;
                events.push(InputEvent::DragEnd);
            } else {
                events.push(InputEvent::Click {
                    button: MouseButton::Left,
                });
            }
            self.pinch_active = false;
            self.pinch_start = None;
        }

        events
    }

    // --- Thumbs-up Scroll ---

    fn process_thumbs_up_scroll(&mut self, hand: &HandResult) -> Option<InputEvent> {
        let t = self.start_time.elapsed().as_secs_f32();
        let raw_y = hand.points[4].1; // thumb tip Y
        let smooth_y = self.scroll_filter.filter(t, raw_y);

        if self.scroll_locked {
            let origin = self.scroll_origin_y.unwrap_or(smooth_y);
            let dy = smooth_y - origin;
            if dy.abs() > SCROLL_DEADZONE / 480.0 {
                let amount = (-dy * SCROLL_SENS) as i32;
                if amount != 0 {
                    return Some(InputEvent::Scroll { amount });
                }
            }
            return None;
        }

        // Lock-in phase
        if self.scroll_lock_start.is_none() {
            self.scroll_lock_start = Some(Instant::now());
            self.scroll_origin_y = Some(smooth_y);
            return None;
        }

        if let Some(start) = self.scroll_lock_start {
            if start.elapsed().as_secs_f32() >= SCROLL_LOCK_SECS {
                self.scroll_locked = true;
                self.scroll_origin_y = Some(smooth_y);
            }
        }

        None
    }

    // --- Two-finger Scroll ---

    fn process_two_finger_scroll(&mut self, hand: &HandResult) -> Option<InputEvent> {
        let raw_y = hand.points[9].1; // middle MCP Y

        let Some(origin) = self.two_finger_origin_y else {
            self.two_finger_origin_y = Some(raw_y);
            return None;
        };

        let dy = raw_y - origin;
        if dy.abs() > SCROLL_DEADZONE / 480.0 {
            let amount = (-dy * SCROLL_SENS) as i32;
            self.two_finger_origin_y = Some(raw_y);
            if amount != 0 {
                return Some(InputEvent::Scroll { amount });
            }
        }

        None
    }

    fn reset_scroll(&mut self) {
        self.scroll_locked = false;
        self.scroll_lock_start = None;
        self.scroll_origin_y = None;
        self.scroll_filter.reset();
        self.two_finger_origin_y = None;
    }

    // --- Gesture Debouncing ---

    fn debounce_gesture(&mut self, gesture: Gesture) -> Gesture {
        self.gesture_buf.push_back(gesture);
        if self.gesture_buf.len() > GESTURE_BUF_SIZE {
            self.gesture_buf.pop_front();
        }
        if self.gesture_buf.len() < 3 {
            return gesture;
        }
        // Majority vote
        let mut counts = [0u8; 7];
        for g in &self.gesture_buf {
            counts[gesture_index(*g)] += 1;
        }
        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        index_to_gesture(max_idx)
    }

    // --- Gaze ---

    fn process_gaze(&mut self, gaze: &GazeResult) -> Option<InputEvent> {
        if gaze.direction == self.last_gaze_dir {
            self.gaze_stable_count = 0;
            return None;
        }
        // Require 3 consecutive frames with new direction (hysteresis)
        self.gaze_stable_count += 1;
        if self.gaze_stable_count >= 3 {
            self.last_gaze_dir = gaze.direction;
            self.gaze_stable_count = 0;
            return Some(InputEvent::GazeSwitch {
                direction: gaze.direction,
            });
        }
        None
    }

    // --- Reset ---

    fn reset_hand_state(&mut self) {
        self.reset_pointer();
        self.reset_scroll();
        if self.drag_active {
            self.drag_active = false;
            // Note: DragEnd event lost if no hand detected — KVM should timeout
        }
        self.pinch_active = false;
        self.pinch_start = None;
        self.gesture_buf.clear();
    }
}

fn gesture_index(g: Gesture) -> usize {
    match g {
        Gesture::Fist => 0,
        Gesture::Pinch => 1,
        Gesture::TwoFingers => 2,
        Gesture::ThumbsUp => 3,
        Gesture::ThumbsDown => 4,
        Gesture::OpenHand => 5,
        Gesture::Pointer => 6,
    }
}

fn index_to_gesture(i: usize) -> Gesture {
    match i {
        0 => Gesture::Fist,
        1 => Gesture::Pinch,
        2 => Gesture::TwoFingers,
        3 => Gesture::ThumbsUp,
        4 => Gesture::ThumbsDown,
        5 => Gesture::OpenHand,
        _ => Gesture::Pointer,
    }
}
