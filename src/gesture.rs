// Gesture types and KVM input events

use serde::Serialize;

/// Hand gesture classification (matches Python hand_tracker.py)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gesture {
    Fist,
    Pinch,
    TwoFingers,
    ThumbsUp,
    ThumbsDown,
    OpenHand,
    Pointer,
}

/// Gaze direction for monitor/PC selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GazeDirection {
    Left,
    Center,
    Right,
}

/// Input events sent to KVM via TCP
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum InputEvent {
    MouseMove { dx: i32, dy: i32 },
    Click { button: MouseButton },
    DoubleClick { button: MouseButton },
    Scroll { amount: i32 },
    DragStart,
    DragEnd,
    GazeSwitch { direction: GazeDirection },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[allow(dead_code)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl std::fmt::Display for Gesture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gesture::Fist => write!(f, "FIST"),
            Gesture::Pinch => write!(f, "PINCH"),
            Gesture::TwoFingers => write!(f, "TWO_FINGERS"),
            Gesture::ThumbsUp => write!(f, "THUMBS_UP"),
            Gesture::ThumbsDown => write!(f, "THUMBS_DOWN"),
            Gesture::OpenHand => write!(f, "OPEN_HAND"),
            Gesture::Pointer => write!(f, "POINTER"),
        }
    }
}

impl std::fmt::Display for GazeDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GazeDirection::Left => write!(f, "LEFT"),
            GazeDirection::Center => write!(f, "CENTER"),
            GazeDirection::Right => write!(f, "RIGHT"),
        }
    }
}

// Serialize GazeDirection for JSON
impl Serialize for GazeDirection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
