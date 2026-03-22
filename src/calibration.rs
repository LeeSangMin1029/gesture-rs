// Gaze calibration config for KVM screen mapping
//
// Maps raw yaw/pitch angles to normalized screen coordinates [0..1]
// and defines monitor zones for KVM switching.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Calibration parameters for gaze → screen mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Yaw angle when looking at screen center (degrees)
    pub yaw_center: f32,
    /// Total yaw range covering the screen width (degrees)
    pub yaw_range: f32,
    /// Pitch angle when looking at screen center (degrees)
    pub pitch_center: f32,
    /// Total pitch range covering the screen height (degrees)
    pub pitch_range: f32,
    /// Monitor zones for KVM switching
    pub monitors: Vec<MonitorZone>,
}

/// A monitor/PC zone defined by yaw angle range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorZone {
    pub name: String,
    /// Minimum yaw angle (degrees, inclusive)
    pub yaw_min: f32,
    /// Maximum yaw angle (degrees, exclusive)
    pub yaw_max: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            yaw_center: 0.0,
            yaw_range: 30.0, // +-15 deg covers screen width
            pitch_center: 0.0,
            pitch_range: 20.0, // +-10 deg covers screen height
            monitors: vec![
                MonitorZone {
                    name: "left".into(),
                    yaw_min: -90.0,
                    yaw_max: -15.0,
                },
                MonitorZone {
                    name: "center".into(),
                    yaw_min: -15.0,
                    yaw_max: 15.0,
                },
                MonitorZone {
                    name: "right".into(),
                    yaw_min: 15.0,
                    yaw_max: 90.0,
                },
            ],
        }
    }
}

impl CalibrationConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Map yaw angle to normalized screen X [0.0 = left, 1.0 = right]
    pub fn yaw_to_screen_x(&self, yaw: f32) -> f32 {
        (0.5 + (yaw - self.yaw_center) / self.yaw_range).clamp(0.0, 1.0)
    }

    /// Map pitch angle to normalized screen Y [0.0 = top, 1.0 = bottom]
    pub fn pitch_to_screen_y(&self, pitch: f32) -> f32 {
        (0.5 - (pitch - self.pitch_center) / self.pitch_range).clamp(0.0, 1.0)
    }

    /// Find which monitor zone the gaze yaw falls into
    pub fn find_monitor(&self, yaw: f32) -> Option<&MonitorZone> {
        self.monitors
            .iter()
            .find(|m| yaw >= m.yaw_min && yaw < m.yaw_max)
    }
}
