// OneEuroFilter — low-latency signal smoothing for pointer coordinates
//
// Adapts cutoff frequency based on signal speed:
// slow movement → heavy smoothing, fast movement → minimal lag

use std::f32::consts::PI;

pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
    t_prev: Option<f32>,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = None;
    }

    pub fn filter(&mut self, t: f32, x: f32) -> f32 {
        let Some(t_prev) = self.t_prev else {
            self.t_prev = Some(t);
            self.x_prev = Some(x);
            return x;
        };

        let t_e = t - t_prev;
        if t_e <= 1e-6 {
            return self.x_prev.unwrap_or(x);
        }

        // Derivative smoothing
        let a_d = alpha(t_e, self.d_cutoff);
        let dx = (x - self.x_prev.unwrap_or(x)) / t_e;
        let dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev;

        // Adaptive cutoff
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = alpha(t_e, cutoff);
        let x_hat = a * x + (1.0 - a) * self.x_prev.unwrap_or(x);

        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        self.t_prev = Some(t);
        x_hat
    }
}

fn alpha(t_e: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / t_e)
}
