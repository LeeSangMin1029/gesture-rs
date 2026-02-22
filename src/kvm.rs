// KVM TCP client — async writer thread with BufWriter + wire format conversion

use crate::gesture::{InputEvent, MouseButton};
use anyhow::{Context, Result};
use std::io::{BufWriter, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

pub struct KvmClient {
    tx: mpsc::Sender<Vec<InputEvent>>,
}

impl KvmClient {
    pub fn connect(addr: &str) -> Result<Self> {
        let stream = TcpStream::connect(addr).context("Failed to connect to KVM server")?;
        stream
            .set_nodelay(true)
            .context("Failed to set TCP_NODELAY")?;
        tracing::info!("KVM connected to {}", addr);

        let (tx, rx) = mpsc::channel::<Vec<InputEvent>>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(stream);
            while let Ok(batch) = rx.recv() {
                for event in &batch {
                    if let Err(e) = write_wire_event(&mut writer, event) {
                        tracing::warn!("KVM write error: {}", e);
                        return;
                    }
                }
                if let Err(e) = writer.flush() {
                    tracing::warn!("KVM flush error: {}", e);
                    return;
                }
            }
        });

        Ok(Self { tx })
    }

    pub fn send_batch(&mut self, events: &[InputEvent]) -> Result<()> {
        self.tx
            .send(events.to_vec())
            .map_err(|_| anyhow::anyhow!("KVM writer thread disconnected"))
    }
}

/// Convert gesture-rs InputEvent → KVM wire format (JSON lines)
fn write_wire_event(w: &mut impl Write, event: &InputEvent) -> Result<()> {
    match event {
        InputEvent::MouseMove { dx, dy } => {
            writeln!(w, r#"{{"type":"mouse_move_delta","dx":{},"dy":{}}}"#, dx, dy)?;
        }
        InputEvent::Click { button } => {
            let btn = button_str(*button);
            writeln!(w, r#"{{"type":"mouse_button","button":"{}","action":"press"}}"#, btn)?;
            writeln!(w, r#"{{"type":"mouse_button","button":"{}","action":"release"}}"#, btn)?;
        }
        InputEvent::DoubleClick { button } => {
            let btn = button_str(*button);
            for _ in 0..2 {
                writeln!(w, r#"{{"type":"mouse_button","button":"{}","action":"press"}}"#, btn)?;
                writeln!(w, r#"{{"type":"mouse_button","button":"{}","action":"release"}}"#, btn)?;
            }
        }
        InputEvent::Scroll { amount } => {
            writeln!(w, r#"{{"type":"mouse_scroll","dx":0,"dy":{}}}"#, amount)?;
        }
        InputEvent::DragStart => {
            writeln!(w, r#"{{"type":"mouse_button","button":"left","action":"press"}}"#)?;
        }
        InputEvent::DragEnd => {
            writeln!(w, r#"{{"type":"mouse_button","button":"left","action":"release"}}"#)?;
        }
        InputEvent::GazeSwitch { direction } => {
            writeln!(w, r#"{{"type":"gaze_switch","direction":"{}"}}"#, direction)?;
        }
    }
    Ok(())
}

fn button_str(b: MouseButton) -> &'static str {
    match b {
        MouseButton::Left => "left",
        MouseButton::Right => "right",
        MouseButton::Middle => "middle",
    }
}
