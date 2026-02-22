// KVM TCP client — sends InputEvent as JSON lines to KVM server

use crate::gesture::InputEvent;
use anyhow::{Context, Result};
use std::io::Write;
use std::net::TcpStream;

pub struct KvmClient {
    stream: TcpStream,
}

impl KvmClient {
    pub fn connect(addr: &str) -> Result<Self> {
        let stream = TcpStream::connect(addr).context("Failed to connect to KVM server")?;
        stream
            .set_nodelay(true)
            .context("Failed to set TCP_NODELAY")?;
        tracing::info!("KVM connected to {}", addr);
        Ok(Self { stream })
    }

    pub fn send(&mut self, event: &InputEvent) -> Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.stream, "{}", json)?;
        Ok(())
    }

    pub fn send_batch(&mut self, events: &[InputEvent]) -> Result<()> {
        for event in events {
            self.send(event)?;
        }
        Ok(())
    }
}
