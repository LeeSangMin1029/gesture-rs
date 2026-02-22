// KVM TCP client — sends InputEvent to KVM server
//
// TODO: Implement after gesture + gaze modules are working

use crate::gesture::InputEvent;
use anyhow::Result;

#[allow(dead_code)]
pub struct KvmClient {
    // stream: tokio::net::TcpStream,
}

#[allow(dead_code)]
impl KvmClient {
    pub async fn connect(_addr: &str) -> Result<Self> {
        tracing::info!("KvmClient: stub (not yet implemented)");
        Ok(Self {})
    }

    pub async fn send(&mut self, _event: &InputEvent) -> Result<()> {
        Ok(())
    }
}
