// Latest-frame-only channel: camera thread writes, main thread reads.
// Old frames are dropped — always processes the most recent frame.

use std::sync::{Arc, Condvar, Mutex};

pub type Frame = (Arc<Vec<u8>>, u32, u32); // (rgb_data, width, height)

struct Inner {
    frame: Option<Frame>,
    closed: bool,
}

pub struct FrameSender {
    inner: Arc<(Mutex<Inner>, Condvar)>,
}

pub struct FrameReceiver {
    inner: Arc<(Mutex<Inner>, Condvar)>,
}

pub fn channel() -> (FrameSender, FrameReceiver) {
    let inner = Arc::new((
        Mutex::new(Inner {
            frame: None,
            closed: false,
        }),
        Condvar::new(),
    ));
    (
        FrameSender {
            inner: inner.clone(),
        },
        FrameReceiver { inner },
    )
}

impl FrameSender {
    /// Replace the current frame (old frame is dropped)
    pub fn send(&self, rgb: Vec<u8>, w: u32, h: u32) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        guard.frame = Some((Arc::new(rgb), w, h));
        cvar.notify_one();
    }
}

impl Drop for FrameSender {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        guard.closed = true;
        cvar.notify_one();
    }
}

impl FrameReceiver {
    /// Block until a new frame is available, return None if channel closed
    pub fn recv(&self) -> Option<Frame> {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        loop {
            if let Some(frame) = guard.frame.take() {
                return Some(frame);
            }
            if guard.closed {
                return None;
            }
            guard = cvar.wait(guard).unwrap();
        }
    }
}
