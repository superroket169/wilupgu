//! Optional CPU<->GPU transfer log, gated by `WILUPGU_LOG_IO=<path>`. Off by
//! default: the hot path is one OnceLock read + branch when unset.
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static SINK: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<std::fs::File>> {
    SINK.get_or_init(|| {
        let path = std::env::var("WILUPGU_LOG_IO").ok()?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("[wilupgu] WILUPGU_LOG_IO: cannot open {path}: {e}"));
        Some(Mutex::new(file))
    })
    .as_ref()
}

/// `dir` is "htod" or "dtoh"; `shader` is the triggering shader's name, or
/// "-" for a transfer not tied to one kernel dispatch (a plain Tensor
/// upload/readback).
pub fn log(dir: &str, shader: &str, bytes: u64) {
    let Some(m) = sink() else { return };
    let mut f = m.lock().unwrap();
    let _ = writeln!(f, "{dir}\t{shader}\t{bytes}");
}
