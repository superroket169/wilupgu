//! Optional CPU<->GPU transfer log, gated by `WILUPGU_LOG_IO=<path>`. Off by
//! default: the hot path is one OnceLock read + branch when unset.
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<File>> {
    SINK.get_or_init(|| open(std::env::var("WILUPGU_LOG_IO").ok()))
        .as_ref()
}

fn open(path: Option<String>) -> Option<Mutex<File>> {
    let path = path?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("[wilupgu] WILUPGU_LOG_IO: cannot open {path}: {e}"));
    Some(Mutex::new(file))
}

fn write_line(out: &mut impl Write, dir: &str, bytes: u64) -> std::io::Result<()> {
    writeln!(out, "{dir}\t{bytes}")
}

/// `dir` is "htod" or "dtoh".
pub fn log(dir: &str, bytes: u64) {
    let Some(m) = sink() else { return };
    let _ = write_line(&mut *m.lock().unwrap(), dir, bytes);
}

#[cfg(test)]
#[path = "../tests/io_log.rs"]
mod tests;
