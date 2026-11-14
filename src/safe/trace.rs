//! What a device did, as the safe layer saw it: transfers, waits, runs.
//! Set with [`DeviceConfig::trace`](super::DeviceConfig), read with
//! [`ComputeDevice::trace`](super::ComputeDevice::trace).

use std::collections::VecDeque;

/// One thing the safe layer asked of the device. Things that happen together
/// come in one batch: a download also waits, so it is `Download` then `Sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Host to device: `bytes` of a tensor's data.
    Upload { bytes: u64 },
    /// Device to host: `bytes` of a tensor's data.
    Download { bytes: u64 },
    /// Host to device: `bytes` of a node's meta words.
    Meta { bytes: u64 },
    /// The host waited for the device to finish the work queued so far.
    Sync,
    /// One run of a node of `shader`, queued.
    Run { shader: &'static str },
    /// `bytes` copied between tensors on the device.
    Copy { bytes: u64 },
    /// A tensor of `bytes` zeroed on the device.
    Clear { bytes: u64 },
    /// A recording queued again.
    Replay,
}

/// Totals of each kind of [`Event`]; constant memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    pub uploads: u64,
    pub upload_bytes: u64,
    pub downloads: u64,
    pub download_bytes: u64,
    pub metas: u64,
    pub meta_bytes: u64,
    pub syncs: u64,
    pub runs: u64,
    pub copies: u64,
    pub copy_bytes: u64,
    pub clears: u64,
    pub clear_bytes: u64,
    pub replays: u64,
}

impl Counts {
    fn add(&mut self, event: &Event) {
        match *event {
            Event::Upload { bytes } => {
                self.uploads += 1;
                self.upload_bytes += bytes;
            }
            Event::Download { bytes } => {
                self.downloads += 1;
                self.download_bytes += bytes;
            }
            Event::Meta { bytes } => {
                self.metas += 1;
                self.meta_bytes += bytes;
            }
            Event::Sync => self.syncs += 1,
            Event::Run { .. } => self.runs += 1,
            Event::Copy { bytes } => {
                self.copies += 1;
                self.copy_bytes += bytes;
            }
            Event::Clear { bytes } => {
                self.clears += 1;
                self.clear_bytes += bytes;
            }
            Event::Replay => self.replays += 1,
        }
    }
}

/// The last `cap` [`Event`]s in order, oldest first. Past `cap`, the oldest
/// one is dropped and counted, so a long run doesn't grow it without end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Events {
    cap: usize,
    events: VecDeque<Event>,
    dropped: u64,
}

impl Events {
    /// Keeps at most `cap` events.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            events: VecDeque::new(),
            dropped: 0,
        }
    }

    /// The most events it keeps.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The kept events, oldest first.
    pub fn events(&self) -> &VecDeque<Event> {
        &self.events
    }

    /// How many events were dropped to stay within `cap`.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    fn push(&mut self, event: Event) {
        if self.events.len() == self.cap {
            if self.events.pop_front().is_none() {
                // `cap` is 0: nothing is kept.
                self.dropped += 1;
                return;
            }
            self.dropped += 1;
        }
        self.events.push_back(event);
    }
}

/// How a device keeps its trace. Chosen in
/// [`DeviceConfig`](super::DeviceConfig); `Off` by default, which costs
/// nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Trace {
    /// Nothing is kept.
    #[default]
    Off,
    /// Totals; give `Counts::default()`.
    Counts(Counts),
    /// The events themselves; give `Events::new(cap)`.
    Events(Events),
}

impl Trace {
    pub(super) fn is_off(&self) -> bool {
        matches!(self, Trace::Off)
    }

    pub(super) fn note(&mut self, events: &[Event]) {
        match self {
            Trace::Off => {}
            Trace::Counts(counts) => events.iter().for_each(|e| counts.add(e)),
            Trace::Events(kept) => events.iter().for_each(|&e| kept.push(e)),
        }
    }

    // What it holds, leaving an empty one of the same kind and `cap`.
    pub(super) fn take(&mut self) -> Trace {
        let empty = match self {
            Trace::Off => Trace::Off,
            Trace::Counts(_) => Trace::Counts(Counts::default()),
            Trace::Events(kept) => Trace::Events(Events::new(kept.cap)),
        };
        std::mem::replace(self, empty)
    }
}

#[cfg(test)]
#[path = "../tests/trace.rs"]
pub(crate) mod tests;
