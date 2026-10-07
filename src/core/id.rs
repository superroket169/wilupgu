use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque identity, tagged with what it identifies at the type level only (`PhantomData`, zero bytes)
/// A `DeviceId` where a `TensorId` is expected doesn't compile.
/// One counter behind every tag, so ids are unique across all of them too.
///
/// `fn() -> T` keeps the id `Send + Sync + Copy` no matter what `T` is; the
/// traits below are written by hand because `derive` would demand them of `T`.
pub struct GlobalId<T> {
    raw: u64,
    _of: PhantomData<fn() -> T>,
}

/// Tag for `DeviceId`. Stand-in until each backend has its own concrete device-id source.
pub enum DeviceTag {}

/// Names one device. Plans (placement, spread) hold this instead of a `&Device`.
pub type DeviceId = GlobalId<DeviceTag>;

/// Tag for `TensorId`.
pub enum TensorTag {}

/// Names one tensor. It is the key of a device's `BufferTable`, and a node's
/// bindings point to tensors with it.
pub type TensorId = GlobalId<TensorTag>;

impl<T> GlobalId<T> {
    /// Crate-only, so users can't mint ids
    /// they get them from specs and devices.
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            raw: NEXT.fetch_add(1, Ordering::Relaxed),
            _of: PhantomData,
        }
    }
}

impl<T> Clone for GlobalId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GlobalId<T> {}

impl<T> PartialEq for GlobalId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<T> Eq for GlobalId<T> {}

impl<T> Hash for GlobalId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<T> std::fmt::Debug for GlobalId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GlobalId({})", self.raw)
    }
}
