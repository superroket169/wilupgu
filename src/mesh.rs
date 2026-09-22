/// How a `NodeSpec` is spread across a mesh's devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelity {
    /// Every device runs its own full, independent copy
    /// Each device's own tensors resolve their sizes independently
    Data,
    /// One logical op is split across devices, each computing its own
    /// slice in parallel (nobody waits)
    ///
    /// sizes are proportioned from a shared total, and the slices need combining afterwards
    /// (a device-to-device or collective transfer, not this enum's concern).
    Tensor,
    /// Sequential stages
    /// a later stage genuinely needs an earlier stage's output before it can run.
    Pipeline,
}
