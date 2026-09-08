// Native Vulkan backend (via the `vulkano` crate), reviving what filuplex
// (~/Documents/Codes/filuplex) was reaching for -- but this time WGSL stays
// the one and only shader source (translated to SPIR-V at pipeline-build time
// via `naga`, cached like WgpuBackend already caches pipelines), and node-to-
// node hazards are tracked from each node's own `Binding`/`TensorMode` list so
// real `vkCmdPipelineBarrier`s can be inserted only where actually needed --
// instead of wgpu's automatic (and, on this hardware, buggy) barrier
// insertion, and instead of filuplex's own answer of a full device-wide
// wait_idle() after every dispatch.
//
// Empty on purpose for now -- this is the reserved slot for that work.
