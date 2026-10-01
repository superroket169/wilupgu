//! Every build-time runtime rule, each one its own function: one rule, one
//! check. Nothing here runs anything -- `graph.rs` calls these, then runs.

use crate::backend::{Backend, Node};
use crate::core::node::NodeSpec;
use crate::core::shader::{BindingRole, MetaKind};
use crate::core::tensor::TensorId;

pub(crate) fn validate_spec<N: Node>(spec: &NodeSpec) -> Result<bool, String> {
    N::validate_workgroups(spec.workgroups())
        .map_err(|e| format!("kernel `{}`: {e}", spec.shader().name))?;

    let layout = spec.shader().layout;
    let name = spec.shader().name;
    let mut covered = vec![false; layout.len()];
    let mut has_dynamic_meta = false;

    for b in spec.bindings() {
        let expected = layout.get(b.slot as usize).ok_or_else(|| {
            format!(
                "Tensor Mode Mismatch: kernel `{name}` binding slot {} out of range (kernel expects {} bindings)",
                b.slot,
                layout.len()
            )
        })?;
        if !expected.accepts(&b.mode) {
            return Err(format!(
                "Tensor Mode Mismatch: kernel `{name}` slot {} expects {:?}, got {:?}",
                b.slot, expected, b.mode
            ));
        }

        covered[b.slot as usize] = true;
        if let BindingRole::Meta {
            kind: MetaKind::Dynamic,
            ..
        } = &b.mode
        {
            has_dynamic_meta = true;
        }
    }

    if !covered.iter().all(|&c| c) {
        return Err(format!(
            "Tensor Mode Mismatch: kernel `{name}` expects {} binding(s), only {} were supplied",
            layout.len(),
            covered.iter().filter(|&&c| c).count()
        ));
    }
    Ok(has_dynamic_meta)
}

pub(crate) fn check_ownership<B: Backend>(ctx: &B, specs: &[NodeSpec]) -> Result<(), String> {
    for (i, spec) in specs.iter().enumerate() {
        for b in spec.bindings() {
            if !ctx.contains(b.tensor) {
                return Err(format!(
                    "Buffer ownership mismatch: dispatch {i} (kernel `{}`) binding slot {} \
                     names tensor {:?}, which isn't allocated on device {:?}",
                    spec.shader().name,
                    b.slot,
                    b.tensor,
                    ctx.device_id()
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn check_hazards(specs: &[NodeSpec]) -> Result<(), String> {
    let mut last_write: std::collections::HashMap<TensorId, usize> =
        std::collections::HashMap::new();

    for (i, spec) in specs.iter().enumerate() {
        for b in spec.bindings() {
            let id = b.tensor;

            match &b.mode {
                BindingRole::Output(_) | BindingRole::InOut(_) => {
                    if let Some(&prev) = last_write.get(&id) {
                        return Err(format!(
                            "Buffer hazard: dispatch {i} writes a buffer already written by \
                             dispatch {prev} with nothing establishing their order"
                        ));
                    }
                    last_write.insert(id, i);
                }
                BindingRole::Accumulate(_) => {
                    last_write.insert(id, i);
                }
                BindingRole::Input(_) | BindingRole::Meta { .. } => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/rules.rs"]
mod tests;
