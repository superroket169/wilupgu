//! Every build-time runtime rule, each one its own function: one rule, one
//! check. Nothing here runs anything -- `graph.rs` calls these, then runs.

use crate::backend::{Backend, Node};
use crate::core::node::NodeSpec;
use crate::core::shader::BindingRole;
use crate::core::tensor::TensorId;

pub(crate) fn validate_spec<N: Node>(spec: &NodeSpec) -> Result<(), String> {
    N::validate_workgroups(spec.workgroups())
        .map_err(|e| format!("kernel `{}`: {e}", spec.shader().name))?;

    let layout = spec.shader().layout;
    let name = spec.shader().name;
    let mut covered = vec![false; layout.len()];

    for b in spec.bindings() {
        // slot 0 is the meta; tensor slots are 1..=layout.len()
        let index = (b.slot as usize).checked_sub(1);
        let expected = index.and_then(|i| layout.get(i)).ok_or_else(|| {
            format!(
                "Tensor Mode Mismatch: kernel `{name}` binding slot {} out of range (kernel expects tensor slots 1..={})",
                b.slot,
                layout.len()
            )
        })?;
        if *expected != b.mode {
            return Err(format!(
                "Tensor Mode Mismatch: kernel `{name}` slot {} expects {:?}, got {:?}",
                b.slot, expected, b.mode
            ));
        }

        covered[b.slot as usize - 1] = true;
    }

    if !covered.iter().all(|&c| c) {
        return Err(format!(
            "Tensor Mode Mismatch: kernel `{name}` expects {} binding(s), only {} were supplied",
            layout.len(),
            covered.iter().filter(|&&c| c).count()
        ));
    }
    Ok(())
}

pub(crate) fn check_meta(spec: &NodeSpec) -> Result<(), String> {
    let declared = spec.shader().meta;
    let name = spec.shader().name;
    if spec.meta().len() != declared.len() {
        return Err(format!(
            "Meta mismatch: shader `{name}` declares {} meta field(s), got {}",
            declared.len(),
            spec.meta().len()
        ));
    }
    for (field, value) in declared.iter().zip(spec.meta()) {
        if field.ty != value.ty() {
            return Err(format!(
                "Meta mismatch: shader `{name}` field `{}` is {:?}, got {:?}",
                field.name,
                field.ty,
                value.ty()
            ));
        }
    }
    Ok(())
}

pub(crate) fn check_shader_code<B: Backend>(specs: &[NodeSpec]) -> Result<(), String> {
    for (i, spec) in specs.iter().enumerate() {
        if !spec.shader().shader_code.has(B::FORMAT) {
            return Err(format!(
                "Missing shader code: dispatch {i} (kernel `{}`) has no {:?} code, \
                 the format this backend runs",
                spec.shader().name,
                B::FORMAT
            ));
        }
    }
    Ok(())
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
                BindingRole::Input(_) => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/rules.rs"]
mod tests;
