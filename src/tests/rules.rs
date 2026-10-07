use super::*;
use crate::backend::dtype::DataKind;
use crate::backend::shader::Workgroups;
use crate::backend::tests::{device_with, ToyNode, COPY_SHADER, META_SHADER};
use crate::backend::Storage;
use crate::tools::core::deferred::{Dynamic, Resolvable};
use crate::tools::core::node::{Binding, MetaSource, MetaValue};

fn meta_spec(meta: Vec<MetaValue>) -> NodeSpec {
    NodeSpec::new(&META_SHADER, meta, vec![], Workgroups::linear(1))
}

#[test]
fn meta_matching_the_declaration_passes() {
    let spec = meta_spec(vec![
        MetaValue::Uint(MetaSource::Once(Resolvable::fixed(4))),
        MetaValue::Float(MetaSource::PerRun(Dynamic::new())),
    ]);
    assert!(check_meta(&spec).is_ok());
}

#[test]
fn meta_with_a_missing_field_is_rejected() {
    let spec = meta_spec(vec![MetaValue::Uint(MetaSource::Once(Resolvable::fixed(
        4,
    )))]);
    let err = check_meta(&spec).unwrap_err();
    assert!(err.contains("declares 2 meta field(s), got 1"), "{err}");
}

#[test]
fn meta_of_the_wrong_type_is_rejected() {
    let spec = meta_spec(vec![
        MetaValue::Uint(MetaSource::Once(Resolvable::fixed(4))),
        MetaValue::Uint(MetaSource::Once(Resolvable::fixed(1))),
    ]);
    let err = check_meta(&spec).unwrap_err();
    assert!(err.contains("field `lr` is Float, got Uint"), "{err}");
}

#[test]
fn slot_zero_is_not_a_tensor_slot() {
    let spec = NodeSpec::new(
        &COPY_SHADER,
        vec![],
        vec![
            Binding::new(0, TensorId::new(), BindingRole::Input(DataKind::F32)),
            Binding::new(2, TensorId::new(), BindingRole::Output(DataKind::F32)),
        ],
        Workgroups::linear(1),
    );
    let err = validate_spec::<ToyNode>(&spec).unwrap_err();
    assert!(err.contains("slot 0 out of range"), "{err}");
}

#[test]
fn check_unused_passes_when_only_the_table_holds_it() {
    let (dev, ids) = device_with(1);
    assert!(check_unused(dev.table(), ids[0], dev.device_id()).is_ok());
}

#[test]
fn check_unused_rejects_a_buffer_a_node_still_holds() {
    let (dev, ids) = device_with(1);
    // stands in for a built node's reference
    let held = dev.table().get(ids[0]).unwrap();
    let err = check_unused(dev.table(), ids[0], dev.device_id()).unwrap_err();
    assert!(err.starts_with("Buffer still in use"), "{err}");
    drop(held);
    assert!(check_unused(dev.table(), ids[0], dev.device_id()).is_ok());
}
