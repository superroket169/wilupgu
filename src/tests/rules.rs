use super::*;
use crate::backend::tests::{ToyNode, COPY_SHADER, META_SHADER};
use crate::core::deferred::{Dynamic, Resolvable};
use crate::core::dtype::DataKind;
use crate::core::node::{Binding, MetaSource, MetaValue};
use crate::core::shader::Workgroups;

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
