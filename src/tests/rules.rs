use super::*;
use crate::backend::tests::{ToyNode, META_SHADER};
use crate::core::node::Binding;
use crate::core::shader::{MetaField, Workgroups};

#[test]
fn static_meta_is_not_flagged_dynamic() {
    let spec = NodeSpec::new(
        &META_SHADER,
        vec![Binding::new(
            0,
            TensorId::new(),
            BindingRole::Meta {
                fields: &[MetaField::Uint],
                kind: MetaKind::Static,
            },
        )],
        Workgroups::linear(1),
    );
    let has_dynamic_meta = validate_spec::<ToyNode>(&spec).unwrap();
    assert!(!has_dynamic_meta, "Static must not be treated as Dynamic");
}
