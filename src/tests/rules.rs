use super::*;
use crate::backend::tests::{ToyNode, META_SHADER};
use crate::node::Binding;
use crate::resolver::Resolvable;
use crate::shader::{MetaField, ResolvedSize, Workgroups};

#[test]
fn maximized_meta_is_not_flagged_dynamic() {
    let spec = NodeSpec::new(
        &META_SHADER,
        vec![Binding::new(
            0,
            TensorId::new(),
            BindingRole::Meta {
                fields: &[MetaField::Uint],
                kind: MetaKind::Maximized(ResolvedSize {
                    size: Resolvable::new(),
                    multiplier: 1,
                    coefficient: 0,
                }),
            },
        )],
        Workgroups::linear(1),
    );
    let has_dynamic_meta = validate_spec::<ToyNode>(&spec).unwrap();
    assert!(
        !has_dynamic_meta,
        "Maximized must not be treated as Dynamic"
    );
}
