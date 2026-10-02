use super::*;
use crate::core::shader::{Shader, ShaderCode, Workgroups};

static EMPTY: Shader = Shader {
    name: "Empty",
    layout: &[],
    shader_code: ShaderCode::NONE,
};

fn spec() -> NodeSpec {
    NodeSpec::new(&EMPTY, vec![], Workgroups::linear(1))
}

#[test]
fn accepts_a_complete_placement() {
    let specs = [spec(), spec()];
    let (d0, d1) = (DeviceId::new(), DeviceId::new());
    let mut p = Placement::new();
    p.assign(&specs[0], vec![d0, d1]).unwrap();
    p.assign(&specs[1], vec![d1]).unwrap();
    assert!(p.validate(&specs).is_ok());
}

#[test]
fn assign_rejects_no_device() {
    let err = Placement::new().assign(&spec(), vec![]).unwrap_err();
    assert!(err.contains("no device"), "{err}");
}

#[test]
fn assign_rejects_duplicate_device() {
    let err = Placement::new()
        .assign(&spec(), {
            let d = DeviceId::new();
            vec![d, d]
        })
        .unwrap_err();
    assert!(err.contains("twice"), "{err}");
}

#[test]
fn rejected_assign_leaves_nothing_behind() {
    let s = spec();
    let mut p = Placement::new();
    let d = DeviceId::new();
    let _ = p.assign(&s, vec![d, d]);
    assert!(p.devices(s.id()).is_none());
}

#[test]
fn validate_rejects_unplaced_node() {
    let specs = [spec()];
    let err = Placement::new().validate(&specs).unwrap_err();
    assert!(err.contains("has no placement"), "{err}");
}

#[test]
fn validate_rejects_node_outside_the_list() {
    let specs = [spec()];
    let stranger = spec();
    let mut p = Placement::new();
    let d = DeviceId::new();
    p.assign(&specs[0], vec![d]).unwrap();
    p.assign(&stranger, vec![d]).unwrap();
    let err = p.validate(&specs).unwrap_err();
    assert!(err.contains("isn't in the spec list"), "{err}");
}
