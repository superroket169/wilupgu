use super::*;

#[test]
fn resolves_and_reads() {
    let r = Resolvable::new();
    r.resolver().resolve(42);
    assert_eq!(*r.value(), 42);
}

#[test]
#[should_panic(expected = "read before it was resolved")]
fn value_before_resolve_panics() {
    let r: Resolvable<u32> = Resolvable::new();
    r.value();
}

#[test]
#[should_panic(expected = "resolver() called more than once")]
fn second_resolver_panics() {
    let r: Resolvable<u32> = Resolvable::new();
    let _first = r.resolver();
    let _second = r.resolver();
}

#[test]
fn clones_share_the_same_resolved_value() {
    let a = Resolvable::new();
    let b = a.clone();
    a.resolver().resolve("batch_size");
    assert_eq!(*b.value(), "batch_size");
}
