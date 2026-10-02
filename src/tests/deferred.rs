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

#[test]
fn fixed_is_resolved_from_the_start() {
    let r = Resolvable::fixed(1024u32);
    assert_eq!(*r.value(), 1024);
}

#[test]
#[should_panic(expected = "resolver() called more than once")]
fn fixed_has_no_resolver_left() {
    Resolvable::fixed(1u32).resolver();
}

#[test]
fn relation_resolves_every_output() {
    let (m, n, k) = (Resolvable::new(), Resolvable::new(), Resolvable::new());
    let (lengths, [len_a, len_b, len_c]) =
        Relation::new([&m, &n, &k], |[m, n, k]| [m * k, k * n, m * n]);
    m.resolver().resolve(2u32);
    n.resolver().resolve(3);
    k.resolver().resolve(5);
    lengths.resolve();
    assert_eq!(
        (*len_a.value(), *len_b.value(), *len_c.value()),
        (10, 15, 6)
    );
}

#[test]
fn relations_chain() {
    let batch = Resolvable::new();
    let (tokens, [len]) = Relation::new([&batch], |[b]| [b * 64 + 1]);
    let (bytes, [size]) = Relation::new([&len], |[l]| [l * 4]);
    batch.resolver().resolve(2u32);
    tokens.resolve();
    bytes.resolve();
    assert_eq!(*size.value(), 129 * 4);
}

#[test]
fn relation_works_for_floats() {
    let base = Resolvable::fixed(0.5f32);
    let (half, [h]) = Relation::new([&base], |[b]| [b / 2.0]);
    half.resolve();
    assert_eq!(*h.value(), 0.25);
}

#[test]
#[should_panic(expected = "read before it was resolved")]
fn relation_before_its_inputs_panics() {
    let batch: Resolvable<u32> = Resolvable::new();
    let (tokens, _) = Relation::new([&batch], |[b]| [b * 64]);
    tokens.resolve();
}

#[test]
#[should_panic(expected = "resolver() called more than once")]
fn relation_output_has_one_source_only() {
    let batch: Resolvable<u32> = Resolvable::new();
    let (_tokens, [len]) = Relation::new([&batch], |[b]| [b * 64]);
    len.resolver();
}
