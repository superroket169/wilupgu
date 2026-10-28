use super::*;

#[test]
fn given_value_reads_back() {
    let r = Resolvable::new();
    r.resolver().unwrap().resolve(42u32);
    assert_eq!(r.value(), Ok(&42));
}

#[test]
fn value_before_it_is_given_fails_then_succeeds_once_given() {
    let r: Resolvable<u32> = Resolvable::new();
    let resolver = r.resolver().unwrap();
    assert!(r.value().is_err());
    resolver.resolve(7);
    assert_eq!(r.value(), Ok(&7));
}

#[test]
fn resolver_can_be_taken_once() {
    let r: Resolvable<u32> = Resolvable::new();
    let _first = r.resolver().unwrap();
    assert!(r.resolver().is_err());
    assert!(
        r.clone().resolver().is_err(),
        "a clone shares the taken resolver"
    );
}

#[test]
fn clones_share_the_given_value() {
    let a = Resolvable::new();
    let b = a.clone();
    a.resolver().unwrap().resolve("batch_size");
    assert_eq!(b.value(), Ok(&"batch_size"));
}

#[test]
fn fixed_is_known_from_the_start_and_has_no_resolver() {
    let r = Resolvable::fixed(1024u32);
    assert_eq!(r.value(), Ok(&1024));
    assert!(r.resolver().is_err());
}

#[test]
fn from_value_is_fixed() {
    let r: Resolvable<usize> = 3.into();
    assert_eq!(r.value(), Ok(&3));
    assert!(r.resolver().is_err());
}

#[test]
fn relation_computes_from_its_inputs() {
    let (m, k) = (Resolvable::new(), Resolvable::new());
    let len = Relation::new([&m, &k], |[m, k]| m * k);
    m.resolver().unwrap().resolve(2u32);
    k.resolver().unwrap().resolve(5);
    assert_eq!(len.value(), Ok(&10));
}

#[test]
fn relation_needs_no_resolve_call_and_follows_its_chain() {
    let batch = Resolvable::new();
    let tokens = Relation::new([&batch], |[b]| b * 64 + 1);
    let bytes = Relation::new([&tokens], |[t]| t * 4);
    batch.resolver().unwrap().resolve(2u32);
    // read the end of the chain first: the middle is computed on the way
    assert_eq!(bytes.value(), Ok(&(129 * 4)));
    assert_eq!(tokens.value(), Ok(&129));
}

#[test]
fn relation_fails_while_an_input_is_missing_then_succeeds() {
    let batch: Resolvable<u32> = Resolvable::new();
    let resolver = batch.resolver().unwrap();
    let tokens = Relation::new([&batch], |[b]| b * 64);
    assert!(tokens.value().is_err());
    resolver.resolve(3);
    assert_eq!(tokens.value(), Ok(&192));
}

#[test]
fn relation_output_can_have_another_type() {
    let n = Resolvable::fixed(1000u32);
    let workgroups = Relation::new([&n], |[n]| (n.div_ceil(256), 1u32, 1u32));
    let inv = Relation::new([&n], |[n]| 1.0 / n as f32);
    assert_eq!(workgroups.value(), Ok(&(4, 1, 1)));
    assert_eq!(inv.value(), Ok(&0.001));
}

#[test]
fn relation_computes_once() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let calls = Arc::new(AtomicU32::new(0));
    let counted = calls.clone();
    let n = Resolvable::fixed(3u32);
    let doubled = Relation::new([&n], move |[n]| {
        counted.fetch_add(1, Ordering::Relaxed);
        n * 2
    });
    let copy = doubled.clone();
    assert_eq!(doubled.value(), Ok(&6));
    assert_eq!(copy.value(), Ok(&6));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn relation_output_has_no_resolver() {
    let n = Resolvable::fixed(1u32);
    assert!(Relation::new([&n], |[n]| n).resolver().is_err());
}

#[test]
fn relation_with_no_inputs_is_a_constant() {
    let r = Relation::new::<u32, _, 0>([], |[]| 9u32);
    assert_eq!(r.value(), Ok(&9));
}

#[test]
fn dynamic_reads_what_was_set() {
    let lr = Dynamic::new();
    lr.set(0.1f32);
    assert_eq!(lr.read(), Ok(0.1));
}

#[test]
fn dynamic_read_before_set_fails() {
    assert!(Dynamic::<u32>::new().read().is_err());
}

#[test]
fn dynamic_reads_the_same_value_until_set_again() {
    let step = Dynamic::new();
    step.set(1u32);
    assert_eq!(step.read(), Ok(1));
    assert_eq!(step.read(), Ok(1));
    step.set(2);
    assert_eq!(step.read(), Ok(2));
}

#[test]
fn dynamic_clones_share_one_value() {
    let step = Dynamic::new();
    let handle = step.clone();
    handle.set(7u32);
    assert_eq!(step.read(), Ok(7));
}
