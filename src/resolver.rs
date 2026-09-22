use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

struct Slot<T> {
    value: OnceLock<T>,
    resolver_taken: AtomicBool,
}

pub struct Resolvable<T> {
    slot: Arc<Slot<T>>,
}

impl<T> Resolvable<T> {
    pub fn new() -> Self {
        Self {
            slot: Arc::new(Slot {
                value: OnceLock::new(),
                resolver_taken: AtomicBool::new(false),
            }),
        }
    }

    /// Hands out the one-shot write authority
    /// Panics if called more than once on this (or a cloned) handle
    pub fn resolver(&self) -> Resolver<T> {
        if self.slot.resolver_taken.swap(true, Ordering::AcqRel) {
            panic!("Resolvable::resolver() called more than once");
        }
        Resolver {
            slot: self.slot.clone(),
        }
    }

    /// Panics if read before a `Resolver` has resolved it
    pub fn value(&self) -> &T {
        self.slot
            .value
            .get()
            .expect("Resolvable read before it was resolved")
    }
}

impl<T> Clone for Resolvable<T> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot.clone(),
        }
    }
}

impl<T> Default for Resolvable<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Resolver<T> {
    slot: Arc<Slot<T>>,
}

impl<T> Resolver<T> {
    pub fn resolve(self, value: T) {
        self.slot
            .value
            .set(value)
            .ok()
            .expect("Resolvable already resolved");
    }
}

#[cfg(test)]
mod tests {
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
}
