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

impl<T: std::fmt::Debug> std::fmt::Debug for Resolvable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.slot.value.get() {
            Some(v) => f.debug_tuple("Resolvable").field(v).finish(),
            None => f.write_str("Resolvable(<unresolved>)"),
        }
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
#[path = "tests/resolver.rs"]
mod tests;
