use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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

    /// Same as:
    /// ```
    /// # use wilupgu::core::deferred::Resolvable;
    /// let n = Resolvable::new();
    /// n.resolver().resolve(1024);
    /// ```
    pub fn fixed(value: T) -> Self {
        let r = Self::new();
        r.resolver().resolve(value);
        r
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

pub struct Relation<T, const I: usize, const O: usize> {
    inputs: [Resolvable<T>; I],
    outputs: [Resolver<T>; O],
    relation: fn([T; I]) -> [T; O],
}

impl<T: Copy, const I: usize, const O: usize> Relation<T, I, O> {
    // The outputs' resolvers are taken here, so this relation is their only possible source.
    pub fn new(
        inputs: [&Resolvable<T>; I],
        relation: fn([T; I]) -> [T; O],
    ) -> (Self, [Resolvable<T>; O]) {
        let outputs: [Resolvable<T>; O] = std::array::from_fn(|_| Resolvable::new());
        let relation = Self {
            inputs: inputs.map(Resolvable::clone),
            outputs: std::array::from_fn(|i| outputs[i].resolver()),
            relation,
        };
        (relation, outputs)
    }

    pub fn resolve(self) {
        let values = self.inputs.each_ref().map(|r| *r.value());
        let results = (self.relation)(values);
        for (resolver, value) in self.outputs.into_iter().zip(results) {
            resolver.resolve(value);
        }
    }
}

struct DynamicCell<T> {
    value: Option<T>,
    // Set by `set`, cleared by a run's read: a run may only read a value set since the last one.
    fresh: bool,
}

pub struct Dynamic<T> {
    cell: Arc<Mutex<DynamicCell<T>>>,
}

impl<T: Copy> Dynamic<T> {
    pub fn new() -> Self {
        Self {
            cell: Arc::new(Mutex::new(DynamicCell {
                value: None,
                fresh: false,
            })),
        }
    }

    pub fn set(&self, value: T) {
        let mut cell = self.cell.lock().unwrap();
        cell.value = Some(value);
        cell.fresh = true;
    }

    pub(crate) fn key(&self) -> usize {
        Arc::as_ptr(&self.cell) as *const () as usize
    }

    // Called once per run per `Dynamic`, however many nodes use it.
    pub(crate) fn read_for_run(&self) -> T {
        let mut cell = self.cell.lock().unwrap();
        let value = cell.value.expect("Dynamic read before it was ever set");
        assert!(
            cell.fresh,
            "Dynamic read again without a set since the last run"
        );
        cell.fresh = false;
        value
    }
}

impl<T> Clone for Dynamic<T> {
    fn clone(&self) -> Self {
        Self {
            cell: self.cell.clone(),
        }
    }
}

impl<T: Copy> Default for Dynamic<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Dynamic<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cell.lock().unwrap().value {
            Some(v) => f.debug_tuple("Dynamic").field(v).finish(),
            None => f.write_str("Dynamic(<unset>)"),
        }
    }
}

#[cfg(test)]
#[path = "../tests/deferred.rs"]
mod tests;
