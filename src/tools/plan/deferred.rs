//! Values that become known later than the code that uses them: the sizes,
//! meta values and workgroup counts a plan is built from.
//!
//! - [`Resolvable`]: one value, fixed, given later by whoever holds its
//!   [`Resolver`] (the user, SysTopology), or computed by a [`Relation`].
//!   Reading it computes whatever it depends on, once.
//! - [`Dynamic`]: a value that changes between runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::Error;

// The computation is `dyn`: it only runs while a plan is built, never on the
// hot path, and it can capture its inputs of any type and count.
type Compute<T> = Box<dyn Fn() -> Result<T, Error> + Send + Sync>;

enum Source<T> {
    // Filled once by the one `Resolver`; `fixed` takes the resolver itself.
    Given { resolver_taken: AtomicBool },
    Computed(Compute<T>),
}

struct Slot<T> {
    value: OnceLock<T>,
    source: Source<T>,
}

/// One value that may not be known yet. Clones share it: a value given or
/// computed through one clone is seen by all of them.
pub struct Resolvable<T> {
    slot: Arc<Slot<T>>,
}

impl<T> Resolvable<T> {
    /// A value given later, through [`Resolvable::resolver`].
    pub fn new() -> Self {
        Self::with_source(Source::Given {
            resolver_taken: AtomicBool::new(false),
        })
    }

    /// A value known from the start. It has no resolver.
    pub fn fixed(value: T) -> Self {
        let r = Self::new();
        r.resolver()
            .expect("a new Resolvable's resolver is free")
            .resolve(value);
        r
    }

    fn with_source(source: Source<T>) -> Self {
        Self {
            slot: Arc::new(Slot {
                value: OnceLock::new(),
                source,
            }),
        }
    }

    /// The one way to give this value. Fails if it was already taken, by
    /// this handle or a clone, or if the value is computed by a [`Relation`].
    pub fn resolver(&self) -> Result<Resolver<T>, Error> {
        match &self.slot.source {
            Source::Computed(_) => Err(Error::Computed),
            Source::Given { resolver_taken } => {
                if resolver_taken.swap(true, Ordering::AcqRel) {
                    return Err(Error::ResolverTaken);
                }
                Ok(Resolver {
                    slot: self.slot.clone(),
                })
            }
        }
    }

    /// The value. A computed one is computed on the first read, from its
    /// inputs (which are read the same way), and kept.
    ///
    /// Fails if this value, or one it is computed from, wasn't given yet.
    /// A later read can still succeed once it is.
    pub fn value(&self) -> Result<&T, Error> {
        if let Some(value) = self.slot.value.get() {
            return Ok(value);
        }
        match &self.slot.source {
            Source::Given { .. } => Err(Error::NotGivenYet),
            Source::Computed(compute) => {
                let value = compute()?;
                // A racing read may have kept its own result; both are equal.
                Ok(self.slot.value.get_or_init(|| value))
            }
        }
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

impl<T> From<T> for Resolvable<T> {
    fn from(value: T) -> Self {
        Self::fixed(value)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Resolvable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.slot.value.get() {
            Some(v) => f.debug_tuple("Resolvable").field(v).finish(),
            None => f.write_str("Resolvable(<not known yet>)"),
        }
    }
}

/// The one-shot right to give a [`Resolvable`] its value.
pub struct Resolver<T> {
    slot: Arc<Slot<T>>,
}

impl<T> Resolver<T> {
    /// Gives the value; every clone of the `Resolvable` sees it.
    pub fn resolve(self, value: T) {
        // Only this resolver can set a given value, and `resolve` takes it by value.
        let _ = self.slot.value.set(value);
    }
}

/// A value computed from others.
pub struct Relation;

impl Relation {
    /// A [`Resolvable`] computed by `f` from `inputs`, when it is first read.
    /// Its inputs share one type; its own can be another, like a workgroup
    /// count from an element count.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<In, Out, const I: usize>(
        inputs: [&Resolvable<In>; I],
        f: impl Fn([In; I]) -> Out + Send + Sync + 'static,
    ) -> Resolvable<Out>
    where
        In: Clone + Send + Sync + 'static,
        Out: Send + Sync + 'static,
    {
        let inputs = inputs.map(Resolvable::clone);
        Resolvable::with_source(Source::Computed(Box::new(move || {
            let values: Vec<In> = inputs
                .iter()
                .map(|input| input.value().cloned())
                .collect::<Result<_, _>>()?;
            let values: [In; I] = values
                .try_into()
                .unwrap_or_else(|_| unreachable!("one value per input"));
            Ok(f(values))
        })))
    }
}

/// A value that changes between runs, like an optimizer's step count. Clones
/// share it.
pub struct Dynamic<T> {
    value: Arc<Mutex<Option<T>>>,
}

impl<T: Copy> Dynamic<T> {
    /// A value not set yet.
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    /// Sets the value; whoever reads it next sees this one.
    pub fn set(&self, value: T) {
        *self.value.lock().unwrap() = Some(value);
    }

    // The value now. Fails only if it was never set.
    pub(crate) fn read(&self) -> Result<T, Error> {
        self.value.lock().unwrap().ok_or(Error::DynamicUnset)
    }
}

impl<T> Clone for Dynamic<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
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
        match &*self.value.lock().unwrap() {
            Some(v) => f.debug_tuple("Dynamic").field(v).finish(),
            None => f.write_str("Dynamic(<unset>)"),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/deferred.rs"]
mod tests;
