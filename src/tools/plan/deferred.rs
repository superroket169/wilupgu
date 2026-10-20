//! Values that become known later than the code that uses them: the sizes,
//! meta values and workgroup counts a plan is built from.
//!
//! - [`Resolvable`]: one value, fixed, given later by whoever holds its
//!   [`Resolver`] (the user, SysTopology), or computed by a [`Relation`].
//!   Reading it computes whatever it depends on, once.
//! - [`Dynamic`]: a value set again before every run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// The computation is `dyn`: it only runs while a plan is built, never on the
// hot path, and it can capture its inputs of any type and count.
type Compute<T> = Box<dyn Fn() -> Result<T, String> + Send + Sync>;

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
    pub fn resolver(&self) -> Result<Resolver<T>, String> {
        match &self.slot.source {
            Source::Computed(_) => {
                Err("resolver: this value is computed by a Relation".to_string())
            }
            Source::Given { resolver_taken } => {
                if resolver_taken.swap(true, Ordering::AcqRel) {
                    return Err("resolver: already taken for this value".to_string());
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
    pub fn value(&self) -> Result<&T, String> {
        if let Some(value) = self.slot.value.get() {
            return Ok(value);
        }
        match &self.slot.source {
            Source::Given { .. } => Err("a value was read before it was given".to_string()),
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

struct DynamicCell<T> {
    value: Option<T>,
    // Set by `set`, cleared by a run's read: a run may only read a value set since the last one.
    fresh: bool,
}

/// A value set again before every run, like an optimizer's step count. Clones
/// share it.
pub struct Dynamic<T> {
    cell: Arc<Mutex<DynamicCell<T>>>,
}

impl<T: Copy> Dynamic<T> {
    /// A value not set yet.
    pub fn new() -> Self {
        Self {
            cell: Arc::new(Mutex::new(DynamicCell {
                value: None,
                fresh: false,
            })),
        }
    }

    /// Sets the value for the next run; the last set before it wins.
    pub fn set(&self, value: T) {
        let mut cell = self.cell.lock().unwrap();
        cell.value = Some(value);
        cell.fresh = true;
    }

    // TODO: used by the plan's run once run.rs is rewritten.
    #[allow(dead_code)]
    pub(crate) fn key(&self) -> usize {
        Arc::as_ptr(&self.cell) as *const () as usize
    }

    // Called once per run per `Dynamic`, however many nodes use it.
    // TODO: used by the plan's run once run.rs is rewritten.
    #[allow(dead_code)]
    pub(crate) fn read_for_run(&self) -> Result<T, String> {
        let mut cell = self.cell.lock().unwrap();
        let value = cell
            .value
            .ok_or("a Dynamic was read before it was ever set")?;
        if !cell.fresh {
            return Err("a Dynamic was read again without a set since the last run".to_string());
        }
        cell.fresh = false;
        Ok(value)
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
#[path = "../../tests/deferred.rs"]
mod tests;
