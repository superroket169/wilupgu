use super::*;
use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{Shader, Workgroups};
use crate::backend::{Backend, Topology};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};
use crate::tools::builtins::{ADD, SCALE};

fn device(trace: Trace) -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig {
            trace,
            ..DeviceConfig::default()
        },
    )
}

fn kept(trace: Trace) -> Vec<Event> {
    match trace {
        Trace::Events(events) => events.events().iter().copied().collect(),
        other => panic!("not an events trace: {other:?}"),
    }
}

#[test]
fn off_keeps_nothing() {
    let dev = device(Trace::Off);
    let x = dev.tensor_from(&F32::wrap(vec![1.0])).unwrap();
    let _ = x.download();
    assert_eq!(dev.trace(), Trace::Off);
}

#[test]
fn counts_add_up_every_kind_of_work() {
    let dev = device(Trace::Counts(Counts::default()));
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    let mut y = dev.alloc(DataKind::F32, 2).unwrap();
    let shader = SCALE.shader_for::<CpuBackend>().unwrap();
    let meta = [MetaValue::Uint(2), MetaValue::Float(2.0)];
    let mut node = dev.node(&shader, &meta, Workgroups::linear(1)).unwrap();
    node.update_meta(&meta).unwrap();
    dev.scope(|s| {
        let (xs, ys, n) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut node)?);
        s.execute(n, &[Arg::Write(xs)])?;
        s.copy(xs, 0, ys, 0, 2)
    })
    .unwrap();
    x.copy_into(0, &mut y, 0, 2).unwrap();
    y.zero().unwrap();
    let _ = x.download();

    let Trace::Counts(counts) = dev.trace() else {
        panic!("a counts trace")
    };
    assert_eq!(
        counts,
        Counts {
            uploads: 1,
            upload_bytes: 8,
            downloads: 1,
            download_bytes: 8,
            metas: 1,
            meta_bytes: 8,
            // the scope's end, `copy_into`, `zero`, the download
            syncs: 4,
            runs: 1,
            // two f32s: 8 bytes each copy, 8 bytes zeroed
            copies: 2,
            copy_bytes: 16,
            clears: 1,
            clear_bytes: 8,
            replays: 0,
        }
    );
}

#[test]
fn events_keep_order_and_drop_past_cap() {
    let mut trace = Trace::Events(Events::new(3));
    trace.note(&[Event::Sync, Event::Upload { bytes: 1 }]);
    trace.note(&[Event::Upload { bytes: 2 }, Event::Upload { bytes: 3 }]);
    trace.note(&[Event::Replay]);
    let Trace::Events(events) = &trace else {
        unreachable!()
    };
    assert_eq!(events.dropped(), 2);
    assert_eq!(
        kept(trace),
        vec![
            Event::Upload { bytes: 2 },
            Event::Upload { bytes: 3 },
            Event::Replay
        ]
    );

    let mut none = Trace::Events(Events::new(0));
    none.note(&[Event::Sync, Event::Sync]);
    let Trace::Events(events) = none else {
        unreachable!()
    };
    assert_eq!((events.events().len(), events.dropped()), (0, 2));
}

#[test]
fn take_trace_leaves_an_empty_one_of_the_same_kind() {
    let dev = device(Trace::Events(Events::new(8)));
    let _ = dev.tensor_from(&F32::wrap(vec![1.0])).unwrap();
    assert_eq!(kept(dev.take_trace()), vec![Event::Upload { bytes: 4 }]);
    assert_eq!(dev.trace(), Trace::Events(Events::new(8)));

    let dev = device(Trace::Counts(Counts::default()));
    let _ = dev.tensor_from(&F32::wrap(vec![1.0])).unwrap();
    assert_ne!(dev.take_trace(), Trace::Counts(Counts::default()));
    assert_eq!(dev.trace(), Trace::Counts(Counts::default()));
}

// A captured run doesn't run, so it isn't a `Run`; its replay is a `Replay`.
#[test]
fn a_capture_runs_nothing_and_a_replay_is_noted() {
    let dev = device(Trace::Events(Events::new(16)));
    let mut x = dev.alloc(DataKind::F32, 1).unwrap();
    let shader = SCALE.shader_for::<CpuBackend>().unwrap();
    let meta = [MetaValue::Uint(1), MetaValue::Float(2.0)];
    let mut node = dev.node(&shader, &meta, Workgroups::linear(1)).unwrap();
    let rec = dev
        .scope(|s| {
            let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
            s.capture(|| s.execute(n, &[Arg::Write(x)]))
        })
        .unwrap();
    assert_eq!(kept(dev.take_trace()), vec![Event::Sync]);
    dev.scope(|s| {
        s.bind(&mut x)?;
        s.bind_node(&mut node)?;
        s.replay(&rec)
    })
    .unwrap();
    assert_eq!(kept(dev.take_trace()), vec![Event::Replay, Event::Sync]);
}

#[test]
fn a_steady_loop_moves_nothing_on_the_cpu() {
    let shader = ADD.shader_for::<CpuBackend>().unwrap();
    a_steady_loop_moves_nothing(CpuBackend::attach(CpuInfo).unwrap(), &shader);
}

/// A loop of runs on tensors already on the device moves no data and waits
/// for nothing: the trace holds the runs, then the one `Sync` at the
/// scope's end. `add` is the `add` builtin in `raw`'s format. This sees what
/// the safe layer asks; a wait hidden inside a backend's `execute_raw` needs
/// a profiler to show.
pub(crate) fn a_steady_loop_moves_nothing<B: Backend + Topology>(raw: B, add: &Shader) {
    const RUNS: usize = 100;
    let dev = ComputeDevice::new(
        raw,
        DeviceConfig {
            trace: Trace::Events(Events::new(2 * RUNS)),
            ..DeviceConfig::default()
        },
    );
    let mut x = dev.tensor_from(&F32::wrap(vec![0.0; 256])).unwrap();
    let mut y = dev.tensor_from(&F32::wrap(vec![1.0; 256])).unwrap();
    let mut node = dev
        .node(add, &[MetaValue::Uint(256)], Workgroups::cover(256, add))
        .unwrap();
    dev.take_trace();

    dev.scope(|s| {
        let (x, y, n) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut node)?);
        for _ in 0..RUNS {
            s.execute(n, &[Arg::Write(x), Arg::Read(y)])?;
        }
        Ok::<(), crate::Error>(())
    })
    .unwrap();

    let mut expected = vec![Event::Run { shader: "add" }; RUNS];
    expected.push(Event::Sync);
    assert_eq!(kept(dev.take_trace()), expected);
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![RUNS as f32; 256]);
}
