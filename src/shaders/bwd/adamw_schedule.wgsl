struct State {
    step: u32,
    lr: f32,
}

struct Config {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: u32,
    max_steps: u32,
}

@group(0) @binding(0) var<storage, read_write> state: State;
@group(0) @binding(1) var<storage, read> cfg: Config;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let t = state.step + 1u;
    state.step = t;

    if (t < cfg.warmup_steps) {
        state.lr = cfg.lr_max * f32(t) / f32(cfg.warmup_steps);
    } else {
        let progress = f32(t - cfg.warmup_steps) / f32(cfg.max_steps - cfg.warmup_steps);
        state.lr = cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1.0 + cos(3.14159265 * progress));
    }
}
