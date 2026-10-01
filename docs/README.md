# wilupgu

Wilupgu is a backend-independent tensor/dispatch library for Rust, aiming to
run GPU math on low-end hardware (e.g. iGPUs) without heavy dependencies. It
has no autograd and no shape system — it focuses purely on GPU compute and on
building compute graphs that run independently of the CPU.

Currently used by [sequexa-core](https://github.com/superroket169/sequexa-core) (sequential model engine).

## License

Licensed under either of [Apache License, Version 2.0](../licenses/LICENSE-APACHE) or
[MIT license](../licenses/LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
