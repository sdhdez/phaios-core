// SPDX-License-Identifier: GPL-3.0-or-later
//! Optional CUDA backend (`--features cuda`).
//!
//! Everything here mirrors a CPU kernel one-for-one: same signature
//! shape, same validation with the same error messages, same output —
//! bit-identical where the maths allows it (integer work and bare
//! multiplies), bounded and tabulated where transcendentals are involved
//! (see `docs/ffi.md` §6). **The CPU implementation is the
//! specification**; every kernel here is tested against it, never the
//! other way round.
//!
//! Built only with the `cuda` cargo feature. The feature adds a
//! build-time requirement (nvcc) but no runtime one beyond the NVIDIA
//! driver, which is dlopened: on machines without it, [`available`] is
//! `false`, [`devices`] is empty, `Context::new` returns
//! `PhaiosError::Backend`, and nothing raises. That last clause is not
//! free — cudarc *panics* when no `libcuda` candidate loads — so every
//! entry point that touches the driver probes for the library first;
//! see `context::driver_present`.

pub mod context;
pub mod kernels;

pub use context::{Context, DeviceImage, DeviceInfo, available, devices};
