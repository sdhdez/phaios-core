// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 13 — Exposure on the CUDA backend.
//!
//! Demonstrates the optional GPU path (build with `--features cuda`):
//! device enumeration, opening a context, running a kernel, and the two
//! properties the backend is built around:
//!
//! 1. **Bit-exactness where the maths allows it.** `exposure` is one
//!    correctly-rounded multiply, so its GPU output equals its CPU
//!    output to the last bit — compared here with `==`, not a tolerance.
//! 2. **Honest transfer economics.** Per-call offload pays a PCIe round
//!    trip; this example prints it next to the CPU time so the numbers
//!    in `docs/architecture.md` can be reproduced on any machine.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0 — CI runs every example, GPU or not.

#[path = "shared/mod.rs"]
mod shared;

use std::time::Instant;

use ndarray::Array3;
use phaios_core::cuda;

fn main() {
    let devices = cuda::devices();
    if devices.is_empty() {
        println!("no CUDA device available, skipping");
        return;
    }
    for d in &devices {
        println!(
            "device {}: {} (cc {}.{}, supported: {})",
            d.ordinal, d.name, d.compute_capability.0, d.compute_capability.1, d.supported
        );
    }
    let Ok(ctx) = cuda::Context::new(0) else {
        println!("device present but unusable, skipping");
        return;
    };
    println!("fingerprint: {}\n", ctx.fingerprint());

    // The Macbeth chart, as in every other example.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    let cpu = phaios_core::exposure::exposure(rgb.view(), 1.0).unwrap();
    let gpu = cuda::kernels::exposure(&ctx, rgb.view(), 1.0).unwrap();
    println!(
        "+1 EV on the Macbeth chart: CPU == GPU bit-for-bit: {}",
        cpu == gpu
    );

    // Transfer economics at export size (24 MP), the number Gate A1 is
    // about. Warm one call first so JIT/module load is excluded.
    let big = Array3::<f32>::from_elem((4323, 5765, 1), 0.5);
    let _ = cuda::kernels::exposure(&ctx, big.view(), 1.0).unwrap();

    let time_best = |f: &mut dyn FnMut() -> Array3<f32>| {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            std::hint::black_box(f());
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        best
    };

    let gpu_ms = time_best(&mut || cuda::kernels::exposure(&ctx, big.view(), 1.0).unwrap());
    let cpu_ms = time_best(&mut || phaios_core::exposure::exposure(big.view(), 1.0).unwrap());
    println!(
        "24 MP mono, per-call offload: GPU {gpu_ms:.1} ms (incl. PCIe round trip), CPU {cpu_ms:.1} ms"
    );
    println!(
        "note: per-call offload of a cheap kernel is transfer-bound by design;\n\
         the win arrives with expensive kernels and the device-resident API."
    );
}
