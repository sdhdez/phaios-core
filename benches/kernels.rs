// SPDX-License-Identifier: GPL-3.0-or-later
//! Criterion benchmarks for phaios-core kernels.
//!
//! One benchmark per kernel on a synthetic 24 MP (5765 × 4323) f32
//! image. All benchmarks use `black_box` to prevent the compiler from
//! optimising away the work.
//!
//! Run with: cargo bench
//! Results are written to target/criterion/. Open
//! target/criterion/report/index.html in a browser.
//!
//! Placeholder — benchmarks implemented in Step 4.

use criterion::{Criterion, criterion_group, criterion_main};

fn placeholder(_c: &mut Criterion) {
    // Replaced by real benchmarks in Step 4.
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
