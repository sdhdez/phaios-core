// SPDX-License-Identifier: GPL-3.0-or-later
// Re-export the target triple so the app can print the CPU backend
// fingerprint (`cpu/<triple>`, docs/ffi.md §6).
fn main() {
    println!(
        "cargo:rustc-env=KV_TARGET={}",
        std::env::var("TARGET").expect("cargo sets TARGET")
    );
}
