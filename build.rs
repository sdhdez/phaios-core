// SPDX-License-Identifier: GPL-3.0-or-later
//! Build script: compiles CUDA kernels to PTX when the `cuda` feature is on.
//!
//! Without the feature this script does nothing, so ordinary builds (and
//! the published wheels) neither require nor look for a CUDA toolkit.
//!
//! With the feature, every `src/cuda/ptx/*.cu` is compiled by `nvcc` to
//! PTX at **`compute_80`** (Ampere). PTX is forward-compatible: the
//! driver JIT-compiles it, optimised, for whatever newer card is present
//! (sm_86, sm_89, sm_90, sm_120, …) and caches the result. One artifact
//! therefore covers RTX 30/40/50 and the A/H-series.
//!
//! The toolkit is a build-time requirement only; at runtime users need
//! nothing but the NVIDIA driver.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run when the `cuda` feature is enabled for this compilation.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = find_nvcc();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let ptx_src_dir = PathBuf::from("src/cuda/ptx");

    println!("cargo:rerun-if-changed=src/cuda/ptx");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let mut compiled = 0_u32;
    for entry in std::fs::read_dir(&ptx_src_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", ptx_src_dir.display()))
    {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("utf-8 file stem");
        let out = out_dir.join(format!("{stem}.ptx"));

        // compute_80: see module doc. Device-only PTX generation barely
        // touches the host compiler, which is why this works even when
        // the system gcc is newer than the toolkit officially supports.
        let status = Command::new(&nvcc)
            .arg("-arch=compute_80")
            .arg("-ptx")
            .arg("-o")
            .arg(&out)
            .arg(&path)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        assert!(
            status.success(),
            "nvcc failed on {} (exit: {status})",
            path.display()
        );
        compiled += 1;
    }

    assert!(
        compiled > 0,
        "cuda feature enabled but no .cu files found in {}",
        ptx_src_dir.display()
    );
}

/// Locate `nvcc`: `$CUDA_PATH/bin`, then PATH, then the Arch default.
///
/// CI shells are not login shells, so `/etc/profile.d/cuda.sh` may not
/// have run; never assume PATH alone.
fn find_nvcc() -> PathBuf {
    if let Some(cuda_path) = std::env::var_os("CUDA_PATH") {
        let candidate = PathBuf::from(cuda_path).join("bin").join("nvcc");
        if candidate.is_file() {
            return candidate;
        }
    }
    if let Ok(output) = Command::new("nvcc").arg("--version").output()
        && output.status.success()
    {
        return PathBuf::from("nvcc");
    }
    let arch_default = PathBuf::from("/opt/cuda/bin/nvcc");
    if arch_default.is_file() {
        return arch_default;
    }
    panic!(
        "the `cuda` feature needs nvcc (CUDA toolkit) at build time; \
         not found via $CUDA_PATH, PATH, or /opt/cuda/bin. \
         Runtime users need only the NVIDIA driver."
    );
}
