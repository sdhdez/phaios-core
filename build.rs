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
//!
//! The script also exports `PHAIOS_NVCC_VERSION` (e.g. `13.4.59`) for
//! `src/cuda/context.rs` to put in the backend fingerprint. The PTX and
//! that string are produced by the same `nvcc` invocation and refreshed
//! together, so the fingerprint always names the toolkit that built the
//! PTX actually embedded in the binary — including when a toolkit is
//! upgraded in place without this script re-running, in which case both
//! stay at the old value and remain consistent with each other.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run when the `cuda` feature is enabled for this compilation.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = find_nvcc();
    println!(
        "cargo:rustc-env=PHAIOS_NVCC_VERSION={}",
        nvcc_version(&nvcc)
    );
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
            // No FMA contraction: Rust does not contract either, and with
            // both sides emitting plain correctly-rounded mul/add, every
            // kernel free of transcendentals agrees with the CPU to the
            // bit instead of to a tolerance.
            .arg("-fmad=false")
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

/// The toolkit release as `<major>.<minor>.<patch>`, e.g. `13.4.59`.
///
/// Parsed from the `V<major>.<minor>.<patch>` token on `nvcc --version`'s
/// "Cuda compilation tools" line — the same three numbers every emitted
/// `.ptx` carries in its own header comment. `release 13.4` alone is not
/// enough: libdevice bodies can change between patch releases, and the
/// bounded kernels inline them.
///
/// Fails the build rather than substituting a placeholder. A fingerprint
/// is a reproducibility key (`docs/ffi.md` §6); an unparsed toolkit
/// would silently weaken it for every render made with this binary.
fn nvcc_version(nvcc: &std::path::Path) -> String {
    let output = Command::new(nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {} --version: {e}", nvcc.display()));
    assert!(
        output.status.success(),
        "{} --version failed (exit: {})",
        nvcc.display(),
        output.status
    );
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .find_map(|tok| {
            let v = tok.trim_end_matches(',').strip_prefix('V')?;
            let mut parts = v.split('.');
            let ok = [parts.next()?, parts.next()?, parts.next()?]
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
            (ok && parts.next().is_none()).then(|| v.to_string())
        })
        .unwrap_or_else(|| {
            panic!(
                "cannot parse a V<major>.<minor>.<patch> release out of \
                 `{} --version`; output was:\n{text}",
                nvcc.display()
            )
        })
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
