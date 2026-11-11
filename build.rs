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
//! Each file is compiled in the precision mode `PRECISION` gives it:
//! strict IEEE arithmetic where speed gains nothing, FMA or fast math
//! where it does. The GPU agrees with the CPU (the specification) to the
//! bit for the kernels in `docs/ffi.md` §6's bit-exact list and within a
//! stated bound for the rest. In the FMA and fast files, and wherever
//! approximate instructions remain (libdevice `cbrtf` in `split_toning`),
//! the driver's JIT compiler finishes part of the arithmetic, so a driver
//! or CUDA update can move their last bits.
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

    let mut seen: Vec<String> = Vec::new();
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
        let mode = PRECISION
            .iter()
            .find(|(name, _)| *name == stem)
            .map(|&(_, mode)| mode)
            .unwrap_or_else(|| {
                panic!(
                    "{} has no entry in build.rs's PRECISION table; \
                     a new CUDA file must choose its precision mode \
                     (CONTRIBUTING.md, kernel checklist)",
                    path.display()
                )
            });
        seen.push(stem.to_string());

        // compute_80: see module doc. Device-only PTX generation barely
        // touches the host compiler, which is why this works even when
        // the system gcc is newer than the toolkit officially supports.
        let status = nvcc_command(&nvcc)
            .arg("-arch=compute_80")
            .args(mode.flags())
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
    }

    assert!(
        !seen.is_empty(),
        "cuda feature enabled but no .cu files found in {}",
        ptx_src_dir.display()
    );
    for (name, _) in PRECISION {
        assert!(
            seen.iter().any(|s| s == name),
            "build.rs's PRECISION table names {name}, but {}/{name}.cu does not exist",
            ptx_src_dir.display()
        );
        assert_eq!(
            PRECISION.iter().filter(|(other, _)| other == name).count(),
            1,
            "build.rs's PRECISION table names {name} more than once"
        );
    }
}

/// How nvcc compiles one `.cu` file. Each mode passes its complete set of
/// precision flags and nothing is shared between modes: an explicit
/// `-fmad=false` (or `-ftz=false`, `-prec-div=true`) overrides what
/// `--use_fast_math` implies, whatever the order, so a common prefix
/// would quietly compile the fast files as strict.
#[derive(Clone, Copy)]
enum Precision {
    /// No FMA contraction, IEEE division and square root, denormals kept.
    /// Every multiply and add is emitted correctly rounded (`mul.rn`,
    /// `add.rn`), which the driver's JIT may not fuse. Rust does not
    /// contract either, so the kernels in `docs/ffi.md` §6's bit-exact
    /// list agree with the CPU to the bit.
    Strict,
    /// FMA contraction on, everything else IEEE. Faster on the guided
    /// filters' multiply-add chains; results move by a few ULP, and the
    /// driver may fuse the remaining multiply-adds.
    Fma,
    /// `--use_fast_math`: FMA contraction, approximate division, square
    /// root and transcendentals (`ex2`/`lg2.approx`), and subnormal
    /// inputs and results flushed to zero. Within one 8-bit code of the
    /// CPU on every image measured.
    Fast,
}

impl Precision {
    fn flags(self) -> &'static [&'static str] {
        match self {
            Precision::Strict => &[
                "-fmad=false",
                "-ftz=false",
                "-prec-div=true",
                "-prec-sqrt=true",
            ],
            Precision::Fma => &[
                "-fmad=true",
                "-ftz=false",
                "-prec-div=true",
                "-prec-sqrt=true",
            ],
            Precision::Fast => &["--use_fast_math"],
        }
    }
}

/// The precision mode of every `src/cuda/ptx/*.cu`. A file missing from
/// this table, or an entry without a file, fails the build.
///
/// Chosen per file from measurements of whole builds compiled in each
/// mode (24 MP, RTX 5070 Ti, nvcc 13.4, 2026-11-11): a file leaves
/// `Strict` only for a gain above run-to-run noise and worth more than
/// about 0.04 ms, at a cost of at most one 8-bit code against the CPU
/// and an error that stays within a fixed (rtol, atol) bound. The
/// figures are each mode's time change against `Strict`.
const PRECISION: &[(&str, Precision)] = &[
    // Fast: transcendentals and divisions dominate these kernels.
    ("zone_system", Precision::Fast), // −47 %
    ("hsl_bw", Precision::Fast),      // −27 % (exact on neutral pixels no longer)
    ("encode_srgb", Precision::Fast), // −14 %
    ("tone_curve", Precision::Fast),  // −13 %, powf only: the .cu pins the affine step
    // Fma: the guided filters' variance and coefficient chains.
    ("local_contrast", Precision::Fma), // −12 % (also C=1 denoise, which runs on these kernels)
    ("denoise", Precision::Fma),        // C=3 −8 %, shared with local_contrast
    // Strict: no mode gains enough to justify the extra error.
    ("vignette", Precision::Strict), // fast: −23 % (0.1 ms), but its error grows with input × amount / feather
    ("blur", Precision::Strict),     // both other modes +20 % on the direct path
    ("sharpen", Precision::Strict),  // element-wise; its time is blur's
    ("glow", Precision::Strict),     // element-wise; its time is blur's
    ("geometry", Precision::Strict), // fast: resize −2 %, straighten −1 % and up to 62 u16 codes on the photos (271 on synthetic HDR)
    ("film_grain", Precision::Strict), // fast: −5 % (0.06 ms) for ~45× the CPU difference
    ("histogram", Precision::Strict), // fast: −3 %, 0.02 ms
    ("highlight_rolloff", Precision::Strict), // fast: −4 %, 0.02 ms
    ("lut", Precision::Strict),      // fast: −2 %
    ("shadow_rolloff", Precision::Strict), // fast: −1 %
    ("exposure", Precision::Strict), // within noise
    ("luminance_bw", Precision::Strict), // within noise
    ("quantize", Precision::Strict), // within noise
    ("hot_pixels", Precision::Strict), // within noise
    ("split_toning", Precision::Strict), // within noise
];

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
    let output = nvcc_command(nvcc)
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

/// An `nvcc` invocation that sees only the flags this script passes.
///
/// nvcc appends `NVCC_APPEND_FLAGS` and prepends `NVCC_PREPEND_FLAGS` to
/// every command line it runs, and its `nvcc.profile` extends
/// `CUDAFE_FLAGS`, `INCLUDES`, `SYSTEM_INCLUDES`, `PTXAS_FLAGS` and
/// `LIBRARIES` from the environment, so a value left in a developer's
/// shell (`--use_fast_math`, a `-D` macro) would change the PTX — and with
/// it the GPU results — while the fingerprint stayed the same. All seven
/// are removed from the child's environment. The list mirrors the `+=`
/// entries of CUDA 13.4's `nvcc.profile`; a later toolkit may add more,
/// and the host preprocessor's own variables (`CPATH` and the like) are
/// out of its reach.
fn nvcc_command(nvcc: &std::path::Path) -> Command {
    let mut cmd = Command::new(nvcc);
    for var in [
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
        "CUDAFE_FLAGS",
        "INCLUDES",
        "SYSTEM_INCLUDES",
        "PTXAS_FLAGS",
        "LIBRARIES",
    ] {
        cmd.env_remove(var);
    }
    cmd
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
