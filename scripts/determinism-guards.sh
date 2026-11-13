#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Static guards for CPU determinism.
#
# The CPU kernels are the specification every other backend is checked
# against. Each wheel computes at its target's fixed default CPU
# (baseline x86-64 on Linux, x86-64 plus SSE3/cmpxchg16b on Windows,
# Apple M1 on macOS), with no per-machine dispatch, whatever CPU the user
# has. These checks fail on the constructs that would make one build
# compute differently from the next, or one wheel compute differently
# from one machine to another:
#
#   - in Rust sources under src/: fused multiply-adds, the algebraic float
#     operations, per-function target features, runtime CPU detection,
#     std::arch/core::arch SIMD intrinsics and ndarray matrix products;
#   - in .cargo/config(.toml) files, workflows and pyproject.toml:
#     `target-cpu` and `target-feature` codegen flags (either spelling).
#
#   bash scripts/determinism-guards.sh          # check this repository
#   bash scripts/determinism-guards.sh ROOT     # check the tree at ROOT
#
# Exit codes: 0 when the tree is clean; 1 when a forbidden construct is
# found, listing file:line for every hit; 2 when ROOT cannot be entered
# or has no src/.
#
# In src/, only lines whose first non-blank characters are `//` are
# skipped, so documentation may name what it forbids. Everything else is
# checked: a trailing comment on a line of code, and a line that starts
# with `*`, which in src/ is code (`*o = ...`, a rustfmt continuation
# such as `* x.mul_add(b, c)`), not a block comment. src/ has no block
# comments; a forbidden name inside one would be flagged.

set -euo pipefail
shopt -s nullglob

ROOT=${1:-"$(dirname "$0")/.."}
cd "$ROOT" 2>/dev/null || { echo "determinism-guards: cannot enter $ROOT" >&2; exit 2; }
[ -d src ] || { echo "determinism-guards: no src/ in $ROOT" >&2; exit 2; }

FAILED=0
flag() {  # $1 = rule and reason, $2 = hits (file:line:text), one per line
  [ -n "$2" ] || return 0
  printf '\n%s\n%s\n' "$1" "$2"
  FAILED=1
}

# Rust sources: hits on lines that are not `//` comments.
rust() {  # $1 = extended regex, $2 = rule and reason
  local hits
  hits=$(grep -rnE --include='*.rs' -- "$1" src/ \
         | grep -vE '^[^:]*:[0-9]+:[[:space:]]*//' | sort -t: -k1,1 -k2,2n || true)
  flag "$2" "$hits"
}

rust 'mul_add' \
  "mul_add: rounds once where a * b + c rounds twice, which silently changes the CPU specification's results. On x86-64 it calls an fma compiled into the wheel (compiler_builtins: the FMA instruction where the CPU has one, a slow software routine otherwise), not the platform libm, so the libm ledger does not see it."
rust 'algebraic_' \
  "f32/f64::algebraic_*: lets the optimiser reassociate and contract, so results depend on the compiler version and the target."
rust 'target_feature *\(' \
  "#[target_feature]: compiles a function for instructions beyond the target's default CPU, callable safely only behind runtime CPU detection."
# Runtime CPU dispatch is how one wheel ends up taking different code
# paths, and computing different bits, on different machines.
rust 'is_[a-z0-9_]+_feature_detected!' \
  "is_*_feature_detected!: runtime CPU dispatch makes one wheel compute differently on different machines."
# `arch::` on its own catches `use std::{arch::...}` and a path through a
# `use std::arch;` import; a renamed import (`use core::arch as a;`) is
# caught on its `use` line.
rust '\b(std|core)::arch\b|\barch::' \
  "std::arch/core::arch: SIMD intrinsics, FMA among them, are machine-specific code. An FMA intrinsic rounds once where a * b + c rounds twice, and one beyond the target's default CPU compiles without #[target_feature] and faults on a CPU that lacks it."
# Coarse on purpose: a 1-D dot stays scalar and is safe, but a grep
# cannot tell it from a 2-D one.
rust '\.dot\(|general_mat_mul' \
  "ndarray matrix products: 2-D dot and general_mat_mul run matrixmultiply's sgemm, which picks an FMA kernel at run time, so one wheel computes different bits on different CPUs."

# Codegen flags. rustc accepts both spellings of an option name, so
# target_cpu is matched as well as target-cpu.
FLAGS='target[-_](cpu|feature)'

# Cargo config at any depth: cargo merges every .cargo/config(.toml) from
# the working directory up. Scratch trees and build output are not ours.
CONFIGS=()
while IFS= read -r f; do CONFIGS+=("$f"); done < <(
  find . \( -name .cache -o -name target -o -name .git \) -prune -o \
    -type f \( -path '*/.cargo/config' -o -path '*/.cargo/config.toml' \) -print | sort)
if [ "${#CONFIGS[@]}" -gt 0 ]; then
  flag "target-cpu/target-feature in a .cargo/config: applies to every build from this tree, the wheels included, which must compute at their target's fixed default CPU." \
    "$(grep -HnE -- "$FLAGS" "${CONFIGS[@]}" | grep -vE '^[^:]*:[0-9]+: *#' || true)"
fi

# Workflows: anywhere in the file, not only on a RUSTFLAGS line. The flag
# reaches rustc as well through CARGO_<target>_RUSTFLAGS,
# CARGO_ENCODED_RUSTFLAGS, a YAML block scalar on the next line, or
# `cargo rustc -- -C`.
WORKFLOWS=(.github/workflows/*.yml .github/workflows/*.yaml)
if [ "${#WORKFLOWS[@]}" -gt 0 ]; then
  flag "target-cpu/target-feature in a workflow: the wheels CI builds must compute at their target's fixed default CPU." \
    "$(grep -HnE -- "$FLAGS" "${WORKFLOWS[@]}" | grep -vE '^[^:]*:[0-9]+: *#' || true)"
fi

# pyproject.toml: [tool.maturin] rustc-args, and anything else in the file
# that would pass the flag to the wheel build.
if [ -f pyproject.toml ]; then
  flag "target-cpu/target-feature in pyproject.toml ([tool.maturin] rustc-args): the wheels must compute at their target's fixed default CPU." \
    "$(grep -HnE -- "$FLAGS" pyproject.toml | grep -vE '^[^:]*:[0-9]+: *#' || true)"
fi

if [ "$FAILED" = 1 ]; then
  printf '\ndeterminism guards: FAILED (docs/ffi.md §6)\n' >&2
  exit 1
fi
echo "determinism guards: src/, cargo config, workflows and pyproject.toml are clean"
