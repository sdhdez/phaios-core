#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Verify the CUDA backend on this machine.
#
# The hosted CI runner has neither nvcc nor an NVIDIA device, so
# .github/workflows/ci.yml deliberately skips every target gated behind
# `required-features = ["cuda"]`. This script is the mirror image of that
# job: it runs exactly the half CI cannot, and nothing else.
#
# It exists so that verifying the GPU backend is not the maintainer's
# private privilege. Anyone with a supported device can run this, and the
# summary it prints at the end is meant to be pasted into an issue or a
# discussion — a second machine, a different card or driver, is real
# evidence that this project cannot generate on its own.
#
#   ./scripts/gpu-verify.sh              # full run
#   ./scripts/gpu-verify.sh --quick      # skip the release-profile examples
#
# Exits non-zero on the first failure. Unlike the examples themselves,
# which skip politely when no device is present, this script treats a
# missing device as an error: you asked for a GPU verification.

set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

step "Environment"
command -v nvcc >/dev/null 2>&1 || fail "nvcc not found — the cuda feature needs the CUDA toolkit at build time"
nvcc --version | tail -2
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — no NVIDIA driver?"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
rustc --version
CRATE_VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
echo "phaios-core ${CRATE_VERSION}"

step "Lint and format (same gates as CI, plus the cuda feature)"
cargo fmt --check
cargo clippy --all-targets --features cuda -- -D warnings

step "Test suite, GPU required"
# PHAIOS_REQUIRE_GPU turns cuda_conformance.rs's device-missing skip into a
# hard panic, so a green run here cannot mean "quietly skipped 54 tests".
PHAIOS_REQUIRE_GPU=1 cargo test --features cuda

step "Examples gated behind --features cuda"
# The exact inverse of the filter in ci.yml: that job runs the targets with
# no required-features, this one runs the targets that have them.
GPU_EXAMPLES=$(cargo metadata --no-deps --format-version=1 \
  | python3 -c "import sys, json; print('\n'.join(sorted(t['name'] for p in json.load(sys.stdin)['packages'] for t in p['targets'] if 'example' in t['kind'] and t.get('required-features'))))")
if [ -z "$GPU_EXAMPLES" ]; then fail "no GPU examples discovered — has required-features gone missing?"; fi
PROFILE=(--release)
[ "$QUICK" = "1" ] && PROFILE=()
echo "$GPU_EXAMPLES" | while read -r ex; do
  printf '\n--- %s ---\n' "$ex"
  cargo run "${PROFILE[@]}" --example "$ex" --features cuda
done

step "Python bindings against the device"
if [ -d .phaios-venv ] && [ -f tests/ffi_gpu.py ]; then
  # shellcheck disable=SC1091
  source .phaios-venv/bin/activate
  if python -c "import phaios_core.gpu" 2>/dev/null; then
    python -m pytest tests/ffi_gpu.py -q
    # On a CUDA build the `gpu` allowlist entry is unused (stubtest checks
    # phaios_core.gpu for real instead of allowing it to be absent).
    python -m mypy.stubtest phaios_core --allowlist tests/stubtest-allowlist.txt --ignore-unused-allowlist
  else
    echo "skipped: the installed wheel has no gpu submodule."
    echo "         rebuild with: maturin develop --release --features cuda"
  fi
else
  echo "skipped: no .phaios-venv; see CLAUDE.md §7 to create one"
fi

step "Summary — please share this if you are reporting a result"
cat <<EOF
phaios-core   ${CRATE_VERSION}
device        $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
nvcc          $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1)
rustc         $(rustc --version | cut -d' ' -f2)
host          $(uname -srm)
result        all GPU checks passed
EOF
