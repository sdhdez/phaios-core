#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Verify the CUDA backend on this machine.
#
# The hosted CI runner has neither nvcc nor an NVIDIA device, so
# .github/workflows/ci.yml deliberately skips every target gated behind
# `required-features = ["cuda"]`. This script is the mirror image of that
# job: it runs the half CI cannot, plus the same fmt, clippy and test
# gates with the cuda feature turned on. Like CI, it passes --locked to
# every cargo command that resolves dependencies, so it tests the
# committed Cargo.lock and fails on a stale one instead of rewriting it.
# .github/workflows/gpu.yml runs this script on a self-hosted runner, but
# only when started by hand, and no such runner exists yet.
#
# It exists so that verifying the GPU backend is not the maintainer's
# private privilege. Anyone with a supported device can run this, and the
# summary it prints at the end is meant to be pasted into an issue or a
# discussion — a second machine, a different card or driver, is real
# evidence that this project cannot generate on its own.
#
#   ./scripts/gpu-verify.sh              # full run
#   ./scripts/gpu-verify.sh --quick      # examples in the dev profile, no release build
#
# Exits non-zero on the first failure. Unlike the examples themselves,
# which skip politely when no device is present, this script treats a
# missing device as an error: you asked for a GPU verification. Each
# example's output is also kept in .cache/gpu-verify/<example>.log, and
# the driver's JIT cache in .cache/cuda-jit unless CUDA_CACHE_PATH is
# already set.
#
# The Python step needs `.phaios-venv` with a CUDA build of the wheel
# installed. With no venv, no phaios_core in it, or a CPU-only build, it
# is skipped and the summary says so; a phaios_core that fails to import
# fails the run. With PHAIOS_REQUIRE_GPU set in the environment (to any
# value, as the Rust tests read it), a skip is a failure too.

set -euo pipefail
cd "$(dirname "$0")/.."

# The driver JIT-compiles the PTX for the device and caches the result,
# by default in ~/.nv/ComputeCache. Keep it in the repository's .cache
# instead; a caller's own setting is respected.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PWD/.cache/cuda-jit}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# Read from the caller's environment, before the test step below sets it
# for cargo alone.
REQUIRE_GPU=0
[ -n "${PHAIOS_REQUIRE_GPU+set}" ] && REQUIRE_GPU=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

step "Environment"
# The nvcc build.rs compiles with: $CUDA_PATH/bin, then PATH, then the
# Arch default.
if [ -n "${CUDA_PATH:-}" ] && [ -f "$CUDA_PATH/bin/nvcc" ]; then
  NVCC=$CUDA_PATH/bin/nvcc
elif command -v nvcc >/dev/null 2>&1; then
  NVCC=$(command -v nvcc)
elif [ -x /opt/cuda/bin/nvcc ]; then
  NVCC=/opt/cuda/bin/nvcc
else
  fail "nvcc not found — the cuda feature needs the CUDA toolkit at build time"
fi
echo "$NVCC"
"$NVCC" --version | tail -2
# The V<major>.<minor>.<patch> token, the same one build.rs puts in the
# fingerprint; "release 13.4" alone would hide the patch level.
NVCC_VERSION=$("$NVCC" --version | sed -n 's/.*, V\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$/\1/p' | tail -1)
[ -n "$NVCC_VERSION" ] || fail "cannot parse a V<major>.<minor>.<patch> release out of \`$NVCC --version\`"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — no NVIDIA driver?"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
rustc --version
CRATE_VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
echo "phaios-core ${CRATE_VERSION}"

step "Lint and format (same gates as CI, plus the cuda feature)"
cargo fmt --check
cargo clippy --locked --all-targets --features cuda -- -D warnings

step "Test suite, GPU required"
# PHAIOS_REQUIRE_GPU turns cuda_conformance.rs's device-missing skip into a
# hard panic, so a green run here cannot mean "quietly skipped the GPU tests".
PHAIOS_REQUIRE_GPU=1 cargo test --locked --features cuda

step "PTX audit"
# Judges one thing: every .ptx in the build the test step used must carry
# the release of the nvcc found above in its header. The examples' check
# below reads the build they run, which in a full run is the release
# build; this one catches a stale dev build beside a fresh release build,
# for example a release build first made, or cleaned, after a toolkit
# change cargo could not see (build.rs, watch_toolkit, lists those).
#
# The rest it prints without judging: precision_tests in
# src/cuda/kernels/mod.rs, run by the test step above, already asserts
# each mode's signature. The counts are here so a report from another
# machine or toolkit shows them. Columns: the file's mode in build.rs's
# PRECISION table, then how many instructions are fused multiply-adds,
# approximate (.approx), flush to zero (.ftz), and float add/sub/mul with
# no rounding modifier — the form the driver's JIT may fuse into an FMA.
PKG_ID=$(cargo metadata --locked --no-deps --format-version=1 \
  | python3 -c "import sys, json; print(next(p['id'] for p in json.load(sys.stdin)['packages'] if p['name'] == 'phaios-core'))")
OUT_DIR=$(cargo build --locked --features cuda --message-format=json \
  | python3 -c "import sys, json; msgs = [json.loads(l) for l in sys.stdin]; print(next((m['out_dir'] for m in msgs if m.get('reason') == 'build-script-executed' and m['package_id'] == sys.argv[1]), ''))" "$PKG_ID")
[ -n "$OUT_DIR" ] || fail "cargo reported no build-script output directory for phaios-core"
python3 - "$OUT_DIR" build.rs "$NVCC_VERSION" "$NVCC" <<'PY'
import pathlib, re, sys

out_dir, build_rs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
nvcc_version, nvcc = sys.argv[3], sys.argv[4]
modes = {name: mode.lower() for name, mode in
         re.findall(r'\(\s*"(\w+)"\s*,\s*Precision::(\w+)\s*\)', build_rs.read_text())}

def fail(msg):
    # The same form as the shell's fail().
    sys.exit(f"\nFAILED: {msg}")

def opcodes(ptx):
    # The same reading of an instruction line as precision_tests.
    for line in ptx.splitlines():
        line = line.lstrip()
        if line.startswith("@"):
            parts = line.split(None, 1)
            line = parts[1].lstrip() if len(parts) > 1 else ""
        op = re.split(r"[\s;]", line, maxsplit=1)[0]
        if op and not op.startswith((".", "/", "$", "{", "}")) and "." in op:
            yield op

def unpinned(op):
    return (op.endswith((".f32", ".f64")) and op.startswith(("add.", "sub.", "mul."))
            and not any(r in op for r in (".rn", ".rz", ".rm", ".rp")))

def toolkit(ptx):
    # nvcc's header line, e.g. "Cuda compilation tools, release 13.4, V13.4.92",
    # and the V<major>.<minor>.<patch> token at its end.
    header = next((l[2:].strip() for l in ptx.splitlines()
                   if l.startswith("// Cuda compilation tools")), None)
    m = re.search(r", V(\d+\.\d+\.\d+)$", header or "")
    return header or "toolkit not stated", m.group(1) if m else None

files = sorted(out_dir.glob("*.ptx"))
if not files:
    fail(f"no .ptx files in {out_dir}")
built_by = {f: toolkit(f.read_text()) for f in files}
print(f"{len(files)} files in {out_dir}")
for header in sorted({h for h, _ in built_by.values()}):
    print(f"compiled by: {header}")
stale = {f.name: v for f, (_, v) in built_by.items() if v != nvcc_version}
if stale:
    which = (f"all {len(files)}" if len(stale) == len(files)
             else f"{len(stale)} of {len(files)} ({', '.join(sorted(stale))})")
    other = ", ".join(sorted({f"nvcc {v}" if v else "a toolkit its header does not name"
                              for v in stale.values()}))
    fail(f"the PTX the tests ran, {which} .ptx files in {out_dir}, was built by "
         f"{other}, but {nvcc} is {nvcc_version}: cargo did not re-run build.rs after "
         f"the toolkit changed (build.rs, watch_toolkit, lists when that happens). "
         f"Run `touch src/cuda/ptx`, which re-runs it in every profile, then this "
         f"script again")
cols = ("fma", "approx", "ftz", "unpinned")
print(f"\n{'file':<18} {'mode':<7}" + "".join(f"{c:>9}" for c in cols))
total = dict.fromkeys(cols, 0)
for f in files:
    ops = list(opcodes(f.read_text()))
    n = {"fma": sum(op.startswith("fma.") for op in ops),
         "approx": sum(".approx" in op for op in ops),
         "ftz": sum(".ftz" in op for op in ops),
         "unpinned": sum(unpinned(op) for op in ops)}
    for c in cols:
        total[c] += n[c]
    print(f"{f.stem:<18} {modes.get(f.stem, '?'):<7}" + "".join(f"{n[c]:>9}" for c in cols))
print(f"{'total':<18} {'':<7}" + "".join(f"{total[c]:>9}" for c in cols))
PY

step "Examples gated behind --features cuda"
# The exact inverse of the filter in ci.yml: that job runs the targets with
# no required-features, this one runs the targets that have them.
GPU_EXAMPLES=$(cargo metadata --locked --no-deps --format-version=1 \
  | python3 -c "import sys, json; print('\n'.join(sorted(t['name'] for p in json.load(sys.stdin)['packages'] for t in p['targets'] if 'example' in t['kind'] and t.get('required-features'))))")
if [ -z "$GPU_EXAMPLES" ]; then fail "no GPU examples discovered — has required-features gone missing?"; fi
PROFILE=(--release)
[ "$QUICK" = "1" ] && PROFILE=()
# The examples print the backend fingerprint ("fingerprint: ..."); it is
# read back from a copy of their output, for the summary and for the
# Python step's comparison.
LOGS=.cache/gpu-verify
mkdir -p "$LOGS"
for ex in $GPU_EXAMPLES; do
  printf '\n--- %s ---\n' "$ex"
  cargo run --locked "${PROFILE[@]}" --example "$ex" --features cuda </dev/null | tee "$LOGS/$ex.log"
done
FINGERPRINT=$(for ex in $GPU_EXAMPLES; do sed -n 's/^fingerprint: //p' "$LOGS/$ex.log"; done | sort -u)
[ -n "$FINGERPRINT" ] || fail "no example printed a \"fingerprint: \" line"
[ "$(printf '%s\n' "$FINGERPRINT" | wc -l)" -eq 1 ] \
  || fail "the examples printed different fingerprints:"$'\n'"$FINGERPRINT"
# The fingerprint names the toolkit that built the embedded PTX. If it is
# not the nvcc build.rs finds now, cargo did not re-run build.rs after the
# toolkit changed (build.rs, watch_toolkit, lists when that happens).
# touch src/cuda/ptx re-runs it in every profile, so the next run
# refreshes the tests' build and the examples' alike.
FP_NVCC=$(printf '%s\n' "$FINGERPRINT" | sed -n 's#.*/nvcc-\([0-9][0-9.]*[0-9]\).*#\1#p')
if [ -n "$FP_NVCC" ] && [ "$FP_NVCC" != "$NVCC_VERSION" ]; then
  fail "the embedded PTX was built by nvcc $FP_NVCC, but $NVCC is $NVCC_VERSION; run \`touch src/cuda/ptx\`, which re-runs build.rs in every profile, then this script again"
fi

step "Python bindings against the device"
PYTHON_RESULT=""
skip() {  # $1 = why, $2 = what to do about it
  if [ "$REQUIRE_GPU" = "1" ]; then
    fail "$1, and PHAIOS_REQUIRE_GPU is set, so the Python step may not be skipped; $2"
  fi
  echo "skipped: $1"
  echo "         $2"
  echo "         (set PHAIOS_REQUIRE_GPU=1 to make this a failure)"
  PYTHON_RESULT="skipped: $1"
}
if [ ! -d .phaios-venv ]; then
  skip "no .phaios-venv" "see README.md \"Build from source\" to create one"
else
  # shellcheck disable=SC1091
  source .phaios-venv/bin/activate
  # Three cases: phaios_core not installed (skip), installed but failing
  # to import (a broken build: fail), importing without the gpu submodule
  # the cuda feature registers (a CPU-only build: skip). Exit status 3
  # means "not installed"; any other failure of the probe falls through to
  # the import, which then reports it.
  FOUND=0
  python -c "import importlib.util, sys; sys.exit(3 if importlib.util.find_spec('phaios_core') is None else 0)" || FOUND=$?
  if [ "$FOUND" = "3" ]; then
    skip "phaios_core is not installed in .phaios-venv" "install a CUDA build with: maturin develop --release --features cuda"
  elif ! IMPORT_ERROR=$(python -c "import phaios_core" 2>&1); then
    printf '%s\n' "$IMPORT_ERROR" >&2
    fail "importing phaios_core in .phaios-venv fails (above): a broken build, not a CPU-only one; rebuild with: maturin develop --release --features cuda"
  elif ! python -c "import sys, phaios_core; sys.exit(not hasattr(phaios_core, 'gpu'))"; then
    skip "the installed wheel has no gpu submodule (a CPU-only build)" "rebuild with: maturin develop --release --features cuda"
  else
    # The wheel in the venv was built separately from the tree tested
    # above. A different fingerprint means it is another build, from
    # another toolkit, and the step below would test that build instead.
    # A build from older sources by the same toolkit has the same
    # fingerprint and is not caught here.
    WHEEL_FINGERPRINT=$(python -c "from phaios_core import gpu; print(gpu.GpuContext(0).fingerprint)")
    echo "wheel fingerprint: $WHEEL_FINGERPRINT"
    if [ "$WHEEL_FINGERPRINT" != "$FINGERPRINT" ]; then
      fail "the installed wheel's fingerprint is \"$WHEEL_FINGERPRINT\", this tree's is \"$FINGERPRINT\": .phaios-venv holds a stale build; rebuild with: maturin develop --release --features cuda"
    fi
    python -m pytest tests/ffi_gpu.py -q
    # On a CUDA build the `gpu` allowlist entry is unused (stubtest checks
    # phaios_core.gpu for real instead of allowing it to be absent).
    python -m mypy.stubtest phaios_core --allowlist tests/stubtest-allowlist.txt --ignore-unused-allowlist
    PYTHON_RESULT="passed (tests/ffi_gpu.py, stubtest; wheel fingerprint matches)"
  fi
fi

if [ "$QUICK" = "1" ]; then RUN="quick (examples in the dev profile)"; else RUN="full"; fi
# Relative to the repository when it lies inside it, so that the block
# does not carry the reporter's home directory.
JIT_CACHE=${CUDA_CACHE_PATH#"$PWD"/}
if [ "${PYTHON_RESULT#skipped}" = "$PYTHON_RESULT" ]; then
  RESULT="all GPU checks passed"
else
  RESULT="GPU checks passed except the Python step, which was skipped"
fi

step "Summary — please share this if you are reporting a result"
cat <<EOF
phaios-core   ${CRATE_VERSION}
device        $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
fingerprint   ${FINGERPRINT}
nvcc          ${NVCC_VERSION}
jit cache     ${JIT_CACHE}
rustc         $(rustc --version | cut -d' ' -f2)
host          $(uname -srm)
run           ${RUN}
python        ${PYTHON_RESULT}
result        ${RESULT}
EOF
