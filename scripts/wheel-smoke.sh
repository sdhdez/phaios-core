#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Install a built phaios-core wheel, or the sdist, into a new virtual
# environment and check that it works there as a user would get it.
#
#   bash scripts/wheel-smoke.sh [--numpy SPEC] DIST PYTHON VENV
#
#   DIST     a .whl, or an sdist (.tar.gz), which uv builds from source
#   PYTHON   anything `uv venv --python` accepts: a version (3.12) or a path
#   VENV     where to create the environment. An existing virtual
#            environment there is replaced; any other existing directory
#            is refused.
#
#   --numpy SPEC   install this requirement as well (for example
#                  'numpy==1.26.*'), to test the floor pyproject.toml
#                  declares rather than the newest release
#
# .github/workflows/wheels.yml runs it on every wheel it builds, on
# Linux, Windows and macOS, and on the sdist; release.yml publishes
# nothing unless every run passed. Locally, keep VENV and uv's cache
# inside the repository (CLAUDE.md §6):
#
#   UV_CACHE_DIR=.cache/uv bash scripts/wheel-smoke.sh \
#       dist/phaios_core-*.whl 3.12 .cache/venvs/smoke
#
# In order, it checks:
#   1. for an sdist, that the archive holds exactly the files
#      `cargo package --list` names for this checkout, plus PKG-INFO. The
#      CPU build reads neither build.rs nor src/cuda/ptx, so installing
#      it would not notice them missing; this does.
#   2. `uv pip install DIST pytest` and nothing else, so numpy can only
#      arrive through the wheel's Requires-Dist. An sdist is built with
#      `--locked` (MATURIN_PEP517_ARGS), so its Cargo.lock must be
#      present and current.
#   3. under `python -I`, which keeps this checkout off sys.path:
#      phaios_core is imported from VENV; its metadata requires numpy;
#      py.typed, __init__.pyi and gpu.pyi are installed beside it;
#      exposure(+1 EV) turns a float32 0.25 image into exactly 0.5; and
#      `from phaios_core import gpu` raises ImportError, since the
#      published wheels are built without the cuda feature.
#   4. tests/ffi.py against the installed package, in the same way.
#
# Needs bash and uv; cargo too for an sdist. Writes VENV and uv's cache,
# nothing else. Exits 0 when every check passes, 1 at the first step that
# fails, 2 on a usage error.

set -euo pipefail

usage() {
  sed -n '7,17p' "$0" >&2
  exit 2
}

NUMPY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --numpy) [ $# -ge 2 ] || usage; NUMPY=$2; shift 2 ;;
    --) shift; break ;;
    -*) echo "wheel-smoke: unknown option $1" >&2; usage ;;
    *) break ;;
  esac
done
[ $# -eq 3 ] || usage
DIST=$1 PYTHON=$2 VENV=$3

step() { printf '\n== %s\n' "$1"; }
fail() { printf '\nwheel-smoke: FAILED: %s\n' "$1" >&2; exit 1; }

[ -f "$DIST" ] || { echo "wheel-smoke: no such file: $DIST" >&2; exit 2; }
case "$DIST" in
  *.whl) KIND=wheel ;;
  *.tar.gz) KIND=sdist ;;
  *) echo "wheel-smoke: $DIST is neither a .whl nor a .tar.gz" >&2; exit 2 ;;
esac
# Absolute, because the rest runs from the repository root.
DIST="$(cd "$(dirname "$DIST")" && pwd)/$(basename "$DIST")"
if [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then
  echo "wheel-smoke: $VENV exists and is not a virtual environment; refusing to replace it" >&2
  exit 2
fi
VENV="$(mkdir -p "$(dirname "$VENV")" && cd "$(dirname "$VENV")" && pwd)/$(basename "$VENV")"
cd "$(dirname "$0")/.."

step "Environment: Python $PYTHON in $VENV"
uv venv --clear --quiet --python "$PYTHON" "$VENV"
if [ -f "$VENV/bin/python" ]; then
  PY=$VENV/bin/python
else
  PY=$VENV/Scripts/python.exe  # Windows
fi
"$PY" -I -c "import sys; print(sys.version)"

if [ "$KIND" = sdist ]; then
  step "sdist contents against cargo package --list"
  # --allow-dirty: CI checks out a clean tree; a local tree with edits
  # gives the list those edits would package, which is what maturin's
  # sdist of the same tree contains.
  cargo package --list --locked --allow-dirty > "$VENV/cargo-package-list.txt" \
    || fail "cargo package --list failed (above)"
  "$PY" -I - "$DIST" "$VENV/cargo-package-list.txt" <<'PY' \
    || fail "the sdist's files differ from cargo package --list (above)"
import sys, tarfile

# cargo writes these two into a .crate; maturin's sdist has PKG-INFO
# instead.
with open(sys.argv[2], encoding="utf-8") as listing:
    expected = {line for line in listing.read().splitlines() if line}
expected = expected - {".cargo_vcs_info.json", "Cargo.toml.orig"} | {"PKG-INFO"}
with tarfile.open(sys.argv[1]) as sdist:
    # Every member sits under one <name>-<version>/ directory.
    found = {m.name.split("/", 1)[1] for m in sdist.getmembers()
             if m.isfile() and "/" in m.name}
for label, names in (("missing from the sdist", expected - found),
                     ("in the sdist but not in the crate", found - expected)):
    for name in sorted(names):
        print(f"{label}: {name}")
if expected != found:
    sys.exit(1)
print(f"{len(found)} files, the crate's list plus PKG-INFO")
PY
fi

step "Install $(basename "$DIST") and pytest${NUMPY:+, with $NUMPY}"
# From a fresh copy inside VENV. uv reuses what it built or unpacked from
# a local file for as long as that file's path and modification time are
# unchanged, even with --refresh or other build constraints; the copy has
# a new modification time, so an sdist is always built here, and a wheel
# always unpacked, from the bytes given.
mkdir -p "$VENV/dist"
cp "$DIST" "$VENV/dist/"
REQS=("$VENV/dist/$(basename "$DIST")" pytest)
[ -n "$NUMPY" ] && REQS+=("$NUMPY")
MATURIN_PEP517_ARGS=--locked uv pip install --python "$PY" "${REQS[@]}" \
  || fail "uv pip install failed (above)"

step "Import, metadata, stubs, one kernel, no GPU"
"$PY" -I - <<'PY' || fail "the installed package is incomplete or wrong (above)"
import importlib.metadata as md
import re
import sys
from pathlib import Path

# This interpreter is VENV's own, so its prefix is VENV.
venv = Path(sys.prefix).resolve()
errors = []


def report():
    for e in errors:
        print(f"error: {e}")
    sys.exit(1 if errors else 0)


try:
    requires = md.requires("phaios-core") or []
except md.PackageNotFoundError:
    errors.append("phaios-core is not installed in the environment")
    report()
# The distribution name, before any version, extra or marker.
if not any(re.split(r"[\s<>=!~;\[(]", r, maxsplit=1)[0].lower() == "numpy"
           for r in requires):
    errors.append(f"Requires-Dist does not name numpy: {requires}")

try:
    import numpy as np
    import phaios_core
except ImportError as e:
    errors.append(f"import failed: {type(e).__name__}: {e}")
    report()

where = Path(phaios_core.__file__).resolve()
print(f"phaios_core {phaios_core.__version__} "
      f"(distribution {md.version('phaios-core')}) from {where}")
print(f"numpy {np.__version__}")
if venv not in where.parents:
    errors.append(f"phaios_core was imported from {where}, outside {venv}")

for name in ("py.typed", "__init__.pyi", "gpu.pyi"):
    if not (where.parent / name).is_file():
        errors.append(f"{name} is not installed beside {where.name}")

out = phaios_core.exposure(np.full((8, 8, 3), 0.25, dtype=np.float32), 1.0)
if not (out.dtype == np.float32 and out.shape == (8, 8, 3)
        and bool((out == np.float32(0.5)).all())):
    errors.append(f"exposure(0.25, +1 EV) gave dtype {out.dtype}, shape "
                  f"{out.shape}, values {np.unique(out)[:5]}; expected 0.5")

try:
    from phaios_core import gpu  # noqa: F401
except ImportError:
    pass
else:
    errors.append("`from phaios_core import gpu` succeeded: this is a CUDA "
                  "build, and the published wheels are CPU-only")

report()
PY

step "tests/ffi.py against the installed package"
# -I keeps the checkout off sys.path, so `import phaios_core` finds the
# package in VENV (pytest adds only tests/ itself); -B and
# no:cacheprovider keep the run from writing into the checkout.
"$PY" -I -B -m pytest tests/ffi.py -q -p no:cacheprovider \
  || fail "tests/ffi.py failed against the installed package (above)"

printf '\nwheel-smoke: %s passed on Python %s\n' "$(basename "$DIST")" "$PYTHON"
