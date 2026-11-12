#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Compare the platform math-library (libm) functions a built extension
# imports against the ledger in tests/libm-imports.txt.
#
# Every function the extension takes from the platform libm (glibc,
# Windows UCRT, Apple's libm) is a place where the same wheel can give
# different bits on different machines: IEEE-754 standardises no
# transcendental, and each libm rounds them its own way. The ledger names
# each imported function and the kernels that reach it. This script
# fails when the set changes, so that a new platform-libm dependency (or
# a removed one) is a deliberate, documented decision rather than a side
# effect of a refactor or a compiler upgrade, which can rewrite
# `2.0_f32.powf(x)` into `exp2f(x)` or merge `sin` and `cos` into `sincos`.
#
#   cargo build --release --lib
#   bash scripts/libm-inventory.sh target/release/libphaios_core.so
#   bash scripts/libm-inventory.sh <extension> [ledger]   # ledger defaults to tests/libm-imports.txt
#
# Names are compared without their symbol versions (`exp`, not
# `exp@GLIBC_2.29`). The versions depend on the glibc the extension was
# linked against (a manylinux_2_17 wheel binds older ones than a build on
# a current distribution) and are printed for information only.
#
# Math functions *defined* inside the extension (Rust's own copies, such
# as the `cbrtf` compiler-builtins provides on Linux x86-64) are listed
# for information too. They are compiled into the wheel and do not come
# from the platform libm. Which functions Rust compiles in may differ
# by target: only Linux x86-64 has been checked, and Rust documents
# f32::cbrt as libc's cbrtf.
#
# Linux: checks, and exits 1 on any difference. macOS: lists the imports
# and exits 0 (record only; untested). Anywhere else: says the check is
# Linux-only and exits 0. Exit 2 means the check could not run.
#
# Needs only binutils (nm), coreutils, grep, sed and awk.

set -euo pipefail
# comm(1) needs both inputs sorted in one collation order.
export LC_ALL=C

usage() {
    echo "usage: $0 <extension .so> [ledger, default tests/libm-imports.txt]" >&2
    exit 2
}

[ $# -ge 1 ] && [ $# -le 2 ] || usage
SO=$1
LEDGER=${2:-$(cd "$(dirname "$0")/.." && pwd)/tests/libm-imports.txt}
[ -f "$SO" ] || { echo "error: no such file: $SO" >&2; exit 2; }
command -v nm >/dev/null 2>&1 || { echo "error: nm (binutils) not found" >&2; exit 2; }

# The C <math.h> functions to look for; each also matches its `f` and
# `l` variants, and glibc's `__<name>_finite` entry points (what
# -ffinite-math-only code called before glibc 2.31).
MATH_NAMES="sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh
exp exp2 expm1 exp10 log log2 log10 log1p logb ilogb pow cbrt sqrt hypot sincos
erf erfc tgamma lgamma fma fmod remainder floor ceil round roundeven trunc rint
nearbyint lround llround lrint llrint ldexp frexp scalbn modf copysign
nextafter fmin fmax fdim"

# Reads symbol names on stdin (with or without an @VERSION suffix) and
# prints those whose base name is a math function, one per line, as
# given (version kept).
math_filter() {
    # On one line: POSIX awk rejects a newline inside a -v value.
    awk -v names="$(printf '%s' "$MATH_NAMES" | tr '\n' ' ')" '
        BEGIN {
            n = split(names, list, /[ \t\n]+/)
            for (i = 1; i <= n; i++) {
                if (list[i] == "") continue
                m[list[i]] = 1; m[list[i] "f"] = 1; m[list[i] "l"] = 1
            }
        }
        {
            sym = $0; base = sym
            sub(/@.*/, "", base)
            core = base
            if (core ~ /^__.+_finite$/) core = substr(core, 3, length(core) - 9)
            if (core in m) print sym
        }'
}

base_names() { sed 's/@.*//' | sort -u; }

OS=$(uname -s)
case $OS in
Linux)
    if ! UNDEF=$(nm -D --undefined-only -- "$SO" 2>&1); then
        printf 'error: cannot read the dynamic symbols of %s:\n%s\n' "$SO" "$UNDEF" >&2
        exit 2
    fi
    IMPORTED=$(printf '%s\n' "$UNDEF" | awk 'NF { print $NF }' | math_filter | sort -u)
    # The full symbol table may be stripped; then there is nothing to list.
    DEFINED=$(nm --defined-only -- "$SO" 2>/dev/null | awk 'NF >= 3 { print $3 }' | math_filter | sort -u || true)
    ;;
Darwin)
    # Untested: no macOS machine was available when this was written.
    IMPORTED=$(nm -u -- "$SO" | sed 's/^_//' | math_filter | sort -u)
    echo "Platform libm functions imported by $SO (macOS: recorded, not checked):"
    printf '%s\n' "${IMPORTED:-(none)}" | sed 's/^/  /'
    exit 0
    ;;
*)
    echo "libm-inventory: the check is Linux-only; nothing checked on $OS."
    exit 0
    ;;
esac

[ -f "$LEDGER" ] || { echo "error: no ledger at $LEDGER" >&2; exit 2; }
# A symbol line starts in column 0; indented lines continue the entry
# above and `#` starts a comment.
EXPECTED=$(awk '/^[A-Za-z_]/ { print $1 }' "$LEDGER" | base_names)

echo "Platform libm functions imported by $SO:"
if [ -n "$IMPORTED" ]; then
    printf '%s\n' "$IMPORTED" | sed 's/^/  /'
else
    echo "  (none)"
fi
echo "Math functions defined inside it (Rust's own code, not the platform libm's):"
if [ -n "$DEFINED" ]; then
    printf '%s\n' "$DEFINED" | tr '\n' ' ' | sed 's/ $//; s/^/  /'
    echo
else
    echo "  (none, or the symbol table is stripped)"
fi

GOT=$(printf '%s\n' "$IMPORTED" | base_names | sed '/^$/d')
ADDED=$(comm -13 <(printf '%s\n' "$EXPECTED") <(printf '%s\n' "$GOT") | sed '/^$/d')
REMOVED=$(comm -23 <(printf '%s\n' "$EXPECTED") <(printf '%s\n' "$GOT") | sed '/^$/d')

if [ -z "$ADDED" ] && [ -z "$REMOVED" ]; then
    echo "libm imports match $LEDGER ($(printf '%s\n' "$GOT" | sed '/^$/d' | wc -l) functions) ✓"
    exit 0
fi

echo
if [ -n "$ADDED" ]; then
    echo "Newly imported from the platform libm, not in the ledger:"
    printf '%s\n' "$ADDED" | sed 's/^/  + /'
fi
if [ -n "$REMOVED" ]; then
    echo "In the ledger, but no longer imported:"
    printf '%s\n' "$REMOVED" | sed 's/^/  - /'
fi
cat <<EOF

The extension's platform-libm imports changed. Each import is a function
whose last bits can differ between the machines a wheel runs on. If the
change is intended:
  1. Update the ledger, $LEDGER:
     add or remove the symbol and, under it, name every kernel that
     reaches it, with file and function, per pixel or once per call.
  2. Update the list of platform-libm functions and kernels in
     docs/ffi.md §6 (Determinism) to match.
If it is not intended, find the code that introduced it. A compiler
upgrade can also rewrite one libm call into another.
EOF
exit 1
