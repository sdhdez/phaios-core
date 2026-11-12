#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Digest the output of every phaios_core kernel, and compare two runs.

The question this answers is "did anything change a single output bit?",
for a dependency upgrade, a new toolchain, another libm, another machine,
or the CUDA backend against the CPU. It drives all 27 kernels (plus the
README pipeline end to end) with identity, typical and extreme
parameters over four deterministic test images, at C = 1 and C = 3
wherever a kernel accepts it, and records a SHA-256 of every output.

Needs numpy and the Python standard library only, and Python >= 3.12.
It makes no network access and writes only to the paths given on the
command line, with one exception: under --gpu the CUDA driver writes its
JIT cache, to $CUDA_CACHE_PATH or, when that is unset, to
~/.nv/ComputeCache (--cuda-cache DIR sets it; see --gpu below). It runs
unchanged against older builds of phaios_core: every case is wrapped,
so an exception (even a PanicException) becomes data in the output
instead of stopping the run.

Usage
-----
    python scripts/kernel_digests.py run --out cpu.jsonl [--size 1024]
        [--gpu] [--cuda-cache DIR] [--cases REGEX] [--ids FILE]
        [--extra-channels] [--dump DIR] [--wheel PATH] [--label TEXT]
        [--strict] [--quiet]
    python scripts/kernel_digests.py compare A.jsonl B.jsonl
        [--dumps DIR_A DIR_B] [--bounds] [--expect-identical]
        [--expect-bounded] [--allow REGEX] [--cases REGEX]
        [--write-ids FILE]
    python scripts/kernel_digests.py selftest

`run` writes JSON Lines, creating the parent directory of --out if
needed. The first line is ``{"kind": "env", ...}``: platform, Python,
numpy and phaios_core versions, the extension module's path and SHA-256,
glibc version, CPU model and flags, GLIBC_TUNABLES, RAYON_NUM_THREADS,
the --label text, the wheel's SHA-256 with --wheel, and with --gpu the
device, driver, ``GpuContext.fingerprint`` and the CUDA_* variables that
steer the driver's JIT and its cache. Then one ``{"kind": "case", ...}``
line per case: id, kernel, backend, image, parameters, input shape and
SHA-256, output shape, dtype and SHA-256 (of the C-order little-endian
bytes), NaN and +-inf counts, and the error, if any, as
``"ExceptionType: message"``. The last line is ``{"kind": "summary",
...}``: case counts, errors, wall time and peak RSS. A file without a
summary line is from a run that died (a Rust abort cannot be caught);
its case lines are flushed one by one, so the last one names the case
before the crash.

Case ids are stable and readable, ``kernel/label/image/cN``, for
example ``exposure/stops=-2/hdr/c3``. Cases labelled ``reject-...``
pass parameters or shapes the kernel must refuse; for them an error is
the expected outcome (``"expect": "error"``), and its message is
digested like any output. ``--strict`` makes `run` exit 1 when any case
errs unexpectedly or a reject case succeeds, or when a histogram field
raises anything but AttributeError (a field an older build lacks; the
case line then lists the others under ``field_errors``). --cases REGEX
(re.search) and --ids FILE (exact ids, one per line, ``#`` comments
allowed, as `compare --write-ids` writes them) select cases; both may be
given. `run` exits 2 when the selection is empty or --ids names an id
the catalogue does not have.

--extra-channels adds C = 2 (channels 0 and 1 of the C = 3 image) and
C = 4 (the C = 3 image followed by its channel 0) on the noise and
specials images, for every non-reject case of the 19 kernels that take
any channel count. It is off by default, so a default run keeps the
case set, and the ``case_lines_sha256``, of earlier runs; with it the
extra ids are only-in-one against a run made without it.

--gpu runs the same cases through ``phaios_core.gpu``: GpuContext(0),
upload, kernel, download. ``gpu.quantize_u8``, ``gpu.quantize_u16`` and
``gpu.histogram`` return host values and are digested as returned. Case
lines then say ``"backend": "gpu"``, and the ids match the CPU run's, so
the two files compare directly. The driver JIT-compiles the PTX and
caches the result on disk; set CUDA_CACHE_PATH (or CUDA_CACHE_DISABLE=1)
or pass --cuda-cache DIR, which sets CUDA_CACHE_PATH before phaios_core
is imported. `run` warns on stderr when neither is set. --dump DIR saves
each output as ``DIR/<case id with "/" as "__">.npy`` (a histogram as
``.npz``, one array per field) for `compare --dumps`. A full dump takes
about 350 MB per backend at --size 256 and about 16 times that at the
default size; --cases or --ids narrows it.

`compare` matches case lines by id and reports, per kernel and overall,
identical / differing / error-changed / only-in-one, after printing the
env and summary fields that differ (the summary's ``case_lines_sha256``
alone tells whether two runs' case lines are byte-identical).
--write-ids FILE writes the ids that are not identical, one per line,
for `run --ids`. With --dumps it first checks that each dump it loads
holds the output its run line digests, and then reports, for each
differing case, and then per kernel and image: the differing-element
count; the largest ULP distance over all differing elements (max_ulp_all:
ordered-integer distance, -0 one step below +0) and over pairs that are
both normal, finite and of one sign (max_ulp_normal); NaN-vs-number, NaN
payload, signed-zero and infinity mismatches; flush-to-zero pairs (ftz:
one side +-0, the other subnormal) and sign changes (both nonzero and
finite, sign bits differ); the largest absolute difference; and how many
u8 and u16 codes change under the crate's own quantisation (``x * max``,
NaN and <= 0 to 0, >= max to max, else ``floor(v + 0.5)``, all in
float32: ``quantize_sample`` in src/quantize.rs, without dither).

--bounds (needs --dumps; one run must be CPU and the other GPU, and the
CPU one is the reference) checks every case against the agreement class
docs/ffi.md section 6 gives its kernel (AGREEMENT below). Exact: no value
may differ (NaN payload and sign bits excepted). Bounded, with (rtol,
atol): the conformance suite's ``worst_violation`` (tests/
cuda_conformance.rs), in float32, max over both-finite elements of
``|x - y| / (atol + rtol * |x|)`` with x the CPU value, must be <= 1, and
no element may change class (NaN against a number, or an infinity
against anything but the same infinity); class changes are counted
apart from the ratio. Pass/fail applies on hdr, ramp and noise. On
specials an exact kernel must still not differ, except tone_curve at
power 1 (section 6 exempts subnormals there); zone_system, encode_srgb,
tone_curve and hsl_bw must keep each element's class (NaN, +inf, -inf,
finite); everything else, and the README pipeline everywhere, is
reported only. Identical cases pass without loading a dump.

`compare` exits 0 when nothing below applies; 1 when --expect-identical
was given and any case differs, errs differently or is missing, or
either run has no summary line, or when --expect-bounded (which implies
--bounds) was given and any case fails its class, errs differently or
is missing, in both cases other than those whose id matches --allow; 2
when the runs cannot be compared: a case whose input or parameters
differ, a run whose setup failed, no case left to compare, an unreadable
file, or with --dumps a dump that does not belong to its run (with
--bounds, also a missing dump or a pair of runs that is not CPU and
GPU).

`selftest` needs neither phaios_core nor a GPU. It checks the generator
against its pinned golden hash, and the comparator on synthetic pairs:
one 1-ULP flip must count as exactly one differing element at one ULP,
and the NaN, signed-zero, infinity and code-boundary cases must be
classified as documented. It runs `compare` and `run` on in-memory
inputs for their exit codes, mirrors the conformance suite's test of
``worst_violation``, and checks AGREEMENT against docs/ffi.md when the
file is beside the script (it skips that check otherwise). It exits
non-zero on any failure.

What is promised
----------------
- The inputs are a pure function of --size and the generator version.
  They come from SplitMix64 in numpy uint64 arithmetic and from
  closed-form fields built only from IEEE-754 operations that are
  correctly rounded (+ - * / sqrt, nextafter) and float32 conversions:
  no numpy RNG stream, no transcendental, no reduction. So they should
  be the same bytes on every platform and numpy version (checked so far
  on x86-64 Linux with numpy 1.26.4 and 2.5.4), and `selftest` and
  every `run` (``generator_ok`` in the env line) check that against the
  pinned hash. Each input's SHA-256 is in every case line, and compare
  refuses runs whose inputs differ.
- Case lines carry no timing, address or path, so rerunning the same
  command on the same build gives byte-identical case lines whenever
  the kernels are deterministic, which the crate promises within one
  backend at any thread count (docs/ffi.md section 6).

What is not promised
--------------------
- Equal digests across backends or machines. CPU against CUDA, only
  the bit-exact kernels of docs/ffi.md section 6 should match. On the
  specials image the six of them that do arithmetic on a NaN (exposure,
  luminance_bw, channel_mixer_bw, color_filter_bw, resize, straighten)
  differ in NaN payload and sign, because GPU arithmetic yields
  0x7fffffff (`compare --dumps` counts such cases as nan-bits only);
  the kernels that pass a NaN through unchanged match. The other
  kernels are bounded, and `compare --dumps --bounds` measures them
  against their committed bounds. CPU across libms or OSes is best
  effort; this script is how it is measured.
- A ULP count is not a bound. Where an output cancels towards zero a
  tiny absolute error is a large ULP distance, even on normal values,
  so --bounds decides on the (rtol, atol) ratio and max_ulp_normal is a
  diagnostic beside it.
- A GPU digest is tied to the fingerprint *and* the driver, which the
  fingerprint does not name; the env line records both.
- The u8/u16 code counts quantise each output as it is. For linear
  (scene-referred) outputs that is not what a viewer sees after
  ``encode_srgb``, and it understates shadow changes.
- The env and summary lines differ between runs (times, RSS, argv);
  `compare` prints their differing fields and ignores them otherwise.
- The case list is part of this file. Adding or changing a case changes
  the set of ids, which `compare` reports as only-in-one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # write only to paths given on the command line

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = 1
GENERATOR = "splitmix64-v1"
GOLDEN_SIZE = 64
# sha256 over every generated input at --size 64; see generator_digest().
GOLDEN_SHA256 = "90c0a6029c111a7f219e83f2f58f00f6424b0cb987de4099d0a0f7631078c93e"

KERNELS = (
    "crop", "orient", "hot_pixels", "denoise", "resize", "straighten",
    "exposure", "luminance_bw", "channel_mixer_bw", "color_filter_bw",
    "hsl_bw", "zone_system", "tone_curve", "blur", "glow",
    "highlight_rolloff", "shadow_rolloff", "quantize_u8", "quantize_u16",
    "histogram", "apply_lut", "local_contrast", "sharpen", "film_grain",
    "split_toning", "vignette", "encode_srgb",
)
IMAGES = ("hdr", "ramp", "noise", "specials")
ANY, RGB, MONO = (1, 3), (3,), (1,)
# --extra-channels: more channel counts for the kernels that take ANY, on these images.
EXTRA_CHANNELS, EXTRA_IMAGES = (2, 4), ("noise", "specials")
U64_MAX = (1 << 64) - 1
SPECIALS_SHAPE = (47, 83)  # odd and landscape: edge blocks, W/H swaps


# ── Deterministic inputs ──────────────────────────────────────────────────────

_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_U = np.uint64


def splitmix64(seed: int, n: int) -> np.ndarray:
    """The first ``n`` outputs of SplitMix64 started from ``seed``.

    Output i is mix(seed + (i + 1) * gamma), exactly the reference
    generator's sequence; uint64 array arithmetic wraps modulo 2**64.
    """
    state = np.arange(1, n + 1, dtype=np.uint64) * _U(_GAMMA) + _U(seed & U64_MAX)
    z = state ^ (state >> _U(30))
    z = z * _U(_M1)
    z = z ^ (z >> _U(27))
    z = z * _U(_M2)
    return z ^ (z >> _U(31))


def _splitmix64_ref(seed: int, n: int) -> list[int]:
    """Scalar reference, in Python integers, for the self-test."""
    out, s = [], seed & U64_MAX
    for _ in range(n):
        s = (s + _GAMMA) & U64_MAX
        z = s
        z = ((z ^ (z >> 30)) * _M1) & U64_MAX
        z = ((z ^ (z >> 27)) * _M2) & U64_MAX
        out.append(z ^ (z >> 31))
    return out


def uniform24(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """float32 in [0, 1): the top 24 bits of each output, times 2**-24 (exact)."""
    n = math.prod(shape)
    top = (splitmix64(seed, n) >> _U(40)).astype(np.float32)
    return (top * np.float32(2.0**-24)).reshape(shape)


def _bits(*words: int) -> np.ndarray:
    return np.array(words, dtype=np.uint32).view(np.float32)


def make_hdr(n: int) -> np.ndarray:
    """(n, n, 3): a smooth gradient from ~1e-6 at the left edge to ~1,
    three highlights, one clipped to an exact 16.0 plateau, hue varying
    with position, and a little multiplicative and additive texture."""
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    yy, xx = y[:, None], x[None, :]
    x2 = xx * xx
    field = (x2 * x2) * (0.25 + 0.75 * yy)

    def bump(cx: float, cy: float, r: float, peak: float) -> np.ndarray:
        # peak / (1 + d^8): flat-topped, with tails short enough to leave
        # the far corner near 1e-6.
        dx, dy = xx - cx, yy - cy
        d2 = (dx * dx + dy * dy) / (r * r)
        d4 = d2 * d2
        return peak / (1.0 + d4 * d4)

    field = field + bump(0.70, 0.30, 0.05, 12.0) + bump(0.25, 0.70, 0.12, 6.0)
    field = field + bump(0.85, 0.82, 0.04, 24.0)
    a, b = 2.0 * xx - 1.0, 2.0 * yy - 1.0
    tints = (0.5 + 0.5 * a, 0.5 + 0.5 * b, 0.5 - 0.25 * (a + b))
    tex = uniform24(0x50480001, (n, n, 3)).astype(np.float64)
    dark = uniform24(0x50480002, (n, n, 3)).astype(np.float64)
    out = np.empty((n, n, 3), dtype=np.float64)
    for c, w in enumerate(tints):
        out[:, :, c] = field * (0.2 + 0.8 * w) * (0.95 + 0.1 * tex[:, :, c]) + 1e-6 * dark[:, :, c]
    return np.minimum(np.maximum(out, 0.0), 16.0).astype(np.float32)


def _ramp_set(max_code: int) -> np.ndarray:
    """Every code k/max and every rounding threshold (k + 0.5)/max, each
    with its float32 neighbours one ULP either side."""
    codes = np.arange(max_code + 1, dtype=np.float32) / np.float32(max_code)
    halves = np.arange(1, 2 * max_code, 2, dtype=np.float32) / np.float32(2 * max_code)
    v = np.concatenate([codes, halves])
    return np.concatenate([np.nextafter(v, np.float32(-np.inf)), v,
                           np.nextafter(v, np.float32(np.inf))])


def make_ramp(n: int) -> np.ndarray:
    """(n, n, 3): a sorted ramp through every u8 code and rounding
    threshold (+-1 ULP) and as many u16 ones as fit (all from n = 629),
    row-major; channels 1 and 2 are the ramp rotated by a third."""
    u8, u16 = np.sort(_ramp_set(255)), np.sort(_ramp_set(65535))
    m = n * n
    if m >= u8.size + u16.size:
        seq = np.resize(np.sort(np.concatenate([u8, u16])), m)
    else:
        k = m - u8.size
        idx = (np.arange(k, dtype=np.int64) * (u16.size - 1)) // max(k - 1, 1)
        seq = np.sort(np.concatenate([u8, u16[idx]]))
    chans = [np.roll(seq, c * (m // 3)) for c in range(3)]
    return np.stack(chans, axis=-1).reshape(n, n, 3)


def noise_shape(n: int) -> tuple[int, int]:
    return (3 * n) // 4 + 3, n + 1  # landscape, odd: not a block multiple


def make_noise(n: int) -> np.ndarray:
    h, w = noise_shape(n)
    return uniform24(0x50480003, (h, w, 3))


SPECIAL_VALUES = np.concatenate([
    _bits(
        0x7FC00000, 0xFFC00000, 0x7FC12345,  # quiet NaN, -NaN, NaN with payload
        0x7F800000, 0xFF800000,              # +inf, -inf
        0x00000000, 0x80000000,              # +0, -0
        0x00000001, 0x007FFFFF, 0x80000001,  # min / max subnormal, -min subnormal
        0x00800000,                          # FLT_MIN
        0x7F7FFFFF, 0xFF7FFFFF,              # +-FLT_MAX
        0x3F800000, 0x3F800001, 0x3F7FFFFF,  # 1, 1 + ulp, 1 - ulp
    ),
    np.array([1e-40, -1e-40, -1e-3, -0.5, -10.0, -1e30, 65504.0, 1e30, 1e8, 0.5],
             dtype=np.float64).astype(np.float32),
])
_DARK = np.array([1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
                  1e-3, 2e-3, 5e-3, 1e-2], dtype=np.float64).astype(np.float32)


def make_specials() -> np.ndarray:
    """(47, 83, 3), whatever --size: rows 0-7 a dark field from 1e-6 to
    1e-2; rows 8-15 bars at 1e0, 1e2, 1e4, 1e6 and 1e8; rows 16-43 noise
    with one sample in eight replaced by a special value (NaN, +-inf,
    +-0, subnormals, FLT_MAX, negatives, huge); row 44 all -1e-3, row 45
    exact zeros, row 46 one special per pixel in every channel."""
    h, w = SPECIALS_SHAPE
    img = uniform24(0x50480004, (h, w, 3))
    tex = uniform24(0x50480005, (h, w, 3))
    cols = np.arange(w)
    img[0:8] = _DARK[cols % _DARK.size][None, :, None] * (np.float32(1.0) + np.float32(0.5) * tex[0:8])
    bars = np.array([1e0, 1e2, 1e4, 1e6, 1e8], dtype=np.float32)
    img[8:16] = bars[(cols * 5) // w][None, :, None] * (np.float32(1.0) + np.float32(0.01) * tex[8:16])
    hsh = splitmix64(0x50480006, h * w * 3).reshape(h, w, 3)
    pick = (hsh & _U(7)) == _U(0)
    pick[:16] = False
    pick[44:] = False
    which = ((hsh >> _U(3)) % _U(SPECIAL_VALUES.size)).astype(np.int64)
    img[pick] = SPECIAL_VALUES[which[pick]]
    img[44] = np.float32(-1e-3)
    img[45] = np.float32(0.0)
    img[46] = SPECIAL_VALUES[cols % SPECIAL_VALUES.size][:, None]
    return img


def min_size() -> int:
    """The smallest --size whose ramp holds every u8 code and threshold."""
    return math.isqrt(_ramp_set(255).size - 1) + 1


def make_images(n: int, extra_channels: bool = False) -> dict[str, np.ndarray]:
    """Every input, keyed "<image>/c<C>". C = 1 is channel 0 of C = 3.
    With ``extra_channels``, also C = 2 (channels 0 and 1) and C = 4 (C = 3
    followed by channel 0) of the EXTRA_IMAGES; these derive from the
    C = 3 bytes by slicing alone, so the golden hash, which covers the
    default set only, still pins them."""
    base = {"hdr": make_hdr(n), "ramp": make_ramp(n), "noise": make_noise(n),
            "specials": make_specials()}
    out = {}
    for name in IMAGES:
        img3 = np.ascontiguousarray(base[name], dtype=np.float32)
        assert img3.dtype == np.float32 and img3.ndim == 3 and img3.shape[2] == 3
        out[f"{name}/c3"] = img3
        out[f"{name}/c1"] = np.ascontiguousarray(img3[:, :, :1])
        if extra_channels and name in EXTRA_IMAGES:
            out[f"{name}/c2"] = np.ascontiguousarray(img3[:, :, :2])
            out[f"{name}/c4"] = np.ascontiguousarray(np.concatenate([img3, img3[:, :, :1]], axis=2))
    return out


def generator_digest(images: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(images):
        a = images[key]
        h.update(f"{key}:{list(a.shape)}:{sha256_array(a)}\n".encode())
    return h.hexdigest()


# ── Digests ───────────────────────────────────────────────────────────────────


def canonical(a: np.ndarray) -> np.ndarray:
    """C-order, little-endian: the byte layout every digest is taken of."""
    a = np.asarray(a)
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(canonical(a).tobytes()).hexdigest()


def array_digest(a: np.ndarray) -> dict:
    c = canonical(a)
    if c.dtype.kind in "fc":
        nan = int(np.count_nonzero(np.isnan(c)))
        posinf = int(np.count_nonzero(np.isposinf(c)))
        neginf = int(np.count_nonzero(np.isneginf(c)))
    else:
        nan = posinf = neginf = 0
    return {"output_shape": list(c.shape), "output_dtype": str(c.dtype.newbyteorder("=")),
            "sha256": hashlib.sha256(c.tobytes()).hexdigest(),
            "nan": nan, "posinf": posinf, "neginf": neginf}


HIST_ARRAYS = ("counts", "cdf", "equalisation_lut")
HIST_LISTS = ("below", "above", "non_finite")
HIST_SCALARS = ("bins", "channels", "min", "max")


def histogram_fields(hist) -> dict[str, np.ndarray | str]:
    """Every numeric field of a Histogram as an array; an error string for
    a field this build lacks or fails to produce."""
    fields: dict[str, np.ndarray | str] = {}

    def grab(name, fn):
        try:
            fields[name] = fn()
        except BaseException as e:  # noqa: BLE001 - PanicException is a BaseException
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            fields[name] = f"{type(e).__name__}: {e}"

    for name in HIST_ARRAYS:
        grab(name, lambda name=name: np.asarray(getattr(hist, name)()))
    for name in HIST_LISTS:
        grab(name, lambda name=name: np.asarray(getattr(hist, name)(), dtype=np.uint64))
    grab("totals", lambda: np.asarray([hist.total(c) for c in range(hist.channels)], dtype=np.uint64))
    for name in HIST_SCALARS:
        dt = np.float32 if name in ("min", "max") else np.uint64
        grab(name, lambda name=name, dt=dt: np.asarray(getattr(hist, name), dtype=dt))
    return fields


def histogram_digest(fields: dict[str, np.ndarray | str]) -> dict:
    per = {}
    nan = posinf = neginf = 0
    for name, v in fields.items():
        if isinstance(v, str):
            per[name] = {"error": v}
            continue
        d = array_digest(v)
        nan, posinf, neginf = nan + d["nan"], posinf + d["posinf"], neginf + d["neginf"]
        per[name] = {"shape": d["output_shape"], "dtype": d["output_dtype"], "sha256": d["sha256"]}
    counts = fields.get("counts")
    joint = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"output_shape": list(counts.shape) if isinstance(counts, np.ndarray) else None,
            "output_dtype": "Histogram", "sha256": joint,
            "nan": nan, "posinf": posinf, "neginf": neginf, "fields": per}


# ── The case catalogue ────────────────────────────────────────────────────────


class P:
    """A parameter object, built lazily as ``phaios_core.<cls>(**kw)``, so
    a class or keyword an older build lacks fails its case, not the run."""

    __slots__ = ("cls", "kw")

    def __init__(self, cls: str, **kw: Any) -> None:
        self.cls, self.kw = cls, kw


@dataclass(frozen=True)
class E:
    """An enum value, built lazily as ``phaios_core.<cls>.<name>``."""
    cls: str
    name: str


@dataclass(frozen=True)
class Case:
    id: str
    kernel: str
    image: str
    channels: int
    args: tuple
    expect: str = "ok"


def build(spec, ph):
    if isinstance(spec, P):
        return getattr(ph, spec.cls)(**{k: build(v, ph) for k, v in spec.kw.items()})
    if isinstance(spec, E):
        return getattr(getattr(ph, spec.cls), spec.name)
    if isinstance(spec, list):
        return [build(v, ph) for v in spec]
    if isinstance(spec, dict):
        return {k: build(v, ph) for k, v in spec.items()}
    return spec


def to_json(spec):
    if isinstance(spec, P):
        return {spec.cls: {k: to_json(v) for k, v in spec.kw.items()}}
    if isinstance(spec, E):
        return f"{spec.cls}.{spec.name}"
    if isinstance(spec, np.ndarray):
        return {"ndarray": {"dtype": str(spec.dtype), "shape": list(spec.shape),
                            "sha256": sha256_array(spec)}}
    if isinstance(spec, dict):
        return {str(k): to_json(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return [to_json(v) for v in spec]
    if isinstance(spec, float) and not math.isfinite(spec):
        return repr(spec)
    return spec


def _luts() -> dict[str, np.ndarray]:
    k256 = np.arange(256, dtype=np.float32)
    t = -1.0 + 17.0 * np.arange(4096, dtype=np.float64) / 4095.0
    tp = np.maximum(t, 0.0)
    return {
        "identity2": np.array([0.0, 1.0], dtype=np.float32),
        "sqrt256": np.sqrt(k256 / np.float32(255)),
        # Non-monotone: solarisation, the one curve tone_curve cannot make.
        "solarise17": np.array([0, .25, .5, .75, 1, .75, .5, .25, 0, .25, .5, .75, 1, .75, .5, .25, 0],
                               dtype=np.float32),
        "hd4096": (tp / (tp + 0.18)).astype(np.float32),
        "len1": np.array([0.5], dtype=np.float32),
        "nan-entry": np.array([0.0, np.nan, 1.0], dtype=np.float32),
    }


ORIENTATIONS = ("Normal", "FlipHorizontal", "Rotate180", "FlipVertical",
                "Transpose", "Rotate90", "Transverse", "Rotate270")
PIPELINES = ("readme-u16", "readme-u8-tpdf", "readme-float")
_LABEL_OK = re.compile(r"^[A-Za-z0-9._=+,-]+$")


def catalog(shapes: dict[str, tuple[int, int]], extra_channels: bool = False) -> list[Case]:
    """Every case, in a fixed order. ``shapes`` maps image name to (H, W);
    only crop and resize parameters depend on it. ``extra_channels`` adds
    C = 2 and C = 4 on EXTRA_IMAGES for every non-reject case of a kernel
    that takes ANY channel count, after that case's C = 1 and C = 3 ids,
    and leaves every other id and its position unchanged."""
    cases: list[Case] = []
    lut = _luts()

    def add(kernel, label, args, chans=ANY, images=IMAGES, expect="ok"):
        assert _LABEL_OK.match(label), label
        for img in images:
            for c in chans:
                a = args(*shapes[img]) if callable(args) else args
                cases.append(Case(f"{kernel}/{label}/{img}/c{c}", kernel, img, c, tuple(a), expect))
        if extra_channels and chans == ANY and expect == "ok":
            for img in images:
                if img not in EXTRA_IMAGES:
                    continue
                for c in EXTRA_CHANNELS:
                    a = args(*shapes[img]) if callable(args) else args
                    cases.append(Case(f"{kernel}/{label}/{img}/c{c}", kernel, img, c, tuple(a), expect))

    def reject(kernel, label, args, c=3, img="noise"):
        add(kernel, "reject-" + label, args, chans=(c,), images=(img,), expect="error")

    # Geometry
    add("crop", "full", lambda h, w: (P("CropParams", x=0, y=0, width=w, height=h),))
    add("crop", "centre-half", lambda h, w: (P("CropParams", x=w // 4, y=h // 4, width=w // 2, height=h // 2),))
    add("crop", "odd-offset", lambda h, w: (P("CropParams", x=3, y=5, width=w - 10, height=h - 7),))
    add("crop", "1x1-corner", lambda h, w: (P("CropParams", x=w - 1, y=h - 1, width=1, height=1),))
    reject("crop", "past-edge", lambda h, w: (P("CropParams", x=1, y=0, width=w, height=h),))

    for o in ORIENTATIONS:
        add("orient", o, (E("Orientation", o),))

    add("hot_pixels", "t=0.04,r=0.02", (P("HotPixelParams", threshold=0.04, relative=0.02),))
    add("hot_pixels", "t=0.1", (P("HotPixelParams", threshold=0.1),))
    add("hot_pixels", "t=0-median", (P("HotPixelParams", threshold=0.0),))
    add("hot_pixels", "t=1e30,r=0.5", (P("HotPixelParams", threshold=1e30, relative=0.5),))
    reject("hot_pixels", "t=-1", (P("HotPixelParams", threshold=-1.0),))

    add("denoise", "none", ())
    add("denoise", "default", (P("DenoiseParams"),))
    add("denoise", "r=4,s=0.01,a=0.6", (P("DenoiseParams", radius=4, noise_sigma=0.01, amount=0.6),))
    add("denoise", "r=1,s=0.001,a=0.3,Bt2020",
        (P("DenoiseParams", radius=1, noise_sigma=0.001, amount=0.3,
           standard=E("LuminanceStandard", "Bt2020")),))
    add("denoise", "r=32,s=0.1,a=1", (P("DenoiseParams", radius=32, noise_sigma=0.1, amount=1.0),))
    add("denoise", "r=0,s=0,a=1", (P("DenoiseParams", radius=0, noise_sigma=0.0, amount=1.0),))
    reject("denoise", "r=33", (P("DenoiseParams", radius=33, noise_sigma=0.01, amount=0.5),))

    for f in ("Area", "Bilinear", "CatmullRom"):
        def rp(w, h, f=f):
            return (P("ResizeParams", width=w, height=h, filter=E("ResizeFilter", f)),)
        add("resize", f"{f},same", lambda h, w, rp=rp: rp(w, h))
        add("resize", f"{f},half", lambda h, w, rp=rp: rp(max(w // 2, 1), max(h // 2, 1)))
        add("resize", f"{f},0.37x0.41", lambda h, w, rp=rp: rp(max(w * 37 // 100, 1), max(h * 41 // 100, 1)))
        add("resize", f"{f},1.5x", lambda h, w, rp=rp: rp(w * 3 // 2, h * 3 // 2))
        add("resize", f"{f},1x1", lambda h, w, rp=rp: rp(1, 1))
    reject("resize", "width=0", (P("ResizeParams", width=0, height=8),))

    for d in (0.0, 0.1, 1.0, 7.3, -13.37, 44.9, -45.0):
        add("straighten", f"deg={d:g}", (P("StraightenParams", degrees=d),))
    reject("straighten", "deg=45.5", (P("StraightenParams", degrees=45.5),))
    reject("straighten", "deg=nan", (P("StraightenParams", degrees=math.nan),))

    # Tone, before B&W
    for label, stops in (("0", 0.0), ("1", 1.0), ("-2", -2.0), ("0.5", 0.5), ("1by3", 1 / 3),
                         ("2.7", 2.7), ("40", 40.0), ("-140", -140.0)):
        add("exposure", f"stops={label}", (stops,))
    reject("exposure", "stops=inf", (math.inf,))
    reject("exposure", "stops=nan", (math.nan,))

    # B&W conversion
    add("luminance_bw", "none", (), chans=RGB)
    for std in ("Bt601", "Bt709", "Bt2020"):
        add("luminance_bw", std, (E("LuminanceStandard", std),), chans=RGB)
    reject("luminance_bw", "c1", (), c=1)

    for label, w in (("equal", (1 / 3, 1 / 3, 1 / 3)), ("0.4,-0.3,1.1", (0.4, -0.3, 1.1)),
                     ("1,0,0", (1.0, 0.0, 0.0)), ("2,-2,0.5", (2.0, -2.0, 0.5)),
                     ("0,0,0", (0.0, 0.0, 0.0))):
        add("channel_mixer_bw", f"w={label}", w, chans=RGB)
    reject("channel_mixer_bw", "c1", (0.3, 0.6, 0.1), c=1)

    add("color_filter_bw", "none", (), chans=RGB)
    for f in ("NoFilter", "Yellow8K2", "Orange21", "Red25A", "Green11X1", "Blue47C5"):
        add("color_filter_bw", f, (E("ColorFilter", f),), chans=RGB)
    for std in ("Bt601", "Bt2020"):
        add("color_filter_bw", f"Red25A,{std}", (E("ColorFilter", "Red25A"), E("LuminanceStandard", std)),
            chans=RGB)
    reject("color_filter_bw", "c1", (E("ColorFilter", "Red25A"),), c=1)

    add("hsl_bw", "zero", (P("HslWeightedParams", hue_weights=[0.0] * 8),), chans=RGB)
    add("hsl_bw", "readme", (P("HslWeightedParams", hue_weights=[0, 0, 0.4, 0.2, 0, -0.5, 0, 0]),), chans=RGB)
    add("hsl_bw", "alt1,s=5",
        (P("HslWeightedParams", hue_weights=[1.0, -1.0] * 4, sigma_deg=5.0),), chans=RGB)
    add("hsl_bw", "s=90,Bt2020",
        (P("HslWeightedParams", hue_weights=[0.3, -0.2, 0.5, 0.0, 0.1, -0.4, 0.2, 0.0],
           standard=E("LuminanceStandard", "Bt2020"), sigma_deg=90.0),), chans=RGB)
    reject("hsl_bw", "s=0", (P("HslWeightedParams", hue_weights=[0.0] * 8, sigma_deg=0.0),))
    reject("hsl_bw", "c1", (P("HslWeightedParams", hue_weights=[0.0] * 8),), c=1)

    # Tone stages
    all11 = {z: round(-1.0 + 0.16 * z, 2) for z in range(11)}
    for label, offs in (("empty", {}), ("z5=0.5", {5: 0.5}), ("all11", all11),
                        ("readme", {3: -0.3, 7: 0.4}), ("z0=3,z10=-3", {0: 3.0, 10: -3.0})):
        add("zone_system", label, (P("ZoneParams", offsets=offs),), chans=MONO)
    reject("zone_system", "z11", (P("ZoneParams", offsets={11: 0.1}),), c=1)
    reject("zone_system", "c3", (P("ZoneParams", offsets={5: 0.5}),), c=3)

    for label, (slope, offset, power) in (
            ("identity", (1.0, 0.0, 1.0)),
            ("p=0.45", (1.0, 0.0, 0.45)), ("p=1by2.2", (1.0, 0.0, 1 / 2.2)),
            ("p=2.2", (1.0, 0.0, 2.2)), ("p=3", (1.0, 0.0, 3.0)),
            ("s=1.15,o=-0.02,p=1", (1.15, -0.02, 1.0)), ("s=0.5,o=0.25,p=1", (0.5, 0.25, 1.0)),
            ("o=-0.05,p=0.45", (1.0, -0.05, 0.45)), ("readme", (1.1, 0.0, 0.9)),
            ("s=8,o=0.01,p=0.1", (8.0, 0.01, 0.1))):
        add("tone_curve", label, (P("ToneCurveParams", slope=slope, offset=offset, power=power),))
    reject("tone_curve", "p=0", (P("ToneCurveParams", power=0.0),))

    add("blur", "none", ())
    for s in (0.0, 0.5, 2.0, 5.9, 6.0, 8.0, 64.0):
        add("blur", f"s={s:g}", (P("BlurParams", sigma=s),))
    reject("blur", "s=-1", (P("BlurParams", sigma=-1.0),))
    reject("blur", "s=5000", (P("BlurParams", sigma=5000.0),))

    add("glow", "none", ())
    add("glow", "default", (P("GlowParams"),))
    for s in (0.5, 2.0, 5.9):
        add("glow", f"t=0.8,s={s:g},a=0.35", (P("GlowParams", threshold=0.8, sigma=s, amount=0.35),))
    add("glow", "veil,t=0,s=64,a=0.06", (P("GlowParams", threshold=0.0, sigma=64.0, amount=0.06),))
    add("glow", "t=0.5,s=8,a=1", (P("GlowParams", threshold=0.5, sigma=8.0, amount=1.0),))
    reject("glow", "a=-0.1", (P("GlowParams", amount=-0.1),))

    add("highlight_rolloff", "default", (P("RolloffParams"),))
    add("highlight_rolloff", "k=0.75,wp=4", (P("RolloffParams", knee=0.75, white_point=4.0),))
    add("highlight_rolloff", "k=0,wp=16", (P("RolloffParams", knee=0.0, white_point=16.0),))
    add("highlight_rolloff", "k=1,wp=1e6", (P("RolloffParams", knee=1.0, white_point=1e6),))
    reject("highlight_rolloff", "k=1.5", (P("RolloffParams", knee=1.5, white_point=4.0),))
    reject("highlight_rolloff", "wp=0.5", (P("RolloffParams", knee=0.5, white_point=0.5),))

    add("shadow_rolloff", "none", ())
    add("shadow_rolloff", "default", (P("ShadowRolloffParams"),))
    for k, s in ((0.2, 0.5), (0.2, 0.6), (0.05, 0.9), (1.0, 1.0)):
        add("shadow_rolloff", f"k={k:g},s={s:g}", (P("ShadowRolloffParams", knee=k, strength=s),))
    reject("shadow_rolloff", "s=2", (P("ShadowRolloffParams", knee=0.2, strength=2.0),))

    # Terminal and analysis
    for q in ("quantize_u8", "quantize_u16"):
        add(q, "none", ())
        add(q, "off", (P("QuantizeParams"),))
        add(q, "off,seed=7", (P("QuantizeParams", dither=E("Dither", "Off"), seed=7),))
        add(q, "tpdf,seed=7", (P("QuantizeParams", dither=E("Dither", "Tpdf"), seed=7),))
        add(q, "tpdf,seed=20260825", (P("QuantizeParams", dither=E("Dither", "Tpdf"), seed=20260825),))
        add(q, "tpdf,seed=max", (P("QuantizeParams", dither=E("Dither", "Tpdf"), seed=U64_MAX),))

    add("histogram", "none", ())
    add("histogram", "bins=256", (P("HistogramParams", bins=256),))
    add("histogram", "bins=65536", (P("HistogramParams", bins=65536),))
    add("histogram", "bins=1024,-1..16", (P("HistogramParams", bins=1024, min=-1.0, max=16.0),))
    add("histogram", "bins=2", (P("HistogramParams", bins=2),))
    reject("histogram", "bins=1", (P("HistogramParams", bins=1),))
    reject("histogram", "min=max", (P("HistogramParams", bins=16, min=0.5, max=0.5),))

    add("apply_lut", "identity2,none", (lut["identity2"],))
    add("apply_lut", "sqrt256", (lut["sqrt256"], P("LutParams")))
    add("apply_lut", "sqrt256,none", (lut["sqrt256"],))
    add("apply_lut", "solarise17,0..8", (lut["solarise17"], P("LutParams", min=0.0, max=8.0)))
    add("apply_lut", "hd4096,-1..16", (lut["hd4096"], P("LutParams", min=-1.0, max=16.0)))
    reject("apply_lut", "len1", (lut["len1"],))
    reject("apply_lut", "nan-entry", (lut["nan-entry"],))

    # Detail
    for label, (r, eps, s) in (("r=8,e=0.01,s=0.4", (8, 0.01, 0.4)), ("r=8,e=0.01,s=0", (8, 0.01, 0.0)),
                               ("r=1,e=1e-4,s=1", (1, 1e-4, 1.0)), ("r=32,e=0.1,s=-0.5", (32, 0.1, -0.5)),
                               ("r=64,e=0,s=2", (64, 0.0, 2.0))):
        add("local_contrast", label, (P("GuidedFilterParams", radius=r, eps=eps), s), chans=MONO)
    reject("local_contrast", "e=-1", (P("GuidedFilterParams", radius=8, eps=-1.0), 0.4), c=1)
    reject("local_contrast", "c3", (P("GuidedFilterParams", radius=8, eps=0.01), 0.4), c=3)

    add("sharpen", "none", ())
    add("sharpen", "default", (P("SharpenParams"),))
    for s in (0.5, 2.0, 5.9):
        add("sharpen", f"a=0.5,s={s:g},t=0.02", (P("SharpenParams", amount=0.5, sigma=s, threshold=0.02),))
    add("sharpen", "a=2,s=8,t=0", (P("SharpenParams", amount=2.0, sigma=8.0, threshold=0.0),))
    add("sharpen", "a=0.5,s=0", (P("SharpenParams", amount=0.5, sigma=0.0),))
    reject("sharpen", "a=-1", (P("SharpenParams", amount=-1.0, sigma=1.0),))

    # Grain, toning, finishing
    for label, seed in (("0", 0), ("20260815", 20260815), ("max", U64_MAX)):
        add("film_grain", f"seed={label}", (P("GrainParams", intensity=0.12, size_pixels=1.5, seed=seed),),
            chans=MONO)
    add("film_grain", "i=0", (P("GrainParams", intensity=0.0, size_pixels=1.5, seed=1),), chans=MONO)
    add("film_grain", "i=1,size=4", (P("GrainParams", intensity=1.0, size_pixels=4.0, seed=2),), chans=MONO)
    add("film_grain", "i=0.25,size=0.5", (P("GrainParams", intensity=0.25, size_pixels=0.5, seed=3),),
        chans=MONO)
    reject("film_grain", "size=0", (P("GrainParams", intensity=0.1, size_pixels=0.0),), c=1)
    reject("film_grain", "c3", (P("GrainParams", intensity=0.1, size_pixels=1.0),), c=3)

    add("split_toning", "default", (P("SplitToningParams"),), chans=MONO)
    add("split_toning", "readme",
        (P("SplitToningParams", shadow_oklab=[0.0, -0.02, -0.04], highlight_oklab=[0.0, 0.03, 0.03]),),
        chans=MONO)
    add("split_toning", "heavy,p=0,b=1",
        (P("SplitToningParams", shadow_oklab=[0.0, 0.1, -0.1], highlight_oklab=[0.0, -0.1, 0.1],
           pivot=0.0, balance=1.0),), chans=MONO)
    add("split_toning", "p=1,b=-1",
        (P("SplitToningParams", shadow_oklab=[0.5, -0.05, 0.02], highlight_oklab=[0.5, 0.04, 0.06],
           pivot=1.0, balance=-1.0),), chans=MONO)
    reject("split_toning", "p=1.5", (P("SplitToningParams", pivot=1.5),), c=1)
    reject("split_toning", "c3", (P("SplitToningParams"),), c=3)

    add("vignette", "default", (P("VignetteParams"),))
    add("vignette", "a=0.35,f=0.8", (P("VignetteParams", amount=0.35, feather=0.8),))
    add("vignette", "a=-1,f=0.5,r=1", (P("VignetteParams", amount=-1.0, feather=0.5, roundness=1.0),))
    add("vignette", "a=1,f=1,r=0.5", (P("VignetteParams", amount=1.0, feather=1.0, roundness=0.5),))
    add("vignette", "a=3,f=0.02", (P("VignetteParams", amount=3.0, feather=0.02),))
    add("vignette", "a=0.5,f=0", (P("VignetteParams", amount=0.5, feather=0.0),))
    reject("vignette", "f=2", (P("VignetteParams", amount=0.3, feather=2.0),))
    reject("vignette", "a=nan", (P("VignetteParams", amount=math.nan),))

    add("encode_srgb", "default", ())

    for p in PIPELINES:
        add("pipeline", p, (), chans=RGB)
    return cases


def readme_pipeline(m, ph, x, variant: str):
    """The README's Python pipeline, verbatim, on module ``m`` (phaios_core
    or phaios_core.gpu) with ``ph``'s parameter classes."""
    x = m.exposure(x, 0.5)
    x = m.hsl_bw(x, ph.HslWeightedParams([0, 0, 0.4, 0.2, 0, -0.5, 0, 0]))
    x = m.zone_system(x, ph.ZoneParams({3: -0.3, 7: 0.4}))
    x = m.local_contrast(x, ph.GuidedFilterParams(radius=8, eps=0.01), 0.4)
    x = m.shadow_rolloff(x, ph.ShadowRolloffParams(knee=0.2, strength=0.5))
    x = m.tone_curve(x, ph.ToneCurveParams(slope=1.1, power=0.9))
    x = m.film_grain(x, ph.GrainParams(intensity=0.12, size_pixels=1.5, seed=20260815))
    x = m.split_toning(x, ph.SplitToningParams([0.0, -0.02, -0.04], [0.0, 0.03, 0.03]))
    x = m.vignette(x, ph.VignetteParams(amount=0.35, feather=0.8))
    x = m.highlight_rolloff(x, ph.RolloffParams(knee=0.75, white_point=4.0))
    x = m.encode_srgb(x)
    if variant == "readme-float":
        return x
    if variant == "readme-u16":
        return m.quantize_u16(x)
    return m.quantize_u8(x, ph.QuantizeParams(ph.Dither.Tpdf, seed=20260825))


def pipeline_params(variant: str):
    terminal = {"readme-u16": "quantize_u16", "readme-u8-tpdf": "quantize_u8(Tpdf, seed=20260825)",
                "readme-float": "encode_srgb"}[variant]
    return {"pipeline": "README.md Python example", "terminal": terminal}


def dump_name(case_id: str) -> str:
    return case_id.replace("/", "__")


# ── run ───────────────────────────────────────────────────────────────────────


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _guard(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - environment probes are best effort
        return default


def _cpu_info() -> tuple[str | None, str | None]:
    model = flags = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if model is None and key in ("model name", "Hardware", "Model"):
                    model = val
                if flags is None and key in ("flags", "Features"):
                    flags = val
                if model is not None and flags is not None:
                    break
    except OSError:
        pass
    return model or platform.processor() or None, flags


def _extension_path(ph) -> str | None:
    inner = sys.modules.get(f"{ph.__name__}.phaios_core")
    f = getattr(inner, "__file__", None)
    if f and not f.endswith(".py"):
        return f
    f = getattr(ph, "__file__", None)
    if f and not f.endswith(".py"):
        return f
    if f:
        found = sorted(p for p in Path(f).parent.iterdir()
                       if p.suffix in (".so", ".pyd") or ".so." in p.name)
        if found:
            return str(found[0])
    return None


def _wheel_extension_sha256(wheel: str) -> str | None:
    with zipfile.ZipFile(wheel) as z:
        for name in sorted(z.namelist()):
            if name.startswith("phaios_core/") and (name.endswith(".so") or name.endswith(".pyd")):
                return hashlib.sha256(z.read(name)).hexdigest()
    return None


def _nvidia_driver() -> str | None:
    """The kernel driver's version number, from /proc (Linux only)."""
    m = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", Path("/proc/driver/nvidia/version").read_text())
    return m.group(1) if m else None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def env_record(ns, ph, ph_error, gen_ok, gpu_fields) -> dict:
    so = _guard(lambda: _extension_path(ph)) if ph is not None else None
    model, flags = _cpu_info()
    rec = {
        "kind": "env", "format": FORMAT, "script": "scripts/kernel_digests.py",
        "generator": GENERATOR, "generator_ok": gen_ok,
        "argv": sys.argv[1:], "size": ns.size, "cases_regex": ns.cases,
        "ids_file": ns.ids, "extra_channels": ns.extra_channels, "label": ns.label,
        "backend": "gpu" if ns.gpu else "cpu",
        "platform": platform.platform(), "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "phaios_core_version": getattr(ph, "__version__", None) if ph is not None else None,
        "phaios_core_import_error": ph_error,
        "module_file": getattr(ph, "__file__", None) if ph is not None else None,
        "extension_file": so,
        "extension_sha256": _guard(lambda: _sha256_file(so)) if so else None,
        "glibc": _guard(lambda: os.confstr("CS_GNU_LIBC_VERSION")),
        "cpu_model": model, "cpu_flags": flags, "cpu_count": os.cpu_count(),
        "GLIBC_TUNABLES": os.environ.get("GLIBC_TUNABLES"),
        "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
    }
    if ns.wheel:
        rec["wheel"] = os.path.basename(ns.wheel)
        rec["wheel_sha256"] = _guard(lambda: _sha256_file(ns.wheel))
        wext = _guard(lambda: _wheel_extension_sha256(ns.wheel))
        rec["wheel_extension_sha256"] = wext
        rec["wheel_matches_module"] = (wext is not None and wext == rec["extension_sha256"])
    if ns.gpu:
        rec.update(gpu_fields)
        for var in CUDA_ENV:
            rec[var] = os.environ.get(var)
    return rec


# The variables that select the device and steer the driver's JIT and its on-disk cache.
CUDA_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_DISABLE",
            "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT")


def cuda_cache_unset() -> bool:
    """True when the driver would fall back to its default cache in $HOME."""
    return not os.environ.get("CUDA_CACHE_PATH") and os.environ.get("CUDA_CACHE_DISABLE") != "1"


def _jsonl(rec: dict) -> str:
    return json.dumps(rec, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _error_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def run_case(case: Case, mod, ph, img: np.ndarray, upload):
    """Run one case; return the host result (ndarray or Histogram)."""
    x = upload(img)
    if case.kernel == "pipeline":
        out = readme_pipeline(mod, ph, x, case.id.split("/")[1])
    else:
        args = tuple(build(a, ph) for a in case.args)
        out = getattr(mod, case.kernel)(x, *args)
    if hasattr(out, "download"):
        out = out.download()
    return out


def read_ids(path: str) -> list[str]:
    """Case ids from a file, one per line; blank lines and ``#`` lines skipped."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def select_cases(cases: list[Case], rx: re.Pattern | None,
                 ids: list[str] | None) -> tuple[list[Case], list[str]]:
    """The cases whose id is in ``ids`` (when given) and matches ``rx`` (when
    given), in catalogue order; and the ids in ``ids`` the catalogue lacks."""
    unknown: list[str] = []
    if ids is not None:
        known = {c.id for c in cases}
        unknown = [i for i in dict.fromkeys(ids) if i not in known]
        want = set(ids)
        cases = [c for c in cases if c.id in want]
    if rx is not None:
        cases = [c for c in cases if rx.search(c.id)]
    return cases, unknown


def histogram_field_errors(per: dict) -> list[str]:
    """The fields of a histogram digest that raised, other than with
    AttributeError, which only says this build lacks the field."""
    return [name for name, v in per.items()
            if isinstance(v, dict) and "error" in v and not str(v["error"]).startswith("AttributeError:")]


def cmd_run(ns) -> int:
    t0 = time.perf_counter()
    floor = min_size()
    if ns.size < floor:
        print(f"--size must be at least {floor}: the ramp image needs that many pixels per side to hold "
              "every u8 code and rounding threshold", file=sys.stderr)
        return 2
    if ns.cuda_cache and not ns.gpu:
        print("--cuda-cache needs --gpu", file=sys.stderr)
        return 2
    try:
        rx = re.compile(ns.cases) if ns.cases else None
        want_ids = read_ids(ns.ids) if ns.ids else None
    except (OSError, UnicodeDecodeError, re.error) as e:
        print(f"cannot select cases: {_error_text(e)}", file=sys.stderr)
        return 2
    gen_ok = generator_digest(make_images(GOLDEN_SIZE)) == GOLDEN_SHA256
    if not gen_ok:
        print("WARNING: the input generator does not reproduce its golden hash; "
              "digests from this run are not comparable with others", file=sys.stderr)

    images = make_images(ns.size, ns.extra_channels)
    shapes = {name: images[f"{name}/c3"].shape[:2] for name in IMAGES}
    cases, unknown = select_cases(catalog(shapes, ns.extra_channels), rx, want_ids)
    if unknown:
        more = f" and {len(unknown) - 10} more" if len(unknown) > 10 else ""
        print(f"--ids names {len(unknown)} id(s) this catalogue lacks"
              f"{' (without --extra-channels)' if not ns.extra_channels else ''}: "
              f"{', '.join(unknown[:10])}{more}", file=sys.stderr)
        return 2
    if not cases:
        print("no case selected: --cases and --ids together match nothing", file=sys.stderr)
        return 2

    if ns.gpu:
        if ns.cuda_cache:
            # Before the first import: the driver reads it when it initialises.
            cache = Path(ns.cuda_cache).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["CUDA_CACHE_PATH"] = str(cache)
        if cuda_cache_unset():
            print("WARNING: neither CUDA_CACHE_PATH nor CUDA_CACHE_DISABLE=1 is set, so the CUDA driver "
                  "writes its JIT cache to ~/.nv/ComputeCache; --cuda-cache DIR puts it elsewhere",
                  file=sys.stderr)

    ph: Any = None
    ph_error: str | None = None
    try:
        import phaios_core as ph  # noqa: PLC0415
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        ph_error = _error_text(e)

    gpu_fields: dict[str, Any] = {}
    mod, upload, setup_error = ph, (lambda a: a), ph_error
    if ns.gpu and ph is not None:
        try:
            from phaios_core import gpu  # noqa: PLC0415
            ctx = gpu.GpuContext(0)
            mod, upload = gpu, ctx.upload
            gpu_fields = {
                "gpu_name": _guard(lambda: ctx.info.name),
                "gpu_compute_capability": _guard(lambda: list(ctx.info.compute_capability)),
                "gpu_fingerprint": _guard(lambda: ctx.fingerprint),
                "nvidia_driver": _guard(_nvidia_driver),
            }
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            setup_error = _error_text(e)
            gpu_fields = {"gpu_error": setup_error}

    input_sha = {k: sha256_array(v) for k, v in images.items()}
    dump = Path(ns.dump) if ns.dump else None
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)

    counts = {"cases": 0, "ok": 0, "errors": 0, "expected_errors": 0,
              "unexpected_errors": 0, "unexpected_successes": 0, "unexpected_field_errors": 0}
    unexpected: list[str] = []
    lines_hash = hashlib.sha256()
    backend = "gpu" if ns.gpu else "cpu"
    with open(ns.out, "w", encoding="utf-8", newline="\n") as out:
        out.write(_jsonl(env_record(ns, ph, ph_error, gen_ok, gpu_fields)) + "\n")
        out.flush()
        if setup_error is None:
            last_kernel = None
            for case in cases:
                if not ns.quiet and case.kernel != last_kernel:
                    print(f"[{time.perf_counter() - t0:7.1f}s] {case.kernel}", file=sys.stderr, flush=True)
                    last_kernel = case.kernel
                key = f"{case.image}/c{case.channels}"
                img = images[key]
                params = (pipeline_params(case.id.split("/")[1]) if case.kernel == "pipeline"
                          else to_json(list(case.args)))
                rec = {"kind": "case", "id": case.id, "kernel": case.kernel, "backend": backend,
                       "image": case.image, "channels": case.channels, "params": params,
                       "expect": case.expect, "input_shape": list(img.shape),
                       "input_sha256": input_sha[key], "output_shape": None, "output_dtype": None,
                       "sha256": None, "nan": None, "posinf": None, "neginf": None, "error": None}
                result = None
                try:
                    result = run_case(case, mod, ph, img, upload)
                    if isinstance(result, np.ndarray):
                        rec.update(array_digest(result))
                    elif hasattr(result, "counts") and hasattr(result, "cdf"):
                        result = histogram_fields(result)
                        rec.update(histogram_digest(result))
                        field_errors = histogram_field_errors(rec["fields"])
                        if field_errors:  # only then: the line stays as it was otherwise
                            rec["field_errors"] = field_errors
                    else:
                        raise TypeError(f"unexpected result type {type(result).__name__}")
                except BaseException as e:  # noqa: BLE001 - PanicException is a BaseException
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    rec["error"] = _error_text(e)
                    result = None
                counts["cases"] += 1
                if rec["error"] is None:
                    counts["ok"] += 1
                    if case.expect == "error":
                        counts["unexpected_successes"] += 1
                        unexpected.append(f"{case.id}: succeeded, expected an error")
                    elif rec.get("field_errors"):
                        counts["unexpected_field_errors"] += 1
                        unexpected.append(f"{case.id}: histogram field(s) raised: "
                                          + "; ".join(f"{f}: {rec['fields'][f]['error']}"
                                                      for f in rec["field_errors"]))
                else:
                    counts["errors"] += 1
                    if case.expect == "error":
                        counts["expected_errors"] += 1
                    else:
                        counts["unexpected_errors"] += 1
                        unexpected.append(f"{case.id}: {rec['error']}")
                line = _jsonl(rec) + "\n"
                lines_hash.update(line.encode())
                out.write(line)
                out.flush()
                if dump is not None and result is not None:
                    stem = dump / dump_name(case.id)
                    if isinstance(result, dict):
                        arrays = {k: canonical(v) for k, v in result.items() if isinstance(v, np.ndarray)}
                        np.savez(str(stem) + ".npz", **arrays)  # type: ignore[arg-type]
                    else:
                        np.save(str(stem) + ".npy", canonical(result), allow_pickle=False)
                del result
        summary: dict[str, Any] = {"kind": "summary", **counts,
                   "setup_error": setup_error,
                   "case_lines_sha256": lines_hash.hexdigest(),
                   "wall_seconds": round(time.perf_counter() - t0, 3),
                   "peak_rss_bytes": _peak_rss_bytes()}
        out.write(_jsonl(summary) + "\n")

    fe = counts["unexpected_field_errors"]
    print(f"{counts['cases']} cases: {counts['ok']} ok, {counts['expected_errors']} expected errors, "
          f"{counts['unexpected_errors']} unexpected errors, {counts['unexpected_successes']} "
          f"unexpected successes{f', {fe} with histogram field errors' if fe else ''}; "
          f"{summary['wall_seconds']} s, peak RSS "
          f"{(summary['peak_rss_bytes'] or 0) / 2**20:.0f} MiB", file=sys.stderr)
    for u in unexpected:
        print(f"  UNEXPECTED {u}", file=sys.stderr)
    if setup_error is not None:
        print(f"setup failed: {setup_error}", file=sys.stderr)
        return 2
    return 1 if (ns.strict and unexpected) else 0


# ── compare ───────────────────────────────────────────────────────────────────


def _ordered(bits_signed: np.ndarray, width: int) -> np.ndarray:
    """Map float bit patterns (as signed ints) to integers ordered like the
    floats, with -0 one step below +0: the ULP distance is a difference."""
    i = bits_signed.astype(np.int64)
    mag = i & np.int64((1 << (width - 1)) - 1)
    return np.where(i < 0, -mag - 1, mag)


def quantize_codes(x: np.ndarray, max_code: int) -> np.ndarray:
    """The crate's quantisation without dither (src/quantize.rs,
    quantize_sample), in float32: v = x * max; NaN and v <= 0 give 0;
    v >= max gives max; else floor(v + 0.5), capped at max."""
    m = np.float32(max_code)
    with np.errstate(invalid="ignore", over="ignore"):
        v = np.asarray(x, dtype=np.float32) * m
        r = np.minimum(np.floor(v + np.float32(0.5)), m)
        r = np.where(v >= m, m, r)
        r = np.where(v > np.float32(0.0), r, np.float32(0.0))  # NaN fails the comparison
    return r.astype(np.int64)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Element-level differences between two outputs of one case."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    if a.dtype.newbyteorder("=") != b.dtype.newbyteorder("="):
        return {"dtype_mismatch": [str(a.dtype), str(b.dtype)]}
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("=")).ravel()
    b = np.ascontiguousarray(b, dtype=b.dtype.newbyteorder("=")).ravel()
    st: dict[str, Any] = {"n": int(a.size)}
    if a.dtype.kind == "f" and a.dtype.itemsize in (4, 8):
        width = 8 * a.dtype.itemsize
        sint = np.int32 if width == 32 else np.int64
        ia, ib = a.view(sint), b.view(sint)
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        both_nan = nan_a & nan_b
        num = ~(nan_a | nan_b)
        na, nb, ina, inb = a[num], b[num], ia[num], ib[num]
        bitdiff = ina != inb
        nan_vs_num = int(np.count_nonzero(nan_a ^ nan_b))
        st["n_diff"] = int(np.count_nonzero(bitdiff)) + nan_vs_num
        st["nan_vs_num"] = nan_vs_num
        st["nan_payload"] = int(np.count_nonzero(both_nan & (ia != ib)))
        oa, ob = _ordered(ina[bitdiff], width), _ordered(inb[bitdiff], width)
        da_, db_ = na[bitdiff], nb[bitdiff]
        tiny = np.finfo(a.dtype).tiny
        fin_d = np.isfinite(da_) & np.isfinite(db_)
        sign_d = np.signbit(da_) != np.signbit(db_)
        normal = fin_d & (np.abs(da_) >= tiny) & (np.abs(db_) >= tiny) & ~sign_d
        if width == 32:
            ulp = np.abs(oa - ob)
            st["max_ulp_all"] = int(ulp.max(initial=0))
            st["max_ulp_normal"] = int(ulp[normal].max(initial=0))
        else:  # int64 differences can overflow: Python integers on the few that differ
            st["max_ulp_all"] = max((abs(int(p) - int(q)) for p, q in zip(oa, ob)), default=0)
            st["max_ulp_normal"] = max((abs(int(p) - int(q)) for p, q in zip(oa[normal], ob[normal])),
                                       default=0)
        st["signed_zero"] = int(np.count_nonzero(bitdiff & (na == 0) & (nb == 0)))
        sub_a = (da_ != 0) & (np.abs(da_) < tiny)
        sub_b = (db_ != 0) & (np.abs(db_) < tiny)
        st["ftz"] = int(np.count_nonzero(((da_ == 0) & sub_b) | ((db_ == 0) & sub_a)))
        st["sign_change"] = int(np.count_nonzero(fin_d & (da_ != 0) & (db_ != 0) & sign_d))
        st["inf_mismatch"] = int(np.count_nonzero(bitdiff & (np.isinf(na) | np.isinf(nb))))
        fin = np.isfinite(na) & np.isfinite(nb)
        d = np.abs(na[fin].astype(np.float64) - nb[fin].astype(np.float64))
        st["max_abs"] = float(d.max(initial=0.0))
        for name, mx in (("u8", 255), ("u16", 65535)):
            qa, qb = quantize_codes(a, mx), quantize_codes(b, mx)
            dq = np.abs(qa - qb)
            st[f"{name}_changed"] = int(np.count_nonzero(dq))
            st[f"{name}_max"] = int(dq.max(initial=0))
    elif a.dtype.kind in "ui":
        m = a != b
        st["n_diff"] = int(np.count_nonzero(m))
        st["max_code"] = max((abs(int(p) - int(q)) for p, q in zip(a[m], b[m])), default=0)
    else:
        st["n_diff"] = int(np.count_nonzero(a.view(np.uint8) != b.view(np.uint8)))
    return st


# ── Agreement classes (compare --bounds) ──────────────────────────────────────

# Copied from docs/ffi.md section 6 ("Determinism"): the kernels of the table
# under "Bit-exact between CPU and CUDA" map to None, those of the table under
# "Bounded, not identical" to their (rtol, atol). `selftest` re-reads both
# tables from docs/ffi.md, when the file is beside this script, and fails when
# they and this copy disagree. tone_curve is in both tables: exact at
# power == 1, bounded otherwise (agreement_class decides). film_grain's row in
# the exact table is its integer hash, not its output, which is bounded.
AGREEMENT: dict[str, tuple[float, float] | None] = {
    "crop": None, "orient": None, "hot_pixels": None, "resize": None, "straighten": None,
    "exposure": None, "luminance_bw": None, "channel_mixer_bw": None, "color_filter_bw": None,
    "vignette": None, "highlight_rolloff": None, "shadow_rolloff": None, "apply_lut": None,
    "histogram": None, "quantize_u8": None, "quantize_u16": None,
    "encode_srgb": (1e-5, 1e-7), "tone_curve": (1e-5, 1e-7), "hsl_bw": (1e-5, 1e-7),
    "zone_system": (1e-5, 1e-7), "split_toning": (1e-5, 1e-7), "blur": (1e-5, 1e-7),
    "glow": (1e-5, 1e-7), "sharpen": (1e-5, 1e-7),
    "local_contrast": (1e-4, 1e-6), "denoise": (1e-4, 1e-6),
    "film_grain": (1e-3, 1e-5),
}
# Section 6, "Non-finite pixels keep their class": checked on the specials image.
CLASS_KEPT = frozenset({"zone_system", "encode_srgb", "tone_curve", "hsl_bw"})
# The images on which the exact and bounded classes are pass/fail.
BOUNDS_IMAGES = ("hdr", "ramp", "noise")


def ffi_agreement_tables(text: str) -> tuple[set[str], dict[str, tuple[float, float]], dict[tuple, str]]:
    """Read docs/ffi.md section 6's two tables: the kernels named (in
    backticks) in the first cell of the bit-exact table's rows, the
    (rtol, atol) of the bounded table's rows, and each row's first cell
    keyed by (table, kernel), which carries a qualifier such as
    ``power == 1``. Raises ValueError when section 6 is missing."""
    start = text.find("\n## 6.")
    if start < 0:
        raise ValueError("docs/ffi.md has no section 6")
    end = text.find("\n## ", start + 1)
    section = text[start:end if end >= 0 else len(text)]
    exact: set[str] = set()
    bounded: dict[str, tuple[float, float]] = {}
    quals: dict[tuple, str] = {}
    table = None
    for line in section.splitlines():
        if not line.startswith("|"):
            table = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Kernel":
            table = ("bounded" if cells[1:3] == ["rtol", "atol"] else
                     "exact" if cells[1].startswith("Why it is exact") else None)
            continue
        if table is None or set(cells[0]) <= set("-: "):
            continue
        for name in re.findall(r"`([a-z0-9_]+)`", cells[0]):
            if name not in KERNELS:
                continue
            quals[(table, name)] = cells[0]
            if table == "exact":
                exact.add(name)
            else:
                bounded[name] = (float(cells[1]), float(cells[2]))
    return exact, bounded, quals


def agreement_class(rec: dict) -> tuple[str, tuple[float, float] | None]:
    """("exact", None), ("bounded", (rtol, atol)) or ("report", None) for a
    case line; "report" for the README pipeline, which has no committed bound."""
    k = rec.get("kernel")
    if k not in AGREEMENT:
        return "report", None
    if k == "tone_curve":
        try:
            power = rec["params"][0]["ToneCurveParams"].get("power", 1.0)
        except (KeyError, IndexError, TypeError, AttributeError):
            return "report", None
        if power == 1:
            return "exact", None
    b = AGREEMENT[k]
    return ("exact", None) if b is None else ("bounded", b)


def class_changes(x: np.ndarray, y: np.ndarray) -> int:
    """Elements whose class (NaN, +inf, -inf, finite) differs between x and y:
    exactly those that tests/cuda_conformance.rs worst_violation scores as
    infinite for their class alone."""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    return int(np.count_nonzero((np.isnan(x) != np.isnan(y)) | (np.isposinf(x) != np.isposinf(y))
                                | (np.isneginf(x) != np.isneginf(y))))


def worst_violation(ref: np.ndarray, other: np.ndarray, rtol: float, atol: float) -> dict:
    """tests/cuda_conformance.rs ``worst_violation``, in the arrays' own float
    type (float32 for every bounded kernel, as in the Rust oracle), with the
    class changes counted apart: ``worst`` is the max over both-finite
    elements of ``|x - y| / (atol + rtol * |x|)``, x from ``ref`` (the CPU);
    ``over`` counts those above 1; ``class_changed`` counts the elements the
    oracle scores infinite for their class; ``oracle`` is what the Rust
    function returns (inf when any class changed)."""
    x = np.asarray(ref).ravel()
    y = np.asarray(other).ravel()
    dt = x.dtype.newbyteorder("=")
    x, y = x.astype(dt, copy=False), y.astype(dt, copy=False)
    fin = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[fin], y[fin]
    with np.errstate(over="ignore", invalid="ignore"):
        v = np.abs(xf - yf) / (dt.type(atol) + dt.type(rtol) * np.abs(xf))
    worst = float(v.max(initial=0))
    cls = class_changes(x, y)
    return {"worst": worst, "over": int(np.count_nonzero(v > 1)), "class_changed": cls,
            "oracle": math.inf if cls else worst}


def judge(rec: dict, ref, other, st: dict | None) -> dict:
    """One case of a CPU-against-GPU comparison, judged by agreement_class.
    ``ref`` and ``other`` are the CPU and GPU dumps (arrays, or histogram
    field dicts); ``st`` is diff_stats(ref, other) for arrays. Returns the
    class, bound, what was checked, the measures, and a verdict: "pass",
    "FAIL" or "report" (nothing committed to check)."""
    cls, bound = agreement_class(rec)
    finite_image = rec.get("image") in BOUNDS_IMAGES
    j: dict[str, Any] = {"class": cls, "bound": bound, "check": None, "worst": None, "over": None,
                         "class_changed": None, "n_diff": None, "max_ulp_normal": None, "verdict": "report"}
    if isinstance(ref, dict) or isinstance(other, dict):  # a histogram: every field must match
        ra = ref if isinstance(ref, dict) else {}
        ob = other if isinstance(other, dict) else {}
        n = 0 if set(ra) == set(ob) else 1
        for f in set(ra) & set(ob):
            fs = diff_stats(ra[f], ob[f])
            n += 1 if ("shape_mismatch" in fs or "dtype_mismatch" in fs) else fs["n_diff"]
        j.update(n_diff=n)
        if cls == "exact":
            j.update(check="exact", verdict="pass" if n == 0 else "FAIL")
        return j
    if st is None or "shape_mismatch" in st or "dtype_mismatch" in st:
        if cls != "report":
            j.update(check="shape", verdict="FAIL")
        return j
    j.update(n_diff=st["n_diff"], max_ulp_normal=st.get("max_ulp_normal"))
    is_float = np.asarray(ref).dtype.kind == "f"
    if cls == "bounded" and is_float:
        assert bound is not None
        j.update(worst_violation(ref, other, *bound))
    elif is_float and rec.get("kernel") in CLASS_KEPT:
        j["class_changed"] = class_changes(ref, other)
    if finite_image:
        if cls == "exact":
            j.update(check="exact", verdict="pass" if st["n_diff"] == 0 else "FAIL")
        elif cls == "bounded" and is_float:
            ok = j["worst"] <= 1 and j["class_changed"] == 0
            j.update(check="bound", verdict="pass" if ok else "FAIL")
        elif cls == "bounded":  # a bounded kernel that returned integers: hold it to exact
            j.update(check="exact", verdict="pass" if st["n_diff"] == 0 else "FAIL")
    else:  # specials (section 6 keeps only these promises on non-finite and extreme input)
        if cls == "exact" and rec.get("kernel") != "tone_curve":  # tone_curve: subnormals exempt
            j.update(check="exact", verdict="pass" if st["n_diff"] == 0 else "FAIL")
        elif rec.get("kernel") in CLASS_KEPT and is_float:
            if j["class_changed"] is None:
                j["class_changed"] = class_changes(ref, other)
            j.update(check="class", verdict="pass" if j["class_changed"] == 0 else "FAIL")
    return j


def dump_mismatch(rec: dict, dump) -> str | None:
    """None when ``dump`` holds the output ``rec`` digests, else why not.
    An array must hash to the line's sha256; a histogram must have exactly
    the line's non-error fields, each hashing to its recorded sha256."""
    fields = rec.get("fields")
    if isinstance(dump, dict):
        if not isinstance(fields, dict):
            return "a histogram dump for a line that is not a histogram"
        want = {f: v.get("sha256") for f, v in fields.items() if isinstance(v, dict) and "error" not in v}
        if set(want) != set(dump):
            extra, lack = sorted(set(dump) - set(want)), sorted(set(want) - set(dump))
            return f"histogram fields differ (dump only: {extra or '-'}; line only: {lack or '-'})"
        bad = sorted(f for f in want if sha256_array(dump[f]) != want[f])
        return f"histogram field(s) {', '.join(bad)} do not hash to the line's" if bad else None
    if isinstance(fields, dict):
        return "an array dump for a histogram line"
    return None if sha256_array(dump) == rec.get("sha256") else "its sha256 is not the line's"


def load_run(path: str) -> tuple[dict, dict | None, dict[str, dict], list[str]]:
    env, summary, cases, order = {}, None, {}, []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("kind", "case")
            if kind == "env":
                env = rec
            elif kind == "summary":
                summary = rec
            else:
                cid = rec["id"]
                if cid in cases:
                    raise ValueError(f"{path}:{n}: duplicate case id {cid}")
                cases[cid] = rec
                order.append(cid)
    return env, summary, cases, order


def classify(a: dict | None, b: dict | None) -> str:
    if a is None:
        return "only-B"
    if b is None:
        return "only-A"
    if a.get("input_sha256") != b.get("input_sha256") or a.get("params") != b.get("params"):
        return "input-mismatch"  # not the same computation: nothing to compare
    if a.get("error") or b.get("error"):
        return "identical" if a.get("error") == b.get("error") else "error-changed"
    same = all(a.get(k) == b.get(k) for k in ("sha256", "output_shape", "output_dtype"))
    return "identical" if same else "differing"


def _load_dump(d: str, cid: str):
    stem = str(Path(d) / dump_name(cid))  # names contain dots: no with_suffix
    if os.path.exists(stem + ".npy"):
        return np.load(stem + ".npy", allow_pickle=False)
    if os.path.exists(stem + ".npz"):
        with np.load(stem + ".npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return None


def _kernel_rank(k: str) -> int:
    return KERNELS.index(k) if k in KERNELS else len(KERNELS)


def _image_rank(i: str) -> int:
    return IMAGES.index(i) if i in IMAGES else len(IMAGES)


def _short(v, n=80) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 3] + "..."


def _print_env_diff(title: str, a: dict | None, b: dict | None) -> None:
    a, b = a or {}, b or {}
    keys = [k for k in dict.fromkeys([*a, *b]) if a.get(k) != b.get(k)]
    print(f"{title}: {len(keys)} differing field(s)")
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if k == "cpu_flags" and isinstance(va, str) and isinstance(vb, str):
            sa, sb = set(va.split()), set(vb.split())
            print(f"  {k}: only A: {' '.join(sorted(sa - sb)) or '-'}; only B: {' '.join(sorted(sb - sa)) or '-'}")
        else:
            print(f"  {k}: A={_short(va)}  B={_short(vb)}")


def cmd_compare(ns) -> int:
    # The verdict lines go to stderr; line-buffer stdout so that, sent to one file, they follow
    # the report they conclude.
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    if (ns.bounds or ns.expect_bounded) and not ns.dumps:
        print("--bounds and --expect-bounded need --dumps", file=sys.stderr)
        return 2
    try:
        run_a, run_b = load_run(ns.a), load_run(ns.b)
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot read runs: {_error_text(e)}", file=sys.stderr)
        return 2
    loaders = None
    if ns.dumps:
        loaders = (lambda cid: _load_dump(ns.dumps[0], cid), lambda cid: _load_dump(ns.dumps[1], cid))
    return compare_runs(ns, run_a, run_b, loaders, (ns.a, ns.b))


def _fmt_ratio(v) -> str:
    """A ratio to three significant digits, a count as an integer, None as "-"."""
    if v is None:
        return "-"
    return str(v) if isinstance(v, int) else f"{v:.3g}"


def compare_runs(ns, run_a, run_b, loaders, names=("A", "B")) -> int:
    """The body of `compare`, on runs as load_run returns them. ``loaders``
    is None or a pair of functions mapping a case id to its dump in A and
    in B (an array, a histogram's field dict, or None when absent)."""
    env_a, sum_a, ca, oa = run_a
    env_b, sum_b, cb, ob = run_b
    bounds = bool(ns.bounds or ns.expect_bounded)
    print(f"A: {names[0]}  ({env_a.get('backend')}, phaios_core {env_a.get('phaios_core_version')}, "
          f"{len(ca)} cases)")
    print(f"B: {names[1]}  ({env_b.get('backend')}, phaios_core {env_b.get('phaios_core_version')}, "
          f"{len(cb)} cases)")
    for tag, s in (("A", sum_a), ("B", sum_b)):
        if s is None:
            print(f"WARNING: {tag} has no summary line: that run did not finish")
    if env_a.get("generator") != env_b.get("generator"):
        print(f"WARNING: generators differ ({env_a.get('generator')} vs {env_b.get('generator')})")
    for tag, env in (("A", env_a), ("B", env_b)):
        if env.get("generator_ok") is False:
            print(f"WARNING: {tag}'s generator missed its golden hash")
    _print_env_diff("env", env_a, env_b)
    _print_env_diff("summary", sum_a, sum_b)
    print()

    failed_setup = [(tag, s["setup_error"]) for tag, s in (("A", sum_a), ("B", sum_b))
                    if s is not None and s.get("setup_error") is not None]
    if failed_setup:
        for tag, err in failed_setup:
            print(f"CANNOT COMPARE: run {tag}'s setup failed: {err}", file=sys.stderr)
        return 2
    ref_is_a = True
    if bounds:
        pair = (env_a.get("backend"), env_b.get("backend"))
        if sorted(map(str, pair)) != ["cpu", "gpu"]:
            print(f"CANNOT COMPARE: --bounds needs one CPU run and one GPU run, not {pair[0]} and {pair[1]}",
                  file=sys.stderr)
            return 2
        ref_is_a = pair[0] == "cpu"

    ids = list(dict.fromkeys([*oa, *ob]))
    try:
        rx = re.compile(ns.cases) if ns.cases else None
        allow = re.compile(ns.allow) if ns.allow else None
    except re.error as e:
        print(f"bad regex: {_error_text(e)}", file=sys.stderr)
        return 2
    if rx is not None:
        ids = [i for i in ids if rx.search(i)]
    if not ids:
        print("CANNOT COMPARE: nothing to compare: "
              + (f"--cases {ns.cases!r} matches no case id" if rx is not None and (oa or ob)
                 else "neither run has a case line"), file=sys.stderr)
        return 2
    cats = ("identical", "differing", "error-changed", "only-A", "only-B", "input-mismatch")
    per: dict[str, dict[str, int]] = {}
    verdict: dict[str, str] = {}
    for cid in ids:
        a, b = ca.get(cid), cb.get(cid)
        either = a if a is not None else b
        assert either is not None
        k = either["kernel"]
        c = classify(a, b)
        verdict[cid] = c
        row = per.setdefault(k, {"cases": 0, **{x: 0 for x in cats}})
        row["cases"] += 1
        row[c] += 1

    hdr = f"{'kernel':<18} {'cases':>6} {'identical':>9} {'differing':>9} {'err-chg':>7} " \
          f"{'only-A':>6} {'only-B':>6} {'input':>5}"
    print(hdr)
    print("-" * len(hdr))
    total = {"cases": 0, **{x: 0 for x in cats}}
    for k, row in per.items():
        for x in total:
            total[x] += row[x]
        print(f"{k:<18} {row['cases']:>6} {row['identical']:>9} {row['differing']:>9} "
              f"{row['error-changed']:>7} {row['only-A']:>6} {row['only-B']:>6} {row['input-mismatch']:>5}")
    print("-" * len(hdr))
    print(f"{'overall':<18} {total['cases']:>6} {total['identical']:>9} {total['differing']:>9} "
          f"{total['error-changed']:>7} {total['only-A']:>6} {total['only-B']:>6} "
          f"{total['input-mismatch']:>5}")

    if ns.write_ids:
        try:
            Path(ns.write_ids).parent.mkdir(parents=True, exist_ok=True)
            with open(ns.write_ids, "w", encoding="utf-8", newline="\n") as f:
                for cid in ids:
                    if verdict[cid] != "identical":
                        f.write(cid + "\n")
        except OSError as e:
            print(f"cannot write --write-ids: {_error_text(e)}", file=sys.stderr)
            return 2

    changed = [i for i in ids if verdict[i] == "error-changed"]
    if changed:
        print("\nerror changed:")
        for cid in changed:
            print(f"  {cid}\n    A: {ca[cid].get('error')}\n    B: {cb[cid].get('error')}")
    only = [i for i in ids if verdict[i] in ("only-A", "only-B")]
    if only:
        print("\nonly in one run:")
        for cid in only:
            print(f"  {verdict[cid]}  {cid}")

    differing = [i for i in ids if verdict[i] == "differing"]
    foreign: list[str] = []      # a dump that does not hold its run line's output, or cannot be read
    missing: list[str] = []      # no dump for a differing case
    judged: dict[str, dict] = {}  # --bounds: case id -> judge()
    if differing and loaders is not None:
        bcols = f" {'class':<7} {'worst':>9} {'>1':>6} {'class!':>6} verdict" if bounds else ""
        print("\ndiffering cases (from dumps; n_diff counts value differences, so a case with\n"
              "n_diff 0 differs only in NaN payload or sign bits, counted under nan-bits"
              + ("; worst: the (rtol, atol)\nratio of tests/cuda_conformance.rs worst_violation over "
                 "both-finite elements, CPU as reference;\n>1: elements above the bound; class!: "
                 "elements changing class (NaN, +inf, -inf, finite)" if bounds else "") + "):")
        print(f"  {'case':<52} {'n_diff':>9} {'of':>9} {'ulp_all':>10} {'ulp_norm':>9} {'max_abs':>10} "
              f"{'nan!=num':>8} {'nan-bits':>8} {'+-0':>4} {'ftz':>4} {'sgn':>4} {'inf':>4} "
              f"{'u8 n/max':>10} {'u16 n/max':>12}{bcols}")
        agg: dict[str, dict] = {}
        for cid in differing:
            try:
                da, db = loaders[0](cid), loaders[1](cid)
            except Exception as e:  # noqa: BLE001 - a corrupt or foreign file
                print(f"  {cid:<52} dump unreadable: {_error_text(e)}")
                foreign.append(cid)
                continue
            if da is None or db is None:
                print(f"  {cid:<52} dump missing in {'A' if da is None else 'B'}")
                missing.append(cid)
                continue
            why = [f"{tag} ({w})" for tag, rec, d in (("A", ca[cid], da), ("B", cb[cid], db))
                   if (w := dump_mismatch(rec, d)) is not None]
            if why:
                print(f"  {cid:<52} dump does not belong to run {' and '.join(why)}")
                foreign.append(cid)
                continue
            x, y = (da, db) if ref_is_a else (db, da)
            if isinstance(da, dict) or isinstance(db, dict):
                da = da if isinstance(da, dict) else {}
                db = db if isinstance(db, dict) else {}
                parts = []
                for f in sorted(set(da) | set(db)):
                    if f not in da or f not in db:
                        parts.append(f"{f}: missing")
                        continue
                    st = diff_stats(da[f], db[f])
                    if st.get("n_diff"):
                        extra = (f" max_ulp_all {st['max_ulp_all']}" if "max_ulp_all" in st
                                 else f" max {st.get('max_code', '?')}")
                        parts.append(f"{f}: {st['n_diff']}/{st['n']}{extra}")
                    elif "shape_mismatch" in st or "dtype_mismatch" in st:
                        parts.append(f"{f}: {st}")
                tail = ""
                if bounds:
                    judged[cid] = j = judge(ca[cid], x, y, None)
                    tail = f"  [{j['class']}: {j['verdict']}]"
                print(f"  {cid:<52} histogram fields: {'; '.join(parts) or 'none differ'}{tail}")
                continue
            st = diff_stats(da, db)
            if bounds:
                judged[cid] = j = judge(ca[cid], x, y, st)  # diff_stats is symmetric
            if "shape_mismatch" in st or "dtype_mismatch" in st:
                print(f"  {cid:<52} {st}" + (f"  [{j['class']}: {j['verdict']}]" if bounds else ""))
                continue
            btail = (f" {j['class']:<7} {_fmt_ratio(j['worst']):>9} {_fmt_ratio(j['over']):>6} "
                     f"{_fmt_ratio(j['class_changed']):>6} {j['verdict']}") if bounds else ""
            k = f"{ca[cid]['kernel']} {ca[cid]['image']}"
            g = agg.setdefault(k, {"cases": 0, "nan_bits_only": 0, "max_ulp_all": 0, "max_ulp_normal": 0,
                                   "max_abs": 0.0, "ftz": 0, "sign_change": 0,
                                   "u8_changed": 0, "u8_max": 0, "u16_changed": 0, "u16_max": 0,
                                   "nan_vs_num": 0, "max_code": 0})
            g["cases"] += 1
            g["nan_bits_only"] += int(st.get("n_diff", 1) == 0)
            for f in ("max_ulp_all", "max_ulp_normal", "u8_max", "u16_max", "max_code"):
                g[f] = max(g[f], st.get(f, 0))
            g["max_abs"] = max(g["max_abs"], st.get("max_abs", 0.0))
            for f in ("u8_changed", "u16_changed", "nan_vs_num", "ftz", "sign_change"):
                g[f] += st.get(f, 0)
            if "max_ulp_all" in st:
                print(f"  {cid:<52} {st['n_diff']:>9} {st['n']:>9} {st['max_ulp_all']:>10} "
                      f"{st['max_ulp_normal']:>9} {st['max_abs']:>10.3g} "
                      f"{st['nan_vs_num']:>8} {st['nan_payload']:>8} {st['signed_zero']:>4} {st['ftz']:>4} "
                      f"{st['sign_change']:>4} {st['inf_mismatch']:>4} {st['u8_changed']:>5}/{st['u8_max']:<4} "
                      f"{st['u16_changed']:>7}/{st['u16_max']:<4}{btail}")
            else:
                print(f"  {cid:<52} {st['n_diff']:>9} {st['n']:>9} {'codes':>10} max code delta "
                      f"{st.get('max_code', '?')}{btail}")
        if agg:
            print("\nper kernel and image, over the differing cases (nan-only: cases differing only\n"
                  "in NaN bits; ulp_norm: over pairs both normal, finite and of one sign; ftz: one side\n"
                  "+-0, the other subnormal; sgn: sign changes between nonzero finite values;\n"
                  "u8/u16: codes changed under the crate's quantisation, int: integer outputs):")
            print(f"  {'kernel image':<27} {'cases':>5} {'nan-only':>8} {'ulp_all':>10} {'ulp_norm':>9} "
                  f"{'max_abs':>10} {'nan!=num':>8} {'ftz':>7} {'sgn':>7} {'u8 chg':>8} {'u8 max':>6} "
                  f"{'u16 chg':>9} {'u16 max':>7} {'int max':>7}")
            for k, g in sorted(agg.items(), key=lambda kv: (_kernel_rank(kv[0].split()[0]),
                                                             _image_rank(kv[0].split()[1]))):
                print(f"  {k:<27} {g['cases']:>5} {g['nan_bits_only']:>8} {g['max_ulp_all']:>10} "
                      f"{g['max_ulp_normal']:>9} {g['max_abs']:>10.3g} {g['nan_vs_num']:>8} {g['ftz']:>7} "
                      f"{g['sign_change']:>7} {g['u8_changed']:>8} {g['u8_max']:>6} "
                      f"{g['u16_changed']:>9} {g['u16_max']:>7} {g['max_code']:>7}")
    elif differing:
        print(f"\n{len(differing)} differing case(s); pass --dumps for element-level detail:")
        for cid in differing:
            print(f"  {cid}")

    bound_fail: list[str] = []
    if bounds:
        ref = "A" if ref_is_a else "B"
        cells: dict[tuple, dict] = {}
        for cid in ids:
            a, b = ca.get(cid), cb.get(cid)
            if verdict[cid] not in ("identical", "differing") or a is None or b is None:
                continue
            if a.get("error") or b.get("error"):
                continue  # the same error on both sides: a reject case, nothing numeric
            if verdict[cid] == "identical":
                cls, bd = agreement_class(a)
                checked = cls != "report" and (a["image"] in BOUNDS_IMAGES or
                                               (cls == "exact" and a["kernel"] != "tone_curve") or
                                               a["kernel"] in CLASS_KEPT)
                j = {"class": cls, "bound": bd, "worst": 0.0 if cls == "bounded" else None, "over": 0,
                     "class_changed": 0, "n_diff": 0, "max_ulp_normal": 0,
                     "verdict": "pass" if checked else "report"}
            elif cid in judged:
                j = judged[cid]
            else:
                continue  # its dump was missing or foreign: reported above, exit 2 below
            key = (a["kernel"], a["image"], j["class"])
            cell = cells.setdefault(key, {"bound": j["bound"], "cases": 0, "differing": 0, "worst": None,
                                          "over": 0, "class_changed": 0, "n_diff": 0, "ulp_norm": 0,
                                          "pass": 0, "fail": 0, "allowed": 0, "report": 0})
            cell["cases"] += 1
            cell["differing"] += int(verdict[cid] == "differing")
            if j["worst"] is not None:
                cell["worst"] = max(cell["worst"] or 0.0, j["worst"])
            for f, g in (("over", "over"), ("class_changed", "class_changed"), ("n_diff", "n_diff")):
                cell[f] += j[g] or 0
            cell["ulp_norm"] = max(cell["ulp_norm"], j["max_ulp_normal"] or 0)
            if j["verdict"] == "FAIL":
                if allow and allow.search(cid):
                    cell["allowed"] += 1
                else:
                    cell["fail"] += 1
                    bound_fail.append(cid)
            else:
                cell[j["verdict"]] += 1
        print(f"\nagreement with docs/ffi.md section 6, per kernel, image and class (reference: {ref}, the CPU\n"
              "run; pass/fail on hdr, ramp and noise; on specials exact kernels except tone_curve, and\n"
              "the class of every element for zone_system, encode_srgb, tone_curve and hsl_bw; worst:\n"
              "worst_violation over both-finite elements; >1: elements above the bound; class!:\n"
              "elements changing class; n_diff: differing values; ulp_norm: max ULP over normal pairs):")
        print(f"  {'kernel':<18} {'image':<9} {'class':<7} {'rtol,atol':<12} {'cases':>5} {'diff':>5} "
              f"{'worst':>9} {'>1':>7} {'class!':>6} {'n_diff':>9} {'ulp_norm':>9}  verdict")
        for (k, img, cls), c in sorted(cells.items(), key=lambda kv: (_kernel_rank(kv[0][0]),
                                                                     _image_rank(kv[0][1]), kv[0][2])):
            bd = "-" if c["bound"] is None else f"{c['bound'][0]:g},{c['bound'][1]:g}"
            if c["fail"]:
                v = f"FAIL ({c['fail']} case{'s' if c['fail'] > 1 else ''})"
            elif c["allowed"]:
                v = f"allowed ({c['allowed']} failing case{'s' if c['allowed'] > 1 else ''} matched by --allow)"
            elif c["pass"]:
                v = "pass" + (f" ({c['report']} report-only)" if c["report"] else "")
            else:
                v = "report"
            print(f"  {k:<18} {img:<9} {cls:<7} {bd:<12} {c['cases']:>5} {c['differing']:>5} "
                  f"{_fmt_ratio(c['worst']):>9} {c['over']:>7} {c['class_changed']:>6} {c['n_diff']:>9} "
                  f"{c['ulp_norm']:>9}  {v}")
        if bound_fail:
            print("\noutside their class (failing cases):")
            for cid in bound_fail:
                j = judged[cid]
                print(f"  {cid}: class {j['class']}, {j['check']} check failed, worst {_fmt_ratio(j['worst'])}, >1 "
                      f"{_fmt_ratio(j['over'])}, class! {_fmt_ratio(j['class_changed'])}, "
                      f"n_diff {_fmt_ratio(j['n_diff'])}")

    if total["input-mismatch"]:
        print(f"\nCANNOT COMPARE: {total['input-mismatch']} case(s) had different inputs or "
              "parameters (different --size, generator or case list)", file=sys.stderr)
        return 2
    if foreign:
        print(f"\nCANNOT COMPARE: {len(foreign)} dump(s) unreadable or not from the run they were "
              "given with (check the order and the directories of --dumps)", file=sys.stderr)
        return 2
    if bounds and missing:
        print(f"\nCANNOT COMPARE: {len(missing)} differing case(s) have no dump, so their bounds "
              "cannot be checked", file=sys.stderr)
        return 2
    rc = 0
    bad = [i for i in ids if verdict[i] != "identical" and not (allow and allow.search(i))]
    if ns.expect_identical and bad:
        print(f"\nFAIL: {len(bad)} case(s) not identical"
              + (f" outside --allow {ns.allow!r}" if allow else ""), file=sys.stderr)
        rc = 1
    unfinished = [tag for tag, s in (("A", sum_a), ("B", sum_b)) if s is None]
    if ns.expect_identical and unfinished:
        print("\nFAIL: " + ("runs A and B have no summary line: neither finished" if len(unfinished) == 2
                            else f"run {unfinished[0]} has no summary line: it did not finish"),
              file=sys.stderr)
        rc = 1
    if bounds:
        unpaired = [i for i in ids if verdict[i] in ("error-changed", "only-A", "only-B")
                    and not (allow and allow.search(i))]
        print(f"\nbounds: {len(bound_fail)} case(s) outside their class"
              + (f" (ids matching --allow {ns.allow!r} not counted)" if allow else "")
              + (f", {len(unpaired)} error-changed or only in one run" if unpaired else ""))
        if ns.expect_bounded and (bound_fail or unpaired):
            print(f"\nFAIL: {len(bound_fail) + len(unpaired)} case(s) not within their class of "
                  "docs/ffi.md section 6" + (f" outside --allow {ns.allow!r}" if allow else ""),
                  file=sys.stderr)
            rc = 1
    if rc:
        return rc
    nonident = sum(1 for i in ids if verdict[i] != "identical")
    if not nonident:
        print("\nidentical")
    elif ns.expect_identical:
        print(f"\n{nonident} case(s) not identical, every one matched by --allow")
    else:
        print(f"\n{nonident} case(s) not identical (reported only: --expect-identical not given)")
    return 0


# ── selftest ──────────────────────────────────────────────────────────────────


def cmd_selftest(_ns) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{(': ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    # Generator
    for seed in (0, 1, 0x123456789ABCDEF, U64_MAX):
        got = [int(v) for v in splitmix64(seed, 8)]
        check(f"splitmix64 vectorised == scalar reference, seed {seed:#x}", got == _splitmix64_ref(seed, 8))
    check("splitmix64(0) first output is the published 0xe220a8397b1dcdaf",
          int(splitmix64(0, 1)[0]) == 0xE220A8397B1DCDAF)
    u = uniform24(7, (1000,))
    check("uniform24 in [0, 1), multiples of 2**-24",
          bool(u.min() >= 0 and u.max() < 1 and np.all((u * np.float32(2**24)) % 1 == 0)))
    g1 = generator_digest(make_images(GOLDEN_SIZE))
    g2 = generator_digest(make_images(GOLDEN_SIZE))
    check("generator is repeatable", g1 == g2)
    check("generator matches its golden hash", g1 == GOLDEN_SHA256, f"got {g1}")
    imgs = make_images(GOLDEN_SIZE)
    sp = imgs["specials/c3"]
    check("specials image holds NaN, +inf, -inf, -0 and subnormals",
          bool(np.isnan(sp).any() and np.isposinf(sp).any() and np.isneginf(sp).any()
               and np.any((sp == 0) & np.signbit(sp))
               and np.any((sp != 0) & (np.abs(sp) < np.finfo(np.float32).tiny))))
    hdr = imgs["hdr/c3"]
    check("hdr image spans about [0, 16], with a 16.0 plateau",
          bool(hdr.min() >= 0 and hdr.min() < 1e-5 and hdr.max() == 16.0 and np.count_nonzero(hdr == 16) > 3),
          f"min {hdr.min()} max {hdr.max()} at-16 {np.count_nonzero(hdr == 16)}")
    r = make_ramp(GOLDEN_SIZE)[:, :, 0].ravel()
    check("ramp holds every u8 code and threshold with both neighbours",
          bool(np.isin(_ramp_set(255), r).all()))
    r1k = make_ramp(640)[:, :, 0].ravel()
    check("ramp at n >= 629 holds every u16 code and threshold with both neighbours",
          bool(np.isin(_ramp_set(65535), r1k).all()))
    floor = min_size()
    r_floor = make_ramp(floor)[:, :, 0].ravel()
    check("min_size() is 40, and the ramp at that size holds every u8 code and threshold",
          floor == 40 and bool(np.isin(_ramp_set(255), r_floor).all()))
    check("one below min_size() the ramp cannot be built (so run refuses that --size)",
          _guard(lambda: make_ramp(floor - 1) is None, True) is True)
    ximgs = make_images(GOLDEN_SIZE, extra_channels=True)
    check("--extra-channels leaves the default inputs, and so the golden hash, unchanged",
          generator_digest({k: v for k, v in ximgs.items() if k in imgs}) == GOLDEN_SHA256)
    check("--extra-channels inputs: c2 is channels 0-1 and c4 is c3 then channel 0, on noise and specials",
          set(ximgs) - set(imgs) == {f"{i}/c{c}" for i in EXTRA_IMAGES for c in EXTRA_CHANNELS}
          and all(np.array_equal(ximgs[f"{i}/c2"], ximgs[f"{i}/c3"][:, :, :2], equal_nan=True)
                  and np.array_equal(ximgs[f"{i}/c4"][:, :, :3], ximgs[f"{i}/c3"], equal_nan=True)
                  and np.array_equal(ximgs[f"{i}/c4"][:, :, 3], ximgs[f"{i}/c3"][:, :, 0], equal_nan=True)
                  for i in EXTRA_IMAGES))

    # Catalogue
    shapes = {name: imgs[f"{name}/c3"].shape[:2] for name in IMAGES}
    cases = catalog(shapes)
    ids = [c.id for c in cases]
    check("case ids are unique", len(ids) == len(set(ids)))
    check("dump names are unique", len({dump_name(i) for i in ids}) == len(ids))
    check("every one of the 27 kernels has a case",
          set(KERNELS) <= {c.kernel for c in cases} and len(KERNELS) == 27)
    check("every case's parameters serialise to strict JSON",
          all(_guard(lambda c=c: bool(_jsonl({"p": to_json(list(c.args))})), False) for c in cases))
    xcases = catalog(shapes, extra_channels=True)
    xids = [c.id for c in xcases]
    extra = [c for c in xcases if c.channels in EXTRA_CHANNELS]
    any_c = {"apply_lut", "blur", "crop", "denoise", "encode_srgb", "exposure", "glow", "highlight_rolloff",
             "histogram", "hot_pixels", "orient", "quantize_u16", "quantize_u8", "resize", "shadow_rolloff",
             "sharpen", "straighten", "tone_curve", "vignette"}
    check("--extra-channels keeps every default id in its order, and its ids are unique",
          [i for i in xids if i in set(ids)] == ids and len(xids) == len(set(xids))
          and len({dump_name(i) for i in xids}) == len(xids))
    check("--extra-channels adds C = 2 and 4 on noise and specials for exactly the 19 any-C kernels, no reject",
          {c.kernel for c in extra} == any_c and all(c.expect == "ok" and c.image in EXTRA_IMAGES for c in extra)
          and len(extra) == 4 * sum(1 for c in cases if c.kernel in any_c and c.expect == "ok"
                                    and c.image == "noise" and c.channels == 3))
    sel, unknown = select_cases(cases, re.compile(r"^exposure/"), ["exposure/stops=1/hdr/c3", "crop/full/hdr/c1",
                                                                   "no/such/id/c3"])
    check("select_cases: --ids and --cases intersect, in catalogue order, and unknown ids are named",
          [c.id for c in sel] == ["exposure/stops=1/hdr/c3"] and unknown == ["no/such/id/c3"])
    check("select_cases: a regex that matches nothing selects nothing",
          select_cases(cases, re.compile("tonecurve"), None)[0] == [])
    check("histogram field errors: AttributeError (field missing in this build) is not one, others are",
          histogram_field_errors({"counts": {"sha256": "x"}, "cdf": {"error": "PanicException: boom"},
                                  "below": {"error": "AttributeError: no below"}}) == ["cdf"])

    # Digest normalisation
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / np.float32(7)
    ref = sha256_array(base)
    check("digest ignores Fortran order", sha256_array(np.asfortranarray(base)) == ref)
    neg1, neg2 = base[::-1], base[:, ::-1, ::-2]
    check("digest ignores negative strides (views whose strides are negative, against their copies)",
          min(neg1.strides) < 0 and min(neg2.strides) < 0
          and sha256_array(neg1) == sha256_array(np.ascontiguousarray(neg1))
          and sha256_array(neg2) == sha256_array(np.ascontiguousarray(neg2)))
    check("digest ignores byte order", sha256_array(base.astype(">f4")) == ref)
    check("digest of a strided view equals its copy",
          sha256_array(base[:, ::2]) == sha256_array(np.ascontiguousarray(base[:, ::2])))

    # Quantisation reference, against src/quantize.rs's own unit tests
    k = np.arange(256, dtype=np.float32)
    check("quantize: k/255 -> code k for every k", bool(np.all(quantize_codes(k / np.float32(255), 255) == k)))
    check("quantize: 1.5, 1e30 and inf -> 255; NaN, -1 and -0 -> 0",
          list(quantize_codes(np.array([1.5, 1e30, np.inf, np.nan, -1.0, -0.0], np.float32), 255))
          == [255, 255, 255, 0, 0, 0])
    check("quantize: 0.5019608 -> 128", int(quantize_codes(np.array([0.5019608], np.float32), 255)[0]) == 128)
    k16 = np.arange(65536, dtype=np.float32)
    check("quantize: k/65535 -> code k for every k",
          bool(np.all(quantize_codes(k16 / np.float32(65535), 65535) == k16)))
    # x * 255 and x * 65535 are exactly 2.5 for these two floats; the crate gives code 3 for both,
    # where round-half-to-even (np.rint) would give 2.
    check("quantize: an exact half rounds up",
          int(quantize_codes(_bits(0x3C20A0A1), 255)[0]) == 3
          and int(quantize_codes(_bits(0x382000A0), 65535)[0]) == 3)

    # Comparator
    rng_free = uniform24(11, (64, 48, 3)) * np.float32(0.9) + np.float32(0.05)
    a = rng_free.copy()
    b = a.copy()
    b[10, 20, 1] = np.nextafter(b[10, 20, 1], np.float32(np.inf))
    st = diff_stats(a, b)
    check("one 1-ULP flip: exactly one differing element at one ULP",
          st["n_diff"] == 1 and st["max_ulp_all"] == 1 and st["nan_vs_num"] == 0, str(st))
    st = diff_stats(a, a.copy())
    check("identical arrays: nothing differs", st["n_diff"] == 0 and st["max_ulp_all"] == 0, str(st))

    def pair(x, y):
        return diff_stats(np.array([0.25, x], np.float32), np.array([0.25, y], np.float32))

    nan2 = _bits(0x7FC12345)[0]
    st = pair(np.float32(np.nan), nan2)
    check("NaN vs NaN with another payload: equal, payload flagged",
          st["n_diff"] == 0 and st["nan_payload"] == 1, str(st))
    st = pair(np.float32(np.nan), np.float32(0.5))
    check("NaN vs number: one difference, flagged", st["n_diff"] == 1 and st["nan_vs_num"] == 1, str(st))
    st = pair(np.float32(0.0), np.float32(-0.0))
    check("+0 vs -0: one ULP, flagged", st["n_diff"] == 1 and st["max_ulp_all"] == 1 and st["signed_zero"] == 1,
          str(st))
    st = pair(np.float32(np.inf), np.finfo(np.float32).max)
    check("+inf vs FLT_MAX: one ULP, inf flagged", st["max_ulp_all"] == 1 and st["inf_mismatch"] == 1, str(st))
    st = pair(_bits(1)[0], _bits(0x80000001)[0])
    check("+min subnormal vs -min subnormal: three steps (through +0 and -0)", st["max_ulp_all"] == 3, str(st))
    st = pair(np.float32(np.inf), np.float32(-np.inf))
    check("+inf vs -inf: one difference, inf flagged", st["n_diff"] == 1 and st["inf_mismatch"] == 1, str(st))
    t = np.float32(255) * np.float32(2)
    lo = np.float32(np.float32(253) / t)  # (126 + 0.5) / 255, a rounding threshold
    below, above = np.nextafter(lo, np.float32(0)), np.nextafter(lo, np.float32(1))
    st = pair(below, above)
    check("pair straddling a u8 threshold: one u8 code changed by one",
          st["u8_changed"] == 1 and st["u8_max"] == 1, str(st))
    st = diff_stats(np.array([1, 2, 3], np.uint16), np.array([1, 5, 3], np.uint16))
    check("integer outputs: count and max code delta", st["n_diff"] == 1 and st["max_code"] == 3, str(st))
    st = diff_stats(np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32))
    check("shape mismatch is reported, not compared", "shape_mismatch" in st)
    st = diff_stats(np.zeros(3, np.float64), np.array([0, 1e-300, 0], np.float64))
    check("float64 path", st["n_diff"] == 1 and st["max_ulp_all"] > 0, str(st))
    st = pair(np.float32(0.0), np.float32(1e-40))
    check("0 vs a subnormal: ftz, and no normal-value ULP",
          st["ftz"] == 1 and st["max_ulp_all"] > 0 and st["max_ulp_normal"] == 0 and st["sign_change"] == 0, str(st))
    st = pair(np.float32(0.5), np.float32(-0.5))
    check("0.5 vs -0.5: a sign change, and no normal-value ULP",
          st["sign_change"] == 1 and st["max_ulp_normal"] == 0 and st["max_ulp_all"] > 2**30, str(st))
    st = pair(np.float32(0.5), np.nextafter(np.float32(0.5), np.float32(0)))
    check("0.5 vs its lower neighbour: one normal-value ULP", st["max_ulp_normal"] == 1 and st["ftz"] == 0,
          str(st))

    # worst_violation, mirroring tests/cuda_conformance.rs
    # worst_violation_does_not_score_garbage_as_agreement
    ok3 = np.array([1.0, 2.0, 3.0], np.float32)
    nan3, inf3 = np.full(3, np.nan, np.float32), np.full(3, np.inf, np.float32)
    off3 = np.array([1.0, 2.0, 3.5], np.float32)

    def oracle(x, y):
        return worst_violation(x, y, 1e-5, 1e-7)["oracle"]

    check("worst_violation: ok/ok, NaN/NaN and +inf/+inf score 0",
          oracle(ok3, ok3) == 0 and oracle(nan3, nan3) == 0 and oracle(inf3, inf3) == 0)
    check("worst_violation: ok/NaN, NaN/ok and ok/inf score inf, as class changes",
          all(math.isinf(oracle(x, y)) and worst_violation(x, y, 1e-5, 1e-7)["class_changed"] == 3
              for x, y in ((ok3, nan3), (nan3, ok3), (ok3, inf3))))
    check("worst_violation: a real difference scores above 1000", oracle(ok3, off3) > 1000)
    fmax = np.array([np.finfo(np.float32).max], np.float32)
    wv = worst_violation(fmax, -fmax, 1e-5, 1e-7)
    check("worst_violation: float32 arithmetic, as in Rust: FLT_MAX against -FLT_MAX overflows to inf",
          math.isinf(wv["worst"]) and wv["class_changed"] == 0, str(wv))
    wv = worst_violation(np.array([0.049465362], np.float32), np.array([0.049464725], np.float32), 1e-5, 1e-7)
    check("worst_violation: hsl_bw/alt1,s=5/noise/c3's worst pixel at --size 128 is 1.071x (1e-5, 1e-7)",
          1.07 < wv["worst"] < 1.072 and wv["over"] == 1 and wv["class_changed"] == 0, str(wv))
    wv = worst_violation(np.array([2.0], np.float32), np.array([3.0], np.float32), 0.0, 1.0)
    check("worst_violation: the reference is the first argument (|x - y| / (atol + rtol |x|))",
          wv["worst"] == 1.0
          and worst_violation(np.array([3.0], np.float32), np.array([2.0], np.float32), 1.0, 0.0)["worst"]
          == np.float32(1) / np.float32(3))

    # Agreement classes
    def crec(kernel, image="noise", power=None):
        params = [{"ToneCurveParams": {"slope": 1.0, "offset": 0.0, "power": power}}] if power is not None else []
        return {"kernel": kernel, "image": image, "params": params}

    check("agreement: every kernel has a class, and only the 27 kernels",
          set(AGREEMENT) == set(KERNELS))
    check("agreement: tone_curve exact at power 1, bounded (1e-5, 1e-7) otherwise",
          agreement_class(crec("tone_curve", power=1.0)) == ("exact", None)
          and agreement_class(crec("tone_curve", power=2.2)) == ("bounded", (1e-5, 1e-7)))
    check("agreement: film_grain (1e-3, 1e-5), denoise (1e-4, 1e-6), histogram exact, pipeline report-only",
          agreement_class(crec("film_grain")) == ("bounded", (1e-3, 1e-5))
          and agreement_class(crec("denoise")) == ("bounded", (1e-4, 1e-6))
          and agreement_class(crec("histogram")) == ("exact", None)
          and agreement_class(crec("pipeline")) == ("report", None))
    ffi = Path(__file__).resolve().parents[1] / "docs" / "ffi.md"
    if ffi.is_file():
        exact_doc, bounded_doc, quals = ffi_agreement_tables(ffi.read_text(encoding="utf-8"))
        want_exact = {k for k, v in AGREEMENT.items() if v is None} | {"tone_curve", "film_grain"}
        check("agreement: the exact kernels are docs/ffi.md section 6's bit-exact table",
              exact_doc == want_exact, f"doc only {sorted(exact_doc - want_exact)}, "
                                       f"here only {sorted(want_exact - exact_doc)}")
        check("agreement: the (rtol, atol) are docs/ffi.md section 6's bounded table",
              bounded_doc == {k: v for k, v in AGREEMENT.items() if v is not None},
              f"doc {bounded_doc}")
        check("agreement: docs/ffi.md qualifies tone_curve (power == 1 / != 1) and film_grain (its hash)",
              "power == 1" in quals.get(("exact", "tone_curve"), "")
              and "power != 1" in quals.get(("bounded", "tone_curve"), "")
              and "hash" in quals.get(("exact", "film_grain"), ""), str(quals))
    else:
        print(f"SKIP  agreement against docs/ffi.md: {ffi} not found")

    def jdg(kernel, image, x, y, power=None):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        return judge(crec(kernel, image, power), x, y, diff_stats(x, y))["verdict"]

    v1 = np.array([0.25, 0.5, 0.75], np.float32)
    v1up = v1.copy()
    v1up[1] = np.nextafter(v1[1], np.float32(1))
    v1far = v1.copy()
    v1far[1] = np.float32(0.5005)
    v1nan = v1.copy()
    v1nan[1] = np.nan
    nanp = v1nan.copy()
    nanp[1] = _bits(0xFFC12345)[0]
    v1inf = v1.copy()
    v1inf[1] = np.inf
    check("judge: an exact kernel passes on equal values, NaN payloads aside, and fails on one ULP",
          jdg("exposure", "noise", v1nan, nanp) == "pass" and jdg("exposure", "noise", v1, v1up) == "FAIL")
    check("judge: a bounded kernel passes one ULP, fails 1e-3, and fails a class change at any ratio",
          jdg("blur", "hdr", v1, v1up) == "pass" and jdg("blur", "hdr", v1, v1far) == "FAIL"
          and jdg("blur", "hdr", v1, v1nan) == "FAIL")
    check("judge: on specials, exact kernels still fail, a class change fails the class-kept kernels, "
          "other bounded kernels and tone_curve's values are report-only",
          jdg("resize", "specials", v1, v1up) == "FAIL" and jdg("hsl_bw", "specials", v1, v1inf) == "FAIL"
          and jdg("hsl_bw", "specials", v1, v1far) == "pass" and jdg("blur", "specials", v1, v1nan) == "report"
          and jdg("tone_curve", "specials", v1, v1up, power=1.0) == "pass"
          and jdg("tone_curve", "specials", v1, v1nan, power=1.0) == "FAIL")
    check("judge: the README pipeline is report-only", jdg("pipeline", "noise", v1, v1far) == "report")
    hfa = {"counts": np.array([1, 2], np.uint64), "min": np.float32(0)}
    check("judge: a histogram passes only when every field matches",
          judge(crec("histogram"), hfa, dict(hfa), None)["verdict"] == "pass"
          and judge(crec("histogram"), hfa, {**hfa, "counts": np.array([1, 3], np.uint64)}, None)["verdict"]
          == "FAIL" and judge(crec("histogram"), hfa, {"counts": hfa["counts"]}, None)["verdict"] == "FAIL")

    # Dumps must belong to their run lines
    line = {"sha256": sha256_array(v1)}
    hline = {"fields": {"counts": {"sha256": sha256_array(hfa["counts"])},
                        "min": {"sha256": sha256_array(hfa["min"])}, "cdf": {"error": "AttributeError: x"}}}
    check("dump check: the right array passes, another array or a histogram fails",
          dump_mismatch(line, v1) is None and dump_mismatch(line, v1up) is not None
          and dump_mismatch(line, hfa) is not None)
    check("dump check: a histogram passes field by field, errored fields skipped, and fails on a lost field",
          dump_mismatch(hline, dict(hfa)) is None and dump_mismatch(hline, {"counts": hfa["counts"]}) is not None
          and dump_mismatch(hline, {**hfa, "counts": np.array([1, 3], np.uint64)}) is not None)

    # compare's exit codes, on runs built in memory
    def mkcase(cid, arr, kernel, image="noise"):
        return {"kind": "case", "id": cid, "kernel": kernel, "image": image, "params": [], "input_sha256": "i",
                "error": None, **array_digest(arr)}

    def mkrun(backend, recs, summary=True, setup_error=None):
        return ({"kind": "env", "backend": backend, "generator": GENERATOR, "generator_ok": True},
                {"kind": "summary", "setup_error": setup_error} if summary else None,
                {r["id"]: r for r in recs}, [r["id"] for r in recs])

    def cmp(ra, rb, dumps=None, **kw):
        opts = {"cases": None, "allow": None, "expect_identical": False, "expect_bounded": False,
                "bounds": False, "write_ids": None, "dumps": None, **kw}
        loaders = None
        if dumps is not None:
            loaders = (lambda cid: dumps[0].get(cid), lambda cid: dumps[1].get(cid))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return compare_runs(argparse.Namespace(**opts), ra, rb, loaders)

    dead = "RuntimeError: backend error: CUDA driver unavailable"
    check("compare: two runs whose setup failed exit 2, even with --expect-identical",
          cmp(mkrun("gpu", [], setup_error=dead), mkrun("gpu", [], setup_error=dead), expect_identical=True) == 2)
    check("compare: two runs with no case line exit 2",
          cmp(mkrun("cpu", []), mkrun("cpu", []), expect_identical=True) == 2)
    ca1 = [mkcase("blur/x/noise/c1", v1, "blur"), mkcase("exposure/x/noise/c1", v1, "exposure")]
    check("compare: a run whose setup failed against a good one exits 2",
          cmp(mkrun("cpu", ca1), mkrun("cpu", [], setup_error=dead)) == 2)
    check("compare: identical runs exit 0, and 2 when --cases matches nothing",
          cmp(mkrun("cpu", ca1), mkrun("cpu", ca1), expect_identical=True) == 0
          and cmp(mkrun("cpu", ca1), mkrun("cpu", ca1), expect_identical=True, cases="defualt") == 2)
    check("compare --expect-identical: a run without a summary line exits 1",
          cmp(mkrun("cpu", ca1), mkrun("cpu", ca1, summary=False), expect_identical=True) == 1
          and cmp(mkrun("cpu", ca1), mkrun("cpu", ca1, summary=False)) == 0)
    cb_near = [mkcase("blur/x/noise/c1", v1up, "blur"), ca1[1]]
    cb_far = [mkcase("blur/x/noise/c1", v1far, "blur"), ca1[1]]
    d_cpu = {"blur/x/noise/c1": v1, "exposure/x/noise/c1": v1}
    d_near, d_far = {**d_cpu, "blur/x/noise/c1": v1up}, {**d_cpu, "blur/x/noise/c1": v1far}
    rcpu, rnear, rfar = mkrun("cpu", ca1), mkrun("gpu", cb_near), mkrun("gpu", cb_far)
    check("compare --dumps: a dump that is not its run's exits 2 (swapped or repeated directories)",
          cmp(rcpu, rnear, dumps=(d_cpu, d_near)) == 0 and cmp(rcpu, rnear, dumps=(d_near, d_cpu)) == 2
          and cmp(rcpu, rnear, dumps=(d_cpu, d_cpu)) == 2)
    check("compare --expect-bounded: within the bound exits 0, beyond it 1, beyond it under --allow 0",
          cmp(rcpu, rnear, dumps=(d_cpu, d_near), expect_bounded=True) == 0
          and cmp(rnear, rcpu, dumps=(d_near, d_cpu), expect_bounded=True) == 0
          and cmp(rcpu, rfar, dumps=(d_cpu, d_far), expect_bounded=True) == 1
          and cmp(rfar, rcpu, dumps=(d_far, d_cpu), expect_bounded=True) == 1
          and cmp(rcpu, rfar, dumps=(d_cpu, d_far), bounds=True) == 0
          and cmp(rcpu, rfar, dumps=(d_cpu, d_far), expect_bounded=True, allow="^blur/") == 0)
    check("compare --bounds: two CPU runs, or a missing dump, exit 2",
          cmp(rcpu, mkrun("cpu", cb_near), dumps=(d_cpu, d_near), bounds=True) == 2
          and cmp(rcpu, rnear, dumps=(d_cpu, {}), bounds=True) == 2)
    check("compare --expect-bounded: a case only in one run fails it",
          cmp(rcpu, mkrun("gpu", cb_near[:1]), dumps=(d_cpu, d_near), expect_bounded=True) == 1)

    # run's exit codes that come before phaios_core is imported (nothing is written)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rc_sel = main(["run", "--out", os.devnull, "--size", str(GOLDEN_SIZE), "--quiet", "--cases", "tonecurve"])
        rc_size = main(["run", "--out", os.devnull, "--size", str(min_size() - 1), "--quiet"])
        rc_cache = main(["run", "--out", os.devnull, "--size", str(GOLDEN_SIZE), "--quiet", "--cuda-cache", "x"])
    check("run: a --cases regex that selects nothing exits 2", rc_sel == 2)
    check("run: --size below min_size() exits 2", rc_size == 2)
    check("run: --cuda-cache without --gpu exits 2", rc_cache == 2)

    rec = {"input_sha256": "x", "sha256": "s", "output_shape": [1], "output_dtype": "float32", "error": None}
    check("classify identical", classify(rec, dict(rec)) == "identical")
    check("classify differing", classify(rec, {**rec, "sha256": "t"}) == "differing")
    check("classify differing shape", classify(rec, {**rec, "output_shape": [2]}) == "differing")
    check("classify error-changed", classify(rec, {**rec, "error": "ValueError: x"}) == "error-changed")
    check("classify same error as identical",
          classify({**rec, "error": "E: m"}, {**rec, "error": "E: m"}) == "identical")
    check("classify only-in-one", classify(None, rec) == "only-B" and classify(rec, None) == "only-A")
    check("classify input mismatch", classify(rec, {**rec, "input_sha256": "y"}) == "input-mismatch")
    check("classify changed parameters as not comparable",
          classify({**rec, "params": [1.0]}, {**rec, "params": [2.0]}) == "input-mismatch")

    print(f"\n{'selftest passed' if not failures else f'{len(failures)} selftest failure(s)'}")
    return 1 if failures else 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="digest every kernel's output to a JSONL file")
    r.add_argument("--out", required=True, help="output .jsonl path")
    r.add_argument("--gpu", action="store_true", help="run through phaios_core.gpu instead of the CPU")
    r.add_argument("--size", type=int, default=1024,
                   help=f"image size N (default 1024, at least {min_size()})")
    r.add_argument("--cases", help="only run case ids matching this regex (re.search)")
    r.add_argument("--ids", metavar="FILE",
                   help="only run the case ids listed in FILE, one per line (as compare --write-ids writes)")
    r.add_argument("--extra-channels", action="store_true",
                   help="also run C = 2 and C = 4 on noise and specials, for the kernels that take any C")
    r.add_argument("--dump", help="save each output under this directory")
    r.add_argument("--wheel", help="record this wheel's sha256 and check it holds the imported module")
    r.add_argument("--label", help="free text recorded in the env line, such as the commit built")
    r.add_argument("--cuda-cache", metavar="DIR",
                   help="with --gpu, set CUDA_CACHE_PATH to DIR before the driver starts")
    r.add_argument("--strict", action="store_true",
                   help="exit 1 on an unexpected error or success, or a histogram field error")
    r.add_argument("--quiet", action="store_true", help="no progress on stderr")
    c = sub.add_parser("compare", help="compare two runs case by case")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--dumps", nargs=2, metavar=("DIR_A", "DIR_B"), help="element-level detail from dumps")
    c.add_argument("--bounds", action="store_true",
                   help="with --dumps, CPU against GPU: check each case against its class in docs/ffi.md §6")
    c.add_argument("--expect-identical", action="store_true",
                   help="exit 1 unless every case is identical and both runs finished")
    c.add_argument("--expect-bounded", action="store_true",
                   help="implies --bounds: exit 1 unless every case is within its class")
    c.add_argument("--allow", help="with --expect-identical or --expect-bounded, tolerate the case ids "
                                   "matching this regex")
    c.add_argument("--cases", help="only compare case ids matching this regex")
    c.add_argument("--write-ids", metavar="FILE",
                   help="write the ids that are not identical to FILE, one per line, for run --ids")
    sub.add_parser("selftest", help="check the generator and the comparator")
    ns = ap.parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "selftest": cmd_selftest}[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
