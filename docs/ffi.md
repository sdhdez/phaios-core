# phaios-core FFI Contract

This document defines the binding contract between the Rust kernels and
the Python `phaios_core` extension module. Read it before adding or
modifying any `#[pyfunction]` or `#[pyclass]` item.

---

## 1. Array layout

All arrays at the FFI boundary obey a single convention:

| Property | Input | Output |
|----------|-------|--------|
| dtype | `numpy.float32` | `numpy.float32` |
| memory order | **any** — C, Fortran, strided, negative strides | always C-contiguous |
| shape | `(H, W, C)` | `(H, W, C)` |
| value range | caller's responsibility; kernels do not clamp | unclamped |
| linearity | **scene-referred linear** except after `encode_srgb` | |

**Layout.** Kernels accept whatever numpy hands them and always allocate
a fresh C-contiguous result. This is not a convenience: `img[::2, ::2]`
— a downsampled preview — is one of the most ordinary arrays a consumer
can produce, and two kernels used to panic on it (see §4).

**Channel count.** C = 1 for single-channel (luminance) arrays, C = 3
for RGB. What each kernel accepts follows from where it sits in the
pipeline:

| Kernel | In | Out | Notes |
|---|---|---|---|
| `crop` | any | same C, smaller H/W | pure index copy; **geometry runs first** |
| `orient` | any | same C, H/W may swap | pure index permutation; before `crop` |
| `hot_pixels` | any | same | index-clamped 3×3 conditional median; channels filtered independently; right after `orient` |
| `denoise` | any | same | guided-filter noise reduction; `C = 3` cross-guided from a shared luminance guide, every other count self-guided per channel; right after `hot_pixels` |
| `straighten` | any | same C, inscribed H/W | Catmull-Rom resampling; after `orient`, before `crop` |
| `resize` | any | same C, target H/W | separable polynomial resampling; last geometry stage or export prep |
| `exposure` | any | same | a scalar multiply; valid before or after the B&W stage |
| `luminance_bw` | 3 | 1 | |
| `channel_mixer_bw` | 3 | 1 | |
| `color_filter_bw` | 3 | 1 | |
| `hsl_bw` | 3 | 1 | needs hue, so it must run before the collapse |
| `zone_system` | 1 | 1 | |
| `local_contrast` | 1 | 1 | |
| `sharpen` | any | same | threshold-gated Gaussian unsharp mask; channels filtered independently |
| `film_grain` | 1 | 1 | |
| `split_toning` | 1 | **3** | the only kernel that adds channels |
| `vignette` | any | same | one factor per pixel, applied to every channel |
| `tone_curve` | any | same | element-wise |
| `blur` | any | same | separable Gaussian; channels filtered independently |
| `glow` | any | same | scattering: halation, diffusion, veiling glare |
| `shadow_rolloff` | any | same | element-wise; the toe, at the start of the tone stages |
| `highlight_rolloff` | any | same | element-wise; the shoulder, last linear stage before `encode_srgb` |
| `encode_srgb` | any | same | element-wise |
| `quantize_u8` | any | same, **`uint8`** | terminal; display-referred input |
| `quantize_u16` | any | same, **`uint16`** | terminal; display-referred input |
| `apply_lut` | any | same | element-wise through a caller-supplied table |
| `histogram` | any | **not an image** | a reduction — returns a `Histogram`, see below |

A kernel given the wrong channel count raises `ValueError`. The
nineteen that accept "any" do so for four distinct reasons.

The four **geometry** kernels (`crop`, `orient`, `straighten`, `resize`)
are channel-agnostic by nature: they move pixels without looking inside
them, and they run first, before the pipeline has decided anything about
colour.

`hot_pixels` and `denoise` are **restoration** kernels: they run at
native resolution, right after `orient` and before anything resamples
or exposes, and their job — defect removal, noise reduction — is a
purely spatial-statistical one that does not depend on what the
channels represent, only on how many pixels' worth of neighbourhood
each one has. `denoise` additionally special-cases `C = 3` (cross-guided
from a shared luminance guide) without rejecting any other count
(self-guided per channel), so "any" holds for it in the same sense it
does for a colour-agnostic geometry kernel, not by accident.

The six **tone** kernels (`exposure`, `vignette`, `tone_curve`,
`shadow_rolloff`, `highlight_rolloff`, `encode_srgb`) run either side of
`split_toning`, so a pipeline need not branch on whether toning is
enabled. `blur`, `glow` and `sharpen` join them for the same reason —
each channel is filtered, scattered or sharpened independently, so none
needs nor imposes a channel count.

The remaining four — `apply_lut`, `quantize_u8`, `quantize_u16` and
`histogram` — are per-sample by construction. The first three map each
sample independently of its neighbours *and* of its channel; `histogram`
does not return an image at all, and simply reports one row of counts
per channel however many there are.

See [`export.md`](export.md) for what a consumer must do with these
codes — greyscale photometric, profile, depth and dither are specified
there, not left to the file layer.

**Reductions.** `histogram` is the first function in the crate that does
not return an image. It returns a `Histogram` object carrying
`(channels, bins)` counts plus separate `below` / `above` / `non_finite`
tallies, and it is exposed on both surfaces — on the GPU it returns host
data, because counts are small and their destination (a display, an
auto-correction) is on the host.

This widens §1's "pure functions on `(H, W, C)` arrays", deliberately.
The justification is that the decisions around a histogram — linear or
display-referred, what range, what counts as clipped — are
specification-shaped, and two consumers answering them independently will
show the photographer different histograms of the same file.

**Integer output.** `quantize_u8` and `quantize_u16` are the only
kernels that do not return `float32`: they return `uint8` and `uint16`
respectively, still `(H, W, C)` and still freshly allocated and
C-contiguous. Two monomorphic functions rather than one function with a
bit-depth argument, so the returned dtype is a static property of the
call rather than something a caller has to inspect. They are terminal —
on the GPU surface they return a numpy array rather than a `GpuImage`,
because a quantised buffer has nowhere further to go on the device.

**Allocation is bounded, and refuses rather than aborts.** No single
array a kernel allocates may exceed **8 GiB**; beyond that the call
raises Python `MemoryError` — the same exception numpy raises for the
same request — instead of attempting it.

The reason is a hard one. A failed `Vec` allocation in Rust calls
`handle_alloc_error`, which **aborts** rather than unwinding: it raises
nothing at all, not even `PanicException`, and takes the interpreter with
it, so a consumer cannot catch it with `except Exception`, `except
BaseException`, or anything else. That is strictly worse than the panic
§2 forbids, and it is reachable from ordinary input — a numpy
`broadcast_to` view has an unbounded logical shape backed by as little as
four bytes, and every kernel sizes its output from that logical shape.

The limit cannot be delegated to the allocator. Measured on Linux with
the default heuristic overcommit on a 60 GiB machine,
`Vec::try_reserve_exact` **succeeded** for a 111 GiB request; the process
died later under the OOM killer when the kernel wrote to the pages. Nor
can it be derived from free memory: that would make the same call succeed
or fail depending on ambient machine state, which is exactly what §2's
purity rule forbids. So it is a constant, and it is documented here
rather than left to be discovered.

For scale: a 24 MP three-channel `f32` frame is 285 MiB, so the limit is
about twenty-eight times the crate's benchmark size and allows roughly a
700 MP RGB image. A legitimate output above it *is* refused; that is a
real limit, and the better failure.

**What this does not bound (v0.2).** Only *single* allocations. Three
things remain a caller's responsibility, and are recorded here as known
gaps rather than solved ones:

- **Peak pipeline footprint.** A kernel may hold several
  full-resolution intermediates at once — `local_contrast` builds f64
  summed-area tables — and a chain of kernels holds more. Each
  allocation may pass the check while the total exhausts memory.
- **Cumulative use across calls.** Nothing tracks what a caller has
  already allocated, so a loop over many frames can exhaust memory with
  every individual call inside the limit.
- **Device memory.** The CUDA path bounds its *host* staging copy but
  relies on the driver returning an out-of-memory error for device
  allocations, which it does — `PhaiosError::Backend`, catchable — so
  this one is a difference in error type rather than a hole.

Bounding a whole pipeline's peak is a design question, not a constant:
it needs either an allocation arena the caller owns or a declared budget
threaded through the API. Deferred past v0.2 deliberately.

**Pixel values must be finite.** This is a precondition, not a validated
input: kernels check their *parameters* and never scan their pixels — a
finiteness pass over a 24 MP frame would cost more than most kernels do.
NaN and ±∞ therefore propagate through the maths rather than raising, and
**with a non-finite sample present the result is unspecified and the
backends may disagree without bound.** The determinism contract in §6
holds for finite input only.

Two things are worth knowing about how the disagreement behaves, because
they are structural rather than incidental:

- *Neighbourhood kernels do not, and the reasons differ per kernel.*
  `blur` and `glow` (which is built on it) diverge on ±∞ **below σ = 6**,
  where the device runs a direct convolution accumulated in
  Kahan-compensated f32: the compensation term computes `∞ − ∞ = NaN`,
  which then poisons every later term, so the device returns NaN where
  the host's uncompensated f64 sum returns ±∞. One infinite sample is
  enough. Dropping the compensation there would trade a divergence on
  input the contract already excludes for a real accuracy loss on input
  it does not.

  At or above σ = 6 both backends accumulate the box passes in f64 and
  agree, including on ±∞ — a sliding window subtracts, so an infinity
  becomes NaN on *both* sides. That path used to accumulate in
  Kahan-compensated f32 and was not merely divergent on non-finite input:
  it missed the committed bound by up to 2.8e7× on ordinary linear
  scene-referred data with a highlight in it, because Kahan's error
  bound scales with `Σ|xᵢ|`, which a bright sample dominates.
  `blur_agrees_within_bound_across_the_dynamic_range` pins it.
- *Element-wise and geometry kernels* stay in agreement anyway. Their
  arithmetic is per-pixel, so a poisoned sample poisons exactly its own
  output on both backends. `non_finite_pixels_agree_across_backends` in
  `tests/cuda_conformance.rs` pins this for `hsl_bw`, which is the
  trickiest of them (an all-infinite pixel makes `delta = ∞ − ∞ = NaN`,
  and NaN fails every comparison — so the neutral guard must be written
  as a positive test, or one backend falls through it).
- *`local_contrast` does not, and cannot cheaply.* The CPU computes its
  box filters from global f64 summed-area tables, so one non-finite
  sample enters every prefix at or after its position and `∞ − ∞ = NaN`
  spreads to the whole image; the CUDA kernel uses separable box passes,
  which confine the damage to a `(4r+1)²` neighbourhood. Measured on a
  96×96 frame at r = 4 with a single ∞ pixel: 9216 non-finite outputs on
  the CPU against 81 on the GPU, with every finite GPU pixel bit-correct.
  Making these agree would mean giving up either the CPU's O(1)-per-pixel
  SAT or the GPU's locality, and neither is worth buying agreement on
  input the contract already excludes.

A consumer that cannot rule out non-finite samples — a dead sensor pixel,
a 0/0 white-balance division, an EXR carrying ∞ — should replace them
before the pipeline, not rely on a kernel to absorb them.

**Size.** H and W are unconstrained. Zero-size arrays are accepted and
return an empty array of the same shape; treating "no pixels" as an
error would push a special case onto every caller. **Exception:** the
resampling kernels `resize` and `straighten` reject an empty *input*
with `ValueError` — there is no meaningful sample to draw from, and
inventing zeros would violate their resampling contract. (`crop` may
still *produce* an empty array from a zero-size rectangle.)

---

## 2. Zero-copy rules

**Inputs** are received as `PyReadonlyArray3<f32>`. The array's backing
store is not copied; the Rust code borrows a read-only view for the
duration of the call.

Do not call `.to_owned()` on an input array. If you need a mutable copy
for scratch work, allocate a fresh `Array3<f32>` with `ndarray::Array3::zeros`
and write into it.

**Outputs** are allocated exactly once:

```rust
let out = Array3::<f32>::zeros((h, w, c_out));
// ... fill out ...
Ok(out.into_pyarray(py).unbind())
```

`PyArray3::from_array` (which copies) is forbidden on the hot path.

Caller guarantees:

- The input array must be alive for the duration of the call (Python
  reference counting ensures this for ordinary calls).
- The dtype must be `float32`. Anything else is a `TypeError` or
  `ValueError` from PyO3 before the kernel runs.

The caller does *not* have to make the array contiguous. Kernels use
`ndarray::Zip`, which walks any layout; `.ascontiguousarray()` before a
call only adds a copy. Note that a kernel reading a strided input is
slower than one reading a contiguous input, but correctness never
depends on it.

---

## 3. Param-object conventions

Kernel parameters are passed as `#[pyclass]` Rust structs. Benefits
over keyword arguments: the struct can be built once and reused across
many calls; it carries its own `__repr__` for debugging; it is typed.

Pattern:

```rust
/// Parameters for the guided filter.
#[pyclass]
#[derive(Clone)]
pub struct GuidedFilterParams {
    /// Filter radius in pixels.
    #[pyo3(get, set)]
    pub radius: u32,
    /// Regularisation term ε.
    #[pyo3(get, set)]
    pub eps: f32,
}

#[pymethods]
impl GuidedFilterParams {
    #[new]
    pub fn new(radius: u32, eps: f32) -> Self {
        Self { radius, eps }
    }
}
```

From Python:

```python
params = phaios_core.GuidedFilterParams(radius=8, eps=0.01)
result = phaios_core.local_contrast(img, params, strength=0.3)
```

Every param struct must:

- Derive `Clone` (the pipeline may clone params for preview rendering).
- Expose all fields for reading — `#[pyo3(get, set)]` for plain fields,
  or a `#[getter]` returning a copy for fields whose Rust type needs
  converting (`ZoneParams.offsets` returns a dict). Consumers serialise
  these into sidecar and preset files; a write-only param object cannot
  round-trip.
- Provide a `#[new]` constructor with positional arguments matching
  the field order.
- Implement `__eq__`, so consumers can compare parameter objects to
  decide whether a cached render is still valid. Note PyO3 makes a class
  unhashable as soon as `__eq__` is defined — correct for value-compared
  objects, but it means params cannot be dict keys.
- Implement `__repr__`.
- Be registered on the module: `m.add_class::<GuidedFilterParams>()?;`

**Validation belongs in the kernel, not the constructor.** Constructors
stay infallible so their signatures remain stable (CLAUDE.md §2); the
kernels already return `Result`, so that is where a zone index outside
0..=10 or a negative `eps` is rejected.

---

## 4. Error handling

All `#[pyfunction]` items return `PyResult<T>`. Never panic at the FFI
boundary.

Conversion chain:

```
PhaiosError (thiserror)
  → impl From<PhaiosError> for PyErr
    → Python raises ValueError (or the appropriate subclass)
```

The `From` impl lives in `src/error.rs` and matches exhaustively, so a
new variant cannot be added without deciding what it becomes in Python.
Variant-to-exception mapping:

| PhaiosError variant | Python exception | Raised for |
|--------------------|-----------------|------------|
| `Shape(_)` | `ValueError` | wrong channel count or dimensionality |
| `Parameter(_)` | `ValueError` | a parameter outside its domain — zone index not in 0..=10, negative `eps`, any non-finite float |
| `Backend(_)` | `RuntimeError` | no CUDA device, driver missing, compute capability below 8.0, or a device operation failed — an environment condition, not a bad argument |
| `Allocation(_)` | `MemoryError` | an array would exceed the 8 GiB single-allocation limit, or its shape overflows a count. Not a `ValueError`: the arguments are well-formed, there is simply too much of them — and it is what numpy raises for the same request |

Add new variants as needed; always map to the most specific Python
exception class.

### Never panic on caller input

PyO3 converts a Rust panic into `pyo3_runtime.PanicException` rather
than aborting the process — but that is not a safety net, because
**`PanicException` inherits from `BaseException`, not `Exception`**. It
passes straight through a consumer's `except Exception:` handler, so in
a GUI it does not surface as a failed operation; it kills the worker
thread.

Two v0.1 kernels called `as_slice().expect("... must be C-contiguous")`
and so panicked on any strided input. Both now accept any layout. The
rule this leaves:

- No kernel may panic on anything the caller can pass. Invalid input is
  a `PhaiosError`, never an `expect`.
- `expect` is acceptable only for invariants the kernel itself
  establishes (a freshly allocated array having the shape it was just
  allocated with), and must be documented on the function.
- Test the boundary, not just the happy path: `tests/ffi.py` runs every
  public kernel through a five-layout matrix — C-contiguous,
  row-strided, column-strided, Fortran-order and reversed — because a
  consumer passing `img[::2, ::2]` is normal usage, not misuse.

---

## 5. GIL release

Any kernel that is O(N) or worse on image pixels must release the GIL
via `py.detach(move || { ... })`. PyO3 0.22 removed `allow_threads`;
the closure must be `Send`. `ArrayView3<f32>` is `Copy + Send` so it
can be captured by `move`.

```rust
#[pyfunction]
pub fn zone_system(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: ZoneParams,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let out = py.detach(move || {
        // computationally expensive work; no Python API calls
        compute(view, &params)
    })?;
    Ok(out.into_pyarray(py).unbind())
}
```

Kernels that do NOT release the GIL must have a doc comment explaining
why (e.g., very small fixed-size output, trivial per-pixel map).

`rayon::par_iter` within `detach` is safe and recommended for
tile-based kernels. Do not spawn rayon work outside `detach`.

---

## 6. Determinism

Determinism is scoped to a **backend** — a named pair of implementation
and target:

- `cpu/<target-triple>`: the reference implementation, together with the
  platform libm it links against.
- `cuda/<device>/cc<maj>.<min>/ptx-compute_80/nvcc-<maj>.<min>.<patch>`:
  a CUDA device, together with the target architecture of this crate's
  PTX and the CUDA toolkit that generated it.
  `GpuContext.fingerprint` returns this string.

  The `nvcc-` segment is load-bearing, not decoration. The PTX is not
  committed: it is regenerated at build time by whatever toolkit is
  present, and the bounded kernels below inline libdevice code for
  `powf`, `expf`, `log2f` and `cbrtf` whose bodies change between
  toolkits. Two builds of the same commit under different toolkits are
  therefore two different backends, and until this segment existed they
  reported the same key — this project moved 13.3 -> 13.4 and driver 610
  -> 615 with the fingerprint unchanged, and nothing would have noticed
  a drift, because the conformance suite compares against the CPU with
  tolerances rather than against a stored golden GPU output. The
  bit-exact kernels listed below use IEEE operations only and agree
  across toolkits regardless.

**Within one backend**, two calls with the same input, parameters and
seed produce **bit-identical** output — in the same process, in another
process, on another machine with the same fingerprint, at any thread
count, block schedule or launch geometry. This is asserted per kernel
by the conformance suite and must never be relaxed.

**Across backends**, output is *bounded*, not identical — and this was
never achievable, GPU or no GPU. IEEE-754 standardises `+ − × ÷ √` and
requires correct rounding; it standardises **no** transcendental. The
CPU kernels do not implement `exp`, `pow`, `log`, `cos` or `cbrt` —
`nm -D` on the built extension shows them resolving to the platform
libm, and the published wheels link three different libms (glibc,
Windows UCRT, Apple's). Two CPU machines with different libms already
disagree in the low bits. The CUDA backend adds one more math library
to that list, not a new category of problem.

What is promised across backends:

- Kernels free of transcendentals are **bit-exact** between CPU and
  CUDA: `exposure`, `luminance_bw`, `channel_mixer_bw`,
  `color_filter_bw`, `vignette`, `tone_curve` at `power == 1`, every
  identity fast path, `film_grain`'s integer hash (asserted over
  2²⁰ coordinates), the geometry kernels `crop` and `orient` (pure
  index permutations), `hot_pixels` (a fixed 19-comparator median-of-9
  sorting network — comparisons and selection only, no arithmetic beyond
  one subtraction and its comparison against the threshold, both
  correctly rounded; `f32::min`/`f32::max` and CUDA's `fminf`/`fmaxf`
  both implement IEEE-754-2008 `minNum`/`maxNum`, so a NaN in the window
  cannot change *which* operations run on either backend, only the
  values inside them — confirmed on hardware by
  `hot_pixels_agrees_bit_for_bit_on_nan_and_inf_input`), the resampling
  kernels `resize` and
  `straighten` (polynomial filters; straighten's sin/cos is computed
  once on the host and shared), and `highlight_rolloff` (a quadratic
  solve — IEEE-754-2008 §5.4.1 requires `sqrt` to be correctly rounded
  just as it does the four arithmetic operations, so a curve built from
  those five alone carries across), `shadow_rolloff` (a cubic in Horner
  form — multiply, add, subtract and one divide), `apply_lut` (subtract, divide,
  multiply, truncate and one linear interpolation), `histogram` (whose
  only float arithmetic is the bin assignment — everything after it is
  integer counting, and integer addition commutes, so the order in which
  the device's atomics complete cannot change a total), and
  `quantize_u8` / `quantize_u16`
  (exact integer hashing for the dither, and `floor(v + 0.5)` for the
  rounding — an exact operation composed with a correctly-rounded one,
  rather than a library rounding routine that host and device could
  implement differently). This is achievable because the PTX is compiled with
  `-fmad=false` — Rust does not contract `a*b+c` into FMA, and with the
  device told the same, every remaining operation is correctly rounded
  on both sides.
- Kernels containing transcendentals hold a committed per-kernel bound,
  asserted against the CPU oracle by `tests/cuda_conformance.rs`:
  (rtol 1e-5, atol 1e-7) for one-`powf`/`expf`/`cbrtf` kernels,
  (rtol 1e-4, atol 1e-6) for `local_contrast` (which reformulates the
  global f64 summed-area tables as separable box filters — dropping the
  *global prefix sum*, but not the f64: the L and L² partial sums stay
  f64 because their difference is the variance, and that subtraction
  cancels. The coefficient sums downstream are Kahan-compensated f32,
  which is enough because nothing there is squared or subtracted),
  (rtol 1e-5, atol 1e-7) for `blur`, whose direct path makes the
  f64-to-Kahan-f32 substitution — sound there because every weight is
  positive and nothing is subtracted — while its box path keeps f64,
  a sliding window being a subtraction, (rtol 1e-5, atol 1e-7) for
  `glow`, which inherits the blur's bound because the blur is the only
  inexact part of it — the threshold subtraction and the weighted add
  either side are correctly rounded, and (rtol 1e-5, atol 1e-7) for
  `sharpen`, for the same reason: its blur is the same shared
  implementation, and the subtract/gate/combine pointwise kernel that
  follows it is free of transcendentals, so the blur is the only
  inexact part of `sharpen` too. (rtol 1e-4, atol 1e-6) for `denoise` —
  `local_contrast`'s own GUIDED_FILTER class, inherited automatically at
  `C = 1` since it is bit-for-bit the same device kernel
  (`local_contrast_device` at `strength = -amount`), and confirmed
  empirically at `C = 3` for the one new cancelling subtraction
  cross-guided adds, `cov(I, p_c)`: worst measured ratios 0.0022×
  (`C = 1`, unit range), 0.0012× (`C = 1`, HDR), 0.0106× (`C = 3`, unit
  range), 0.0012× (`C = 3`, both the uniform- and partial-channel-bar
  HDR sweeps, up to the required 1e8 highlight). `radius` is capped at
  `MAX_RADIUS = 32` on both backends — `denoise` sums each window
  directly from its own pixels rather than through a summed-area table
  (docs/architecture.md §22), so cost is linear in `radius` and no
  larger case is ever asserted. (rtol 1e-3, atol 1e-5) for
  `film_grain`'s Box–Muller half, whose splitmix64 hash underneath is
  exact and is asserted over a coordinate grid by `hash_grid`, not by
  the identity case in `examples/23_gpu_selftest.rs` — that one is a
  device-to-device copy and never launches the grain kernel.
  A driver update that regresses accuracy fails the suite rather than
  being absorbed.

**The backend fingerprint is part of the reproducibility key.** A
consumer that promises exact reproduction from a settings file must
either record the fingerprint alongside the parameters, or designate
`cpu` as the archival backend and use CUDA for interactive work.
`phaios` desktop should do the latter and record the fingerprint
regardless, so a preview render is identifiable as one.

### Explicit seeds

No global RNG. No thread-local RNG. Every kernel that uses randomness
takes an explicit `seed: u64` — in `film_grain` it is a field of
`GrainParams`, so it travels with the rest of the settings and lands in
the consumer's sidecar file automatically.

### Position-keyed noise beats a seeded generator

`film_grain` does not run a sequential generator at all. Each pixel's
noise is a hash of `(seed, x, y)`:

```rust
let bits = splitmix64(seed ^ splitmix64(mix(x, y)));
let noise = box_muller(bits);
```

The point is that no pixel depends on another's state, so the output is
identical whatever the thread count — verified against rayon pools of 1,
2, 3 and 8 threads.

The alternative, a `SeedableRng` per tile keyed by tile index, is
reproducible only as long as the tiling never changes. The tile size
would become part of the output contract without ever being written
down, and changing it later would alter every image the consumer had
already rendered. Prefer a scheme with no hidden parameters.

### Ordered reductions

A seed is not sufficient. f32 addition is not associative, so any
reduction over an unordered collection has to be given an order.

`zone_system` shipped in v0.1 summing its Gaussian terms in `HashMap`
iteration order, which depends on the per-instance `RandomState` seed.
The result: eight processes given identical input produced eight
different images — 12.8% of pixels off by 1 ULP, and 23 in 65536
differing by one code after 16-bit quantisation. Small enough to be
invisible, large enough to break checksums and reproducible renders.

So:

- Sort before reducing over a `HashMap`, `HashSet` or anything else
  without a defined order.
- Fix the reduction order when parallelising a sum — rayon's `sum()`
  over a parallel iterator does not guarantee one. Where an order cannot
  be fixed cheaply, accumulate in f64 so the result is insensitive to it.
- Assert it: a determinism test comparing repeated calls belongs beside
  any kernel with a seed or an unordered reduction.

---

## 7. Version lockstep

The Rust crate version in `Cargo.toml` and the Python wheel version in
`pyproject.toml` must be identical at all times. CI verifies this with:

```sh
CRATE=$(cargo metadata --no-deps --format-version=1 \
  | python -c "import sys,json; print(json.load(sys.stdin)['packages'][0]['version'])")
WHEEL=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
[ "$CRATE" = "$WHEEL" ] || { echo "version mismatch: $CRATE vs $WHEEL"; exit 1; }
```

When bumping the version for a release, update both files in the same
commit.

## 8. Type stubs

The typing contract, in brief — see `CONTRIBUTING.md` §"Type stubs" for
how it's enforced:

- Images are `NDArray[np.float32]`; shape and channel-count rules are
  documented in each kernel's docstring, not the type (numpy's typing
  has no shape parameter). `quantize_u8`/`quantize_u16` are the
  exception, returning `NDArray[np.uint8]`/`NDArray[np.uint16]`.
- Param objects are unhashable (§3: PyO3 drops `__hash__` once `__eq__`
  is defined) and declare `__hash__: ClassVar[None]`, the typeshed idiom
  a type checker recognises.
- `gpu` is typed but optional at runtime — present only in a
  `--features cuda` build. Consumers guard the import.
- The package uses maturin's mixed layout (`python-source = "python"`)
  solely to carry these stubs (maturin's pure-Rust layout supports only
  a single root-level stub, not a submodule one); `__init__.py` in
  `python/phaios_core/` is byte-identical to what maturin generates on
  its own, so this changes packaging, not runtime behaviour.
