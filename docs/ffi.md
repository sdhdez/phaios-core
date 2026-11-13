# phaios-core FFI Contract

This document defines the binding contract between the Rust kernels and
the Python `phaios_core` extension module. Read it before adding or
modifying any `#[pyfunction]` or `#[pyclass]` item.

---

## 1. Array layout

All arrays at the FFI boundary obey a single convention. This section is
the normative contract; [kernels.md](kernels.md) explains, per kernel,
what the shapes below mean for an image.

| Property | Input | Output |
|----------|-------|--------|
| dtype | `numpy.float32` | `numpy.float32` |
| memory order | **any** — C, Fortran, strided, negative strides | always C-contiguous |
| shape | `(H, W, C)` | `(H, W, C)` |
| value range | caller's responsibility; kernels do not clamp | unclamped |
| linearity | **scene-referred linear** except after `encode_srgb` | |

**Layout.** Kernels accept whatever numpy hands them and always allocate
a fresh C-contiguous result. This is not a convenience: `img[::2, ::2]`,
a downsampled preview, is one of the most ordinary arrays a consumer can
produce.

**Channel count.** C = 1 for single-channel (luminance) arrays, C = 3
for RGB. What each kernel accepts follows from where it sits in the
pipeline:

| Kernel | In | Out | Notes |
|---|---|---|---|
| `orient` | any | same C, H/W may swap | pure index permutation; **geometry runs first** |
| `hot_pixels` | any | same | index-clamped 3×3 conditional median; channels filtered independently; right after `orient` |
| `denoise` | any | same | guided-filter noise reduction; `C = 3` cross-guided from a shared luminance guide, every other count self-guided per channel; right after `hot_pixels` |
| `straighten` | any | same C, inscribed H/W | Catmull-Rom resampling; after `orient`, before `crop` |
| `crop` | any | same C, smaller H/W | pure index copy; after `orient`, so the rectangle is expressed in the upright frame |
| `resize` | any | same C, target H/W | separable polynomial resampling; last geometry stage or export prep |
| `exposure` | any | same | a scalar multiply; valid before or after the B&W stage |
| `luminance_bw` | 3 | 1 | |
| `channel_mixer_bw` | 3 | 1 | |
| `color_filter_bw` | 3 | 1 | |
| `hsl_bw` | 3 | 1 | needs hue, so it must run before the collapse |
| `zone_system` | 1 | 1 | |
| `blur` | any | same | separable Gaussian; channels filtered independently |
| `glow` | any | same | scattering: halation, diffusion, veiling glare |
| `local_contrast` | 1 | 1 | |
| `sharpen` | any | same | threshold-gated Gaussian unsharp mask; channels filtered independently |
| `shadow_rolloff` | any | same | element-wise; the toe, at the start of the tone stages |
| `tone_curve` | any | same | element-wise |
| `film_grain` | 1 | 1 | |
| `split_toning` | 1 | **3** | the only kernel that adds channels |
| `vignette` | any | same | one factor per pixel, applied to every channel |
| `highlight_rolloff` | any | same | element-wise; the shoulder, last linear stage before `encode_srgb` |
| `encode_srgb` | any | same | element-wise |
| `apply_lut` | any | same | element-wise through a caller-supplied table |
| `histogram` | any | **not an image** | a reduction — returns a `Histogram`, see below |
| `quantize_u8` | any | same, **`uint8`** | terminal; display-referred input |
| `quantize_u16` | any | same, **`uint16`** | terminal; display-referred input |

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

The limit cannot be delegated to the allocator. Under Linux's default
heuristic overcommit, `Vec::try_reserve_exact` succeeds for a request far
larger than the machine's memory, and the process dies under the OOM
killer later, when the pages are written. Nor can the limit be derived
from free memory: that would make the same call succeed or fail
depending on ambient machine state, which is exactly what §2's purity
rule forbids. So it is a constant, documented here rather than left to
be discovered. A user-facing summary of what each failure looks like in
Python is in [kernels.md](kernels.md#errors).

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
  becomes NaN on *both* sides. f64 is required there rather than
  Kahan-compensated f32: Kahan's error bound scales with `Σ|xᵢ|`, which a
  bright sample dominates, so a compensated sliding window misses the
  committed bound on ordinary scene-referred data with a highlight in it
  (architecture.md §16). `blur_agrees_within_bound_across_the_dynamic_range`
  pins it.
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
  spreads over the quadrant below and right of it, widened by 2r; the
  CUDA kernel uses separable box passes, which confine the damage to a
  `(4r+1)²` neighbourhood. Measured on a 96×96 frame at r = 4 with a
  single ∞ pixel at its centre: 3136 non-finite outputs on the CPU
  against 289 on the GPU, the whole `(4r+1)²` neighbourhood, and every pixel finite on both within the
  (1e-4, 1e-6) bound.
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
for scratch work, allocate a fresh `Array3<f32>` with `alloc::zeros3`
and write into it.

**Outputs** are allocated exactly once:

```rust
let out = alloc::zeros3::<f32>((h, w, c_out))?;
// ... fill out ...
Ok(out.into_pyarray(py).unbind())
```

Never `ndarray::Array3::zeros` on a shape derived from caller input: a
failed `Vec` allocation *aborts* rather than unwinding, which no `except`
can catch, and a zero-stride numpy broadcast reaches that from four bytes
of storage. `alloc::zeros3` applies the 8 GiB single-allocation bound
(`src/alloc.rs`) and returns `PhaiosError::Allocation`, which Python sees
as `MemoryError`. `alloc::check_shape` is the same bound without the
allocation, for the paths that hand the shape to something else.

`PyArray3::from_array` (which copies) is forbidden on the hot path.

Caller guarantees:

- The input array must be alive for the duration of the call (Python
  reference counting ensures this for ordinary calls).
- The dtype must be native-endian `float32`, and each array argument
  must have the number of dimensions its binding declares. Anything
  else is a `TypeError` raised by rust-numpy while it extracts the
  argument, before the kernel runs. The message spans two lines:
  `type mismatch:\n from=float64, to=float32`, or
  `dimensionality mismatch:\n from=2, to=3`.

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
#[pyclass(from_py_object)]
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

- Derive `Clone` (the pipeline may clone params for preview rendering)
  and declare `#[pyclass(from_py_object)]`. Since PyO3 0.30 no
  `FromPyObject` is derived unless asked. The CPU bindings take params
  as `PyRef<'_, T>`, which needs none, but enum arguments and several
  GPU bindings in `src/gpu_py.rs` take them by value, and those do not
  compile without the attribute (the GPU ones only under
  `--features cuda`). Declaring it on every param struct keeps either
  form available.
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

**Validation belongs in the kernel, not the constructor.** A `#[new]`
returning `PyResult` changes the constructor's signature, and a shipped
signature is frozen until the next major version, so constructors stay
infallible. The kernels already return `Result`, so that is where a zone
index outside 0..=10 or a negative `eps` is rejected.

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
| `Shape(_)` | `ValueError` | wrong channel count, or a shape a kernel cannot accept (a wrong *number* of dimensions never gets this far: rust-numpy rejects it, and a wrong dtype, with `TypeError` while extracting the argument — §2) |
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

The rules:

- No kernel may panic on anything the caller can pass, including an
  unusual array layout. Invalid input is a `PhaiosError`, never an
  `expect`.
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
via `py.detach(move || { ... })`. PyO3 0.26 renamed `allow_threads` to
`detach` and 0.28 removed the old name; the closure must be `Send`. `ArrayView3<f32>` is `Copy + Send` so it
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
  present, and the bounded kernels below inline toolkit code — libdevice
  `cbrtf` in `split_toning` and `logf`/`cosf` in `film_grain`, nvcc's
  fast-math intrinsics in the four files compiled with fast math — that
  changes between toolkits. Two builds of the same commit under
  different toolkits are two different backends, and without this
  segment they would report the same key. Nothing else would catch the
  drift: the conformance suite compares against the CPU with tolerances,
  not against a stored golden GPU output. The bit-exact kernels listed
  below use IEEE operations only and agree across toolkits regardless.

  Each CUDA file is compiled in one of three precision modes, chosen per
  file by `build.rs` (`PRECISION`) from measured speed gains and listed in
  [gpu.md](gpu.md#precision-modes): **strict** (no FMA contraction, IEEE
  division and square root), **FMA** (`local_contrast`, `denoise`) and
  **fast math** (`zone_system`, `hsl_bw`, `encode_srgb`, `tone_curve`).
  The modes are part of the source, not of the fingerprint. In the FMA
  and fast-math files, and wherever approximate instructions remain
  (libdevice `cbrtf` in `split_toning`), the driver's JIT compiler
  finishes part of the arithmetic — it may fuse a multiply with an add,
  and it lowers the approximate instructions — so a driver update can
  move their last bits while the fingerprint stays the same. The
  fingerprint does not name the driver.

**Within one backend and driver**, two calls with the same input,
parameters and seed produce **bit-identical** output — in the same
process, in another process, on another machine with the same
fingerprint and driver, at any thread count, block schedule or launch
geometry. This is asserted per kernel by the conformance suite and must
never be relaxed.

**Across backends**, output is *bounded*, not identical — and this was
never achievable, GPU or no GPU. IEEE-754 standardises `+ − × ÷ √` and
requires correct rounding; it standardises **no** transcendental. The
CPU kernels do not implement their transcendentals either: the built
extension imports `powf`, `expf`, `exp2f`, `logf`, `log2f`, `cosf`,
and f64 `exp` and `sincos`, from the platform libm, and the published
wheels link three different libms (glibc, Windows UCRT, Apple's).
Per pixel they serve `encode_srgb`, `tone_curve` at `power != 1`,
`zone_system`, `hsl_bw` and `film_grain`; once per call, on the host,
`exposure`'s gain, `straighten`'s angle, and the Gaussian taps of
`blur` below σ = 6, and so of `glow` and `sharpen`.
`tests/libm-imports.txt` names the call sites, and
`scripts/libm-inventory.sh` fails CI when the set changes.
`split_toning`'s cube root is not among them on Linux x86-64, where
this was checked: Rust compiles its own `cbrtf`, from
compiler-builtins, into the extension. On the macOS and Windows wheels
this is unverified, and Rust documents `f32::cbrt` as libc's `cbrtf`.
Two CPU machines with different libms can disagree in the low bits.
The CUDA backend adds one more math library to that list, not a new
category of problem. On top of that, the CUDA backend gives up
exactness on purpose where it pays: the files compiled with FMA or
fast math trade some ULP for up to half their run time. On the digest
test images that is at most 90 ULP where a result does not cancel
(`tone_curve`'s `powf`). Where it cancels towards zero, as the guided
filters' `L + s·(L − q)` and `hsl_bw` under opposing hue weights do,
the count runs to millions (1.4 million in `denoise` on the ramp image)
at an absolute error below 1e-6. Every bounded kernel stays within one
8-bit code of the CPU on every finite test image.

What is promised across backends. The four classes below are summarised
once, as a table of kernels per class, in
[gpu.md](gpu.md#agreement-classes); this section is the normative source
and says why each kernel sits where it does.

**Bit-exact between CPU and CUDA.** These kernels contain no
per-pixel transcendental (any they have is one host-side scalar both
backends share), and every operation in them is correctly rounded on
both sides: Rust does not contract `a*b+c` into an FMA, their files are
compiled strict (no FMA contraction, IEEE division and square root), and
`tone_curve`, compiled with fast math, pins its affine step to the CPU's
two roundings in the source. Strict and free of transcendentals is not
enough on its own: `blur`, `glow` and `sharpen` are both, yet bounded,
because the device reformulates the algorithm. The list decides, not a
rule. NaN bits are the one exception: where a GPU kernel computes on a
NaN, the NaN comes back as `0x7fffffff` whatever its payload and sign,
while the CPU propagates the input's bits. On the digest test images
that is the only difference for `exposure`, `luminance_bw`,
`channel_mixer_bw`, `color_filter_bw`, `resize` and `straighten`.

| Kernel | Why it is exact |
|---|---|
| `exposure`, `luminance_bw`, `channel_mixer_bw`, `color_filter_bw`, `vignette` | multiplies and adds only, plus `vignette`'s divisions and square root, which IEEE-754 also rounds correctly. `exposure`'s gain `2^stops` is one host-side `exp2f` from the platform libm, shared by both backends |
| `tone_curve` at `power == 1`, unless an input or the affine result is subnormal | the power branch is skipped and the affine step is pinned to the CPU's two roundings |
| every identity fast path | the input is returned or copied |
| `film_grain`'s integer hash | integer arithmetic, asserted over 2²⁰ coordinates |
| `crop`, `orient` | pure index permutations, no arithmetic on pixel values |
| `resize`, `straighten` | polynomial filters; `straighten`'s sin/cos is one host-side `sincos` from the platform libm, computed once and shared by both backends |
| `hot_pixels` | a fixed 19-comparator median-of-9 sorting network: comparisons and selection only, plus one subtraction and its comparison against the threshold. `f32::min`/`max` and CUDA's `fminf`/`fmaxf` both implement IEEE-754-2008 `minNum`/`maxNum`, so a NaN in the window cannot change *which* operations run, only the values inside them. Confirmed on hardware by `hot_pixels_agrees_bit_for_bit_on_nan_and_inf_input` |
| `highlight_rolloff` | a quadratic solve; IEEE-754-2008 §5.4.1 requires `sqrt` to be correctly rounded, as it does the four arithmetic operations |
| `shadow_rolloff` | a cubic in Horner form: multiply, add, subtract, one divide |
| `apply_lut` | subtract, divide, multiply, truncate, one linear interpolation |
| `histogram` | the only float arithmetic is the bin assignment; everything after is integer counting, and integer addition commutes, so the order in which the device's atomics complete cannot change a total |
| `quantize_u8`, `quantize_u16` | exact integer hashing for the dither, and `floor(v + 0.5)` for the rounding: an exact operation composed with a correctly-rounded one, rather than a library rounding routine host and device could implement differently |

**Bounded, not identical.** Each bound is committed per kernel and
asserted against the CPU oracle by `tests/cuda_conformance.rs`. A driver
or toolkit update that regresses accuracy fails the suite rather than
being absorbed.

| Kernel | rtol | atol | Why |
|---|---|---|---|
| `encode_srgb`, `tone_curve` at `power != 1`, `hsl_bw`, `zone_system` | 1e-5 | 1e-7 | a small fixed number of `powf`/`expf`/`log2f` calls per pixel, the GPU side compiled with fast math: approximate `ex2`/`lg2`, approximate division, FMA contraction. `hsl_bw`'s neutral pixels skip the hue weights but not the contracted dot product, so they are not exact any more. Its hue and chroma divisions are correctly rounded (`__fdiv_rn`): approximated, they missed the bound by 2.1× under opposing hue weights and differed by the full range next to `FLT_MAX`, −1e30 or −inf (`hsl_bw_agrees_under_opposing_weights_and_on_extreme_pixels`) |
| `split_toning` | 1e-5 | 1e-7 | one cube root per pixel, from two different implementations: on the CPU Rust's own `cbrtf`, compiled into the extension on Linux x86-64 (unverified on the macOS and Windows wheels, which may call the platform libm's), and libdevice `cbrtf` on the GPU (compiled strict) |
| `blur` | 1e-5 | 1e-7 | the direct path substitutes Kahan-compensated f32 for the CPU's f64, sound there because every weight is positive and nothing is subtracted; the box path keeps f64 on both sides, a sliding window being a subtraction |
| `glow` | 1e-5 | 1e-7 | inherits `blur`'s bound: the threshold subtraction and the weighted add either side are correctly rounded, so the blur is the only inexact part |
| `sharpen` | 1e-5 | 1e-7 | the same shared blur, and the subtract/gate/combine pass after it is free of transcendentals. Open: where the output cancels towards zero beside a bright blur, half a one-ULP difference in the blur can exceed this atol. Measured once, on one element: 1.13× (output 5.7e-4 beside a blur of 3.5, the digest harness's HDR image at size 256, sigma 5.9) |
| `local_contrast` | 1e-4 | 1e-6 | compiled with FMA contraction, and the device reformulates the global f64 summed-area tables as separable box filters. It drops the *global prefix sum*, not the f64: the L and L² partial sums stay f64 because their difference is the variance and that subtraction cancels. The coefficient sums downstream are Kahan-compensated f32, which is enough because nothing there is squared or subtracted |
| `denoise` | 1e-4 | 1e-6 | compiled with FMA contraction; `local_contrast`'s class, inherited at `C = 1` because it is bit-for-bit the same device kernel (`local_contrast_device` at `strength = -amount`), and confirmed empirically at `C = 3`, where cross-guided adds two cancelling subtractions: `cov(I, p_c)`, and the intercept `b_c = mean(p_c) − a_c·mean(I)`, which next to a sample far above its neighbours cancels a huge product against an O(1) mean. `var(I)`, `cov(I, p_c)` and `b_c` are pinned to the CPU's two roundings (`__dmul_rn`, `__dsub_rn`): fused by the driver, `b_c` kept the product's rounding residue and missed the bound by up to 4e24× from about 1e12 |
| `film_grain`'s Box–Muller half | 1e-3 | 1e-5 | one `logf`, one `sqrtf` and one `cosf` per sample, the loosest transcendental chain in the crate; the splitmix64 hash underneath is exact and is asserted over a coordinate grid by `hash_grid` |

Fast math flushes subnormal inputs and results to zero, so in the
fast-math kernels a subnormal pixel reads as 0.0 on the GPU and a
subnormal result comes out as 0.0: a 1e-40 pixel through `tone_curve`
at power 0.3 gives 1e-12 on the CPU and 0.0 on the GPU, far below one
16-bit code. Non-finite pixels keep their class
(`fast_math_kernels_agree_on_special_values`): `zone_system` and
`encode_srgb` return NaN where the CPU does, though NaN payload and
sign bits may differ; `tone_curve` clamps NaN and −inf to 0 on both
backends; ±inf comes out where the CPU returns it. `hsl_bw` agrees on
them to the bit (§1).

`denoise`'s worst measured ratios, on the strict build before
`local_contrast` and `denoise` moved to FMA, were 0.0022x (`C = 1`,
unit range), 0.0012x (`C = 1`, HDR), 0.0106x (`C = 3`, unit range) and
0.0012x (`C = 3`, both HDR sweeps, up to the required 1e8 highlight);
the FMA build stays within the same bound in the conformance suite.
Next to one isolated sample far above an ordinary or a dark field, the
bound is asserted at radius 1, 2 and 4 up to 1e38
(`denoise_cross_guided_agrees_next_to_an_isolated_extreme_sample`).
At radius 8 even a strict build misses it between about 1e22 and 1e31.
That far beyond any photograph the specification is ill-conditioned
itself: evaluated exactly at radius 1, the CPU's result is 0.8× the
bound off at 1e13 and 3500× at 1e30.
Its
`radius` is capped at `MAX_RADIUS = 32` on both backends, because it
sums each window directly from its own pixels rather than through a
summed-area table (architecture.md §22), so cost is linear in `radius`
and no larger case is ever asserted.

**The backend fingerprint is part of the reproducibility key.** A
consumer that promises exact reproduction from a settings file must
either record the fingerprint alongside the parameters, or designate
`cpu` as the archival backend and use CUDA for interactive work.
`phaios` desktop should do the latter and record the fingerprint
regardless, so a preview render is identifiable as one. A GPU render
re-made after a driver update can differ in the last bits of the FMA
and fast-math kernels and of `split_toning` even under the same
fingerprint. At 8 bits that is expected to stay within about one code;
it has not been measured across drivers.

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
Eight processes given identical input produced eight different images:
invisible, and enough to break every checksum and reproducible render.

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
