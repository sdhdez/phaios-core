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
| `exposure` | any | same | a scalar multiply; valid before or after the B&W stage |
| `luminance_bw` | 3 | 1 | |
| `channel_mixer_bw` | 3 | 1 | |
| `color_filter_bw` | 3 | 1 | |
| `hsl_bw` | 3 | 1 | needs hue, so it must run before the collapse |
| `zone_system` | 1 | 1 | |
| `local_contrast` | 1 | 1 | |
| `film_grain` | 1 | 1 | |
| `split_toning` | 1 | **3** | the only kernel that adds channels |
| `vignette` | any | same | one factor per pixel, applied to every channel |
| `tone_curve` | any | same | element-wise |
| `encode_srgb` | any | same | element-wise |

A kernel given the wrong channel count raises `ValueError`. The four
that accept "any" do so deliberately: they run either side of
`split_toning`, and a pipeline should not have to branch on whether
toning is enabled.

**Size.** H and W are unconstrained. Zero-size arrays are accepted and
return an empty array of the same shape; treating "no pixels" as an
error would push a special case onto every caller.

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
- Test the boundary, not just the happy path: `tests/ffi.py` calls every
  kernel with C-contiguous, row-strided, column-strided, Fortran-order
  and reversed inputs.

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

Two calls with the same input, the same parameters and the same seed
must produce **bit-identical** output — in the same process, in another
process, and on another machine with the same target.

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
