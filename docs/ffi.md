# phaios-core FFI Contract

This document defines the binding contract between the Rust kernels and
the Python `phaios_core` extension module. Read it before adding or
modifying any `#[pyfunction]` or `#[pyclass]` item.

---

## 1. Array layout

All arrays at the FFI boundary obey a single convention:

| Property | Value |
|----------|-------|
| dtype | `numpy.float32` |
| memory order | C-contiguous (row-major) |
| shape | `(H, W, C)` where C ∈ {1, 3} |
| value range | caller's responsibility; kernels do not clamp |
| linearity | **scene-referred linear** except after `encode_srgb` |

C = 1 for single-channel (luminance) arrays; C = 3 for RGB. No other
channel counts are accepted — kernels raise `ValueError` on mismatch.

The height H and width W are unconstrained positive integers. Zero-size
arrays are rejected.

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
- The input array must be C-contiguous. Pass `.ascontiguousarray()` from
  Python if unsure.

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
- Expose all fields with `#[pyo3(get, set)]`.
- Provide a `#[new]` constructor with positional arguments matching
  the field order.
- Be registered on the module: `m.add_class::<GuidedFilterParams>()?;`

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

The `From` impl lives in `src/error.rs`. Variant-to-exception mapping:

| PhaiosError variant | Python exception |
|--------------------|-----------------|
| `Shape(_)` | `ValueError` |

Add new variants as needed; always map to the most specific Python
exception class.

Internal Rust panics (e.g. index-out-of-bounds on a checked invariant)
are acceptable within the Rust side but must be documented on the
function. PyO3 0.17+ converts Rust panics into `PanicException` at the
FFI boundary rather than aborting the process, but do not rely on this:
panics in kernel code indicate bugs, not user errors.

---

## 5. GIL release

Any kernel that is O(N) or worse on image pixels must release the GIL:

```rust
#[pyfunction]
pub fn zone_system(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: ZoneParams,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    py.allow_threads(|| {
        // computationally expensive work here; no Python API calls
        let out = compute(view, &params)?;
        Ok(out)
    })
    .map(|arr| arr.into_pyarray(py).unbind())
}
```

Kernels that do NOT release the GIL must have a doc comment explaining
why (e.g., very small fixed-size output, trivial per-pixel map).

`rayon::par_iter` within `allow_threads` is safe and recommended for
tile-based kernels. Do not spawn rayon work outside `allow_threads`.

---

## 6. RNG determinism (reserved for v0.2)

v0.1 contains no kernels with randomness. When grain is added in v0.2,
the pattern is:

```rust
/// `seed`: explicit RNG seed for deterministic output. Two calls with
/// the same `seed`, `params`, and input produce identical output.
pub fn film_grain(img: ..., params: GrainParams, seed: u64) -> ...
```

No global RNG. No thread-local RNG. Every random kernel takes `seed: u64`
as an explicit parameter. This rule is load-bearing for reproducibility.

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
