// SPDX-License-Identifier: GPL-3.0-or-later
//! Bounded array allocation.
//!
//! Every kernel allocates its output — and often an intermediate — from
//! the *logical* shape of the caller's input. That is normally the same
//! thing as its physical size, but a numpy view need not be backed by
//! anything like the memory its shape implies: `np.broadcast_to` and
//! `np.lib.stride_tricks.as_strided` produce zero-stride views whose
//! logical extent is unbounded while their storage may be four bytes.
//!
//! Allocating that shape directly is fatal in a way §2's rules do not
//! tolerate. A failed `Vec` allocation calls `handle_alloc_error`, which
//! **aborts** rather than unwinding: it raises nothing at all — not even
//! `PanicException` — and takes the interpreter with it, so a consumer
//! cannot catch it with `except Exception`, `except BaseException`, or
//! anything else. That is strictly worse than the panic the rule names,
//! and for the same reason: a caller's bad input must not kill the
//! process around it.
//!
//! # Why a constant and not the allocator
//!
//! The obvious fix is fallible allocation, and it does not work. Measured
//! on Linux with the default heuristic overcommit (`overcommit_memory =
//! 0`) on a 60 GiB machine, `Vec::try_reserve_exact` **succeeded** for a
//! 111 GiB request; only `usize::MAX / 8` failed. The allocation is
//! granted, the kernel then writes to every element, and the OOM killer
//! ends the process — uncatchable again, just later. The allocator cannot
//! be relied on to refuse, so the limit has to be ours.
//!
//! # Why a constant and not the free memory
//!
//! Sizing the limit from available RAM would make the same call succeed
//! or fail depending on what else the machine happens to be doing, which
//! is exactly the ambient state §2 forbids: a kernel's result must depend
//! on its arguments and nothing else. A fixed constant keeps behaviour
//! reproducible across machines and across runs.
//!
//! # The trade-off, stated
//!
//! [`MAX_ALLOCATION_BYTES`] is a real ceiling: a legitimate output above
//! it is refused. It is set far above any photographic frame — see the
//! constant — and refusing is the better failure, but it is a limit and
//! it is documented rather than left to be discovered.
//!
//! # What this does *not* bound
//!
//! Only single allocations. A kernel holding several full-resolution
//! intermediates can still exceed this in total, and a caller looping
//! over many images can still exhaust memory. Bounding a whole pipeline's
//! peak footprint is a larger design question, deferred past v0.2 — see
//! `docs/ffi.md`.

use ndarray::{Array2, Array3};

use crate::error::PhaiosError;

/// Largest single array allocation a kernel will attempt, in bytes.
///
/// 8 GiB. A 24 MP three-channel `f32` frame — the crate's benchmark
/// size — is 285 MiB, so this is roughly twenty-eight times the largest
/// image the crate was built for, and allows about a 700 MP RGB frame.
/// No real photograph approaches it; a zero-stride broadcast reaches it
/// immediately.
pub const MAX_ALLOCATION_BYTES: usize = 8 << 30;

/// Element count of a shape, or an error if it overflows `usize`.
fn element_count(shape: (usize, usize, usize)) -> Result<usize, PhaiosError> {
    shape
        .0
        .checked_mul(shape.1)
        .and_then(|v| v.checked_mul(shape.2))
        .ok_or_else(|| {
            PhaiosError::Allocation(format!(
                "shape ({}, {}, {}) has more elements than usize can count",
                shape.0, shape.1, shape.2
            ))
        })
}

/// Check that `count` elements of `T` are within the budget, naming
/// `shape` in the error so the caller can see what was asked for.
fn check_budget<T>(count: usize, shape: impl std::fmt::Debug) -> Result<usize, PhaiosError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| PhaiosError::Allocation(format!("{shape:?} overflows a byte count")))?;
    if bytes > MAX_ALLOCATION_BYTES {
        return Err(PhaiosError::Allocation(format!(
            "{shape:?} needs {} MiB, above the {} MiB limit for a single array \
             (a zero-stride view such as numpy's broadcast_to reaches this from \
             very little actual data)",
            bytes >> 20,
            MAX_ALLOCATION_BYTES >> 20
        )));
    }
    Ok(bytes)
}

/// Check that a `(H, W, C)` shape of `T` is within the budget, without
/// allocating.
///
/// For call sites that hand the allocation to someone else —
/// `ArrayView::as_standard_layout` on the CUDA upload path, which
/// materialises a host copy of the logical shape and would abort on the
/// same input the kernels now refuse; and [`crate::blur::blur`], which must
/// bound the caller's shape *before* its first pass reads it.
///
/// # Errors
/// [`PhaiosError::Allocation`] on the same terms as [`zeros3`].
pub(crate) fn check_shape<T>(shape: (usize, usize, usize)) -> Result<(), PhaiosError> {
    let n = element_count(shape)?;
    check_budget::<T>(n, shape)?;
    Ok(())
}

/// Allocate a zero-filled `(H, W, C)` array, or fail with
/// [`PhaiosError::Allocation`] rather than aborting the process.
///
/// The replacement for `Array3::zeros` everywhere a shape derives from
/// caller input. `try_reserve` is used underneath as a second line, for
/// the requests an allocator will genuinely refuse.
///
/// # Errors
/// [`PhaiosError::Allocation`] if the shape overflows a count, or needs
/// more than [`MAX_ALLOCATION_BYTES`].
pub(crate) fn zeros3<T: Clone + Default>(
    shape: (usize, usize, usize),
) -> Result<Array3<T>, PhaiosError> {
    let n = element_count(shape)?;
    check_budget::<T>(n, shape)?;

    let mut data: Vec<T> = Vec::new();
    data.try_reserve_exact(n).map_err(|e| {
        PhaiosError::Allocation(format!(
            "the allocator refused {:?} ({} MiB): {e}",
            shape,
            (n * std::mem::size_of::<T>()) >> 20
        ))
    })?;
    data.resize(n, T::default());

    Array3::from_shape_vec(shape, data).map_err(|e| {
        PhaiosError::Allocation(format!("cannot build an array of shape {shape:?}: {e}"))
    })
}

/// Allocate a zero-filled `(rows, cols)` array, bounded as [`zeros3`].
///
/// # Errors
/// [`PhaiosError::Allocation`] if the shape overflows a count, or needs
/// more than [`MAX_ALLOCATION_BYTES`].
pub(crate) fn zeros2<T: Clone + Default>(shape: (usize, usize)) -> Result<Array2<T>, PhaiosError> {
    let n = shape.0.checked_mul(shape.1).ok_or_else(|| {
        PhaiosError::Allocation(format!(
            "shape ({}, {}) has more elements than usize can count",
            shape.0, shape.1
        ))
    })?;
    check_budget::<T>(n, shape)?;

    let mut data: Vec<T> = Vec::new();
    data.try_reserve_exact(n)
        .map_err(|e| PhaiosError::Allocation(format!("the allocator refused {shape:?}: {e}")))?;
    data.resize(n, T::default());

    Array2::from_shape_vec(shape, data).map_err(|e| {
        PhaiosError::Allocation(format!("cannot build an array of shape {shape:?}: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_shapes_are_allocated() {
        let a = zeros3::<f32>((4, 5, 3)).unwrap();
        assert_eq!(a.dim(), (4, 5, 3));
        assert!(a.iter().all(|v| *v == 0.0));
        assert_eq!(zeros3::<f32>((0, 5, 3)).unwrap().dim(), (0, 5, 3));
    }

    #[test]
    fn a_24_megapixel_frame_is_comfortably_inside_the_budget() {
        // The crate's benchmark size, three channels: this must never be
        // the thing the limit catches.
        let n = 4323 * 5764 * 3;
        assert!(n * std::mem::size_of::<f32>() < MAX_ALLOCATION_BYTES / 20);
    }

    #[test]
    fn an_oversized_shape_is_refused_rather_than_attempted() {
        // The zero-stride broadcast case: 100000 x 100000 x 3 f32.
        let err = zeros3::<f32>((100_000, 100_000, 3)).unwrap_err();
        assert!(matches!(err, PhaiosError::Allocation(_)));
        assert!(err.to_string().contains("above the"), "{err}");
    }

    #[test]
    fn an_overflowing_shape_is_refused() {
        let err = zeros3::<f32>((usize::MAX, 2, 2)).unwrap_err();
        assert!(matches!(err, PhaiosError::Allocation(_)));
    }

    #[test]
    fn the_budget_accounts_for_element_size() {
        // The same element count is inside the budget as u16 and outside
        // it as f64: the limit is bytes, not elements.
        let n = (3usize << 30) / 2;
        assert!(check_shape::<u16>((n, 1, 1)).is_ok());
        assert!(matches!(
            check_shape::<f64>((n, 1, 1)).unwrap_err(),
            PhaiosError::Allocation(_)
        ));
    }

    #[test]
    fn check_shape_agrees_with_the_allocating_form() {
        // The CUDA upload path uses `check_shape` where the kernels use
        // `zeros3`; they must refuse the same shapes.
        for shape in [(4, 5, 3), (100_000, 100_000, 3), (usize::MAX, 2, 2)] {
            assert_eq!(
                check_shape::<f32>(shape).is_err(),
                zeros3::<f32>(shape).is_err(),
                "disagreement on {shape:?}"
            );
        }
    }
}
