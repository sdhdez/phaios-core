// SPDX-License-Identifier: GPL-3.0-or-later
//! GPU kernels, one module per CPU kernel they mirror.

mod exposure;
mod local_contrast;

pub use exposure::exposure;
pub use local_contrast::local_contrast;
