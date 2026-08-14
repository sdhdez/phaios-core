// SPDX-License-Identifier: GPL-3.0-or-later
//! Exposure compensation kernel.
//!
//! Multiplies every pixel value by `2^stops`. This is the first stage
//! of the pipeline after the consumer delivers scene-referred linear
//! f32 RGB data.
//!
//! Not yet implemented — planned for v0.2 (see `CHANGELOG.md`). The
//! module exists so the pipeline documented in `docs/architecture.md`
//! §1 has a home for it; it exports nothing today, and consumers apply
//! their own exposure in the meantime.
