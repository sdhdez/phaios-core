# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-side FFI smoke tests for phaios_core.

Verifies that each PyO3 binding is callable from Python, returns a
float32 C-contiguous array, and raises a clear exception on bad input.

Placeholder — full tests implemented in Step 3 after the kernels are
exposed via PyO3 in src/lib.rs.

Run with: pytest tests/ffi.py
"""

import phaios_core


def test_import():
    """The module must be importable after `maturin develop`."""
    assert phaios_core is not None
