from __future__ import annotations

import numpy as np

from scripts.invention_c3 import icc_2_1


def test_icc_perfect_agreement() -> None:
    matrix = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    assert abs(icc_2_1(matrix) - 1.0) < 1e-9


def test_icc_disagreement_is_lower() -> None:
    matrix = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 1.0], [3.0, 7.0, 5.0]])
    value = icc_2_1(matrix)
    assert np.isfinite(value)
    assert value < 1.0
