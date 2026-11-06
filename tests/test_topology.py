from __future__ import annotations

import numpy as np

from topobpe.topology import (
    betti_curve,
    compare_diagrams,
    persistence_summaries,
    rips_diagrams,
    validate_distance_matrix,
)


def _points(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(24, 5))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def test_distance_validation_is_symmetric_and_nonnegative() -> None:
    matrix = np.array([[0.0, -1.0], [2.0, 0.0]])
    result = validate_distance_matrix(matrix)
    assert np.allclose(result, result.T)
    assert np.all(result >= 0.0)
    assert np.allclose(np.diag(result), 0.0)


def test_ripser_and_gudhi_agree_on_small_cloud() -> None:
    points = _points()
    distance = validate_distance_matrix(1.0 - points @ points.T)
    report = compare_diagrams(distance, maxdim=1, tolerance=1e-6)
    assert report["dimensions"]["0"]["agree"]
    assert report["dimensions"]["1"]["agree"]


def test_persistence_summaries_are_finite_and_normalized() -> None:
    distance = validate_distance_matrix(np.abs(np.subtract.outer(range(10), range(10))))
    diagrams = rips_diagrams(distance, maxdim=1)
    summaries = persistence_summaries(diagrams[1], [0.1, 0.25], "h1")
    assert summaries["h1_num_bars"] >= 0
    assert summaries["h1_total_persistence"] >= 0
    assert summaries["h1_persistence_entropy"] >= 0
    assert all(np.isfinite(value) for value in summaries.values())
    curve = betti_curve(diagrams[1], np.linspace(0.0, 1.0, 20))
    assert curve.shape == (20,)
    assert np.all(curve >= 0)
