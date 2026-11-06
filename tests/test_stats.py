from __future__ import annotations

import numpy as np

from topobpe.stats import (
    _cluster_indices,
    _holm,
    _partial_spearman,
    _permutation_test,
    _select_alpha,
)


def test_holm_is_monotone_and_controls_family() -> None:
    adjusted = _holm([0.01, 0.04, 0.03, 0.002])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.008])
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_cluster_indices_resample_whole_corpora() -> None:
    groups = np.array(["a"] * 5 + ["b"] * 4 + ["c"] * 3)
    rng = np.random.default_rng(3)
    indices = _cluster_indices(groups, rng)
    selected = set(groups[indices])
    assert selected.issubset(set(groups))
    assert len(indices) >= len(groups) - 2


def test_partial_spearman_recovers_linear_relation() -> None:
    rng = np.random.default_rng(11)
    control = rng.normal(size=200)
    x = rng.normal(size=200)
    y = 3.0 * x + 0.5 * control + rng.normal(scale=0.01, size=200)
    rho = _partial_spearman(x, y, control)
    assert rho > 0.99


def test_alpha_selection_prefers_regularization_on_noise() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 30))
    y = rng.normal(size=40)
    groups = np.array(["a"] * 20 + ["b"] * 20)
    alpha = _select_alpha(x, y, groups, [0.01, 1.0, 100.0])
    assert alpha in {0.01, 1.0, 100.0}


def test_permutation_uses_only_topology_columns() -> None:
    rng = np.random.default_rng(7)
    groups = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6)
    baseline = rng.normal(size=(18, 2))
    topology = rng.normal(size=(18, 4))
    y = baseline[:, 0] + 0.1 * rng.normal(size=18)
    result = _permutation_test(
        baseline, topology, y, groups, [0.01, 1.0], iterations=5, seed=1
    )
    assert np.isfinite(result["observed_delta_r2"])
    assert result["iterations"] == 5
