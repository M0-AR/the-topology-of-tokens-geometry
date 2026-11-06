from __future__ import annotations

from typing import Any

import numpy as np
import ripser
from persim import bottleneck


def validate_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def rips_diagrams(distance_matrix: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    matrix = validate_distance_matrix(distance_matrix)
    if matrix.shape[0] < 2:
        return [np.zeros((0, 2), dtype=np.float64) for _ in range(maxdim + 1)]
    result = ripser.ripser(matrix, distance_matrix=True, maxdim=maxdim)
    return [np.asarray(diagram, dtype=np.float64) for diagram in result["dgms"]]


def gudhi_diagrams(
    distance_matrix: np.ndarray,
    maxdim: int = 1,
    max_edge_length: float | None = None,
) -> list[np.ndarray]:
    import gudhi

    matrix = validate_distance_matrix(distance_matrix)
    if matrix.shape[0] < 2:
        return [np.zeros((0, 2), dtype=np.float64) for _ in range(maxdim + 1)]
    if max_edge_length is None:
        max_edge_length = float(matrix.max()) * 1.01 + 1e-9
    complex_ = gudhi.RipsComplex(distance_matrix=matrix, max_edge_length=max_edge_length)
    simplex_tree = complex_.create_simplex_tree(max_dimension=maxdim + 1)
    pairs = simplex_tree.persistence()
    diagrams: list[list[tuple[float, float]]] = [[] for _ in range(maxdim + 1)]
    for dimension, (birth, death) in pairs:
        if dimension <= maxdim:
            diagrams[dimension].append((float(birth), float(death)))
    return [np.asarray(sorted(diagram), dtype=np.float64) for diagram in diagrams]


def compare_diagrams(
    distance_matrix: np.ndarray,
    maxdim: int = 1,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    rips = rips_diagrams(distance_matrix, maxdim=maxdim)
    gudhi = gudhi_diagrams(distance_matrix, maxdim=maxdim)
    report: dict[str, Any] = {"tolerance": tolerance, "dimensions": {}}
    for dimension in range(maxdim + 1):
        left = rips[dimension]
        right = gudhi[dimension]
        left_finite = left[np.isfinite(left[:, 1])] if left.size else left
        right_finite = right[np.isfinite(right[:, 1])] if right.size else right
        left_infinite = int(np.sum(~np.isfinite(left[:, 1]))) if left.size else 0
        right_infinite = int(np.sum(~np.isfinite(right[:, 1]))) if right.size else 0
        left_positive = left_finite[(left_finite[:, 1] - left_finite[:, 0]) > tolerance]
        right_positive = right_finite[(right_finite[:, 1] - right_finite[:, 0]) > tolerance]
        left_zero = int(len(left_finite) - len(left_positive))
        right_zero = int(len(right_finite) - len(right_positive))
        distance = float(bottleneck(left_positive, right_positive))
        report["dimensions"][str(dimension)] = {
            "ripser_finite": int(len(left_finite)),
            "gudhi_finite": int(len(right_finite)),
            "ripser_zero_persistence": left_zero,
            "gudhi_zero_persistence": right_zero,
            "ripser_infinite": left_infinite,
            "gudhi_infinite": right_infinite,
            "bottleneck_distance": distance,
            "agree": bool(left_infinite == right_infinite and distance <= tolerance),
        }
    return report


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))
    return float(np.trapz(values, grid))


def betti_curve(
    diagram: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    counts = np.zeros_like(grid, dtype=np.float64)
    for index, threshold in enumerate(grid):
        finite = np.sum((births <= threshold) & (deaths > threshold) & np.isfinite(deaths))
        infinite = np.sum((~np.isfinite(deaths)) & (births <= threshold))
        counts[index] = finite + infinite
    return counts


def _threshold_name(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def persistence_summaries(
    diagram: np.ndarray,
    thresholds: list[float],
    prefix: str,
) -> dict[str, float]:
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.size == 0:
        diagram = np.zeros((0, 2), dtype=np.float64)
    finite_mask = np.isfinite(diagram[:, 1])
    finite = diagram[finite_mask]
    lifetimes = np.clip(finite[:, 1] - finite[:, 0], 0.0, None) if len(finite) else np.zeros(0)
    total = float(lifetimes.sum()) if len(lifetimes) else 0.0
    maximum = float(lifetimes.max()) if len(lifetimes) else 0.0
    mean = float(lifetimes.mean()) if len(lifetimes) else 0.0
    std = float(lifetimes.std()) if len(lifetimes) else 0.0
    if total > 0:
        probs = lifetimes[lifetimes > 0] / total
        entropy = float(-(probs * np.log(probs)).sum())
    else:
        entropy = 0.0
    finite_deaths = finite[:, 1] if len(finite) else np.zeros(0)
    max_death = float(finite_deaths.max()) if len(finite_deaths) else 0.0
    grid = np.unique(
        np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.asarray(thresholds, dtype=np.float64),
                np.asarray([max_death], dtype=np.float64),
            ]
        )
    )
    betti = betti_curve(diagram, grid)
    area = _trapz(betti, grid)
    summaries: dict[str, float] = {
        f"{prefix}_num_bars": float(len(diagram)),
        f"{prefix}_num_finite_bars": float(len(finite)),
        f"{prefix}_num_infinite_bars": float(np.sum(~finite_mask)),
        f"{prefix}_total_persistence": total,
        f"{prefix}_max_persistence": maximum,
        f"{prefix}_mean_persistence": mean,
        f"{prefix}_std_persistence": std,
        f"{prefix}_persistence_entropy": entropy,
        f"{prefix}_betti_area": area,
    }
    for threshold in thresholds:
        summaries[f"{prefix}_count_{_threshold_name(threshold)}"] = float(
            np.sum(lifetimes > threshold)
        )
        summaries[f"{prefix}_betti_at_{_threshold_name(threshold)}"] = float(
            betti_curve(diagram, np.asarray([threshold], dtype=np.float64))[0]
        )
    return summaries


def mean_nearest_neighbor_distance(distance_matrix: np.ndarray) -> float:
    matrix = validate_distance_matrix(distance_matrix)
    if matrix.shape[0] < 2:
        return 0.0
    values = matrix.copy()
    np.fill_diagonal(values, np.inf)
    nearest = values.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(finite.mean()) if len(finite) else 0.0


def persistence_diagram_payload(diagram: np.ndarray) -> list[list[float | None]]:
    payload: list[list[float | None]] = []
    for birth, death in np.asarray(diagram, dtype=np.float64):
        payload.append(
            [
                float(birth),
                None if not np.isfinite(death) else float(death),
            ]
        )
    return payload
