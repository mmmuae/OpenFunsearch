"""Lattice reduction benchmark for FunSearch.

This module provides a deterministic evaluation function and a solid baseline
reduction routine so the FunSearch agent starts from a meaningful solution.
"""

from __future__ import annotations

import numpy as np
import funsearch

# CONFIGURATION
DIMENSION = 10
NUM_TEST_CASES = 5
LOWER_BOUND = -100
UPPER_BOUND = 100
DELTA = 0.75  # Lovász condition parameter for LLL


def _generate_random_basis(seed: int) -> np.ndarray:
    """Generate a random integer lattice basis with deterministic seeding."""
    rng = np.random.default_rng(seed)
    return rng.integers(LOWER_BOUND, UPPER_BOUND, size=(DIMENSION, DIMENSION)).astype(float)


def _compute_gram_schmidt(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Gram–Schmidt orthogonalization and coefficients.

    Returns (B, mu, norm_sq) where B is the orthogonalized basis, mu contains the
    projection coefficients, and norm_sq stores squared norms of the rows in B.
    """
    n = basis.shape[0]
    ortho = np.zeros_like(basis)
    mu = np.zeros((n, n), dtype=float)
    norm_sq = np.zeros(n, dtype=float)

    for i in range(n):
        ortho[i] = basis[i]
        for j in range(i):
            if norm_sq[j] <= 0.0:
                continue
            mu[i, j] = np.dot(basis[i], ortho[j]) / norm_sq[j]
            ortho[i] -= mu[i, j] * ortho[j]
        norm_sq[i] = np.dot(ortho[i], ortho[i])
    return ortho, mu, norm_sq


def _lll_reduce(basis: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """Perform a lightweight LLL reduction for a square basis matrix."""
    reduced = np.array(basis, dtype=float, copy=True)
    n = reduced.shape[0]
    ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

    def refresh(k: int) -> None:
        nonlocal ortho, mu, norm_sq
        ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

    k = 1
    while k < n:
        # Size reduction step
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                reduced[k] -= np.round(mu[k, j]) * reduced[j]
                refresh(k)

        # Lovász condition
        if norm_sq[k] < (delta - mu[k, k - 1] ** 2) * norm_sq[k - 1]:
            reduced[[k, k - 1]] = reduced[[k - 1, k]]
            refresh(k)
            k = max(k - 1, 1)
        else:
            k += 1

    return reduced


@funsearch.run
def evaluate(program) -> float:
    """Judge function for FunSearch lattice experiments."""
    scores: list[float] = []

    for seed in range(NUM_TEST_CASES):
        original_basis = _generate_random_basis(seed=seed)

        try:
            candidate = program.reduce_basis(original_basis.copy())
        except Exception:
            return -1000.0  # Program crashed

        if not isinstance(candidate, np.ndarray):
            return -1000.0  # Invalid output type

        if candidate.shape != original_basis.shape:
            return -1000.0  # Dimension mismatch

        # Volume preservation check
        det_orig = np.abs(np.linalg.det(original_basis))
        det_new = np.abs(np.linalg.det(candidate))
        if not np.isfinite(det_new) or not np.isclose(det_orig, det_new, rtol=1e-4):
            return -1000.0

        # Score shortest vector (higher is better)
        norms = np.linalg.norm(candidate, axis=1)
        shortest = np.min(norms)
        shortest = max(shortest, 1e-6)  # Prevent log(0)
        scores.append(-np.log(shortest))

    return float(np.mean(scores))


@funsearch.evolve
def reduce_basis(basis: np.ndarray) -> np.ndarray:
    """Provide an LLL-inspired baseline as a strong starting point."""
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("Basis must be a square 2D array")

    # Ensure float computations and copy to avoid side effects.
    working_basis = np.array(basis, dtype=float, copy=True)
    reduced = _lll_reduce(working_basis)

    # Sort by vector norm to present shorter vectors first.
    norms = np.linalg.norm(reduced, axis=1)
    order = np.argsort(norms)
    return reduced[order]
