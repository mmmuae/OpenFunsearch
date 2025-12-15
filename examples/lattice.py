"""Lattice reduction benchmark for FunSearch.

This module avoids external dependencies so it can run inside FunSearch
sandboxes without extra installation steps. It provides a deterministic
evaluation function and a solid baseline reduction routine so the FunSearch
agent starts from a meaningful solution.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence

import funsearch

# CONFIGURATION
DIMENSION = 10
NUM_TEST_CASES = 5
LOWER_BOUND = -100
UPPER_BOUND = 100
DELTA = 0.75  # Lovász condition parameter for LLL

Vector = List[float]
Matrix = List[Vector]


def _generate_random_basis(seed: int) -> Matrix:
    """Generate a random integer lattice basis with deterministic seeding."""
    rng = random.Random(seed)
    return [
        [float(rng.randint(LOWER_BOUND, UPPER_BOUND)) for _ in range(DIMENSION)]
        for _ in range(DIMENSION)
    ]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm_sq(v: Sequence[float]) -> float:
    return _dot(v, v)


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(row) for row in matrix]


def _compute_gram_schmidt(basis: Matrix) -> tuple[Matrix, Matrix, List[float]]:
    """Return Gram–Schmidt orthogonalization and coefficients.

    Returns (B, mu, norm_sq) where B is the orthogonalized basis, mu contains the
    projection coefficients, and norm_sq stores squared norms of the rows in B.
    """
    n = len(basis)
    ortho: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    mu: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    norm_sq: List[float] = [0.0 for _ in range(n)]

    for i in range(n):
        ortho[i] = list(basis[i])
        for j in range(i):
            if norm_sq[j] <= 0.0:
                continue
            mu[i][j] = _dot(basis[i], ortho[j]) / norm_sq[j]
            for k in range(n):
                ortho[i][k] -= mu[i][j] * ortho[j][k]
        norm_sq[i] = _norm_sq(ortho[i])
    return ortho, mu, norm_sq


def _swap_rows(matrix: Matrix, i: int, j: int) -> None:
    matrix[i], matrix[j] = matrix[j], matrix[i]


def _lll_reduce(basis: Matrix, delta: float = DELTA) -> Matrix:
    """Perform a lightweight LLL reduction for a square basis matrix."""
    reduced = _copy_matrix(basis)
    n = len(reduced)
    ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

    def refresh(_: int) -> None:
        nonlocal ortho, mu, norm_sq
        ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

    k = 1
    while k < n:
        # Size reduction step
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > 0.5:
                factor = round(mu[k][j])
                for t in range(n):
                    reduced[k][t] -= factor * reduced[j][t]
                refresh(k)

        # Lovász condition
        if norm_sq[k] < (delta - mu[k][k - 1] ** 2) * norm_sq[k - 1]:
            _swap_rows(reduced, k, k - 1)
            refresh(k)
            k = max(k - 1, 1)
        else:
            k += 1

    return reduced


def _determinant(matrix: Matrix) -> float:
    """Compute determinant via Gaussian elimination with pivoting."""
    n = len(matrix)
    working = _copy_matrix(matrix)
    det = 1.0
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(working[r][i]))
        pivot = working[pivot_row][i]
        if abs(pivot) < 1e-12:
            return 0.0
        if pivot_row != i:
            _swap_rows(working, i, pivot_row)
            det *= -1.0
        det *= pivot
        for r in range(i + 1, n):
            factor = working[r][i] / pivot
            for c in range(i, n):
                working[r][c] -= factor * working[i][c]
    return det


def _matrix_from_candidate(candidate: Iterable[Iterable[float]]) -> Matrix:
    matrix = [list(row) for row in candidate]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Basis must be a non-empty square matrix")
    return matrix


@funsearch.run
def evaluate(program) -> float:
    """Judge function for FunSearch lattice experiments."""
    scores: list[float] = []

    for seed in range(NUM_TEST_CASES):
        original_basis = _generate_random_basis(seed=seed)

        try:
            candidate = program.reduce_basis(_copy_matrix(original_basis))
        except Exception:
            return -1000.0  # Program crashed

        try:
            matrix = _matrix_from_candidate(candidate)
        except Exception:
            return -1000.0  # Invalid output structure

        if len(matrix) != DIMENSION or any(len(row) != DIMENSION for row in matrix):
            return -1000.0  # Dimension mismatch

        det_orig = abs(_determinant(original_basis))
        det_new = abs(_determinant(matrix))
        if det_new == 0.0 or abs(det_orig - det_new) > max(1e-4, 1e-4 * det_orig):
            return -1000.0

        norms = [math.sqrt(_norm_sq(row)) for row in matrix]
        shortest = max(min(norms), 1e-6)
        scores.append(-math.log(shortest))

    return float(sum(scores) / len(scores))


@funsearch.evolve
def reduce_basis(basis: Iterable[Iterable[float]]) -> Matrix:
    """Provide an LLL-inspired baseline as a strong starting point."""
    matrix = _matrix_from_candidate(basis)
    if len(matrix) != DIMENSION:
        raise ValueError("Basis must match configured dimension")

    reduced = _lll_reduce(matrix)

    norms = [_norm_sq(row) for row in reduced]
    order = sorted(range(len(reduced)), key=lambda i: norms[i])
    return [reduced[i] for i in order]
