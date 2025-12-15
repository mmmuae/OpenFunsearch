"""
Lattice reduction benchmark for FunSearch.

Goal: evolve reduce_basis() to make the shortest basis vector as small as possible
while preserving the lattice volume (determinant magnitude).

No external deps (standard library only). Determinant checks are EXACT (integer).
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

import funsearch

NUM_TEST_CASES = 5
LOWER_BOUND = -100
UPPER_BOUND = 100

# LLL delta in (0.25, 1). Higher = stronger reduction, more work.
DELTA = 0.99

Vector = List[float]
Matrix = List[Vector]


def _is_finite(x: float) -> bool:
  return isinstance(x, (int, float)) and math.isfinite(float(x))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
  return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm_sq(v: Sequence[float]) -> float:
  return _dot(v, v)


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
  return [list(map(float, row)) for row in matrix]


def _matrix_from_candidate(candidate: Iterable[Iterable[float]], n: int) -> Matrix:
  m = [list(row) for row in candidate]
  if not m or len(m) != n or any(len(row) != n for row in m):
    raise ValueError("Candidate must be a non-empty n x n matrix")
  out: Matrix = []
  for row in m:
    r: Vector = []
    for x in row:
      fx = float(x)
      if not _is_finite(fx):
        raise ValueError("Non-finite entry in candidate matrix")
      r.append(fx)
    out.append(r)
  return out


def _as_int_matrix(matrix: Matrix) -> List[List[int]]:
  # Convert floats to ints if they are extremely close to integers.
  out: List[List[int]] = []
  for row in matrix:
    r: List[int] = []
    for x in row:
      rx = int(round(float(x)))
      if abs(float(x) - float(rx)) > 1e-8:
        raise ValueError("Matrix entry is not integer-like (breaks exact det check)")
      r.append(rx)
    out.append(r)
  return out


def _det_bareiss_int(a: List[List[int]]) -> int:
  # Bareiss algorithm (fraction-free Gaussian elimination), exact for integer matrices.
  n = len(a)
  if n == 0 or any(len(row) != n for row in a):
    raise ValueError("det: matrix must be non-empty and square")

  m = [row[:] for row in a]
  det_sign = 1
  prev = 1

  for k in range(n - 1):
    # Find non-zero pivot
    pivot_row = k
    while pivot_row < n and m[pivot_row][k] == 0:
      pivot_row += 1
    if pivot_row == n:
      return 0
    if pivot_row != k:
      m[k], m[pivot_row] = m[pivot_row], m[k]
      det_sign *= -1

    pivot = m[k][k]
    if pivot == 0:
      return 0

    for i in range(k + 1, n):
      for j in range(k + 1, n):
        num = m[i][j] * pivot - m[i][k] * m[k][j]
        if prev != 1:
          # Bareiss guarantees exact divisibility here for integer matrices.
          q, r = divmod(num, prev)
          if r != 0:
            raise ValueError("det: non-exact division (unexpected)")
          m[i][j] = q
        else:
          m[i][j] = num
      m[i][k] = 0

    prev = pivot

  return det_sign * m[n - 1][n - 1]


def _generate_non_singular_basis(seed: int, n: int) -> Matrix:
  # Deterministically retry a few times to avoid singular matrices.
  # This keeps evaluation stable and avoids -1000 due to det_orig == 0.
  for t in range(64):
    rng = random.Random(seed + 1000003 * t)
    basis: Matrix = [
      [float(rng.randint(LOWER_BOUND, UPPER_BOUND)) for _ in range(n)]
      for _ in range(n)
    ]
    try:
      det = _det_bareiss_int(_as_int_matrix(basis))
    except Exception:
      det = 0
    if det != 0:
      return basis
  # Worst-case fallback: identity basis (non-singular).
  return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _compute_gram_schmidt(basis: Matrix) -> Tuple[Matrix, Matrix, List[float]]:
  # Returns (ortho, mu, norm_sq) for Gram–Schmidt on ROW vectors.
  n = len(basis)
  ortho: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
  mu: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
  norm_sq: List[float] = [0.0 for _ in range(n)]

  for i in range(n):
    ortho[i] = list(map(float, basis[i]))
    for j in range(i):
      if norm_sq[j] <= 1e-18:
        continue
      mu[i][j] = _dot(basis[i], ortho[j]) / norm_sq[j]
      for k in range(n):
        ortho[i][k] -= mu[i][j] * ortho[j][k]
    norm_sq[i] = _norm_sq(ortho[i])

  return ortho, mu, norm_sq


def _swap_rows(m: Matrix, i: int, j: int) -> None:
  m[i], m[j] = m[j], m[i]


def _lll_reduce(basis: Matrix, delta: float = DELTA) -> Matrix:
  # LLL-style reduction. Row operations are integer (via rounded mu) -> preserves det exactly.
  reduced = _copy_matrix(basis)
  n = len(reduced)

  ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

  def refresh() -> None:
    nonlocal ortho, mu, norm_sq
    ortho, mu, norm_sq = _compute_gram_schmidt(reduced)

  k = 1
  while k < n:
    # Size reduction
    for j in range(k - 1, -1, -1):
      if norm_sq[j] <= 1e-18:
        continue
      if abs(mu[k][j]) > 0.5:
        r = int(round(mu[k][j]))
        if r != 0:
          for t in range(n):
            reduced[k][t] -= float(r) * reduced[j][t]
          refresh()

    # Lovász condition
    if norm_sq[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * norm_sq[k - 1]:
      _swap_rows(reduced, k, k - 1)
      refresh()
      k = max(k - 1, 1)
    else:
      k += 1

  return reduced


@funsearch.run
def evaluate(program, n: int = 8) -> float:
  # Penalize invalid outputs with -1000, otherwise score by -log(shortest_norm).
  try:
    n_int = int(n)
  except Exception:
    return -1000.0
  if n_int < 2 or n_int > 48:
    return -1000.0

  scores: List[float] = []

  for seed in range(NUM_TEST_CASES):
    original = _generate_non_singular_basis(seed=seed, n=n_int)

    try:
      cand = program.reduce_basis(_copy_matrix(original))
    except Exception:
      return -1000.0

    try:
      reduced = _matrix_from_candidate(cand, n=n_int)
      # Ensure “integer-like” so determinant comparisons are meaningful/exact.
      original_i = _as_int_matrix(original)
      reduced_i = _as_int_matrix(reduced)
      det_orig = _det_bareiss_int(original_i)
      det_new = _det_bareiss_int(reduced_i)
    except Exception:
      return -1000.0

    if det_orig == 0 or det_new == 0:
      return -1000.0
    if abs(det_orig) != abs(det_new):
      return -1000.0

    norms = [math.sqrt(_norm_sq(row)) for row in reduced]
    shortest = max(min(norms), 1e-6)
    scores.append(-math.log(shortest))

  return float(sum(scores) / len(scores))


@funsearch.evolve
def reduce_basis(basis: Iterable[Iterable[float]]) -> Matrix:
  # Strong baseline: LLL reduce + final sort by norm.
  m = [list(row) for row in basis]
  n = len(m)
  if n == 0 or any(len(row) != n for row in m):
    raise ValueError("Basis must be a non-empty square matrix")

  mat = _matrix_from_candidate(m, n=n)
  reduced = _lll_reduce(mat, delta=DELTA)

  norms = [_norm_sq(row) for row in reduced]
  order = sorted(range(n), key=lambda i: norms[i])
  return [reduced[i] for i in order]
