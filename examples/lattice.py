"""
OBJECTIVE: Discover a novel and highly effective lattice basis reduction algorithm.

The goal is to implement the `reduce_basis` function to find a reduced basis `B` for a given lattice basis `A`.

## GOAL & SCORING:
The score is defined by the `evaluate` function: it is based on the negative logarithm of the norm of the shortest vector found, `-1.0 * log(norm(B[0]))`.
MAXIMIZING this score means MINIMIZING the length of the shortest vector in the reduced basis (B[0]). A higher score is always better.

## INSTRUCTIONS FOR EVOLUTION:
1.  **Maximize Creativity and Novelty:** Do not simply re-implement existing algorithms like LLL (Lenstra-Lenstra-Lovász) or BKZ (Block Korkine-Zolotarev). Instead, combine concepts, introduce new reduction strategies, or use non-standard techniques to achieve a better reduction.
2.  **Constraint Freedom:** You are only constrained by Python syntax and the function signature `def reduce_basis(basis: np.ndarray) -> np.ndarray:`. **You may use any functions or libraries already imported at the top of the file (e.g., `numpy` functions like `linalg.det`, `linalg.norm`, etc.).**
3.  **Core Requirement:** The reduced basis must span the same lattice. This is checked by verifying that the absolute determinant remains unchanged (within floating point tolerance) and that the basis consists of linear integer combinations of the original vectors.
4.  **Iterative Improvement:** The current `reduce_basis` is a basic, non-optimal implementation. Propose a modification that significantly improves the reduction quality by yielding a shorter first vector. Focus on sophisticated swapping, pivoting, or iterative reduction loops.
5.  **Code Style:** Prefer concise, clean, and highly vectorized NumPy code for efficiency.

## CURRENT BASELINE CODE TO IMPROVE:
# The LLM will see the existing `reduce_basis` function here and be asked to provide a new version.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

import funsearch

# Judge configuration
NUM_TEST_CASES = 5
LOWER_BOUND = -100
UPPER_BOUND = 100

# LLL delta in (0.25, 1). Higher -> stronger reduction, more work.
DELTA = 0.99

# Candidate must be "integer-like" within this tolerance.
INT_TOL = 1e-6

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


def _round_int_matrix(matrix: Matrix) -> List[List[int]]:
  out: List[List[int]] = []
  for row in matrix:
    r: List[int] = []
    for x in row:
      fx = float(x)
      rx = int(round(fx))
      if abs(fx - float(rx)) > INT_TOL:
        raise ValueError("Entry not integer-like enough for exact det check")
      r.append(rx)
    out.append(r)
  return out


def _det_bareiss_int(a: List[List[int]]) -> int:
  n = len(a)
  if n == 0 or any(len(row) != n for row in a):
    raise ValueError("det: matrix must be non-empty and square")

  m = [row[:] for row in a]
  det_sign = 1
  prev = 1

  for k in range(n - 1):
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
          q, r = divmod(num, prev)
          if r != 0:
            # For true integer matrices, Bareiss should divide exactly.
            # If this happens, treat as singular/invalid.
            return 0
          m[i][j] = q
        else:
          m[i][j] = num
      m[i][k] = 0

    prev = pivot

  return det_sign * m[n - 1][n - 1]


def _generate_non_singular_basis(seed: int, n: int) -> Matrix:
  # Try multiple deterministic attempts; fallback to identity if unlucky.
  for t in range(128):
    rng = random.Random(seed + 1000003 * t)
    basis: Matrix = [
      [float(rng.randint(LOWER_BOUND, UPPER_BOUND)) for _ in range(n)]
      for _ in range(n)
    ]
    try:
      det = _det_bareiss_int(_round_int_matrix(basis))
    except Exception:
      det = 0
    if det != 0:
      return basis

  return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _compute_gram_schmidt(basis: Matrix) -> Tuple[Matrix, Matrix, List[float]]:
  """Classical Gram-Schmidt for row vectors.

  Returns:
    ortho: orthogonalized vectors (not normalized)
    mu: projection coefficients, mu[i][j] = <b_i, b*_j> / ||b*_j||^2
    norm_sq: squared norms of ortho vectors
  """
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


def _lll_reduce(basis: Matrix, delta: float = DELTA, max_steps: int = 10_000) -> Matrix:
  """Basic (but robust) floating LLL using only integer row operations.

  All basis updates are of the form:
    b_k <- b_k - r * b_j   (r is an integer)
    swap(b_k, b_{k-1})
  which preserves the lattice volume up to sign (unimodular ops).
  """
  reduced = _copy_matrix(basis)
  n = len(reduced)

  ortho, mu, ns = _compute_gram_schmidt(reduced)

  def refresh() -> None:
    nonlocal ortho, mu, ns
    ortho, mu, ns = _compute_gram_schmidt(reduced)

  k = 1
  steps = 0
  while k < n and steps < max_steps:
    steps += 1

    # Size reduction (integer steps).
    for j in range(k - 1, -1, -1):
      if ns[j] <= 1e-18:
        continue
      r = int(round(mu[k][j]))
      if r != 0:
        for t in range(n):
          reduced[k][t] -= float(r) * reduced[j][t]
        refresh()

    # Lovász condition.
    if ns[k - 1] <= 1e-18:
      k += 1
      continue
    if ns[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * ns[k - 1]:
      _swap_rows(reduced, k, k - 1)
      refresh()
      k = max(k - 1, 1)
    else:
      k += 1

  return reduced


def _local_first_vector_improve(basis: Matrix, window: int = 3, rounds: int = 3) -> None:
  """Deterministically hill-climb the first vector via unimodular row ops.

  Only applies operations of the form:
    b0 <- b0 + c * bi,  c in [-window, window]
  which keeps det unchanged (det is invariant under adding integer multiples of
  other rows to a row).
  """
  n = len(basis)
  if n < 2:
    return

  for _ in range(rounds):
    v0 = basis[0]
    best_v0 = v0[:]
    best_ns = _norm_sq(v0)
    improved = False

    for i in range(1, n):
      vi = basis[i]
      denom = _norm_sq(vi)
      if denom <= 1e-18:
        continue

      # Center the search near the best real-valued coefficient.
      k0 = int(round(_dot(v0, vi) / denom))
      # Avoid pathological huge coefficients that can amplify float round-off.
      if k0 > 50:
        k0 = 50
      elif k0 < -50:
        k0 = -50

      for dk in range(-window, window + 1):
        k = k0 + dk
        if k == 0:
          continue
        cand = [v0[t] - float(k) * vi[t] for t in range(n)]
        ns = _norm_sq(cand)
        if ns + 1e-12 < best_ns:
          best_ns = ns
          best_v0 = cand
          improved = True

    if not improved:
      break

    # Commit as a true unimodular update by rewriting b0.
    basis[0] = best_v0


@funsearch.run
def evaluate(program=None, n: int = 8) -> float:
  """Evaluate a candidate reducer.

  The sandbox may call this function as ``evaluate(8)`` (only a dimension
  argument) or ``evaluate(program, 8)`` (program plus dimension). To keep both
  call styles working, we detect when the first argument is actually the
  dimension and fall back to this module's baseline ``reduce_basis`` when no
  program object is supplied.
  """
  if program is None or isinstance(program, (int, float, str)):
    dim_arg = program if program is not None else n
    program_obj = None
  else:
    dim_arg = n
    program_obj = program

  try:
    n_int = int(dim_arg)
  except Exception:
    return -1000.0
  if n_int < 2 or n_int > 48:
    return -1000.0

  reducer = None
  if program_obj is not None and hasattr(program_obj, "reduce_basis"):
    reducer = getattr(program_obj, "reduce_basis")
  if not callable(reducer):
    reducer = reduce_basis

  scores: List[float] = []
  for seed in range(NUM_TEST_CASES):
    original = _generate_non_singular_basis(seed=seed, n=n_int)

    try:
      cand = reducer(_copy_matrix(original))
    except Exception:
      return -1000.0

    try:
      reduced = _matrix_from_candidate(cand, n=n_int)

      # Exact determinant comparison on rounded int matrices.
      orig_i = _round_int_matrix(original)
      red_i = _round_int_matrix(reduced)

      det_orig = _det_bareiss_int(orig_i)
      det_new = _det_bareiss_int(red_i)
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
  """Baseline reducer that stays strictly within unimodular row operations.

  This is intentionally conservative: FunSearch needs a *valid* starting point.
  If the baseline violates the determinant / integrality checks, the initial
  analysis produces no clusters and the run aborts.

  Strategy:
    1) Run a robust LLL pass (integer row ops only).
    2) Try a small deterministic local search that only updates b0 <- b0 + c*bi.
    3) Run one more LLL pass and return vectors sorted by norm.
  """
  mat = [list(map(float, row)) for row in basis]
  n = len(mat)
  if n == 0 or any(len(row) != n for row in mat):
    raise ValueError("basis must be non-empty square")

  red = _lll_reduce(mat, delta=DELTA)
  _local_first_vector_improve(red, window=3, rounds=3)
  red = _lll_reduce(red, delta=DELTA)

  norms = [_norm_sq(row) for row in red]
  order = sorted(range(n), key=lambda i: norms[i])
  return [red[i] for i in order]
