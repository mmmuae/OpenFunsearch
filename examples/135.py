"""Elliptic-curve scalar search heuristic for Puzzle 135.

Guidelines for the LLM (keep these instructions intact):
- Always return a single, complete Python module that starts at column 0.
  Repeat the imports, ECC primitives, and ``evaluate`` exactly as provided; do
  not wrap code in markdown, prose, or extra indentation.
- Only adjust the body of ``heuristic_priority``; keep function signatures and
  top-level layout unchanged. Avoid adding helper functions or globals.
- Match the indentation style of ``examples/capset.py`` by using **two spaces**
  for each block level—never tabs or mixed spacing—and finish with a trailing
  newline.
- Keep ``evaluate`` as the authoritative scoring function and avoid altering
  its logic or randomness handling.
- Prefer short, deterministic arithmetic; avoid I/O, randomness, or external
  state inside ``heuristic_priority``.
"""

import numpy as np
import types
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (Gx, Gy)


def mod_inv(a, n):
  return pow(a, n - 2, n)


def decompress_pubkey(pk_hex):
  prefix = pk_hex[:2]
  x = int(pk_hex[2:], 16)
  y_sq = (pow(x, 3, P) + 7) % P
  y = pow(y_sq, (P + 1) // 4, P)
  if (prefix == "02" and y % 2 != 0) or (prefix == "03" and y % 2 == 0):
    y = P - y
  return (x, y)


def point_add(p1, p2):
  if p1 is None:
    return p2
  if p2 is None:
    return p1
  x1, y1 = p1
  x2, y2 = p2
  if x1 == x2 and y1 != y2:
    return None
  if x1 == x2:
    m = (3 * x1 * x1 * mod_inv(2 * y1, P)) % P
  else:
    m = ((y2 - y1) * mod_inv(x2 - x1, P)) % P
  x3 = (m * m - x1 - x2) % P
  y3 = (m * (x1 - x3) - y1) % P
  return (x3, y3)


def scalar_mult(point, k):
  result = None
  addend = point
  while k:
    if k & 1:
      result = point_add(result, addend)
    addend = point_add(addend, addend)
    k >>= 1
  return result


crypto = types.SimpleNamespace(
  P=P,
  N=N,
  A=A,
  B=B,
  G=G,
  mod_inv=mod_inv,
  decompress_pubkey=decompress_pubkey,
  point_add=point_add,
  scalar_mult=scalar_mult,
)

import funsearch

# Puzzle 135 Target Data
TARGET_PK_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
TARGET_Q = crypto.decompress_pubkey(TARGET_PK_HEX)
RANGE_START = 0x4000000000000000000000000000000000


@funsearch.run
def evaluate(program) -> float:
  """Score a candidate ``heuristic_priority`` implementation.

  The evaluator exercises the heuristic on a smaller window that mimics the
  135-bit challenge. The score rewards correct scalar guesses and proximity to
  the target point, encouraging concise and well-indented proposals from the
  LLM.
  """

  score = 0.0
  for _ in range(5):
    # Generate a random k in a test range
    test_k = np.random.randint(1, 2**40)
    test_Q = crypto.scalar_mult(crypto.G, test_k)

    # Ask AI to predict the 'energy' or 'likelihood' of a candidate scalar
    # based on the bitwise interaction between points.
    candidates = []
    for i in range(1, 500):
      p_val = crypto.scalar_mult(crypto.G, i)
      p_score = program.heuristic_priority(
        p_val[0], p_val[1], test_Q[0], test_Q[1]
      )
      candidates.append((p_score, i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    # Check if the correct relation is in the top picks
    found = any(k == test_k for _, k in candidates[:5])
    if found:
      score += 20.0

    # Add 'partial credit' based on bitwise proximity of top guesses
    best_guess_x = crypto.scalar_mult(crypto.G, candidates[0][1])[0]
    bit_match = bin(best_guess_x ^ test_Q[0]).count("0")
    score += bit_match / 256.0

  return float(score)


@funsearch.evolve
def heuristic_priority(px, py, qx, qy) -> float:
  """Baseline heuristic for ranking candidate scalars.

  Keep this function short, deterministic, and free of side effects so the
  sandbox can parse it without indentation errors.
  """

  shared_bits = bin(px ^ qx).count("0")
  return float(shared_bits)
