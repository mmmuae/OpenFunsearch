"""Advanced multi-feature elliptic-curve search heuristic for Puzzle 135.

This variant exposes 30+ mathematical features to enable FunSearch to discover
genius combinations that humans haven't thought of. The feature space combines:
- Bitwise patterns (XOR, AND, OR, Hamming distances)
- Field arithmetic (modular inverses, GCD patterns, quadratic residues)
- Geometric properties (slope estimates, coordinate ratios, cross products)
- Number theoretic (digit sums, Jacobi symbols, factorization hints)
- Statistical patterns (entropy, variance in bit chunks)
- Coordinate relationships (x-y correlations, negation patterns)

Guidelines for the LLM (CRITICAL - keep these instructions intact):
- **MAKE BOLD CHANGES**: Don't just tweak constants—try completely different
  feature combinations, mathematical operations, and scoring strategies on each
  iteration. Explore radically different approaches.
- **COMBINE FEATURES CREATIVELY**: Use products, ratios, sums, XORs, and nested
  conditions. Look for non-obvious patterns across multiple feature dimensions.
- **EXPERIMENT AGGRESSIVELY**: Test counterintuitive ideas like inverse
  relationships, negative weights, polynomial combinations, and threshold-based
  logic. The goal is to find hidden mathematical shortcuts.
- Always return a single, complete Python module that starts at column 0.
  Repeat the imports, ECC primitives, feature computation, and ``evaluate``
  exactly as provided; do not wrap code in markdown, prose, or extra indentation.
- Only adjust the body of ``priority``; keep function signatures and
  top-level layout unchanged. You can use ANY of the features in the features
  dict in creative ways, but don't add helper functions or globals.
- Match the indentation style by using **two spaces** for each block level—
  never tabs or mixed spacing—and finish with a trailing newline.
- Keep ``evaluate`` as the authoritative scoring function and avoid altering
  its logic or randomness handling.
- The priority function receives a rich 'features' dictionary with
  30+ pre-computed mathematical properties. Your job is to combine them cleverly.
- Prefer deterministic arithmetic; avoid I/O, randomness, or external state
  inside ``priority``.
- **TRY EVERYTHING**: Each iteration should explore fundamentally different
  mathematical territory. Don't converge too quickly—keep exploring!
"""

import numpy as np
import types
import math

# secp256k1 curve parameters
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


def gcd(a, b):
  while b:
    a, b = b, a % b
  return a


def jacobi_symbol(a, n):
  """Compute Jacobi symbol (a/n) - generalization of Legendre symbol."""
  if n <= 0 or n % 2 == 0:
    return 0
  a = a % n
  result = 1
  while a != 0:
    while a % 2 == 0:
      a //= 2
      if n % 8 in [3, 5]:
        result = -result
    a, n = n, a
    if a % 4 == 3 and n % 4 == 3:
      result = -result
    a = a % n
  return result if n == 1 else 0


def popcount(x):
  """Count number of 1-bits."""
  return bin(x).count('1')


def compute_features(px, py, qx, qy):
  """Compute comprehensive mathematical features for the heuristic.

  Returns a dictionary with 30+ features covering multiple mathematical domains.
  """
  features = {}

  # === BITWISE PATTERNS ===
  features['x_xor_popcount'] = popcount(px ^ qx)
  features['y_xor_popcount'] = popcount(py ^ qy)
  features['x_and_popcount'] = popcount(px & qx)
  features['y_and_popcount'] = popcount(py & qy)
  features['x_or_popcount'] = popcount(px | qx)
  features['y_or_popcount'] = popcount(py | qy)
  features['x_hamming'] = bin(px ^ qx).count('0')  # Shared bits
  features['y_hamming'] = bin(py ^ qy).count('0')

  # === COORDINATE RELATIONSHIPS ===
  features['xy_cross_p'] = (px * py) % P
  features['xy_cross_q'] = (qx * qy) % P
  features['xy_cross_diff'] = abs(((px * qy) % P) - ((py * qx) % P))
  features['coord_sum_p'] = (px + py) % P
  features['coord_sum_q'] = (qx + qy) % P
  features['coord_diff_p'] = (px - py) % P
  features['coord_diff_q'] = (qx - qy) % P

  # === MODULAR ARITHMETIC ===
  features['x_diff_mod_p'] = (px - qx) % P
  features['y_diff_mod_p'] = (py - qy) % P
  features['x_sum_mod_p'] = (px + qx) % P
  features['y_sum_mod_p'] = (py + qy) % P
  features['x_prod_mod_p'] = (px * qx) % P
  features['y_prod_mod_p'] = (py * qy) % P

  # === FIELD INVERSES ===
  try:
    px_inv = mod_inv(px if px != 0 else 1, P)
    qx_inv = mod_inv(qx if qx != 0 else 1, P)
    py_inv = mod_inv(py if py != 0 else 1, P)
    qy_inv = mod_inv(qy if qy != 0 else 1, P)
    features['x_inv_diff'] = (px_inv - qx_inv) % P
    features['y_inv_diff'] = (py_inv - qy_inv) % P
    features['x_ratio'] = (px * qx_inv) % P
    features['y_ratio'] = (py * qy_inv) % P
  except:
    features['x_inv_diff'] = 0
    features['y_inv_diff'] = 0
    features['x_ratio'] = 0
    features['y_ratio'] = 0

  # === GCD PATTERNS ===
  features['gcd_x'] = gcd(px, qx)
  features['gcd_y'] = gcd(py, qy)
  features['gcd_x_p'] = gcd(px, P)
  features['gcd_y_p'] = gcd(py, P)

  # === JACOBI SYMBOLS (quadratic residue patterns) ===
  features['jacobi_px'] = jacobi_symbol(px, P)
  features['jacobi_py'] = jacobi_symbol(py, P)
  features['jacobi_qx'] = jacobi_symbol(qx, P)
  features['jacobi_qy'] = jacobi_symbol(qy, P)

  # === DIGIT SUMS (patterns in different bases) ===
  features['digit_sum_x_b256'] = sum(int(d) for d in hex(px)[2:])
  features['digit_sum_y_b256'] = sum(int(d) for d in hex(py)[2:])
  features['digit_sum_qx_b256'] = sum(int(d) for d in hex(qx)[2:])
  features['digit_sum_qy_b256'] = sum(int(d) for d in hex(qy)[2:])

  # === BIT PATTERNS AT DIFFERENT POSITIONS ===
  features['x_low_32_xor'] = (px ^ qx) & 0xFFFFFFFF
  features['x_high_32_xor'] = ((px ^ qx) >> 224) & 0xFFFFFFFF
  features['y_low_32_xor'] = (py ^ qy) & 0xFFFFFFFF
  features['y_high_32_xor'] = ((py ^ qy) >> 224) & 0xFFFFFFFF

  # === GEOMETRIC/SLOPE PATTERNS ===
  # Approximate slope-like quantity (avoiding full modular division for speed)
  features['delta_y'] = (qy - py) % P
  features['delta_x'] = (qx - px) % P
  features['slope_numerator'] = features['delta_y']
  features['slope_denominator'] = features['delta_x'] if features['delta_x'] != 0 else 1

  # === STATISTICAL PATTERNS ===
  # Entropy-like measures (count transitions in binary representation)
  px_bin = bin(px)[2:]
  qx_bin = bin(qx)[2:]
  features['px_transitions'] = sum(1 for i in range(len(px_bin)-1) if px_bin[i] != px_bin[i+1])
  features['qx_transitions'] = sum(1 for i in range(len(qx_bin)-1) if qx_bin[i] != qx_bin[i+1])

  # === PARITY AND DIVISIBILITY ===
  features['x_parity_match'] = 1 if (px % 2) == (qx % 2) else 0
  features['y_parity_match'] = 1 if (py % 2) == (qy % 2) else 0
  features['x_mod_3'] = (px % 3) - (qx % 3)
  features['y_mod_3'] = (py % 3) - (qy % 3)
  features['x_mod_7'] = (px % 7) - (qx % 7)
  features['y_mod_7'] = (py % 7) - (qy % 7)

  # === NEGATION PATTERNS (important for ECC: -P = (x, -y)) ===
  features['y_negation_match'] = 1 if (py + qy) % P == 0 else 0
  features['y_negation_diff'] = ((py + qy) % P)

  # === COMBINED COMPLEXITY MEASURES ===
  features['total_popcount'] = popcount(px) + popcount(py) + popcount(qx) + popcount(qy)
  features['xor_product'] = ((px ^ qx) * (py ^ qy)) % P
  features['coord_similarity'] = features['x_hamming'] + features['y_hamming']

  return features


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
def evaluate(seed: int) -> float:
  """Score a candidate ``priority`` implementation.

  This evaluator tests the heuristic on multiple scenarios with diverse
  mathematical properties. It rewards both exact matches and proximity,
  encouraging the LLM to discover creative feature combinations.
  """

  # Rebuild the ECC namespace if the sandbox omits module globals.
  global crypto, P, N
  if "crypto" not in globals():
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    A = 0
    B = 7
    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    G = (Gx, Gy)
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

  rng = np.random.default_rng(seed)
  score = 0.0

  # Test on 10 diverse scenarios (increased from 5 for better evaluation)
  for test_round in range(10):
    # Vary the test range to cover different magnitudes
    if test_round < 3:
      test_k = rng.integers(1, 2**20)  # Small keys
    elif test_round < 6:
      test_k = rng.integers(2**20, 2**40)  # Medium keys
    else:
      test_k = rng.integers(2**40, 2**60)  # Larger keys

    test_Q = crypto.scalar_mult(crypto.G, test_k)

    # Rank candidate scalars using the evolved heuristic
    candidates = []
    for i in range(1, 1000):  # Increased candidate pool for harder challenge
      p_val = crypto.scalar_mult(crypto.G, i)
      features = compute_features(p_val[0], p_val[1], test_Q[0], test_Q[1])
      p_score = priority(features)
      candidates.append((p_score, i))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # High reward for finding the exact match in top 10
    found_top10 = any(k == test_k for _, k in candidates[:10])
    if found_top10:
      score += 50.0

    # Medium reward for finding in top 50
    found_top50 = any(k == test_k for _, k in candidates[:50])
    if found_top50 and not found_top10:
      score += 20.0

    # Small reward for finding in top 100
    found_top100 = any(k == test_k for _, k in candidates[:100])
    if found_top100 and not found_top50:
      score += 5.0

    # Partial credit: bitwise proximity of best guess
    best_guess_x = crypto.scalar_mult(crypto.G, candidates[0][1])[0]
    bit_match = bin(best_guess_x ^ test_Q[0]).count("0")
    score += bit_match / 256.0

    # Bonus: reward diversity in top picks (avoid getting stuck on one pattern)
    top_5_scalars = [k for _, k in candidates[:5]]
    diversity = len(set(top_5_scalars))
    score += diversity * 0.5

    # Extra bonus: check if top picks have any useful mathematical relationship
    # to the target (e.g., factors, sums, differences)
    for _, candidate_k in candidates[:5]:
      if candidate_k != test_k:
        # Check if there's a simple relationship
        if (test_k % candidate_k == 0) or (candidate_k % test_k == 0):
          score += 2.0
        if abs(test_k - candidate_k) < 100:
          score += 1.0

  return float(score)


@funsearch.evolve
def priority(features: dict) -> float:
  """Advanced multi-feature heuristic for ranking candidate scalars.

  The 'features' dictionary contains 30+ pre-computed mathematical properties.
  Your mission: combine them in creative, non-obvious ways to discover hidden
  patterns in the elliptic curve discrete logarithm problem.

  Available feature categories:
  - Bitwise: x_xor_popcount, y_xor_popcount, x_hamming, y_hamming, etc.
  - Coordinate: xy_cross_p, coord_sum_p, coord_diff_p, etc.
  - Modular: x_diff_mod_p, x_prod_mod_p, x_ratio, y_ratio, etc.
  - Field inverses: x_inv_diff, y_inv_diff
  - GCD patterns: gcd_x, gcd_y, gcd_x_p
  - Jacobi symbols: jacobi_px, jacobi_py, jacobi_qx, jacobi_qy
  - Digit sums: digit_sum_x_b256, digit_sum_y_b256
  - Bit positions: x_low_32_xor, x_high_32_xor
  - Geometric: delta_x, delta_y, slope_numerator
  - Statistical: px_transitions, qx_transitions
  - Parity: x_parity_match, y_parity_match, x_mod_3, x_mod_7
  - Negation: y_negation_match, y_negation_diff
  - Combined: total_popcount, xor_product, coord_similarity

  EXPLORE AGGRESSIVELY:
  - Try polynomial combinations: a*f1^2 + b*f2*f3 + c*f4
  - Use conditionals: if f1 > threshold then weight1 else weight2
  - Combine distant features: bitwise AND geometric AND number theoretic
  - Test inverse relationships: 1/f1, -f2, f3 XOR f4
  - Look for resonance: (f1 * f2) % some_modulus
  - Be creative with thresholds, products, ratios, and XORs!

  Baseline starter (YOU MUST IMPROVE ON THIS!):
  """

  # Simple baseline: combine coordinate similarity with modular patterns
  # THIS IS JUST A STARTING POINT - EVOLVE IT RADICALLY!
  base_score = float(features['coord_similarity'])

  # Add some basic feature combinations
  if features['x_parity_match'] and features['y_parity_match']:
    base_score += 10.0

  # Reward GCD patterns
  if features['gcd_x'] > 1:
    base_score += 5.0

  # Consider Jacobi symbols
  if features['jacobi_px'] == features['jacobi_qx']:
    base_score += 3.0

  return base_score
