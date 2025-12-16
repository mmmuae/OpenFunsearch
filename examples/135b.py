"""Advanced ECDLP attack heuristic for Bitcoin Puzzle 135.

THIS IS NOT A TOY. This is a real attack on Puzzle 135 using every known
mathematical approach to ECDLP. FunSearch's job: discover which mathematical
properties can narrow the 2^135 keyspace.

Target: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
Range: 0x4000000000000000000000000000000000 to 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
Prize: Real Bitcoin

Mathematical arsenal exposed to FunSearch:
- Baby-step Giant-step collision patterns
- Pollard's Rho/Kangaroo distinguished point properties
- Lattice reduction indicators (short vector patterns)
- Frobenius endomorphism (secp256k1 GLV decomposition)
- Negation symmetry (k vs N-k)
- Montgomery ladder patterns
- Point addition chain properties
- Quadratic residue patterns
- Modular arithmetic relationships
- Birthday paradox collision indicators
- Differential addition patterns

Guidelines for the LLM:
- **THIS IS REAL**: You're attacking an actual cryptographic problem
- **EXPLORE EVERYTHING**: Combine features in ways cryptographers haven't tried
- **LOOK FOR STRUCTURE**: ECDLP has hidden structure waiting to be found
- **BE RADICAL**: Try counterintuitive combinations, negative correlations
- Always return a single, complete Python module starting at column 0
- Only adjust the body of ``priority``; keep function signatures unchanged
- Use **two spaces** per indentation level
- The priority function receives 60+ mathematical features about candidate keys
- Your job: find which feature combinations predict proximity to the real key
"""

import numpy as np
import types
import math

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
  return bin(x).count('1')


def compute_features(candidate_k, candidate_point, target_point):
  """Compute 80+ features for ECDLP analysis.

  Args:
    candidate_k: The candidate private key being evaluated
    candidate_point: k*G (the candidate public key)
    target_point: TARGET_Q (the actual Puzzle 135 public key)

  Returns:
    Dictionary with mathematical features that might correlate with key proximity
  """
  px, py = candidate_point
  qx, qy = target_point
  features = {}

  # === KEY PROPERTIES ===
  features['k_value'] = candidate_k
  features['k_bitlength'] = candidate_k.bit_length()
  features['k_popcount'] = popcount(candidate_k)
  features['k_low_bits'] = candidate_k & 0xFFFFFFFF
  features['k_high_bits'] = (candidate_k >> 100) & 0xFFFFFFFF
  features['k_mod_small_primes'] = (candidate_k % 2) + (candidate_k % 3) + (candidate_k % 5) + (candidate_k % 7)

  # === COORDINATE BITWISE PATTERNS ===
  features['x_xor_popcount'] = popcount(px ^ qx)
  features['y_xor_popcount'] = popcount(py ^ qy)
  features['x_and_popcount'] = popcount(px & qx)
  features['y_and_popcount'] = popcount(py & qy)
  features['x_hamming'] = bin(px ^ qx).count('0')
  features['y_hamming'] = bin(py ^ qy).count('0')
  features['xy_combined_hamming'] = features['x_hamming'] + features['y_hamming']

  # === COORDINATE ARITHMETIC ===
  features['x_diff'] = abs(px - qx) if px > qx else abs(qx - px)
  features['y_diff'] = abs(py - qy) if py > qy else abs(qy - py)
  features['x_sum'] = (px + qx) % P
  features['y_sum'] = (py + qy) % P
  features['x_prod'] = (px * qx) % P
  features['y_prod'] = (py * qy) % P
  features['xy_cross_prod'] = ((px * qy) % P + (py * qx) % P) % P

  # === NEGATION PATTERNS (k and N-k relationship) ===
  features['y_negation_match'] = 1 if (py + qy) % P == 0 else 0
  features['x_match'] = 1 if px == qx else 0
  features['mirror_k'] = N - candidate_k
  features['mirror_k_distance'] = min(candidate_k, N - candidate_k)

  # === MODULAR PATTERNS ===
  features['x_mod_256'] = (px % 256) - (qx % 256)
  features['y_mod_256'] = (py % 256) - (qy % 256)
  features['x_mod_65536'] = (px % 65536) - (qx % 65536)
  features['y_mod_65536'] = (py % 65536) - (qy % 65536)

  # === FIELD INVERSES ===
  try:
    px_inv = mod_inv(px if px != 0 else 1, P)
    qx_inv = mod_inv(qx if qx != 0 else 1, P)
    features['x_inv_prod'] = (px_inv * qx) % P
    features['x_ratio'] = (px * qx_inv) % P
  except:
    features['x_inv_prod'] = 0
    features['x_ratio'] = 0

  # === JACOBI SYMBOLS (quadratic residue patterns) ===
  features['jacobi_px'] = jacobi_symbol(px, P)
  features['jacobi_py'] = jacobi_symbol(py, P)
  features['jacobi_qx'] = jacobi_symbol(qx, P)
  features['jacobi_qy'] = jacobi_symbol(qy, P)
  features['jacobi_match'] = 1 if (features['jacobi_px'] == features['jacobi_qx'] and
                                     features['jacobi_py'] == features['jacobi_qy']) else 0

  # === BABY-STEP GIANT-STEP PATTERNS ===
  # Look for patterns in how coordinates relate to stepping patterns
  features['x_step_pattern'] = (px % 1000) - (qx % 1000)
  features['y_step_pattern'] = (py % 1000) - (qy % 1000)
  features['coord_alternating'] = ((px ^ qx) ^ (py ^ qy)) % 1000

  # === DISTINGUISHED POINT PROPERTIES (Pollard's methods) ===
  # Check if points have "distinguished" properties
  features['px_leading_zeros'] = len(bin(px)) - len(bin(px).rstrip('0'))
  features['qx_leading_zeros'] = len(bin(qx)) - len(bin(qx).rstrip('0'))
  features['px_trailing_zeros'] = len(bin(px)) - len(bin(px).lstrip('0'))
  features['distinguished_diff'] = features['px_leading_zeros'] - features['qx_leading_zeros']

  # === DIGIT SUMS (additive patterns) ===
  features['digit_sum_x'] = sum(int(d, 16) for d in hex(px)[2:])
  features['digit_sum_y'] = sum(int(d, 16) for d in hex(py)[2:])
  features['digit_sum_qx'] = sum(int(d, 16) for d in hex(qx)[2:])
  features['digit_sum_qy'] = sum(int(d, 16) for d in hex(qy)[2:])
  features['digit_sum_diff'] = abs(features['digit_sum_x'] - features['digit_sum_qx'])

  # === BIT POSITION ANALYSIS ===
  features['x_low_64_xor'] = (px ^ qx) & 0xFFFFFFFFFFFFFFFF
  features['x_high_64_xor'] = ((px ^ qx) >> 192) & 0xFFFFFFFFFFFFFFFF
  features['bit_pattern_match'] = sum(1 for i in range(256) if ((px >> i) & 1) == ((qx >> i) & 1))

  # === SLOPE/GEOMETRIC PATTERNS ===
  features['delta_x'] = (qx - px) % P
  features['delta_y'] = (qy - py) % P
  if features['delta_x'] != 0:
    try:
      features['slope_approx'] = (features['delta_y'] * mod_inv(features['delta_x'], P)) % P
    except:
      features['slope_approx'] = 0
  else:
    features['slope_approx'] = 0

  # === STATISTICAL/ENTROPY ===
  px_bin = bin(px)[2:]
  qx_bin = bin(qx)[2:]
  features['px_transitions'] = sum(1 for i in range(len(px_bin)-1) if px_bin[i] != px_bin[i+1])
  features['qx_transitions'] = sum(1 for i in range(len(qx_bin)-1) if qx_bin[i] != qx_bin[i+1])
  features['transition_similarity'] = abs(features['px_transitions'] - features['qx_transitions'])

  # === PARITY AND DIVISIBILITY ===
  features['x_parity_match'] = 1 if (px % 2) == (qx % 2) else 0
  features['y_parity_match'] = 1 if (py % 2) == (qy % 2) else 0
  features['both_parity_match'] = features['x_parity_match'] * features['y_parity_match']

  # === GCD PATTERNS ===
  features['gcd_x'] = gcd(px, qx)
  features['gcd_y'] = gcd(py, qy)
  features['gcd_large'] = 1 if features['gcd_x'] > 1 or features['gcd_y'] > 1 else 0

  # === FROBENIUS/GLV DECOMPOSITION HINTS ===
  # secp256k1 has efficient endomorphism, might reveal structure
  lambda_val = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
  features['glv_k_mod'] = candidate_k % lambda_val
  features['glv_pattern'] = (px * lambda_val) % P

  # === COLLISION/BIRTHDAY PARADOX INDICATORS ===
  features['collision_indicator'] = ((px ^ qx) * (py ^ qy)) % 65536
  features['birthday_hash'] = ((px + qx) ^ (py + qy)) % 65536

  # === LATTICE-BASED HINTS ===
  # Short vector patterns
  features['coord_magnitude'] = (px % 10000) + (py % 10000)
  features['target_magnitude'] = (qx % 10000) + (qy % 10000)
  features['magnitude_ratio'] = features['coord_magnitude'] / (features['target_magnitude'] + 1)

  # === COMBINED COMPLEXITY MEASURES ===
  features['total_popcount'] = popcount(px) + popcount(py) + popcount(qx) + popcount(qy)
  features['xor_product'] = ((px ^ qx) * (py ^ qy)) % P
  features['combined_similarity'] = features['xy_combined_hamming'] + features['both_parity_match'] * 10

  # === RANGE POSITION ===
  RANGE_START = 0x4000000000000000000000000000000000
  RANGE_END = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
  features['range_position'] = (candidate_k - RANGE_START) / (RANGE_END - RANGE_START) if candidate_k >= RANGE_START else 0
  features['in_puzzle_range'] = 1 if RANGE_START <= candidate_k <= RANGE_END else 0

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

# THE REAL TARGET
TARGET_PK_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
TARGET_Q = crypto.decompress_pubkey(TARGET_PK_HEX)
RANGE_START = 0x4000000000000000000000000000000000
RANGE_END = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


@funsearch.run
def evaluate(seed: int) -> float:
  """Evaluate priority function against REAL Puzzle 135 target.

  Tests if heuristic can identify mathematical patterns by comparing
  candidate points (k*G for small k) against the real TARGET_Q.
  If the heuristic finds features that correlate with the target,
  it might reveal exploitable structure in ECDLP.
  """

  global crypto, P, N, TARGET_Q, RANGE_START, RANGE_END
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
    TARGET_PK_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
    TARGET_Q = crypto.decompress_pubkey(TARGET_PK_HEX)
    RANGE_START = 0x4000000000000000000000000000000000
    RANGE_END = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

  rng = np.random.default_rng(seed)
  score = 0.0

  # Strategy: Test against REAL Puzzle 135 target (TARGET_Q)
  # Use small candidate keys (fast to compute) but measure how well
  # the heuristic identifies patterns when comparing to the real target

  candidates = []

  # Generate candidates using small keys (1 to 2000) for fast computation
  # But compare their points against the REAL TARGET_Q
  for i in range(1, 2001):
    point = crypto.scalar_mult(crypto.G, i)
    features = compute_features(i, point, TARGET_Q)
    priority_score = priority(features)
    candidates.append((priority_score, i, point))

  # Sort by priority score (higher = more promising according to heuristic)
  candidates.sort(key=lambda x: x[0], reverse=True)

  # SCORING: Reward heuristics that identify patterns correlating with TARGET_Q

  # 1. Coordinate proximity of top-ranked candidates
  for i, (_, k, point) in enumerate(candidates[:100]):
    px, py = point
    qx, qy = TARGET_Q

    # Hamming distance - closer is better
    x_hamming = bin(px ^ qx).count('0')
    y_hamming = bin(py ^ qy).count('0')
    hamming_score = (x_hamming + y_hamming) / 512.0

    # Weight by rank (higher rank = more reward)
    rank_weight = (100 - i) / 100.0
    score += hamming_score * rank_weight * 10.0

  # 2. Check for CRITICAL patterns
  top_point = candidates[0][2]
  px, py = top_point
  qx, qy = TARGET_Q

  # Negation check (k and N-k relationship)
  if (py + qy) % P == 0 and px == qx:
    score += 10000.0  # JACKPOT

  # X-coordinate match (extremely rare)
  if px == qx:
    score += 5000.0

  # 3. Statistical pattern detection
  # If top picks have better hamming distance than random, reward it
  top_10_hamming = []
  bottom_10_hamming = []

  for _, k, point in candidates[:10]:
    px, py = point
    h = bin(px ^ qx).count('0') + bin(py ^ qy).count('0')
    top_10_hamming.append(h)

  for _, k, point in candidates[-10:]:
    px, py = point
    h = bin(px ^ qx).count('0') + bin(py ^ qy).count('0')
    bottom_10_hamming.append(h)

  avg_top = sum(top_10_hamming) / len(top_10_hamming)
  avg_bottom = sum(bottom_10_hamming) / len(bottom_10_hamming)

  # If top picks are genuinely better than bottom (found pattern!)
  if avg_top > avg_bottom:
    score += (avg_top - avg_bottom) * 5.0

  # 4. Diversity bonus (explore different candidates)
  top_20_keys = [k for _, k, _ in candidates[:20]]
  diversity = len(set(top_20_keys))
  score += diversity * 0.5

  return float(score)


@funsearch.evolve
def priority(features: dict) -> float:
  """ECDLP attack heuristic - find patterns that reveal the private key.

  You have 80+ mathematical features. Your mission: discover which combinations
  correlate with proximity to the real Puzzle 135 solution.

  Feature categories available:
  - Key properties: k_value, k_bitlength, k_popcount, k_mod_small_primes
  - Coordinate patterns: x_hamming, y_hamming, xy_combined_hamming
  - Negation: y_negation_match, x_match, mirror_k_distance
  - Modular: x_mod_256, x_mod_65536, x_ratio
  - Jacobi: jacobi_px, jacobi_qx, jacobi_match
  - Baby-step: x_step_pattern, coord_alternating
  - Distinguished points: px_leading_zeros, distinguished_diff
  - Digit sums: digit_sum_diff
  - Bit positions: x_low_64_xor, x_high_64_xor, bit_pattern_match
  - Geometric: slope_approx, delta_x, delta_y
  - Statistical: px_transitions, transition_similarity
  - Parity: x_parity_match, both_parity_match
  - GCD: gcd_x, gcd_y, gcd_large
  - GLV/Frobenius: glv_k_mod, glv_pattern
  - Collision: collision_indicator, birthday_hash
  - Lattice: magnitude_ratio, coord_magnitude
  - Range: range_position, in_puzzle_range

  FIND THE PATTERN. BREAK THE CODE.
  """

  # Baseline: coordinate similarity
  score = float(features['xy_combined_hamming'])

  # Negation check is critical in ECC
  if features['y_negation_match']:
    score += 1000.0

  # Parity matching
  score += features['both_parity_match'] * 5.0

  # Jacobi symbol matching
  score += features['jacobi_match'] * 3.0

  # Bit pattern matching
  score += features['bit_pattern_match'] * 0.1

  return score
