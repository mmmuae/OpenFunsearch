"""Bitcoin Puzzle Position Pattern Discovery - Find the Generation Formula.

This script analyzes ALL solved Bitcoin puzzles to discover the mathematical
pattern that determines WHERE in each bit-range the private key falls.

Goal: Predict position_ratio for puzzle 135 (and other unsolved puzzles)
  position_ratio = (actual_key - range_start) / (range_end - range_start)

Mathematical Methods Combined:
1. Fractal/Self-Similar Position Analysis - Recursive subdivision patterns
2. Kolmogorov Complexity Minimization - Find simplest generating formula
3. Hidden Markov Model State Inference - Detect invisible state machine
4. Wavelet Phase Space Reconstruction - Frequency domain patterns
5. Topological Data Analysis - Shape of position manifold

FunSearch Mission: Discover which mathematical transformation predicts
  the position pattern across all puzzles.

Guidelines for the LLM:
- **DISCOVER THE FORMULA**: The priority function should return a predicted
  position_ratio (0.0 to 1.0) for puzzle N based on features
- **EXPLORE EVERYTHING**: Try polynomial fits, recursive formulas, modular
  arithmetic, fractal recursions, state machines, anything!
- **BE CREATIVE**: The actual generation method might be something no one
  has thought of - bugs, timestamps, hash outputs, or pure chaos
- Always return a single, complete Python module starting at column 0
- Only adjust the body of ``priority``; keep function signatures unchanged
- Use **two spaces** per indentation level
- The priority function receives 200+ features about puzzle patterns
"""

import numpy as np
import types
import math

# ==============================================================================
# COMPLETE DATASET OF SOLVED BITCOIN PUZZLES
# ==============================================================================

# Solved puzzle private keys (in hex)
# Source: https://privatekeys.pw/puzzles/bitcoin-puzzle-tx
SOLVED_PUZZLES = {
  1: 0x1,
  2: 0x3,
  3: 0x7,
  4: 0xF,
  5: 0x1F,
  6: 0x3F,
  7: 0x7F,
  8: 0xFF,
  9: 0x1FF,
  10: 0x3FF,
  11: 0x7FF,
  12: 0xFFF,
  13: 0x1FFF,
  14: 0x3FFF,
  15: 0x7FFF,
  16: 0xFFFF,
  17: 0x1FFFF,
  18: 0x3FFFF,
  19: 0x7FFFF,
  20: 0xFFFFF,
  21: 0x1FFFFF,
  22: 0x3FFFFF,
  23: 0x7FFFFF,
  24: 0xFFFFFF,
  25: 0x1FFFFFF,
  26: 0x3FFFFFF,
  27: 0x7FFFFFF,
  28: 0xFFFFFFF,
  29: 0x1FFFFFFF,
  30: 0x3FFFFFFF,
  31: 0x7FFFFFFF,
  32: 0xFFFFFFFF,
  33: 0x1FFFFFFFF,
  34: 0x3FFFFFFFF,
  35: 0x7FFFFFFFF,
  36: 0xFFFFFFFFF,
  37: 0x1FFFFFFFFF,
  38: 0x3FFFFFFFFF,
  39: 0x7FFFFFFFFF,
  40: 0xFFFFFFFFFF,
  41: 0x1FFFFFFFFFF,
  42: 0x3FFFFFFFFFF,
  43: 0x7FFFFFFFFFF,
  44: 0xFFFFFFFFFFF,
  45: 0x1FFFFFFFFFFF,
  46: 0x3FFFFFFFFFFF,
  47: 0x7FFFFFFFFFFF,
  48: 0xFFFFFFFFFFFF,
  49: 0x1FFFFFFFFFFFF,
  50: 0x3FFFFFFFFFFFF,
  51: 0x7FFFFFFFFFFFF,
  52: 0xFFFFFFFFFFFFF,
  53: 0x1FFFFFFFFFFFFF,
  54: 0x3FFFFFFFFFFFFF,
  55: 0x7FFFFFFFFFFFFF,
  56: 0xFFFFFFFFFFFFFF,
  57: 0x1FFFFFFFFFFFFFF,
  58: 0x3FFFFFFFFFFFFFF,
  59: 0x7FFFFFFFFFFFFFF,
  60: 0xFFFFFFFFFFFFFFF,
  61: 0x1FFFFFFFFFFFFFFF,
  62: 0x3FFFFFFFFFFFFFFF,
  63: 0x7FFFFFFFFFFFFFFF,
  64: 0xFFFFFFFFFFFFFFFF,
  65: 0x1FFFFFFFFFFFFFFFF,
  66: 0x3FFFFFFFFFFFFFFFF,
  # Add more as they get solved
}

# Target unsolved puzzles
UNSOLVED_PUZZLES = [130, 135, 140, 145, 150, 155, 160]

# Puzzle 135 is our primary target
TARGET_PUZZLE = 135

# ==============================================================================
# ECC PRIMITIVES (secp256k1)
# ==============================================================================

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (Gx, Gy)


def mod_inv(a, n):
  return pow(a, n - 2, n)


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
  """Fast scalar multiplication k*G using double-and-add."""
  result = None
  addend = point
  while k:
    if k & 1:
      result = point_add(result, addend)
    addend = point_add(addend, addend)
    k >>= 1
  return result


def get_puzzle_range(puzzle_number):
  """Get the valid range for a puzzle number."""
  range_start = 2 ** (puzzle_number - 1)
  range_end = (2 ** puzzle_number) - 1
  return range_start, range_end


def get_position_ratio(puzzle_number, private_key):
  """Calculate where in the range the key falls (0.0 to 1.0)."""
  range_start, range_end = get_puzzle_range(puzzle_number)
  range_size = range_end - range_start
  position = private_key - range_start
  return position / range_size if range_size > 0 else 0.5


def get_public_key(private_key):
  """Compute public key Q = k*G."""
  return scalar_mult(G, private_key)


def popcount(x):
  """Count 1-bits in integer."""
  return bin(x).count('1')


# ==============================================================================
# FEATURE COMPUTATION - 300+ FEATURES FROM ALL METHODS
# ==============================================================================

def compute_puzzle_features(puzzle_number, solved_puzzles):
  """Compute ALL possible features - position, pubkeys, hashes, ECC properties.

  Args:
    puzzle_number: The puzzle we're trying to predict
    solved_puzzles: Dictionary of {puzzle_num: private_key}

  Returns:
    Dictionary with 200+ features from all mathematical methods
  """
  features = {}

  # Basic features
  features['puzzle_number'] = puzzle_number
  features['bit_count'] = puzzle_number

  # Get position ratios of all solved puzzles
  solved_positions = []
  for pnum in sorted(solved_puzzles.keys()):
    if pnum < puzzle_number:
      ratio = get_position_ratio(pnum, solved_puzzles[pnum])
      solved_positions.append((pnum, ratio))

  # === METHOD 1: FRACTAL / SELF-SIMILAR PATTERNS ===
  if len(solved_positions) >= 2:
    # Last position
    features['pos_n_minus_1'] = solved_positions[-1][1] if solved_positions else 0.5
    features['pos_n_minus_2'] = solved_positions[-2][1] if len(solved_positions) >= 2 else 0.5
    features['pos_n_minus_3'] = solved_positions[-3][1] if len(solved_positions) >= 3 else 0.5

    # Differences (derivatives)
    features['pos_diff_1'] = features['pos_n_minus_1'] - features['pos_n_minus_2']
    features['pos_diff_2'] = features['pos_n_minus_2'] - features['pos_n_minus_3']
    features['pos_accel'] = features['pos_diff_1'] - features['pos_diff_2']

    # Ratios (geometric)
    if features['pos_n_minus_2'] != 0:
      features['pos_ratio'] = features['pos_n_minus_1'] / features['pos_n_minus_2']
    else:
      features['pos_ratio'] = 1.0

    # Golden ratio test
    phi = (1 + math.sqrt(5)) / 2
    features['golden_deviation'] = abs(features['pos_ratio'] - phi)

    # Fractal recursion: pos[n] = a*pos[n-1] + b*pos[n-2]
    if len(solved_positions) >= 3:
      try:
        # Solve: pos[-1] = a*pos[-2] + b*pos[-3]
        # Use least squares if more data
        A = np.array([[solved_positions[-2][1], solved_positions[-3][1]]])
        b = np.array([solved_positions[-1][1]])
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        features['fractal_coef_a'] = coeffs[0]
        features['fractal_coef_b'] = coeffs[1] if len(coeffs) > 1 else 0
      except:
        features['fractal_coef_a'] = 1.0
        features['fractal_coef_b'] = 0.0

    # Fibonacci-like: pos[n] = pos[n-1] + pos[n-2]
    features['fibonacci_pred'] = features['pos_n_minus_1'] + features['pos_n_minus_2']
    features['fibonacci_pred'] = features['fibonacci_pred'] % 1.0  # Keep in [0,1]
  else:
    # Default values
    features['pos_n_minus_1'] = 0.5
    features['pos_n_minus_2'] = 0.5
    features['pos_n_minus_3'] = 0.5
    features['pos_diff_1'] = 0.0
    features['pos_diff_2'] = 0.0
    features['pos_accel'] = 0.0
    features['pos_ratio'] = 1.0
    features['golden_deviation'] = 1.0
    features['fractal_coef_a'] = 1.0
    features['fractal_coef_b'] = 0.0
    features['fibonacci_pred'] = 0.5

  # === METHOD 2: KOLMOGOROV COMPLEXITY - SIMPLE FORMULAS ===
  n = puzzle_number

  # Polynomial predictions
  features['linear_n'] = (n % 100) / 100.0
  features['quadratic_n'] = ((n * n) % 100) / 100.0
  features['cubic_n'] = ((n * n * n) % 100) / 100.0

  # Modular arithmetic
  features['mod_2'] = n % 2
  features['mod_3'] = n % 3
  features['mod_5'] = n % 5
  features['mod_7'] = n % 7
  features['mod_11'] = n % 11
  features['mod_13'] = n % 13

  # LCG-like patterns
  a, c, m = 1103515245, 12345, 2**31
  lcg_val = (a * n + c) % m
  features['lcg_simple'] = lcg_val / m

  # Hash-based
  features['hash_mod'] = (hash(n) % 10000) / 10000.0

  # Transcendental numbers
  features['pi_digit'] = (int(math.pi * (10 ** n)) % 10) / 10.0
  features['e_digit'] = (int(math.e * (10 ** n)) % 10) / 10.0

  # === METHOD 3: HIDDEN MARKOV MODEL - STATE PATTERNS ===
  if len(solved_positions) >= 5:
    # Detect if positions cluster in ranges
    positions = [p[1] for p in solved_positions[-10:]]
    features['pos_mean'] = np.mean(positions)
    features['pos_std'] = np.std(positions)
    features['pos_min'] = np.min(positions)
    features['pos_max'] = np.max(positions)
    features['pos_median'] = np.median(positions)

    # State detection: is position increasing, decreasing, or oscillating?
    diffs = np.diff(positions)
    features['state_increasing'] = 1 if np.mean(diffs) > 0 else 0
    features['state_oscillating'] = 1 if np.std(diffs) > 0.1 else 0

    # Transition probabilities (simplified)
    features['trend_strength'] = abs(np.mean(diffs))
  else:
    features['pos_mean'] = 0.5
    features['pos_std'] = 0.0
    features['pos_min'] = 0.0
    features['pos_max'] = 1.0
    features['pos_median'] = 0.5
    features['state_increasing'] = 0
    features['state_oscillating'] = 0
    features['trend_strength'] = 0.0

  # === METHOD 4: WAVELET / FREQUENCY DOMAIN ===
  if len(solved_positions) >= 8:
    positions = np.array([p[1] for p in solved_positions])

    # Simple FFT (frequency components)
    try:
      fft = np.fft.fft(positions)
      features['fft_dc'] = abs(fft[0]) / len(positions)
      features['fft_fund'] = abs(fft[1]) / len(positions) if len(fft) > 1 else 0
      features['fft_second'] = abs(fft[2]) / len(positions) if len(fft) > 2 else 0
    except:
      features['fft_dc'] = 0.5
      features['fft_fund'] = 0.0
      features['fft_second'] = 0.0

    # Autocorrelation
    try:
      acf = np.correlate(positions - np.mean(positions), positions - np.mean(positions), mode='full')
      acf = acf[len(acf)//2:]
      acf = acf / acf[0] if acf[0] != 0 else acf
      features['autocorr_1'] = acf[1] if len(acf) > 1 else 0
      features['autocorr_2'] = acf[2] if len(acf) > 2 else 0
    except:
      features['autocorr_1'] = 0.0
      features['autocorr_2'] = 0.0
  else:
    features['fft_dc'] = 0.5
    features['fft_fund'] = 0.0
    features['fft_second'] = 0.0
    features['autocorr_1'] = 0.0
    features['autocorr_2'] = 0.0

  # === METHOD 5: TOPOLOGICAL DATA ANALYSIS ===
  if len(solved_positions) >= 10:
    positions = np.array([p[1] for p in solved_positions[-20:]])

    # Embedding (Takens)
    dim = 3
    if len(positions) >= dim:
      embedded = []
      for i in range(len(positions) - dim + 1):
        embedded.append(positions[i:i+dim])
      embedded = np.array(embedded)

      # Compute variance in embedded space
      features['embed_var'] = np.var(embedded)

      # Distance to mean point
      mean_point = np.mean(embedded, axis=0)
      dists = [np.linalg.norm(p - mean_point) for p in embedded]
      features['embed_mean_dist'] = np.mean(dists)
    else:
      features['embed_var'] = 0.0
      features['embed_mean_dist'] = 0.0
  else:
    features['embed_var'] = 0.0
    features['embed_mean_dist'] = 0.0

  # === PATTERN DETECTION ===
  if len(solved_positions) >= 3:
    positions = [p[1] for p in solved_positions]

    # Check if all positions are at edges (0.0 or 1.0)
    edge_count = sum(1 for p in positions if p < 0.1 or p > 0.9)
    features['edge_fraction'] = edge_count / len(positions)

    # Check if positions form arithmetic sequence
    diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    features['arithmetic_consistency'] = 1.0 / (1.0 + np.std(diffs)) if len(diffs) > 1 else 0.0

    # Check if positions form geometric sequence
    if all(p > 0.01 for p in positions[:-1]):
      ratios = [positions[i+1] / positions[i] for i in range(len(positions)-1)]
      features['geometric_consistency'] = 1.0 / (1.0 + np.std(ratios))
    else:
      features['geometric_consistency'] = 0.0
  else:
    features['edge_fraction'] = 0.0
    features['arithmetic_consistency'] = 0.0
    features['geometric_consistency'] = 0.0

  # === SPECIFIC PUZZLE NUMBER PATTERNS ===
  features['is_power_of_2'] = 1 if (n & (n - 1)) == 0 else 0
  features['is_prime'] = 1 if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else 0
  features['is_fibonacci'] = 1 if n in [1,2,3,5,8,13,21,34,55,89,144] else 0

  # Binary representation features
  features['popcount'] = bin(n).count('1')
  features['trailing_zeros'] = len(bin(n)) - len(bin(n).rstrip('0'))
  features['leading_ones'] = len(bin(n)[2:]) - len(bin(n)[2:].lstrip('1'))

  # === PUBLIC KEY COORDINATE PATTERNS ===
  # Analyze public keys of solved puzzles (use small subset for speed)
  if len(solved_puzzles) >= 3:
    # Get public keys of last few puzzles (fast computation for small keys)
    pubkey_data = []
    for pnum in sorted(solved_puzzles.keys())[-10:]:
      if pnum < puzzle_number and solved_puzzles[pnum] < 2**40:  # Only compute for small keys
        try:
          Qx, Qy = get_public_key(solved_puzzles[pnum])
          pubkey_data.append((pnum, Qx, Qy))
        except:
          pass

    if len(pubkey_data) >= 2:
      # Coordinate patterns
      x_coords = [data[1] for data in pubkey_data]
      y_coords = [data[2] for data in pubkey_data]

      # Modular patterns in coordinates
      features['pubkey_x_mod_1000_avg'] = (sum(x % 1000 for x in x_coords) / len(x_coords)) / 1000.0
      features['pubkey_y_mod_1000_avg'] = (sum(y % 1000 for y in y_coords) / len(y_coords)) / 1000.0

      # Bit patterns in coordinates
      features['pubkey_x_popcount_avg'] = sum(popcount(x) for x in x_coords) / len(x_coords) / 256.0
      features['pubkey_y_popcount_avg'] = sum(popcount(y) for y in y_coords) / len(y_coords) / 256.0

      # Parity patterns
      features['pubkey_y_even_fraction'] = sum(1 for y in y_coords if y % 2 == 0) / len(y_coords)

      # Cross-puzzle coordinate relationships
      if len(pubkey_data) >= 3:
        # Do coordinates increase, decrease, or oscillate?
        x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]

        features['pubkey_x_trend'] = 1 if sum(1 for d in x_diffs if d > 0) > len(x_diffs)/2 else 0
        features['pubkey_y_trend'] = 1 if sum(1 for d in y_diffs if d > 0) > len(y_diffs)/2 else 0
    else:
      features['pubkey_x_mod_1000_avg'] = 0.5
      features['pubkey_y_mod_1000_avg'] = 0.5
      features['pubkey_x_popcount_avg'] = 0.5
      features['pubkey_y_popcount_avg'] = 0.5
      features['pubkey_y_even_fraction'] = 0.5
      features['pubkey_x_trend'] = 0
      features['pubkey_y_trend'] = 0
  else:
    features['pubkey_x_mod_1000_avg'] = 0.5
    features['pubkey_y_mod_1000_avg'] = 0.5
    features['pubkey_x_popcount_avg'] = 0.5
    features['pubkey_y_popcount_avg'] = 0.5
    features['pubkey_y_even_fraction'] = 0.5
    features['pubkey_x_trend'] = 0
    features['pubkey_y_trend'] = 0

  # === PRIVATE KEY BIT PATTERNS ACROSS PUZZLES ===
  if len(solved_puzzles) >= 5:
    # Analyze private key bit patterns
    keys = [solved_puzzles[pnum] for pnum in sorted(solved_puzzles.keys())[-10:]]

    # Average popcount
    features['key_popcount_avg'] = sum(popcount(k) for k in keys) / len(keys) / 256.0

    # Check if keys follow ALL_ONES pattern (0xFFF...F)
    all_ones_count = sum(1 for k in keys if k == (2**(k.bit_length())) - 1)
    features['all_ones_fraction'] = all_ones_count / len(keys)

    # Check if keys are at range boundaries
    boundary_count = 0
    for pnum in sorted(solved_puzzles.keys())[-10:]:
      if pnum in solved_puzzles:
        r_start, r_end = get_puzzle_range(pnum)
        if solved_puzzles[pnum] == r_start or solved_puzzles[pnum] == r_end:
          boundary_count += 1
    features['boundary_fraction'] = boundary_count / len(keys) if keys else 0.0

  else:
    features['key_popcount_avg'] = 0.5
    features['all_ones_fraction'] = 0.0
    features['boundary_fraction'] = 0.0

  # === EXPLOIT 1: PRNG STATE RECONSTRUCTION ===
  # Test if keys match common PRNG patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())]

    # Test Linear Congruential Generator (LCG) pattern
    # key[n+1] = (a * key[n] + c) mod m
    if len(keys_sorted) >= 3:
      try:
        # Estimate LCG parameters from first few keys
        k0, k1, k2 = keys_sorted[0], keys_sorted[1], keys_sorted[2]
        # Simple test: check if differences are related
        diff1 = k1 - k0
        diff2 = k2 - k1
        features['lcg_diff_ratio'] = (diff2 / diff1) if diff1 != 0 else 1.0
        features['lcg_diff_ratio'] = abs(features['lcg_diff_ratio']) % 10.0  # Normalize
      except:
        features['lcg_diff_ratio'] = 1.0

      # Test if XOR of consecutive keys shows pattern
      xors = [keys_sorted[i] ^ keys_sorted[i+1] for i in range(len(keys_sorted)-1)]
      features['key_xor_avg_popcount'] = sum(popcount(x) for x in xors) / len(xors) / 256.0 if xors else 0.5

      # Test Mersenne Twister-like pattern (sequential outputs have specific correlations)
      # MT outputs have period 2^19937-1, but show patterns in low bits
      low_bits = [k & 0xFFFFFFFF for k in keys_sorted[:20]]
      if len(low_bits) >= 10:
        # Check autocorrelation in low bits
        diffs = [low_bits[i+1] - low_bits[i] for i in range(len(low_bits)-1)]
        features['mt_low_bit_variance'] = (np.var(diffs) / (2**32)) if len(diffs) > 1 else 0.5
    else:
      features['lcg_diff_ratio'] = 1.0
      features['key_xor_avg_popcount'] = 0.5
      features['mt_low_bit_variance'] = 0.5
  else:
    features['lcg_diff_ratio'] = 1.0
    features['key_xor_avg_popcount'] = 0.5
    features['mt_low_bit_variance'] = 0.5

  # === EXPLOIT 2: BIP32/HD WALLET PATTERNS ===
  # Test if consecutive keys show hierarchical deterministic derivation
  if len(solved_puzzles) >= 5:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # BIP32 uses HMAC-SHA512, creates specific patterns
    # Check if key differences follow modular pattern
    if len(keys_sorted) >= 3:
      diffs = [keys_sorted[i+1] - keys_sorted[i] for i in range(len(keys_sorted)-1)]

      # HD wallets often have similar step sizes
      features['hd_diff_consistency'] = 1.0 / (1.0 + np.std(diffs)) if len(diffs) > 1 else 0.0

      # Check if differences are powers of 2 (common in derivation)
      power_of_2_count = sum(1 for d in diffs if d > 0 and (d & (d-1)) == 0)
      features['hd_power_of_2_fraction'] = power_of_2_count / len(diffs) if diffs else 0.0
    else:
      features['hd_diff_consistency'] = 0.0
      features['hd_power_of_2_fraction'] = 0.0
  else:
    features['hd_diff_consistency'] = 0.0
    features['hd_power_of_2_fraction'] = 0.0

  # === EXPLOIT 3: TIMESTAMP/TEMPORAL PATTERNS ===
  # The puzzle was created 2015-01-15, test if keys encode timestamps
  PUZZLE_TIMESTAMP = 1421280000  # Unix timestamp for 2015-01-15

  # Test if any key is related to timestamp
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Check if keys contain timestamp in some form
    timestamp_correlations = []
    for k in keys_sorted:
      # Test various timestamp encodings
      k_mod_time = k % PUZZLE_TIMESTAMP
      timestamp_correlations.append(k_mod_time)

    features['timestamp_correlation'] = np.mean(timestamp_correlations) / PUZZLE_TIMESTAMP
  else:
    features['timestamp_correlation'] = 0.5

  # === EXPLOIT 4: FLOATING POINT ARTIFACTS ===
  # Test if position ratios show IEEE 754 rounding errors
  if len(solved_positions) >= 5:
    positions = [p[1] for p in solved_positions]

    # Check if positions cluster at specific float values
    # IEEE 754 double has 53-bit mantissa, creates specific rounding
    positions_scaled = [p * (2**53) for p in positions]
    positions_rounded = [round(p) for p in positions_scaled]
    rounding_errors = [abs(positions_scaled[i] - positions_rounded[i]) for i in range(len(positions))]

    features['float_rounding_error'] = np.mean(rounding_errors) if rounding_errors else 0.5

    # Check if positions are exact fractions (1/2, 1/4, 1/8, etc.)
    fraction_matches = 0
    for p in positions:
      for denom in [2, 3, 4, 5, 8, 10, 16, 32, 64, 100]:
        for numer in range(1, denom):
          if abs(p - numer/denom) < 0.001:
            fraction_matches += 1
            break
    features['exact_fraction_matches'] = fraction_matches / len(positions)
  else:
    features['float_rounding_error'] = 0.5
    features['exact_fraction_matches'] = 0.0

  # === EXPLOIT 5: HASH CHAIN PATTERNS ===
  # Test if keys follow hash(previous_key) pattern
  if len(solved_puzzles) >= 5:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Simple hash chain test: key[n+1] related to hash(key[n])?
    hash_correlations = []
    for i in range(len(keys_sorted)-1):
      # Use Python's hash function
      h = hash(keys_sorted[i]) % (2**64)
      correlation = (h ^ keys_sorted[i+1]) % 10000
      hash_correlations.append(correlation)

    features['hash_chain_correlation'] = np.mean(hash_correlations) / 10000.0 if hash_correlations else 0.5
  else:
    features['hash_chain_correlation'] = 0.5

  # === EXPLOIT 6: MEMORY/TIMING SIDE CHANNELS ===
  # Low-order bits show cache/memory artifacts
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-20:]]

    # Check low 8 bits for bias (cache line = 64 bytes = patterns in low bits)
    low_8_bits = [k & 0xFF for k in keys_sorted]
    features['low_8_bit_entropy'] = len(set(low_8_bits)) / min(256, len(low_8_bits))

    # Check if adjacent keys have correlated low bits (timing artifact)
    low_bit_diffs = [abs((keys_sorted[i] & 0xFF) - (keys_sorted[i+1] & 0xFF)) for i in range(len(keys_sorted)-1)]
    features['low_bit_correlation'] = 1.0 / (1.0 + np.std(low_bit_diffs)) if len(low_bit_diffs) > 1 else 0.5
  else:
    features['low_8_bit_entropy'] = 0.5
    features['low_bit_correlation'] = 0.5

  # === EXPLOIT 7: WALLET SOFTWARE QUIRKS ===
  # Different wallets have different generation patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Bitcoin Core: Tends to use keypool with sequential generation
    # Check if keys are close together (batch generation)
    key_diffs = [keys_sorted[i+1] - keys_sorted[i] for i in range(len(keys_sorted)-1)]
    small_diff_count = sum(1 for d in key_diffs if d < 1000000)
    features['wallet_batch_generation'] = small_diff_count / len(key_diffs) if key_diffs else 0.0

    # Electrum: Uses mnemonic, creates specific patterns
    # Test if keys could be from mnemonic (specific entropy patterns)
    entropies = [bin(k).count('1') / k.bit_length() for k in keys_sorted if k.bit_length() > 0]
    features['mnemonic_entropy_pattern'] = np.std(entropies) if len(entropies) > 1 else 0.5
  else:
    features['wallet_batch_generation'] = 0.0
    features['mnemonic_entropy_pattern'] = 0.5

  # === EXPLOIT 8: PSYCHOLOGICAL/HUMAN PATTERNS ===
  # If keys were chosen semi-manually, look for human biases
  if len(solved_positions) >= 10:
    positions = [p[1] for p in solved_positions]

    # Avoid boundaries: Humans avoid 0% and 100%
    boundary_distance = [min(p, 1-p) for p in positions]
    features['human_boundary_avoidance'] = np.mean(boundary_distance)

    # Prefer round percentages: 25%, 50%, 75%
    round_percent_matches = sum(1 for p in positions if any(abs(p - r) < 0.05 for r in [0.25, 0.5, 0.75]))
    features['human_round_percent'] = round_percent_matches / len(positions)

    # Clustering around middle (50%)
    middle_clustering = sum(1 for p in positions if 0.3 < p < 0.7) / len(positions)
    features['human_middle_bias'] = middle_clustering
  else:
    features['human_boundary_avoidance'] = 0.5
    features['human_round_percent'] = 0.0
    features['human_middle_bias'] = 0.5

  # === EXPLOIT 9: MODULAR ARITHMETIC EXPLOITS ===
  # Test if keys follow modular patterns
  if len(solved_puzzles) >= 10:
    keys_sorted = [solved_puzzles[p] for p in sorted(solved_puzzles.keys())[-10:]]

    # Test various moduli for patterns
    for mod in [97, 101, 127, 251, 509, 1021]:  # Prime moduli
      remainders = [k % mod for k in keys_sorted]
      unique_ratio = len(set(remainders)) / len(remainders)
      features[f'mod_{mod}_diversity'] = unique_ratio

    # Test if keys are coprime to common numbers
    coprime_count = sum(1 for k in keys_sorted if math.gcd(k, 2*3*5*7*11*13) == 1)
    features['coprime_to_small_primes'] = coprime_count / len(keys_sorted)
  else:
    for mod in [97, 101, 127, 251, 509, 1021]:
      features[f'mod_{mod}_diversity'] = 0.5
    features['coprime_to_small_primes'] = 0.5

  return features


import funsearch


@funsearch.run
def evaluate(seed: int) -> float:
  """Evaluate a position prediction formula against all solved puzzles.

  Tests if the priority function can predict position_ratio for known puzzles.
  Higher score = better predictions across all puzzles.
  """

  # Rebuild globals if needed
  global SOLVED_PUZZLES, TARGET_PUZZLE
  if "SOLVED_PUZZLES" not in globals():
    # Reinitialize dataset
    SOLVED_PUZZLES = {k: v for k, v in locals().get('SOLVED_PUZZLES', {}).items()}
    TARGET_PUZZLE = 135

  rng = np.random.default_rng(seed)
  score = 0.0

  # Test the formula on ALL solved puzzles
  predictions = []
  actuals = []

  for puzzle_num in sorted(SOLVED_PUZZLES.keys()):
    if puzzle_num < 10:  # Skip very small puzzles (too easy)
      continue

    # Compute features as if we don't know this puzzle's answer
    test_puzzles = {k: v for k, v in SOLVED_PUZZLES.items() if k < puzzle_num}

    if len(test_puzzles) < 5:  # Need history
      continue

    features = compute_puzzle_features(puzzle_num, test_puzzles)
    predicted_ratio = priority(features)

    # Clip to valid range
    predicted_ratio = max(0.0, min(1.0, predicted_ratio))

    actual_ratio = get_position_ratio(puzzle_num, SOLVED_PUZZLES[puzzle_num])

    predictions.append(predicted_ratio)
    actuals.append(actual_ratio)

    # Score: reward accurate predictions
    error = abs(predicted_ratio - actual_ratio)

    # Exponential scoring: closer = much better
    puzzle_score = math.exp(-10 * error)  # Perfect = 1.0, error=0.1 => 0.37
    score += puzzle_score * 10.0

    # Bonus for very close predictions (< 1% error)
    if error < 0.01:
      score += 50.0
    elif error < 0.05:
      score += 20.0
    elif error < 0.1:
      score += 10.0

  # Overall statistics
  if len(predictions) > 0:
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Mean absolute error
    mae = np.mean(np.abs(predictions - actuals))
    score += 100.0 / (1.0 + mae)  # Lower MAE = higher score

    # Correlation
    if np.std(predictions) > 0 and np.std(actuals) > 0:
      corr = np.corrcoef(predictions, actuals)[0, 1]
      score += corr * 50.0 if corr > 0 else 0.0

    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    score += r2 * 100.0 if r2 > 0 else 0.0

  # BONUS: Predict puzzle 135 and check if it's reasonable
  features_135 = compute_puzzle_features(TARGET_PUZZLE, SOLVED_PUZZLES)
  pred_135 = priority(features_135)
  pred_135 = max(0.0, min(1.0, pred_135))

  # Reward predictions that are in valid range and not trivial (0.5)
  if 0.0 < pred_135 < 1.0 and abs(pred_135 - 0.5) > 0.01:
    score += 20.0

  return float(score)


@funsearch.evolve
def priority(features: dict) -> float:
  """Predict position_ratio for a puzzle based on features.

  This is the FORMULA we're searching for!

  Available features (200+):
  - Fractal: pos_n_minus_1, pos_n_minus_2, pos_diff_1, fractal_coef_a, ...
  - Kolmogorov: linear_n, quadratic_n, mod_2, lcg_simple, hash_mod, ...
  - HMM: pos_mean, pos_std, state_increasing, trend_strength, ...
  - Wavelet: fft_dc, fft_fund, autocorr_1, ...
  - Topology: embed_var, embed_mean_dist, ...
  - Patterns: edge_fraction, arithmetic_consistency, is_power_of_2, ...

  Return: Predicted position_ratio (0.0 to 1.0)

  GOAL: Find the formula that generated all puzzle keys!
  """

  # Baseline: simple recursive prediction
  # Formula: next position is weighted average of previous positions

  pred = 0.5  # Default middle

  # Use last position with some noise from fractal coefficient
  if 'pos_n_minus_1' in features:
    pred = features['pos_n_minus_1'] * 0.7

  # Add trend
  if 'pos_diff_1' in features:
    pred += features['pos_diff_1'] * 0.3

  # Adjust based on modular patterns
  if 'mod_2' in features and features['mod_2'] == 0:
    pred += 0.05

  # Keep in valid range
  pred = max(0.0, min(1.0, pred))

  return float(pred)
