"""Bitcoin Puzzle Private Key Finder

Return code for Python method priority_vX, where every iteration of priority_vX improves on previous iterations.

DIRECTIONS TO EXPLORE (non-exhaustive):
- Modular arithmetic (mod small primes, mod curve order N, CRT)
- Bit patterns (popcount, runs, transitions, density, palindromes)
- Hex patterns (nibble distributions, repeated sequences, symmetry)
- Algebraic properties (smoothness, quadratic residues, Legendre symbol)
- Relationships (k vs chunk_start, k vs range boundaries, k vs N)
- ECC geometry (how k*G relates to wild kangaroo paths on curve)
- Number theory (continued fractions, convergents, Farey sequences)
- Dynamical systems (orbits, fixed points, attractors in the walk)
- Information theory (entropy, compressibility, Kolmogorov complexity)
- Spectral analysis (Fourier intuition, periodicity, harmonics)
- Topology (shape of the data, clustering, manifold structure)
- Statistical patterns (distribution tails, outlier structure)


Use two spaces as indentation level.
"""
import math
import numpy as np
from typing import Tuple, List
import funsearch

# ============================================================================
# DATASET
# Format: (puzzle_number, private_key)
# ============================================================================
SOLVED_KEYS = [
  (1, 0x1), (2, 0x3), (3, 0x7), (4, 0x8), (5, 0x15), (6, 0x31), (7, 0x4c),
  (8, 0xe0), (9, 0x1d3), (10, 0x202), (11, 0x483), (12, 0xa7b), (13, 0x1460),
  (14, 0x2930), (15, 0x68f3), (16, 0xc936), (17, 0x1764f), (18, 0x3080d),
  (19, 0x5749f), (20, 0xd2c55), (21, 0x1ba534), (22, 0x2de40f), (23, 0x556e52),
  (24, 0xdc2a04), (25, 0x1fa5ee5), (26, 0x340326e), (27, 0x6ac3875),
  (28, 0xd916ce8), (29, 0x17e2551e), (30, 0x3d94cd64), (31, 0x7d4fe747),
  (32, 0xb862a62e), (33, 0x1a96ca8d8), (34, 0x34a65911d), (35, 0x4aed21170),
  (36, 0x9de820a7c), (37, 0x1757756a93), (38, 0x22382facd0), (39, 0x4b5f8303e9),
  (40, 0xe9ae4933d6), (41, 0x153869acc5b), (42, 0x2a221c58d8f),
  (43, 0x6bd3b27c591), (44, 0xe02b35a358f), (45, 0x122fca143c05),
  (46, 0x2ec18388d544), (47, 0x6cd610b53cba), (48, 0xade6d7ce3b9b),
  (49, 0x174176b015f4d), (50, 0x22bd43c2e9354), (51, 0x75070a1a009d4),
  (52, 0xefae164cb9e3c), (53, 0x180788e47e326c), (54, 0x236fb6d5ad1f43),
  (55, 0x6abe1f9b67e114), (56, 0x9d18b63ac4ffdf), (57, 0x1eb25c90795d61c),
  (58, 0x2c675b852189a21), (59, 0x7496cbb87cab44f), (60, 0xfc07a1825367bbe),
  (61, 0x13c96a3742f64906), (62, 0x363d541eb611abee), (63, 0x7cce5efdaccf6808),
  (64, 0xf7051f27b09112d4), (65, 0x1a838b13505b26867), (66, 0x2832ed74f2b5e35ee),
  (67, 0x730fc235c1942c1ae), (68, 0xbebb3940cd0fc1491), (69, 0x101d83275fb2bc7e0c),
  (70, 0x349b84b6431a6c4ef1), (75, 0x4c5ce114686a1336e07),
  (80, 0xea1a5c66dcc11b5ad180), (85, 0x11720c4f018d51b8cebba8),
  (90, 0x2ce00bb2136a445c71e85bf), (95, 0x527a792b183c7f64a0e8b1f4),
  (100, 0xaf55fc59c335c8ec67ed24826), (105, 0x16f14fc2054cd87ee6396b33df3),
  (110, 0x35c0d7234df7deb0f20cf7062444), (115, 0x60f4d11574f5deee49961d9609ac6),
  (120, 0xb10f22572c497a836ea187f2e1fc23),
  (125, 0x1c533b6bb7f0804e09960225e44877ac),
  (130, 0x33e7665705359f04f28b88cf897c603c9),
]

def get_training_data(target_n: int) -> List[Tuple[int, int]]:
  """Returns puzzles < target_n. No leakage."""
  return [(n, k) for n, k in SOLVED_KEYS if n < target_n]

def get_puzzle_range(n: int) -> Tuple[int, int]:
  """Range [2^(n-1), 2^n - 1]."""
  return (1 << (n - 1)), ((1 << n) - 1)

def bits_to_int(bits: Tuple[int, ...]) -> int:
  r = 0
  for b in bits:
    r = (r << 1) | b
  return r

# ============================================================================
# FUNSEARCH ENTRY POINT
# ============================================================================

@funsearch.run
def evaluate(dummy: int) -> float:
  """Evaluates priority function across all puzzles. Returns score [0, 100]."""
  total_score = 0.0
  count = 0
  
  # Evaluate on every puzzle in the dataset
  for puzzle_n, solution in SOLVED_KEYS:
    try:
      guess = solve(puzzle_n)
      total_score += score_single(puzzle_n, guess, solution)
    except Exception:
      # If solve crashes entirely (rare), we give 0 for this puzzle
      pass
    count += 1
    
  # Avoid ZeroDivisionError if dataset is empty
  return total_score / count if count > 0 else 0.0

def score_single(n: int, guess: int, solution: int) -> float:
  """Score single puzzle [0, 100]."""
  if guess == solution:
    return 100.0

  low, high = get_puzzle_range(n)
  
  # Bit accuracy: Leading bits matched (weight 40)
  xor = guess ^ solution
  bits_matched = max(0, n - xor.bit_length()) if xor else n
  
  # Safety: n cannot be 0 here (puzzles start at 1), but good practice
  safe_n = n if n > 0 else 1
  bit_score = (bits_matched / safe_n) ** 1.5 * 40

  # Log proximity: Distance in value space (weight 60)
  distance = abs(guess - solution)
  log_prox = 0.0
  if distance > 0:
    log_prox = max(0.0, 1.0 - math.log2(1 + distance) / safe_n)
  prox_score = (log_prox ** 2) * 60

  return min(100.0, bit_score + prox_score)

# ============================================================================
# SOLVER (Bit-by-Bit Beam Search with Guardrails)
# ============================================================================

def solve(n: int) -> int:
  """
  Constructs the key bit-by-bit using the priority function.
  Acts as a 'Sandboxed Harness' that prevents the LLM from crashing the runner.
  """
  # 1. Initialize safe defaults
  if n < 1: 
    return 0
  low, _ = get_puzzle_range(n)
  
  # 2. Prepare Data
  # We handle the empty check HERE so priority() doesn't have to.
  training_data = get_training_data(n)
  if not training_data:
    # Fallback: if no training data, return lowest in range
    return low

  # 3. Beam Search Setup
  beam_width = 8 
  candidates = [(1,)] # MSB is always 1

  # 4. Iterative Construction
  for _ in range(n - 1):
    next_candidates = []
    
    for bits in candidates:
      next_candidates.append(bits + (0,))
      next_candidates.append(bits + (1,))
    
    scored_candidates = []
    for cand in next_candidates:
      # --- SANDBOX START ---
      try:
        # We pass only valid, pre-checked data to the LLM code
        p_score = priority(cand, n, training_data)
        
        # Guard against invalid return types (NaN, Inf, None)
        if not isinstance(p_score, (int, float)) or math.isnan(p_score) or math.isinf(p_score):
          p_score = -1e9
      except Exception:
        # If the LLM code divides by zero or crashes, we penalize it without killing the script
        p_score = -1e9
      # --- SANDBOX END ---
      
      scored_candidates.append((p_score, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored_candidates[:beam_width]]

  if not candidates:
    return low

  return bits_to_int(candidates[0])

# ============================================================================
# EVOLVABLE FUNCTION
# ============================================================================

@funsearch.evolve
def priority(partial_key: Tuple[int, ...], n: int, training_data: List[Tuple[int, int]]) -> float:
  """
  Assigns a priority score to a partial key candidate.
  The goal is to find a pattern in how keys are distributed. The key was generated for each puzzle to be within its range  puzzle n: [2^(n-1), 2^n - 1]

  You are allowed to use any known or unknown method to find the key

  Args:
    partial_key: Tuple of bits (0 or 1) representing the MSB-first prefix.
    n: The target puzzle number.

  Returns:
    float: Priority score. Higher is better.
  """
  
  return 0.0