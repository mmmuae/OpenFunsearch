import math
import numpy as np
from typing import Tuple, List

# ============================================================================
# DATASET: THE COMPLETE FOUNDATIONAL KNOWLEDGE (Puzzles 1-130)
# ============================================================================
SOLVED_DATA = [
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

# --- SECP256K1 PARAMETERS ---
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

# ============================================================================
# CRYPTOGRAPHIC UTILITIES
# ============================================================================

def inverse(a, m=P):
  return pow(a, m - 2, m)

def tonelli_shanks(n, p):
  if pow(n, (p - 1) // 2, p) != 1: return None
  if p % 4 == 3: return pow(n, (p + 1) // 4, p)
  s, q = 0, p - 1
  while q % 2 == 0: s += 1; q //= 2
  z = 2
  while pow(z, (p - 1) // 2, p) != p - 1: z += 1
  c, r, t, m = pow(z, q, p), pow(n, (q + 1) // 2, p), pow(n, q, p), s
  while t != 1:
    i, temp = 1, pow(t, 2, p)
    while temp != 1: i += 1; temp = pow(temp, 2, p)
    b = pow(c, 2 ** (m - i - 1), p)
    m, c, t, r = i, pow(b, 2, p), (t * pow(b, 2, p)) % p, (r * b) % p
  return r

def uncompress_pubkey(pub_hex: str) -> Tuple[int, int]:
  pub_hex = pub_hex.strip()
  x = int(pub_hex[2:66] if pub_hex.startswith("04") else pub_hex[2:], 16)
  y_sq = (pow(x, 3, P) + 7) % P
  y = tonelli_shanks(y_sq, P)
  if pub_hex.startswith(("02", "03")):
    if (pub_hex.startswith("02") and y % 2 != 0) or (pub_hex.startswith("03") and y % 2 == 0):
      y = P - y
  return (x, y)

def point_mul(k):
  """Fast Scalar Multiplication."""
  res = None; temp = G
  while k:
    if k & 1:
      if res is None: res = temp
      else:
        m = (temp[1] - res[1]) * inverse(temp[0] - res[0])
        nx = (m*m - res[0] - temp[0]) % P
        ny = (m*(res[0] - nx) - res[1]) % P
        res = (nx, ny)
    # Double
    m = (3 * temp[0] * temp[0]) * inverse(2 * temp[1])
    nx = (m*m - 2*temp[0]) % P
    ny = (m*(temp[0] - nx) - temp[1]) % P
    temp = (nx, ny)
    k >>= 1
  return res

# ============================================================================
# LATTICE & MANIFOLD LOGIC
# ============================================================================

def solve_forensic(target_n: int, pub_hex: str):
  print(f"\n\033[94m[*] Initializing Forensic Manifold for Puzzle #{target_n}...")
  
  # 1. Point Recovery
  target_pt = uncompress_pubkey(pub_hex)
  
  # 2. Log-Offset Manifold Regression (Causal: Only use keys < target_n)
  training = [k for k in SOLVED_DATA if k[0] < target_n]
  if not training: training = SOLVED_DATA
  
  lors = [math.log2(max(1, tk - (1 << (tn-1)) + 1)) / tn for tn, tk in training]
  ns = [tn for tn, tk in training]
  
  # Use a Localized Gaussian Weighting to predict the LOR for target_n
  weights = np.array([math.exp(-0.01 * (tn - target_n)**2) for tn in ns])
  X = np.column_stack([np.array(ns)**2, np.array(ns), np.ones(len(ns))])
  W = np.diag(weights)
  
  # Solve Weighted Least Squares
  try:
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ lors)
    predicted_lor = theta[0]*(target_n**2) + theta[1]*target_n + theta[2]
  except:
    predicted_lor = np.average(lors, weights=weights)

  # 3. Calculate Archetype Key
  low_range = 1 << (target_n - 1)
  offset = int(2 ** (target_n * predicted_lor))
  predicted_k = low_range + offset
  
  # 4. Point Comparison (The "Truth" Check)
  pred_pt = point_mul(predicted_k)
  # Measure the "Field Distance" in bit-strength
  dist_x = abs(pred_pt[0] - target_pt[0])
  
  # 5. Output Forensic Report
  print("-" * 60)
  print(f"\033[92m[+] ANALYSIS COMPLETE FOR PUZZLE {target_n}\033[0m")
  print("-" * 60)
  print(f"Predicted LOR (Log-Offset Ratio): {predicted_lor:.6f}")
  print(f"Predicted Archetype (Hex):        {hex(predicted_k).upper()}")
  
  # Structural Analysis
  match = [k for n, k in SOLVED_DATA if n == target_n]
  if match:
    xor = predicted_k ^ match[0]
    acc = target_n - bin(xor).count('1')
    leading = target_n - xor.bit_length() if xor != 0 else target_n
    print(f"\033[93m[*] Historical Accuracy:\033[0m")
    print(f"    - Leading Bits:  {leading}")
    print(f"    - Bit Accuracy:  {acc}/{target_n} ({(acc/target_n)*100:.2f}%)")

  print(f"\n\033[1;94m[!] RECOMMENDED RANGE SEARCH (FOR BITCRACK/KANGAROO):\033[0m")
  # We calculate the search range as +/- 2^20 from the archetype
  # which covers the local manifold variance.
  r_min = max(low_range, predicted_k - (1 << 32))
  r_max = min((1 << target_n) - 1, predicted_k + (1 << 32))
  print(f"Range: {hex(r_min)}:{hex(r_max)}")
  print("-" * 60)

if __name__ == "__main__":
  import sys
  try:
    p_num = int(input("[?] Puzzle Number: "))
    p_key = input("[?] Public Key: ")
    solve_forensic(p_num, p_key)
  except Exception as e:
    print(f"[Error] {e}")