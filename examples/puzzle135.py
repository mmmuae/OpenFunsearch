"""
Puzzle 135 Kangaroo Gap Analysis - Discover why some chunks have lower gaps.

OBJECTIVE: Find patterns that predict which chunks have "FIT" (small) gaps.
Hypothesis: Chunks with FIT gaps may be closer to the actual private key.

PUZZLE 135 INFO:
  Range: 0x6000000000000000000000000000000000 - 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
  Public key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
  Bit length: 135

OBSERVED DATA (32 chunks, each ~42.5 trillion keys):
=====================================================
chunk | status | rel_pos | pop | dens  | rough | mod3 | mod7 | score | k_est_hex_suffix
------|--------|---------|-----|-------|-------|------|------|-------|------------------
   0  | LARGE  |  0.636  |  67 | 0.496 | 43.2  |   1  |   3  |  -1.0 | ...D5522ECE
   1  | LARGE  |  0.586  |  73 | 0.541 | 31.5  |   2  |   0  |  -2.0 | ...6F215FB
   2  | LARGE  |  0.379  |  65 | 0.481 | 49.0  |   2  |   3  |  -1.5 | ...995EF2C
   3  | FIT    |  0.310  |  81 | 0.600 | 31.8  |   1  |   3  |   0.5 | ...B77E3FD
   4  | FIT    |  0.067  |  60 | 0.444 | 33.8  |   1  |   5  |   0.5 | ...91BC131
   8  | FIT    |  0.864  |  63 | 0.467 | 29.6  |   0  |   3  |   2.0 | ...240A80
   9  | FIT    |  0.771  |  65 | 0.481 | 38.6  |   0  |   5  |   1.5 | ...4DC9E2
  10  | FIT    |  0.118  |  68 | 0.504 | 49.9  |   0  |   5  |   0.5 | ...D3AD8C
  18  | FIT    |  0.942  |  72 | 0.533 | 35.9  |   0  |   5  |   0.5 | ...D84F49
  19  | FIT    |  0.247  |  68 | 0.504 | 36.4  |   0  |   2  |   0.5 | ...D5B87A
  20  | FIT    |  0.422  |  67 | 0.496 | 31.5  |   1  |   4  |   0.5 | ...9F91B3
  22  | FIT    |  0.259  |  68 | 0.504 | 38.2  |   0  |   2  |   1.5 | ...E95AFB
  26  | FIT    |  0.685  |  68 | 0.504 | 49.0  |   0  |   5  |   0.5 | ...E74D63
  28  | FIT    |  0.114  |  63 | 0.467 | 37.3  |   0  |   5  |   1.5 | ...417F10

PATTERNS OBSERVED:
- FIT chunks cluster: [3,4], [8,9,10], [18,19,20], [22], [26], [28]
- FIT chunks often have mod3=0 (8/11 = 73%)
- FIT chunks often have mod7 in {2,3,5}
- Hamming distance between consecutive FIT k_est values tends lower (54-66)
- FIT chunks 8,9 have more low nibbles (20 vs ~16 average)

FEATURES TO EXPLORE:
- Hamming distance patterns between k_est values
- XOR relationships with chunk boundaries
- Nibble distribution (low 0-7 vs high 8-F)
- Bit run lengths and entropy
- Modular arithmetic residues
- Distance from "golden ratio" positions
- Correlation with previous chunk's k_est
"""

import math
import funsearch

# Raw chunk data: (chunk_idx, start_hex, k_est_hex, rel_pos, popcount, density, 
#                  roughness, mod3, mod7, mod11, mod13, score, is_fit)
CHUNKS = [
    (0,  "6000000000000000000000000000000000", "601457C8D9C990C2B195F723F6D5522ECE", 0.636, 67, 0.496, 43.2, 1, 3, 1, 8, -1.0, False),
    (1,  "6020000000000000000000000000000000", "6032BD63E7A446E99656379BC8E6F215FB", 0.586, 73, 0.541, 31.5, 2, 0, 8, 6, -2.0, False),
    (2,  "6040000000000000000000000000000000", "604C1F839E88CCA8FD29335D061995EF2C", 0.379, 65, 0.481, 49.0, 2, 3, 5, 4, -1.5, False),
    (3,  "6060000000000000000000000000000000", "6069E8F5CEC79C73A699CD33B76B77E3FD", 0.310, 81, 0.600, 31.8, 1, 3, 10, 1, 0.5, True),
    (4,  "6080000000000000000000000000000000", "6082265918495394BA71E6A8ECE91BC131", 0.067, 60, 0.444, 33.8, 1, 5, 1, 12, 0.5, True),
    (5,  "60A0000000000000000000000000000000", "60B8C8483146D4F1D110C4C19516A7C60A", 0.774, 55, 0.407, 52.8, 1, 4, 3, 0, -1.0, False),
    (6,  "60C0000000000000000000000000000000", "60D79A0B3939F58B63AEA7E5836B2A387B", 0.738, 72, 0.533, 40.9, 2, 6, 6, 11, -1.5, False),
    (7,  "60E0000000000000000000000000000000", "60E2ED70CDF68A51360F532985D7AB2508", 0.091, 65, 0.481, 48.8, 1, 2, 7, 5, -1.0, False),
    (8,  "6100000000000000000000000000000000", "611BA7833FB7379C657E151418B6240A80", 0.864, 63, 0.467, 29.6, 0, 3, 2, 1, 2.0, True),
    (9,  "6120000000000000000000000000000000", "6138ADF02235F435F5779460605C4DC9E2", 0.771, 65, 0.481, 38.6, 0, 5, 2, 9, 1.5, True),
    (10, "6140000000000000000000000000000000", "614782CABF74B61C08A2DF1D12EAD3AD8C", 0.118, 68, 0.504, 49.9, 0, 5, 7, 2, 0.5, True),
    (11, "6160000000000000000000000000000000", "61681B06E6A2F38FF51F4C6C0C24E93AF8", 0.252, 62, 0.459, 45.4, 2, 2, 7, 9, -1.5, False),
    (12, "6180000000000000000000000000000000", "619F6CEE3D60A3F2B3E79C58768A89DC52", 0.776, 66, 0.489, 42.2, 0, 5, 9, 3, -2.0, False),
    (13, "61A0000000000000000000000000000000", "61B8EA0EEE8866A0B2BC91687F6B94FB44", 0.778, 62, 0.459, 44.2, 2, 0, 4, 6, -2.5, False),
    (14, "61C0000000000000000000000000000000", "61C869E3EFC2CE5F9DEF0BB2A8B40C7F9A", 0.263, 70, 0.519, 47.8, 1, 1, 0, 5, -1.0, False),
    (15, "61E0000000000000000000000000000000", "61EEA0E0889F92AEDCAE2C36BE9DEAC60D", 0.456, 65, 0.481, 33.8, 1, 6, 6, 4, -1.5, False),
    (16, "6200000000000000000000000000000000", "6205A5BA26F1AD8C48A1DC54A3A5A4AAF9", 0.176, 60, 0.444, 36.6, 1, 1, 6, 2, -1.0, False),
    (17, "6220000000000000000000000000000000", "6227C9B31E69D7CB1BFE8098B0D3E9A377", 0.243, 69, 0.511, 50.2, 0, 3, 3, 11, -1.0, False),
    (18, "6240000000000000000000000000000000", "625EB0B7D47C91CD4E5A1AB66215D84F49", 0.942, 72, 0.533, 35.9, 0, 5, 2, 12, 0.5, True),
    (19, "6260000000000000000000000000000000", "6267E7C16E54E1DBAB67FE2EF2D1D5B87A", 0.247, 68, 0.504, 36.4, 0, 2, 6, 10, 0.5, True),
    (20, "6280000000000000000000000000000000", "628D7B85E5D1E1BA3F0B0636C6B19F91B3", 0.422, 67, 0.496, 31.5, 1, 4, 3, 8, 0.5, True),
    (21, "62A0000000000000000000000000000000", "62A18DB45307D05136D9FD4563BD2852C7", 0.049, 66, 0.489, 35.6, 1, 0, 1, 1, -2.0, False),
    (22, "62C0000000000000000000000000000000", "62C84A8C64B7B2958B0C513EED92E95AFB", 0.259, 68, 0.504, 38.2, 0, 2, 9, 6, 1.5, True),
    (23, "62E0000000000000000000000000000000", "62EE31863DA5E96EFE02FBEA77B723FC07", 0.444, 78, 0.578, 45.0, 1, 6, 5, 8, -1.0, False),
    (24, "6300000000000000000000000000000000", "631898EA617332863496B29060B2A5E220", 0.769, 56, 0.415, 34.0, 2, 6, 2, 4, 0.0, False),
    (25, "6320000000000000000000000000000000", "632D9B5D2A41126E868AAB2B3F602FE25E", 0.425, 67, 0.496, 45.6, 1, 4, 0, 5, -1.0, False),
    (26, "6340000000000000000000000000000000", "6355E9CE383FC243E529116933D0E74D63", 0.685, 68, 0.504, 49.0, 0, 5, 0, 1, 0.5, True),
    (27, "6360000000000000000000000000000000", "6365BCC5D8ADF81894FEEB4E838FBF31EC", 0.179, 76, 0.563, 34.9, 2, 4, 3, 1, 0.0, False),
    (28, "6380000000000000000000000000000000", "6383A92DF46275F82368108DE593417F10", 0.114, 63, 0.467, 37.3, 0, 5, 6, 1, 1.5, True),
    (29, "63A0000000000000000000000000000000", "63A5EF33262CFBD4D576BA07E336470B5F", 0.185, 76, 0.563, 41.7, 2, 5, 7, 3, -1.5, False),
    (30, "63C0000000000000000000000000000000", "63DBCD291CA3A5CD555967BE80231F0898", 0.869, 66, 0.489, 41.2, 2, 3, 7, 7, -1.5, False),
    (31, "63E0000000000000000000000000000000", "63E1BB72D65B70B9A620082C90755EBDA4", 0.054, 66, 0.489, 41.1, 1, 0, 4, 3, -3.0, False),
]


def hex_to_int(h):
  """Convert hex string to int."""
  return int(h, 16)


def hamming_distance(a, b):
  """Count differing bits between two integers."""
  return bin(a ^ b).count('1')


def count_nibbles(n, low=True):
  """Count nibbles in range 0-7 (low=True) or 8-F (low=False)."""
  h = hex(n)[2:].upper()
  if low:
    return sum(1 for c in h if c in '01234567')
  return sum(1 for c in h if c in '89ABCDEF')


def bit_runs(n):
  """Count number of runs (alternations) in binary representation."""
  b = bin(n)[2:]
  runs = 1
  for i in range(1, len(b)):
    if b[i] != b[i-1]:
      runs += 1
  return runs


def compute_features(chunk_idx, chunks_data):
  """Compute rich feature set for a chunk."""
  chunk = chunks_data[chunk_idx]
  start = hex_to_int(chunk[1])
  k_est = hex_to_int(chunk[2])
  
  features = {
      # Basic from data
      "idx": chunk_idx,
      "rel_pos": chunk[3],
      "popcount": chunk[4],
      "density": chunk[5],
      "roughness": chunk[6],
      "mod3": chunk[7],
      "mod7": chunk[8],
      "mod11": chunk[9],
      "mod13": chunk[10],
      "composite_score": chunk[11],
      
      # Derived hex/bit features
      "k_est": k_est,
      "xor_with_start": start ^ k_est,
      "xor_popcount": bin(start ^ k_est).count('1'),
      "low_nibbles": count_nibbles(k_est, low=True),
      "high_nibbles": count_nibbles(k_est, low=False),
      "bit_runs": bit_runs(k_est),
      
      # Low bits analysis
      "low_16_bits": k_est & 0xFFFF,
      "low_32_bits": k_est & 0xFFFFFFFF,
      "low_8_bits": k_est & 0xFF,
      
      # Modular features
      "k_mod_17": k_est % 17,
      "k_mod_31": k_est % 31,
      "k_mod_127": k_est % 127,
      "k_mod_257": k_est % 257,
      
      # Position features
      "idx_mod_4": chunk_idx % 4,
      "idx_mod_8": chunk_idx % 8,
      "is_even_chunk": chunk_idx % 2 == 0,
  }
  
  # Features from neighboring chunks
  if chunk_idx > 0:
    prev_k = hex_to_int(chunks_data[chunk_idx - 1][2])
    features["hamming_to_prev"] = hamming_distance(k_est, prev_k)
    features["diff_to_prev"] = k_est - prev_k
  else:
    features["hamming_to_prev"] = 0
    features["diff_to_prev"] = 0
    
  if chunk_idx < len(chunks_data) - 1:
    next_k = hex_to_int(chunks_data[chunk_idx + 1][2])
    features["hamming_to_next"] = hamming_distance(k_est, next_k)
  else:
    features["hamming_to_next"] = 0
    
  # Cluster detection: are neighbors also FIT?
  neighbors_fit = 0
  if chunk_idx > 0 and chunks_data[chunk_idx - 1][12]:
    neighbors_fit += 1
  if chunk_idx < len(chunks_data) - 1 and chunks_data[chunk_idx + 1][12]:
    neighbors_fit += 1
  features["neighbors_fit_count"] = neighbors_fit
  
  return features


@funsearch.run
def evaluate(unused: int) -> float:
  """Score the priority function on predicting FIT vs LARGE chunks."""
  total_score = 0.0
  correct = 0
  total = 0
  
  for idx in range(len(CHUNKS)):
    features = compute_features(idx, CHUNKS)
    actual_fit = CHUNKS[idx][12]
    
    # Get prediction (should return probability of FIT)
    pred_score = priority(features)
    
    # Validate
    if not isinstance(pred_score, (int, float)) or not math.isfinite(pred_score):
      total_score -= 10
      total += 1
      continue
    
    pred_score = max(0.0, min(1.0, float(pred_score)))
    predicted_fit = pred_score > 0.5
    
    # Scoring
    if predicted_fit == actual_fit:
      correct += 1
      # Bonus for confidence
      if actual_fit:
        total_score += 10 + 20 * pred_score  # Reward high confidence for FIT
      else:
        total_score += 10 + 20 * (1 - pred_score)  # Reward low score for LARGE
    else:
      # Penalty proportional to confidence in wrong answer
      if actual_fit:
        total_score -= 5 + 10 * (1 - pred_score)  # Missed a FIT
      else:
        total_score -= 5 + 10 * pred_score  # False positive
    
    total += 1
  
  # Accuracy bonus
  accuracy = correct / total if total > 0 else 0
  total_score += 50 * accuracy
  
  # Special bonus for finding patterns in FIT chunks
  fit_indices = [i for i in range(len(CHUNKS)) if CHUNKS[i][12]]
  fit_predictions = []
  for idx in fit_indices:
    features = compute_features(idx, CHUNKS)
    fit_predictions.append(priority(features))
  
  if fit_predictions:
    avg_fit_pred = sum(fit_predictions) / len(fit_predictions)
    if avg_fit_pred > 0.6:
      total_score += 30  # Good at identifying FIT chunks
  
  return total_score


@funsearch.evolve
def priority(features: dict) -> float:
  """Predict probability that this chunk has a FIT (small) gap.

  GOAL: Discover what makes some chunks have smaller gaps!
  This could reveal where the actual private key is hiding.

  Available features:
    Basic: idx, rel_pos, popcount, density, roughness, mod3, mod7, mod11, mod13
    Hex: k_est, xor_with_start, xor_popcount, low_nibbles, high_nibbles, bit_runs
    Low bits: low_16_bits, low_32_bits, low_8_bits
    Modular: k_mod_17, k_mod_31, k_mod_127, k_mod_257
    Position: idx_mod_4, idx_mod_8, is_even_chunk
    Neighbors: hamming_to_prev, hamming_to_next, diff_to_prev, neighbors_fit_count

  KNOWN FIT PATTERNS:
    - FIT chunks: [3,4], [8,9,10], [18,19,20], [22], [26], [28]
    - 73% of FIT have mod3 == 0
    - FIT often have mod7 in {2, 3, 5}
    - FIT tend to cluster (neighbors_fit_count > 0)
    - FIT chunks 8,9 have more low_nibbles (~20 vs ~16)
    - Lower hamming_to_prev correlates with FIT (54-66 range)

  Returns:
    float in [0, 1]: probability of being FIT (higher = more likely FIT)
  """
  # Simple baseline using observed patterns
  score = 0.3  # Base probability
  
  # mod3 == 0 is strong FIT indicator
  if features["mod3"] == 0:
    score += 0.25
  
  # mod7 in {2, 3, 5}
  if features["mod7"] in (2, 3, 5):
    score += 0.15
  
  # Clustering bonus
  if features["neighbors_fit_count"] > 0:
    score += 0.1
  
  # composite_score from original data
  if features["composite_score"] > 0:
    score += 0.1
  
  return min(1.0, max(0.0, score))
