"""BTC Puzzle 135: Why Do Certain Chunks Produce Low Kangaroo Gaps?

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.

THE MYSTERY:
When running Pollard's Kangaroo on puzzle 135, we scan different chunks
of the keyspace. Some chunks produce extremely low gaps (tame and wild
kangaroos nearly collide), while others produce huge gaps. The difference
spans 26 orders of magnitude. This is NOT random.

What property of of that chunk from the keyspace causes kangaroos to nearly collide within it?

THE DATA:
  - start: chunk starting position
  - k_est: derived from the near-collision (where tame/wild got closest)
  - chunk_bits: size of chunk being scanned (2^64, 2^130, etc.) helps you to know the end of the scan.
  
The cause is somewhere in the chunk, the region's relationship to the
public key, the actual private key location, or curve dynamics, or something not yet discovered.

PUZZLE 135 PARAMETERS:
  Public Key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
  PUBKEY_X = 0x145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
  Range: 0x4000000000000000000000000000000000 to 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF (It's known the key exists in this range and it's 135 bits)
  Curve Order N: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
  Field Prime P: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

DATA STRUCTURE:
  - LOW_RANK: chunks in bottom 15% of lgap WITHIN their chunk_bits group
  - HIGH_RANK: chunks in top 15% of lgap WITHIN their chunk_bits group
  - lgap Normalized by chunk size (apples-to-apples comparison)

YOUR MISSION:
Primary: Discover what makes LOW_RANK chunks produce near-collisions. Secondary : discover a pattern, it might help predict where the actual
private key is located. Low-gap chunks could be "attracted" to the key.

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

"""

import funsearch
import math

# =============================================================================
# PUZZLE 135 CONSTANTS
# =============================================================================

# The public key (compressed, 02 = even y)
PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
PUBKEY_X = 0x145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

# secp256k1 parameters
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# Puzzle 135 range
RANGE_MIN = 0x4000000000000000000000000000000000
RANGE_MAX = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

# =============================================================================
# DATA: Chunks ranked by gap within each chunk_bits group
# Format: (k_est_hex, start_hex, chunk_bits)
#   - start_hex: defines the chunk region (PRIMARY - this is what we're analyzing)
#   - k_est_hex: where near-collision occurred (SECONDARY - useful clue)
#   - chunk_bits: chunk size (for normalization)
# LOW_RANK = chunks with lowest gaps in their size group (INTERESTING)
# HIGH_RANK = chunks with highest gaps in their size group (CONTRAST)
# =============================================================================

LOW_RANK = [
    ("6E2D041B59DE42F57F83E13AAAE88E7A60", "6E2D041B59DE42F57F8300000000000000", 56),
    ("6B08839FC3BC068A3C004B2815B66EA9C3", "6B08839FC3BC068A3C0000000000000000", 60),
    ("609474AA04271A2F70AE6C634142976DB1", "609474AA04271A2F700000000000000000", 64),
    ("528EA7CEDDF6ED3F63E005F06D62CA2CEA", "528EA7CEDDF6ED00000000000000000000", 80),
    ("6355E9CE383FC243E529116933D0E74D63", "6340000000000000000000000000000000", 125),
    ("4EDC8854AA6ED8CBCF21D2D252D431CBDA", "4C00000000000000000000000000000000", 130),
    ("56F20B20021EBCECB5000000007D6674CB", "56F20B20021EBCECB50000000000000000", 64),
    ("5D90FD9D5AF04D31EBEFE46C8E2BB573C3", "5D90FD9D5AF04D31EB0000000000000000", 64),
    ("57A59B74D0B359F34A6C3A3BA7BCBFF16A", "57A59B74D0B359F34A0000000000000000", 64),
    ("4C9A0CFD6D9C09EB529807971117507026", "4C9A0CFD6D9C09EB520000000000000000", 64),
    ("4AA11E389054C5A9A0FFFFFFFCD45715E5", "4AA11E389054C5A9A10000000000000000", 64),
    ("611E98A18172F392F162DF1DC04DB4F60B", "611E98A18172F392F10000000000000000", 64),
    ("74947D9503F8E98B1A00000005231502FE", "74947D9503F8E98B1A0000000000000000", 64),
    ("633C9337F9D6BB7028D5FDA160826BDBD1", "633C9337F9D6BB70280000000000000000", 64),
    ("5726AE3BAC550CE5B600000005D1A92EBA", "5726AE3BAC550CE5B60000000000000000", 64),
    ("5054306B8534BD21FD0CAAB9F42ADFD7D7", "5054306B8534BD21FD0000000000000000", 64),
    ("5ECC6C4875BAC032C6000000097F53A0A1", "5ECC6C4875BAC032C60000000000000000", 64),
    ("5103C48EC3B60824A103CFBA500EA2F62D", "5103C48EC3B60824A10000000000000000", 64),
    ("5117E03FDB30E43680CFA6FAAF546C9248", "5117E03FDB30E436800000000000000000", 64),
    ("4D7D9337CBC657D152FEDCDF1A20737F33", "4D7D9337CBC657D1520000000000000000", 64),
    ("5B60DD42D4C7A4FFFFFFFFF561C6050413", "5B60DD42D4C7A500000000000000000000", 80),
    ("51338E6C9C55209591C32656B0A1B5851C", "51338E6C9C552095910000000000000000", 64),
    ("5FD16BE55D9392ECAAB6B997E23C7BAA9C", "5FD16BE55D9392ECAA0000000000000000", 64),
    ("5AE97CB65E671504A9FFFFFFF34742D32D", "5AE97CB65E671504AA0000000000000000", 64),
    ("566BA29CF83279032534FDAEB493C01046", "566BA29CF8327903250000000000000000", 64),
    ("48FFDA42883B2439A3316431830EB9B85E", "48FFDA42883B2439A30000000000000000", 64),
    ("49774CA503C7A9547B03176EE9F4D82C23", "49774CA503C7A9547B0000000000000000", 64),
    ("64C9CF1872DA06EDF051BB3FFE440CAEEC", "64C9CF1872DA06EDF00000000000000000", 64),
    ("5A2F6C828329E92510E4AAE530FE14A92D", "5A2F6C828329E925100000000000000000", 64),
    ("4B85F1F99068DE58A2FFFFFFEE24700B4D", "4B85F1F99068DE58A30000000000000000", 64),
    ("56047DA71615985FDBFE6663870F5119EE", "5400000000000000000000000000000000", 130),
    ("5F2429A520890AF3BCFFFFFFEDE430C596", "5F2429A520890AF3BD0000000000000000", 64),
    ("46CFB2A4E1B3AB5F52C23C253ADAFBC36B", "46CFB2A4E1B3AB5F520000000000000000", 64),
    ("517CF6A45A0664E7A53BC25C895BD08789", "517CF6A45A0664E7A50000000000000000", 64),
    ("4EF62059C9BD4CC488D2BF9E2EC8E36D74", "4EF62059C9BD4CC4880000000000000000", 64),
    ("5A10ACFF496FA6774AB040127F4A6B9330", "5A10ACFF496FA6774A0000000000000000", 64),
    ("464AF07CC6BEF9000000000B108D3E8AF8", "464AF07CC6BEF900000000000000000000", 80),
    ("54D52FDF6BE9B90035F2B0A4B99D30ADB5", "54D52FDF6BE9B900350000000000000000", 64),
    ("5A4F97FF20B4C3A20B24972F6AB008FC2F", "5A4F97FF20B4C3A20B0000000000000000", 64),
    ("52E514D3B3F2E5153745C0427FA23CBF3D", "52E514D3B3F2E515370000000000000000", 64),
    ("61D7E0E664367C44F8B9A4D709050A1CB4", "61D7E0E664367C44F80000000000000000", 64),
    ("5F99544C718056C58C356F60CB31ECAC5D", "5F99544C718056C58C0000000000000000", 64),
    ("5AE4F0DEB76F520C778EC9583DECD03F1F", "5AE4F0DEB76F520C770000000000000000", 64),
    ("63F3A6E9A943C53C760000001A3193A3CA", "63F3A6E9A943C53C760000000000000000", 64),
    ("60E2ED70CDF68A51360F532985D7AB2508", "60E0000000000000000000000000000000", 125),
    ("60E629F6CE12DA7770FCCFAF86B878DEEE", "60E629F6CE12DA77700000000000000000", 64),
    ("60BB631D9C88A5BFDCB3EDED1F370A0661", "60BB631D9C88A5BFDC0000000000000000", 64),
    ("56F5F1A4AA9A305691622D44A592312C70", "56F5F1A4AA9A3056910000000000000000", 64),
    ("6423ABB69FB07BCF2F90CBF216A4A70CA8", "6423ABB69FB07BCF2F0000000000000000", 64),
    ("54595F139E243AED430000001CC210ED4F", "54595F139E243AED430000000000000000", 64),
]

HIGH_RANK = [
    ("400000000000000000A400154FBFAF16CB", "400000000000000000A400000000000000", 56),
    ("6A7688C54585BBA6893FFFCC5B59BCBBFB", "6A7688C54585BBA6894000000000000000", 60),
    ("4A52C68110EBA7F382C40E0139F6BD7099", "4A52C68110EBA7F3820000000000000000", 64),
    ("79A3E4B4BB6D89FFFFFFF4D30ACD03D036", "79A3E4B4BB6D8A00000000000000000000", 80),
    ("60D79A0B3939F58B63AEA7E5836B2A387B", "60C0000000000000000000000000000000", 125),
    ("7B58AA7E0EDEF9F5BE4F22A78B7ADFEF24", "7800000000000000000000000000000000", 130),
    ("52DBF6AD57E5558A0B00059E34930A55DF", "52DBF6AD57E5558A0B0000000000000000", 64),
    ("47210B68AD7F4D4F14149C01C0CC3C98FC", "47210B68AD7F4D4F140000000000000000", 64),
    ("55E525FA48D760928F000514A4108A7B5E", "55E525FA48D760928F0000000000000000", 64),
    ("77A8A12FD494A7245512D45CD0127D9909", "77A8A12FD494A724550000000000000000", 64),
    ("4B0E444D0D2E97EA7FE69D722569CEE6F0", "4B0E444D0D2E97EA7F0000000000000000", 64),
    ("742A462DD40BAAABE6CE83A42162A3C453", "742A462DD40BAAABE60000000000000000", 64),
    ("54BDAF75953FE38BF300040C4277594686", "54BDAF75953FE38BF30000000000000000", 64),
    ("5105108500A45D43F50003E9EF4AFC3637", "5105108500A45D43F50000000000000000", 64),
    ("54FF162DD32098C47DD75600E7B6005EE2", "54FF162DD32098C47D0000000000000000", 64),
    ("51607C51A4FA29482D0003D2D145AA857E", "51607C51A4FA29482D0000000000000000", 64),
    ("514EFC318B325913440003A4CAC8B59FFC", "514EFC318B325913440000000000000000", 64),
    ("57753289097198EAB80003620281FBCF68", "57753289097198EAB80000000000000000", 64),
    ("54A6936FD99A9FDDEA000351133A09E9A9", "54A6936FD99A9FDDEA0000000000000000", 64),
    ("5061A434D1614CF84B00034DCF8388493A", "5061A434D1614CF84B0000000000000000", 64),
    ("4CF5889FC46FDC0000000A80A03F2EC718", "4CF5889FC46FDC00000000000000000000", 80),
    ("5334B4E5B3E20AFBA500032F91E3707C7C", "5334B4E5B3E20AFBA50000000000000000", 64),
    ("530337B5747CC8D90900030FD408B25FDC", "530337B5747CC8D9090000000000000000", 64),
    ("57F3D7F8318753960400030AEDEF0F1CF6", "57F3D7F831875396040000000000000000", 64),
    ("4AF6784F342A319EF4D6D7B18A26BF6A6B", "4AF6784F342A319EF40000000000000000", 64),
    ("555A900DB722EEA7E00002F6FFAF77E40E", "555A900DB722EEA7E00000000000000000", 64),
    ("4A280B8A0383969AC44A820B76915277FE", "4A280B8A0383969AC40000000000000000", 64),
    ("4969665B245546C80355994CB23E0EF9E4", "4969665B245546C8030000000000000000", 64),
    ("5369F313234DAAA7E70002A8A39ED33DA7", "5369F313234DAAA7E70000000000000000", 64),
    ("542A3410622284D90F0002A27D1E66C59F", "542A3410622284D90F0000000000000000", 64),
]


# =============================================================================
# ROBUST EVALUATION - DeepMind style
# =============================================================================

@funsearch.run  
def evaluate(seed: int) -> float:
  """Robust evaluation using rank-based scoring with caps."""
  low_scores = []
  high_scores = []
  
  # Score LOW_RANK entries (should get HIGHER priority scores)
  for k_hex, start_hex, chunk_bits in LOW_RANK:
    k = int(k_hex, 16)
    start = int(start_hex, 16)
    try:
      s = priority(k, start, chunk_bits)
      # Cap individual scores to prevent gaming
      if isinstance(s, (int, float)) and math.isfinite(s):
        s = max(-100.0, min(100.0, float(s)))
      else:
        s = 0.0
    except:
      s = 0.0
    low_scores.append(s)
  
  # Score HIGH_RANK entries (should get LOWER priority scores)
  for k_hex, start_hex, chunk_bits in HIGH_RANK:
    k = int(k_hex, 16)
    start = int(start_hex, 16)
    try:
      s = priority(k, start, chunk_bits)
      if isinstance(s, (int, float)) and math.isfinite(s):
        s = max(-100.0, min(100.0, float(s)))
      else:
        s = 0.0
    except:
      s = 0.0
    high_scores.append(s)
  
  # --- ROBUST METRICS ---
  
  # 1. Median-based separation (robust to outliers)
  low_sorted = sorted(low_scores)
  high_sorted = sorted(high_scores)
  median_low = low_sorted[len(low_sorted) // 2]
  median_high = high_sorted[len(high_sorted) // 2]
  separation = median_low - median_high
  
  # 2. Rank-based accuracy (AUC-like)
  # Count how many low_scores beat how many high_scores
  wins = 0
  for ls in low_scores:
    for hs in high_scores:
      if ls > hs:
        wins += 1
      elif ls == hs:
        wins += 0.5
  auc = wins / (len(low_scores) * len(high_scores))
  
  # 3. Threshold-free accuracy using median split
  threshold = (median_low + median_high) / 2
  correct_low = sum(1 for s in low_scores if s > threshold)
  correct_high = sum(1 for s in high_scores if s <= threshold)
  accuracy = (correct_low + correct_high) / (len(low_scores) + len(high_scores))
  
  # 4. Tail separation (10th percentile low vs 90th percentile high)
  p10_low = low_sorted[len(low_sorted) // 10] if len(low_sorted) >= 10 else low_sorted[0]
  p90_high = high_sorted[9 * len(high_sorted) // 10] if len(high_sorted) >= 10 else high_sorted[-1]
  tail_separation = p10_low - p90_high
  
  # Combined score (all components capped and weighted)
  score = 0.0
  score += min(20.0, max(-20.0, separation)) * 2      # Separation: -40 to +40
  score += auc * 50                                    # AUC: 0 to 50
  score += accuracy * 30                               # Accuracy: 0 to 30
  score += min(10.0, max(-10.0, tail_separation))     # Tail: -10 to +10
  
  # Total possible: ~130, baseline random: ~40
  return score


def priority(k: int, start: int, chunk_bits: int) -> float:
  """Score a chunk. Higher = more likely to produce low gap (near-collision).

  Args:
    k: The k_est value - WHERE the near-collision occurred within chunk.
       This is a measurement/clue, not the cause. But it tells you where
       tame and wild kangaroos got closest - could reveal patterns.
    start: The chunk start position - defines the region being scanned.
       This is the PRIMARY input. What makes this region special?
    chunk_bits: Size of chunk (56, 60, 64, 80, 125, or 130)

  Available constants:
    Public Key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
    PUBKEY_X = 0x145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
    Range: 0x4000000000000000000000000000000000 to 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF (It's known the key exists in this range and it's 135 bits)
    Curve Order N: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    Field Prime P: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

  EXPLORE FREELY - the answer could involve:
    PRIMARY (chunk region):
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

    SECONDARY (k_est as clue):
      - Offset of k within the chunk (k - start) - where did collision happen?
      - Bit/hex patterns in k that might indicate structure
      - Relationship between k and start
    
    SPECULATION:
      - Could low-gap chunks be near the actual private key?
      - If you find a pattern, it might help predict key location!

  Returns:
    float: higher = chunk more likely to produce low gap
  """
  """Improved version of `priority_v1`."""
  chunk_size = 2 ** chunk_bits
  offset = abs(k - start)
  offset_normalized = offset / chunk_size

  # 1. Proximity to multiples of N (stronger signal)
  n_mod_start = start % N
  n_dist = min(n_mod_start, N - n_mod_start)
  n_score = 1.0 - (n_dist / N)

  # 2. Proximity to multiples of P
  p_mod_start = start % P
  p_dist = min(p_mod_start, P - p_mod_start)
  p_score = 1.0 - (p_dist / P)

  # 3. Smoothness (number of small prime factors)
  small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

  def smoothness(n):
    factors = 0
    for p in small_primes:
      while n % p == 0:
        factors += 1
        n //= p
    return factors

  smooth_score = smoothness(start) / 14.0  # Normalize by number of primes

  # 4. Bit pattern (fewer transitions = better)
  bits = bin(start)[2:]
  transitions = sum(1 for i in range(len(bits)-1) if bits[i] != bits[i+1])
  bit_pattern_score = 1.0 / (1.0 + transitions / len(bits))

  # 5. Alignment with pubkey (how close is start to a value that would align well with pubkey)
  pubkey_mod_start = (PUBKEY_X - start) % N
  pubkey_dist = min(pubkey_mod_start, N - pubkey_mod_start)
  pubkey_score = 1.0 - (pubkey_dist / N)

  # 6. Boundary proximity
  range_dist = min(abs(start - RANGE_MIN), abs(start - RANGE_MAX))
  boundary_score = 1.0 - (range_dist / (RANGE_MAX - RANGE_MIN))

  # 7. Offset from k (where collision occurred)
  center = start + chunk_size // 2
  offset_from_center = abs(k - center)
  center_score = 1.0 - (offset_from_center / (chunk_size // 2))

  # 8. Entropy-based pattern (lower entropy = more predictable)
  def entropy(n):
    s = bin(n)[2:]
    counts = {}
    for c in s:
      counts[c] = counts.get(c, 0) + 1
    total = len(s)
    ent = 0
    for v in counts.values():
      p = v / total
      ent -= p * math.log2(p)
    return ent / math.log2(total)  # Normalized entropy
  entropy_score = 1.0 - entropy(start)

  # 9. Quadratic Residue Properties (Legendre symbol)
  def legendre(a, p):
    return pow(a, (p - 1) // 2, p)

  try:
    # If 1, it's a QR; if -1, non-QR
    qr_score = 1.0 - abs(legendre(start % P, P) - 1)
  except:
    qr_score = 0.0

  # 10. Distance to N mod P (could hint at alignment with curve structure)
  n_mod_p = start % P
  n_mod_p_dist = min(n_mod_p, P - n_mod_p)
  n_mod_p_score = 1.0 - (n_mod_p_dist / P)

  # 11. Bit density (number of 1s vs 0s)
  popcount = bin(start).count('1')
  density = popcount / len(bin(start)[2:])
  density_score = 1.0 - abs(density - 0.5)  # Closer to 0.5 = balanced

  # 12. Periodicity (if start has repeated patterns)
  hex_str = hex(start)[2:]
  if len(hex_str) > 10:
    # Look for repeating substrings in hex
    max_repeats = 0
    for i in range(1, len(hex_str) // 2 + 1):
      substr = hex_str[:i]
      count = hex_str.count(substr)
      if count > 1:
        max_repeats = max(max_repeats, count)
    repeat_score = max_repeats / (len(hex_str) // 2)
  else:
    repeat_score = 0.0

  # 13. Enhanced Kangaroo Path Feature
  # Measure how k_est aligns with the expected path of kangaroos
  expected_path = (start + chunk_size // 2) % N
  path_alignment = 1.0 - abs((k - expected_path) % N / N)

  # 14. Curve Structure Alignment
  # Check if start aligns with known curve properties
  curve_alignment = 1.0 - abs((start % N) - (PUBKEY_X % N)) / N

  # Combine all scores
  final_score = (
      3.0 * n_score +
      2.0 * p_score +
      1.5 * smooth_score +
      1.0 * bit_pattern_score +
      1.0 * pubkey_score +
      0.5 * boundary_score +
      1.0 * center_score +
      0.5 * entropy_score +
      0.5 * qr_score +
      0.3 * n_mod_p_score +
      0.4 * density_score +
      0.2 * repeat_score +
      0.3 * path_alignment +
      0.2 * curve_alignment
  )

  return final_score

