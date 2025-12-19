"""
Bitcoin Puzzle Pattern Discovery - Find the key generation formula.

OBJECTIVE: Discover f(n) that predicts private_key (or its position/properties).

SOLVED PUZZLE DATA - Multiple views for pattern discovery:
=========================================================
n   | ratio   | key_hex_suffix | low16  | popcount | key
----|---------|----------------|--------|----------|----
 10 | 0.00391 | 0x0202         |    514 |     2    | 514
 11 | 0.12805 | 0x0483         |   1155 |     4    | 1155
 12 | 0.31021 | 0x0a7b         |   2683 |     8    | 2683
 13 | 0.27350 | 0x1460         |   5216 |     4    | 5216
 14 | 0.28714 | 0x2930         |  10544 |     5    | 10544
 15 | 0.63987 | 0x68f3         |  26867 |     9    | 26867
 16 | 0.57198 | 0xc936         |  51510 |     8    | 51510
 17 | 0.46215 | 0x764f         |  30287 |    11    | 95823
 18 | 0.51573 | 0x080d         |   2061 |     6    | 198669
 19 | 0.36389 | 0x749f         |  29855 |    12    | 357535
 20 | 0.64665 | 0x2c55         |  11349 |    10    | 863317
...
 64 | 0.92984 | 0x12d4         |   4820 |    30    | 17799667357578236628
 65 | 0.65712 | 0x6867         |  26727 |    29    | 30568377312064202855
 66 | 0.25622 | 0x35ee         |  13806 |    38    | 46346217550346335726
 67 | 0.79784 | 0xc1ae         |  49582 |    31    | 132656943602386256302
 68 | 0.49009 | 0x1491         |   5265 |    33    | 219898266213316039825
 69 | 0.00721 | 0x7e0c         |  32268 |    35    | 297274491920375905804
 70 | 0.64398 | 0x4ef1         |  20209 |    34    | 970436974005023690481

LOOK FOR:
- LCG patterns: key[n] = (a * key[n-1] + c) mod m
- Hex patterns in suffixes
- Popcount relationships to n
- Modular arithmetic on keys
- Ratio patterns (sin, polynomial, hash-based)
"""

import math
import funsearch

# (n, private_key) - raw data
KEYS = [
    (1, 1), (2, 3), (3, 7), (4, 8), (5, 21), (6, 49), (7, 76), (8, 224),
    (9, 467), (10, 514), (11, 1155), (12, 2683), (13, 5216), (14, 10544),
    (15, 26867), (16, 51510), (17, 95823), (18, 198669), (19, 357535),
    (20, 863317), (21, 1811764), (22, 3007503), (23, 5598802), (24, 14428676),
    (25, 33185509), (26, 54538862), (27, 111949941), (28, 227634408),
    (29, 400708894), (30, 1033162084), (31, 2102388551), (32, 3093472814),
    (33, 7137437912), (34, 14133072157), (35, 20112871792), (36, 42387769980),
    (37, 100251560595), (38, 146971536592), (39, 323724968937),
    (40, 1003651412950), (41, 1458252205147), (42, 2895374552463),
    (43, 7409811047825), (44, 15404761757071), (45, 19996463086597),
    (46, 51408670348612), (47, 119666659114170), (48, 191206974700443),
    (49, 409118905032525), (50, 611140496167764), (51, 2058769515153876),
    (52, 4216495639600700), (53, 6763683971478124), (54, 9974455244496707),
    (55, 30045390491869460), (56, 44218742292676575), (57, 138245758910846492),
    (58, 199976667976342049), (59, 525070384258266191),
    (60, 1135041350219496382), (61, 1425787542618654982),
    (62, 3908372542507822062), (63, 8993229949524469768),
    (64, 17799667357578236628), (65, 30568377312064202855),
    (66, 46346217550346335726), (67, 132656943602386256302),
    (68, 219898266213316039825), (69, 297274491920375905804),
    (70, 970436974005023690481),
    (75, 22538323240989823823367), (80, 1105520030589234487939456),
    (85, 21090315766411506144426920), (90, 868012190417726402719548863),
    (95, 25525831956644113617013748212), (100, 868221233689326498340379183142),
    (105, 29083230144918045706788529192435),
    (110, 1090246098153987172547740458951748),
    (115, 31464123230573852164273674364426950),
    (120, 919343500840980333540511050618764323),
    (125, 37650549717742544505774009877315221420),
    (130, 1103873984953507439627945351144005829577),
]

def get_features(n, key):
  """Compute features for a puzzle."""
  rmin = 1 << (n - 1)  # 2^(n-1)
  rmax = (1 << n) - 1  # 2^n - 1
  rsize = rmax - rmin
  return {
      "n": n,
      "key": key,
      "ratio": (key - rmin) / rsize if rsize > 0 else 0.5,
      "offset": key - rmin,           # distance from range start
      "low16": key & 0xFFFF,          # low 16 bits
      "low8": key & 0xFF,             # low 8 bits
      "high_bits": key >> (n - 8) if n > 8 else key,  # high 8 bits
      "popcount": bin(key).count('1'),
      "mod7": key % 7,
      "mod13": key % 13,
      "mod127": key % 127,
      "n_mod3": n % 3,
      "n_mod5": n % 5,
  }

def build_history(up_to_n):
  """Build feature history for puzzles before n."""
  history = {}
  for n, key in KEYS:
    if n < up_to_n:
      history[n] = get_features(n, key)
  return history


@funsearch.run
def evaluate(unused: int) -> float:
  """Score the priority function across all puzzles."""
  total_score = 0.0
  count = 0

  for n, actual_key in KEYS:
    if n < 10:  # skip trivial
      continue

    history = build_history(n)
    if len(history) < 5:
      continue

    # Get actual values
    actual = get_features(n, actual_key)

    # Predict
    predicted = priority(n, history)

    # Score based on what was predicted
    # Option 1: If predicting ratio
    if isinstance(predicted, float) and 0 <= predicted <= 1:
      error = abs(predicted - actual["ratio"])
      score = math.exp(-10 * error) * 10
      if error < 0.01: score += 50
      elif error < 0.05: score += 20
      elif error < 0.10: score += 10
      total_score += score

    # Option 2: If predicting key directly (partial credit for closeness)
    elif isinstance(predicted, int) and predicted > 0:
      rmin = 1 << (n - 1)
      rmax = (1 << n) - 1
      if rmin <= predicted <= rmax:
        pred_ratio = (predicted - rmin) / (rmax - rmin)
        error = abs(pred_ratio - actual["ratio"])
        score = math.exp(-10 * error) * 10
        if predicted == actual_key:
          score += 1000  # Jackpot!
        total_score += score
      else:
        total_score -= 5  # Out of range penalty

    count += 1

  return total_score / count if count > 0 else -100.0


@funsearch.evolve
def priority(n: int, history: dict) -> float:
  """Predict position ratio (or key) for puzzle n.

  Args:
    n: puzzle number (bits)
    history: dict of {puzzle_num: features_dict} for all solved puzzles < n
             Each features_dict has: n, key, ratio, offset, low16, low8,
             high_bits, popcount, mod7, mod13, mod127, n_mod3, n_mod5

  SAMPLE DATA (ratio values to predict):
    n=10: 0.004  n=15: 0.640  n=20: 0.647  n=25: 0.978  n=30: 0.924
    n=35: 0.171  n=40: 0.826  n=45: 0.137  n=50: 0.086  n=55: 0.668
    n=60: 0.969  n=65: 0.657  n=69: 0.007  n=70: 0.644

  SAMPLE DATA (low16 hex patterns):
    n=10: 0x0202  n=15: 0x68f3  n=20: 0x2c55  n=25: 0x5ee5  n=30: 0xcd64
    n=35: 0x1170  n=40: 0x33d6  n=45: 0x3c05  n=50: 0x9354  n=55: 0xe114

  SAMPLE DATA (popcount):
    n=10: 2   n=20: 10  n=30: 16  n=40: 22  n=50: 24  n=60: 32  n=70: 34

  Returns:
    float in [0,1] for ratio prediction, OR int for direct key prediction
  """
  # Get previous puzzle's features
  prev_n = max(k for k in history.keys())
  prev = history[prev_n]

  # Simple baseline: follow previous ratio with noise
  return prev["ratio"] * 0.7 + 0.15 + (n % 10) * 0.02
