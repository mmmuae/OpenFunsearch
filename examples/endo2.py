"""Endomorphism Pattern Discovery for Bitcoin Puzzles

BACKGROUND:
secp256k1 has a special efficiently-computable endomorphism where:
  - λ is a cube root of unity mod N (curve order): λ³ ≡ 1 (mod N)
  - β is a cube root of unity mod P (field prime): β³ ≡ 1 (mod P)
  
For any private key k with public point P = k*G:
  - k * λ mod N is the private key for the point (β * Px mod P, Py)
  - k * λ² mod N is the private key for the point (β² * Px mod P, Py)

These three values (k, k*λ, k*λ²) form a cycle and give three equivalent 
representations of the same discrete log relationship.

THE INSIGHT:
The puzzle creator generated 160 private keys. If there's ANY bias or pattern
in how they were generated, it might be more visible when we examine all three
representations together. The endomorphism "unfolds" the key into a richer
structure where hidden patterns may become apparent.

YOUR MISSION:
Discover a function that identifies properties unique to real puzzle keys
versus random keys. Even weak signals (60-70% accuracy) could narrow the
135-bit search space significantly.

DATA: 82 solved puzzle private keys with their public keys.
"""

import funsearch
import math

# =============================================================================
# SECP256K1 CONSTANTS
# =============================================================================

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Curve order
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F  # Field prime
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798  # Generator x
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8  # Generator y

# Endomorphism constants (verified: λ³ ≡ 1 mod N, β³ ≡ 1 mod P)
LAMBDA = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
LAMBDA2 = pow(LAMBDA, 2, N)  # λ²
BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
BETA2 = pow(BETA, 2, P)  # β²

# =============================================================================
# MINIMAL SECP256K1 OPERATIONS (for synthetic public keys)
# =============================================================================


def _mod_inv(a: int, p: int = P) -> int:
  """Return the modular inverse using Fermat's little theorem."""
  return pow(a, p - 2, p)


def _point_double(p):
  if p is None:
    return None
  x, y = p
  if y == 0:
    return None
  m = ((3 * x * x) * _mod_inv(2 * y)) % P
  x3 = (m * m - 2 * x) % P
  y3 = (m * (x - x3) - y) % P
  return (x3, y3)


def _point_add(p1, p2):
  """Add two points on secp256k1. Points are (x, y) or None for infinity."""
  if p1 is None:
    return p2
  if p2 is None:
    return p1
  x1, y1 = p1
  x2, y2 = p2
  if x1 == x2 and (y1 + y2) % P == 0:
    return None
  if p1 == p2:
    return _point_double(p1)
  m = ((y2 - y1) * _mod_inv(x2 - x1)) % P
  x3 = (m * m - x1 - x2) % P
  y3 = (m * (x1 - x3) - y1) % P
  return (x3, y3)


def _scalar_mult(k: int, point) -> int:
  """Return x-coordinate of k * point using double-and-add."""
  result = None
  addend = point
  while k:
    if k & 1:
      result = _point_add(result, addend)
    addend = _point_double(addend)
    k >>= 1
  return result[0] if result else 0

# =============================================================================
# SOLVED PUZZLE DATA: (bits, private_key, compressed_public_key)
# 82 puzzles from bits 1-130 (with gaps at 71-74, 76-79, 81-84, etc.)
# =============================================================================

SOLVED_KEYS = [
  (1, 0x1, '0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798'),
  (2, 0x3, '02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9'),
  (3, 0x7, '025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc'),
  (4, 0x8, '022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01'),
  (5, 0x15, '02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5'),
  (6, 0x31, '03f2dac991cc4ce4b9ea44887e5c7c0bce58c80074ab9d4dbaeb28531b7739f530'),
  (7, 0x4c, '0296516a8f65774275278d0d7420a88df0ac44bd64c7bae07c3fe397c5b3300b23'),
  (8, 0xe0, '0308bc89c2f919ed158885c35600844d49890905c79b357322609c45706ce6b514'),
  (9, 0x1d3, '0243601d61c836387485e9514ab5c8924dd2cfd466af34ac95002727e1659d60f7'),
  (10, 0x202, '03a7a4c30291ac1db24b4ab00c442aa832f7794b5a0959bec6e8d7fee802289dcd'),
  (11, 0x483, '038b05b0603abd75b0c57489e451f811e1afe54a8715045cdf4888333f3ebc6e8b'),
  (12, 0xa7b, '038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c'),
  (13, 0x1460, '03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d'),
  (14, 0x2930, '03b4f1de58b8b41afe9fd4e5ffbdafaeab86c5db4769c15d6e6011ae7351e54759'),
  (15, 0x68f3, '02fea58ffcf49566f6e9e9350cf5bca2861312f422966e8db16094beb14dc3df2c'),
  (16, 0xc936, '029d8c5d35231d75eb87fd2c5f05f65281ed9573dc41853288c62ee94eb2590b7a'),
  (17, 0x1764f, '033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3'),
  (18, 0x3080d, '020ce4a3291b19d2e1a7bf73ee87d30a6bdbc72b20771e7dfff40d0db755cd4af1'),
  (19, 0x5749f, '0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19'),
  (20, 0xd2c55, '033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c'),
  (21, 0x1ba534, '031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e'),
  (22, 0x2de40f, '023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43'),
  (23, 0x556e52, '03f82710361b8b81bdedb16994f30c80db522450a93e8e87eeb07f7903cf28d04b'),
  (24, 0xdc2a04, '036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c'),
  (25, 0x1fa5ee5, '03057fbea3a2623382628dde556b2a0698e32428d3cd225f3bd034dca82dd7455a'),
  (26, 0x340326e, '024e4f50a2a3eccdb368988ae37cd4b611697b26b29696e42e06d71368b4f3840f'),
  (27, 0x6ac3875, '031a864bae3922f351f1b57cfdd827c25b7e093cb9c88a72c1cd893d9f90f44ece'),
  (28, 0xd916ce8, '03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7'),
  (29, 0x17e2551e, '026caad634382d34691e3bef43ed4a124d8909a8a3362f91f1d20abaaf7e917b36'),
  (30, 0x3d94cd64, '030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b'),
  (31, 0x7d4fe747, '0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28'),
  (32, 0xb862a62e, '0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69'),
  (33, 0x1a96ca8d8, '03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579'),
  (34, 0x34a65911d, '033cdd9d6d97cbfe7c26f902faf6a435780fe652e159ec953650ec7b1004082790'),
  (35, 0x4aed21170, '02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d'),
  (36, 0x9de820a7c, '02b3e772216695845fa9dda419fb5daca28154d8aa59ea302f05e916635e47b9f6'),
  (37, 0x1757756a93, '027d2c03c3ef0aec70f2c7e1e75454a5dfdd0e1adea670c1b3a4643c48ad0f1255'),
  (38, 0x22382facd0, '03c060e1e3771cbeccb38e119c2414702f3f5181a89652538851d2e3886bdd70c6'),
  (39, 0x4b5f8303e9, '022d77cd1467019a6bf28f7375d0949ce30e6b5815c2758b98a74c2700bc006543'),
  (40, 0xe9ae4933d6, '03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4'),
  (41, 0x153869acc5b, '03b357e68437da273dcf995a474a524439faad86fc9effc300183f714b0903468b'),
  (42, 0x2a221c58d8f, '03eec88385be9da803a0d6579798d977a5d0c7f80917dab49cb73c9e3927142cb6'),
  (43, 0x6bd3b27c591, '02a631f9ba0f28511614904df80d7f97a4f43f02249c8909dac92276ccf0bcdaed'),
  (44, 0xe02b35a358f, '025e466e97ed0e7910d3d90ceb0332df48ddf67d456b9e7303b50a3d89de357336'),
  (45, 0x122fca143c05, '026ecabd2d22fdb737be21975ce9a694e108eb94f3649c586cc7461c8abf5da71a'),
  (46, 0x2ec18388d544, '03fd5487722d2576cb6d7081426b66a3e2986c1ce8358d479063fb5f2bb6dd5849'),
  (47, 0x6cd610b53cba, '023a12bd3caf0b0f77bf4eea8e7a40dbe27932bf80b19ac72f5f5a64925a594196'),
  (48, 0xade6d7ce3b9b, '0291bee5cf4b14c291c650732faa166040e4c18a14731f9a930c1e87d3ec12debb'),
  (49, 0x174176b015f4d, '02591d682c3da4a2a698633bf5751738b67c343285ebdc3492645cb44658911484'),
  (50, 0x22bd43c2e9354, '03f46f41027bbf44fafd6b059091b900dad41e6845b2241dc3254c7cdd3c5a16c6'),
  (51, 0x75070a1a009d4, '028c6c67bef9e9eebe6a513272e50c230f0f91ed560c37bc9b033241ff6c3be78f'),
  (52, 0xefae164cb9e3c, '0374c33bd548ef02667d61341892134fcf216640bc2201ae61928cd0874f6314a7'),
  (53, 0x180788e47e326c, '020faaf5f3afe58300a335874c80681cf66933e2a7aeb28387c0d28bb048bc6349'),
  (54, 0x236fb6d5ad1f43, '034af4b81f8c450c2c870ce1df184aff1297e5fcd54944d98d81e1a545ffb22596'),
  (55, 0x6abe1f9b67e114, '0385a30d8413af4f8f9e6312400f2d194fe14f02e719b24c3f83bf1fd233a8f963'),
  (56, 0x9d18b63ac4ffdf, '033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a'),
  (57, 0x1eb25c90795d61c, '02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea'),
  (58, 0x2c675b852189a21, '0311569442e870326ceec0de24eb5478c19e146ecd9d15e4666440f2f638875f42'),
  (59, 0x7496cbb87cab44f, '0241267d2d7ee1a8e76f8d1546d0d30aefb2892d231cee0dde7776daf9f8021485'),
  (60, 0xfc07a1825367bbe, '0348e843dc5b1bd246e6309b4924b81543d02b16c8083df973a89ce2c7eb89a10d'),
  (61, 0x13c96a3742f64906, '0249a43860d115143c35c09454863d6f82a95e47c1162fb9b2ebe0186eb26f453f'),
  (62, 0x363d541eb611abee, '03231a67e424caf7d01a00d5cd49b0464942255b8e48766f96602bdfa4ea14fea8'),
  (63, 0x7cce5efdaccf6808, '0365ec2994b8cc0a20d40dd69edfe55ca32a54bcbbaa6b0ddcff36049301a54579'),
  (64, 0xf7051f27b09112d4, '03100611c54dfef604163b8358f7b7fac13ce478e02cb224ae16d45526b25d9d4d'),
  (65, 0x1a838b13505b26867, '0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b'),
  (66, 0x2832ed74f2b5e35ee, '024ee2be2d4e9f92d2f5a4a03058617dc45befe22938feed5b7a6b7282dd74cbdd'),
  (67, 0x730fc235c1942c1ae, '0212209f5ec514a1580a2937bd833979d933199fc230e204c6cdc58872b7d46f75'),
  (68, 0xbebb3940cd0fc1491, '031fe02f1d740637a7127cdfe8a77a8a0cfc6435f85e7ec3282cb6243c0a93ba1b'),
  (69, 0x101d83275fb2bc7e0c, '024babadccc6cfd5f0e5e7fd2a50aa7d677ce0aa16fdce26a0d0882eed03e7ba53'),
  (70, 0x349b84b6431a6c4ef1, '0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483'),
  (75, 0x4c5ce114686a1336e07, '03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755'),
  (80, 0xea1a5c66dcc11b5ad180, '037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc'),
  (85, 0x11720c4f018d51b8cebba8, '0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a'),
  (90, 0x2ce00bb2136a445c71e85bf, '035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5'),
  (95, 0x527a792b183c7f64a0e8b1f4, '02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045'),
  (100, 0xaf55fc59c335c8ec67ed24826, '03d2063d40402f030d4cc71331468827aa41a8a09bd6fd801ba77fb64f8e67e617'),
  (105, 0x16f14fc2054cd87ee6396b33df3, '03bcf7ce887ffca5e62c9cabbdb7ffa71dc183c52c04ff4ee5ee82e0c55c39d77b'),
  (110, 0x35c0d7234df7deb0f20cf7062444, '0309976ba5570966bf889196b7fdf5a0f9a1e9ab340556ec29f8bb60599616167d'),
  (115, 0x60f4d11574f5deee49961d9609ac6, '0248d313b0398d4923cdca73b8cfa6532b91b96703902fc8b32fd438a3b7cd7f55'),
  (120, 0xb10f22572c497a836ea187f2e1fc23, '02ceb6cbbcdbdf5ef7150682150f4ce2c6f4807b349827dcdbdd1f2efa885a2630'),
  (125, 0x1c533b6bb7f0804e09960225e44877ac, '0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e'),
  (130, 0x33e7665705359f04f28b88cf897c603c9, '03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852'),
]

# =============================================================================
# PRECOMPUTE ENDOMORPHISM TRANSFORMS
# For each key k, compute the triplet (k, k*λ mod N, k*λ² mod N)
# =============================================================================


def _build_transformed_data():
  transformed = []
  for bits, k, pubkey in SOLVED_KEYS:
    k_lambda = (k * LAMBDA) % N
    k_lambda2 = (k * LAMBDA2) % N
    # Also extract pubkey x-coordinate for potential analysis
    pubkey_x = int(pubkey[2:], 16)  # Skip the 02/03 prefix
    transformed.append({
      'bits': bits,
      'k': k,
      'k_lambda': k_lambda,
      'k_lambda2': k_lambda2,
      'pubkey_x': pubkey_x,
      'pubkey_prefix': pubkey[:2],  # 02 or 03 (y parity)
    })
  return transformed


try:
  TRANSFORMED_DATA
except NameError:
  TRANSFORMED_DATA = None


def _get_transformed_data():
  global TRANSFORMED_DATA
  if TRANSFORMED_DATA is None:
    TRANSFORMED_DATA = _build_transformed_data()
  return TRANSFORMED_DATA


# =============================================================================
# ROBUST EVALUATION - DeepMind Style
# =============================================================================

@funsearch.run
def evaluate(seed: int) -> float:
  """Robust evaluation comparing real puzzle keys vs random keys.
  
  Uses multiple metrics to prevent gaming:
  1. Median separation (robust to outliers)
  2. AUC/rank-based (threshold-free)
  3. Consistency (variance comparison)
  4. Accuracy at optimal threshold
  5. Tail behavior (extreme percentiles)
  """
  import random
  rng = random.Random(seed)

  transformed_data = _get_transformed_data()

  # Score real puzzle keys
  real_scores = []
  for data in transformed_data:
    try:
      s = priority(
        data['bits'],
        data['k'],
        data['k_lambda'],
        data['k_lambda2'],
        data['pubkey_x']
      )
      if isinstance(s, (int, float)) and math.isfinite(s):
        s = max(-100.0, min(100.0, float(s)))
      else:
        s = 0.0
    except:
      s = 0.0
    real_scores.append(s)
  
  # Generate fake keys (random in same bit ranges) and score them
  fake_scores = []
  for data in transformed_data:
    bits = data['bits']
    # Random key in same bit range [2^(bits-1), 2^bits - 1]
    if bits == 1:
      fake_k = 1
    else:
      fake_k = rng.randint(2**(bits-1), 2**bits - 1)
    
    fake_k_lambda = (fake_k * LAMBDA) % N
    fake_k_lambda2 = (fake_k * LAMBDA2) % N
    fake_pubkey_x = _scalar_mult(fake_k, (Gx, Gy))
    
    try:
      s = priority(bits, fake_k, fake_k_lambda, fake_k_lambda2, fake_pubkey_x)
      if isinstance(s, (int, float)) and math.isfinite(s):
        s = max(-100.0, min(100.0, float(s)))
      else:
        s = 0.0
    except:
      s = 0.0
    fake_scores.append(s)
  
  # -------------------------------------------------------------------------
  # METRIC 1: Median Separation (robust to outliers)
  # -------------------------------------------------------------------------
  real_sorted = sorted(real_scores)
  fake_sorted = sorted(fake_scores)
  real_median = real_sorted[len(real_sorted) // 2]
  fake_median = fake_sorted[len(fake_sorted) // 2]
  separation = real_median - fake_median
  
  # -------------------------------------------------------------------------
  # METRIC 2: AUC (rank-based, threshold-free)
  # Count how often real scores beat fake scores
  # -------------------------------------------------------------------------
  wins = 0
  total_pairs = len(real_scores) * len(fake_scores)
  for rs in real_scores:
    for fs in fake_scores:
      if rs > fs:
        wins += 1
      elif rs == fs:
        wins += 0.5
  auc = wins / total_pairs if total_pairs > 0 else 0.5
  
  # -------------------------------------------------------------------------
  # METRIC 3: Consistency (lower variance in real = pattern found)
  # -------------------------------------------------------------------------
  real_var = sum((s - real_median)**2 for s in real_scores) / len(real_scores)
  fake_var = sum((s - fake_median)**2 for s in fake_scores) / len(fake_scores)
  # Ratio < 1 means real is more consistent
  consistency = 1.0 - (real_var / (fake_var + 0.001))
  
  # -------------------------------------------------------------------------
  # METRIC 4: Classification Accuracy at optimal threshold
  # -------------------------------------------------------------------------
  threshold = (real_median + fake_median) / 2
  correct_real = sum(1 for s in real_scores if s > threshold)
  correct_fake = sum(1 for s in fake_scores if s <= threshold)
  accuracy = (correct_real + correct_fake) / (len(real_scores) + len(fake_scores))
  
  # -------------------------------------------------------------------------
  # METRIC 5: Tail Separation (10th percentile real vs 90th percentile fake)
  # -------------------------------------------------------------------------
  p10_real = real_sorted[len(real_sorted) // 10] if len(real_sorted) >= 10 else real_sorted[0]
  p90_fake = fake_sorted[9 * len(fake_sorted) // 10] if len(fake_sorted) >= 10 else fake_sorted[-1]
  tail_sep = p10_real - p90_fake
  
  # -------------------------------------------------------------------------
  # COMBINE SCORES (all capped to prevent gaming)
  # -------------------------------------------------------------------------
  score = 0.0
  
  # Separation: up to ±30 points
  score += max(-30.0, min(30.0, separation * 3))
  
  # AUC: 0 to 40 points (baseline 0.5 = 20 points)
  score += (auc - 0.5) * 80  # Range: -40 to +40
  
  # Consistency: up to 15 points
  score += max(-15.0, min(15.0, consistency * 15))
  
  # Accuracy: 0 to 30 points (baseline 0.5 = 15 points)
  score += accuracy * 30
  
  # Tail separation: up to ±10 points
  score += max(-10.0, min(10.0, tail_sep))
  
  # Baseline random (balanced scores): ~30, max possible: ~125
  return score


# =============================================================================
# THE FUNCTION TO EVOLVE
# =============================================================================

@funsearch.evolve
def priority(bits: int, k: int, k_lambda: int, k_lambda2: int, pubkey_x: int) -> float:
  """Score how likely a key is to be a real Bitcoin puzzle key.
  
  The endomorphism gives us three equivalent representations of the same
  discrete log. Patterns that exist in the original key might be more
  visible (or differently visible) in the transformed versions.
  
  Args:
    bits: The puzzle number / bit length (1-130 for solved puzzles)
    k: The private key
    k_lambda: k * λ mod N (first endomorphism transform)
    k_lambda2: k * λ² mod N (second endomorphism transform)
    pubkey_x: The x-coordinate of the public key (for reference)
  
  Key insight: λ³ ≡ 1 (mod N), so the three values form a cycle.
  
  Available constants:
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    LAMBDA = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
    Gx, Gy (generator point)
  
  EXPLORATION DIRECTIONS:
  
  1. TRIPLET RELATIONSHIPS:
     - min(k, k_lambda, k_lambda2) - smallest representation
     - Distances between the three values
     - XOR / AND / OR patterns across the triplet
     - GCD(k, k_lambda), GCD(k, k_lambda2), GCD(k_lambda, k_lambda2)
     
  2. MODULAR PATTERNS:
     - k mod small_primes (3, 7, 11, 13, ...)
     - Patterns that persist across all three representations
     - Quadratic residues, Legendre symbols
     
  3. BIT PATTERNS:
     - Popcount (number of 1 bits) in k, k_lambda, k_lambda2
     - Bit runs, transitions, palindromes
     - Common bits across all three representations
     - Leading/trailing zeros
     
  4. ALGEBRAIC STRUCTURE:
     - Relationship to N (curve order)
     - Smoothness (size of largest prime factor)
     - Continued fraction expansion
     
  5. CROSS-PUZZLE PATTERNS:
     - How does the pattern change with puzzle number (bits)?
     - Relationships between consecutive puzzle keys
     
  6. PUBLIC KEY RELATIONSHIPS:
     - pubkey_x relation to k, k_lambda, k_lambda2
     - pubkey_x mod small values
  
  Returns:
    float: higher = more likely to be a real puzzle key
  """
  popcounts = [v.bit_count() for v in (k, k_lambda, k_lambda2)]
  mean_pop = sum(popcounts) / 3.0
  normalized_pop = mean_pop / max(bits, 1)

  # Spread close to 0 means the triplet shares similar popcounts.
  triplet_spread = (max(popcounts) - min(popcounts)) / max(bits, 1)

  # Preference for balanced residues across the cycle.
  cycle_residue = ((k % bits) + (k_lambda % bits) + (k_lambda2 % bits)) / (3.0 * bits)

  # Lightweight public-key signal: low-byte variation is enough to avoid ties.
  pub_low = (pubkey_x & 0xFF) / 255.0

  score = 0.0
  score += (normalized_pop - 0.5) * 10.0
  score += (1.0 - min(triplet_spread, 1.0)) * 5.0
  score += (0.5 - cycle_residue) * 3.0
  score += (pub_low - 0.5) * 2.0
  score -= bits * 0.02

  return score
