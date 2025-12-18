"""Endomorphism pattern discovery for Bitcoin puzzle keys.

Return code for Python method priority_vX, where every iteration improves on previous.

Make only small changes.
Try to make the code short.
Use two spaces as indentation level.
"""
import math
import random
import funsearch

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
LAMBDA = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
LAMBDA2 = pow(LAMBDA, 2, N)
BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee

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

TRANSFORMED = []
for bits, k in SOLVED_KEYS:
  k_l = (k * LAMBDA) % N
  k_l2 = (k * LAMBDA2) % N
  TRANSFORMED.append((bits, k, k_l, k_l2))


@funsearch.run
def evaluate(seed: int) -> float:
  """Evaluates priority function by comparing real vs random keys."""
  rng = random.Random(seed)
  real_scores = []
  fake_scores = []

  for bits, k, k_l, k_l2 in TRANSFORMED:
    try:
      s = priority(bits, k, k_l, k_l2)
      s = max(-100.0, min(100.0, float(s))) if math.isfinite(s) else 0.0
    except:
      s = 0.0
    real_scores.append(s)

    fake_k = rng.randint(2**(bits-1), 2**bits - 1) if bits > 1 else 1
    fake_kl = (fake_k * LAMBDA) % N
    fake_kl2 = (fake_k * LAMBDA2) % N
    try:
      s = priority(bits, fake_k, fake_kl, fake_kl2)
      s = max(-100.0, min(100.0, float(s))) if math.isfinite(s) else 0.0
    except:
      s = 0.0
    fake_scores.append(s)

  real_sorted = sorted(real_scores)
  fake_sorted = sorted(fake_scores)
  real_med = real_sorted[len(real_sorted) // 2]
  fake_med = fake_sorted[len(fake_sorted) // 2]

  wins = sum(1 for r in real_scores for f in fake_scores if r > f)
  wins += sum(0.5 for r in real_scores for f in fake_scores if r == f)
  auc = wins / (len(real_scores) * len(fake_scores))

  thresh = (real_med + fake_med) / 2
  correct = sum(1 for s in real_scores if s > thresh)
  correct += sum(1 for s in fake_scores if s <= thresh)
  acc = correct / (len(real_scores) + len(fake_scores))

  score = 0.0
  score += max(-30.0, min(30.0, (real_med - fake_med) * 3))
  score += (auc - 0.5) * 80
  score += acc * 30
  return score


@funsearch.evolve
def priority(bits: int, k: int, k_l: int, k_l2: int) -> float:
  """Returns priority score. Higher = more likely real puzzle key.

  Args:
    bits: puzzle number (bit length)
    k: private key
    k_l: k * lambda mod N
    k_l2: k * lambda^2 mod N

  The triplet (k, k_l, k_l2) forms a cycle under the endomorphism.
  Discover patterns that distinguish real keys from random ones.
  """
  return 0.0