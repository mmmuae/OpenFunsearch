"""Elliptic-curve scalar search heuristic for Puzzle 135.

Guidelines for the LLM (keep these instructions intact):
- Always return a single, complete Python module that starts at column 0.
  Repeat the imports and ``evaluate`` exactly as provided; do not wrap code in
  markdown, prose, or extra indentation.
- Only adjust the body of ``heuristic_priority``; keep function signatures and
  top-level layout unchanged. Avoid adding helper functions or globals.
- Use plain Python with 4-space indentation inside functions—never tabs or
  mixed spacing—and finish with a trailing newline.
- Keep ``evaluate`` as the authoritative scoring function and avoid altering
  its logic or randomness handling.
- Prefer short, deterministic arithmetic; avoid I/O, randomness, or external
  state inside ``heuristic_priority``.
"""

import ecc_backend as crypto
import numpy as np

import funsearch

# Puzzle 135 Target Data
TARGET_PK_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
TARGET_Q = crypto.decompress_pubkey(TARGET_PK_HEX)
RANGE_START = 0x4000000000000000000000000000000000


@funsearch.run
def evaluate(program) -> float:
    """Score a candidate ``heuristic_priority`` implementation.

    The evaluator exercises the heuristic on a smaller window that mimics the
    135-bit challenge. The score rewards correct scalar guesses and proximity
    to the target point, encouraging concise and well-indented proposals from
    the LLM.
    """

    score = 0.0
    for _ in range(5):
        # Generate a random k in a test range
        test_k = np.random.randint(1, 2**40)
        test_Q = crypto.scalar_mult(crypto.G, test_k)

        # Ask AI to predict the 'energy' or 'likelihood' of a candidate scalar
        # based on the bitwise interaction between points.
        candidates = []
        for i in range(1, 500):
            p_val = crypto.scalar_mult(crypto.G, i)
            p_score = program.heuristic_priority(
                p_val[0], p_val[1], test_Q[0], test_Q[1]
            )
            candidates.append((p_score, i))

        candidates.sort(key=lambda x: x[0], reverse=True)
        # Check if the correct relation is in the top picks
        found = any(k == test_k for _, k in candidates[:5])
        if found:
            score += 20.0

        # Add 'partial credit' based on bitwise proximity of top guesses
        best_guess_x = crypto.scalar_mult(crypto.G, candidates[0][1])[0]
        bit_match = bin(best_guess_x ^ test_Q[0]).count("0")
        score += bit_match / 256.0

    return float(score)


@funsearch.evolve
def heuristic_priority(px, py, qx, qy) -> float:
    """Baseline heuristic for ranking candidate scalars.

    Keep this function short, deterministic, and free of side effects so the
    sandbox can parse it without indentation errors.
    """

    shared_bits = bin(px ^ qx).count("0")
    return float(shared_bits)
