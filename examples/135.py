import ecc_backend as crypto
import numpy as np
import funsearch

# Puzzle 135 Target Data
TARGET_PK_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
TARGET_Q = crypto.decompress_pubkey(TARGET_PK_HEX)
RANGE_START = 0x4000000000000000000000000000000000

@funsearch.run
def evaluate(program) -> float:
    # Testing the evolved heuristic on a smaller 40-bit window 
    # that mimics the structure of the 135-bit problem.
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
            p_score = program.heuristic_priority(p_val[0], p_val[1], test_Q[0], test_Q[1])
            candidates.append((p_score, i))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Check if the correct relation is in the top picks
        found = any(k == test_k for _, k in candidates[:5])
        if found: score += 20.0
        
        # Add 'partial credit' based on bitwise proximity of top guesses
        best_guess_x = crypto.scalar_mult(crypto.G, candidates[0][1])[0]
        bit_match = bin(best_guess_x ^ test_Q[0]).count('0')
        score += (bit_match / 256.0)
        
    return float(score)

@funsearch.evolve
def heuristic_priority(px, py, qx, qy) -> float:
    """
    Find a novel mathematical priority that identifies 
    points share a scalar relationship.
    """
    # Placeholder: The AI will replace this with novel logic.
    return float(bin(px & qx).count('1'))