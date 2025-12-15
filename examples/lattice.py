import numpy as np
import funsearch

# CONFIGURATION
DIMENSION = 10
NUM_TEST_CASES = 5
LOWER_BOUND = -100
UPPER_BOUND = 100

def _generate_random_basis(seed):
    """Generates a random lattice basis."""
    # IMPORT INSIDE to ensure Sandbox sees it
    import numpy as np
    rng = np.random.default_rng(seed)
    # CRITICAL FIX: .astype(float) prevents "Cannot cast ufunc" errors
    return rng.integers(LOWER_BOUND, UPPER_BOUND, size=(DIMENSION, DIMENSION)).astype(float)

@funsearch.run
def evaluate(program) -> float:
    """The Judge Function."""
    import numpy as np
    scores = []
    
    for i in range(NUM_TEST_CASES):
        # Generate fresh basis
        original_basis = _generate_random_basis(seed=i)
        
        try:
            # Run the AI-generated code
            # .copy() is crucial so we can verify against original later if needed
            reduced_basis = program.reduce_basis(original_basis.copy())
            
            # 1. Sanity Check: The determinant must remain the same (volume preserved)
            det_orig = np.abs(np.linalg.det(original_basis))
            det_new = np.abs(np.linalg.det(reduced_basis))
            
            # Allow for tiny floating point errors (1e-4)
            if not np.isclose(det_orig, det_new, rtol=1e-4):
                return -1000.0 # INVALID: Destroyed the lattice structure
            
            # 2. Score: We want the shortest possible first vector.
            # Shorter vector = Higher Score (because we negate)
            shortest_vec = np.linalg.norm(reduced_basis[0])
            
            # Prevent log(0) error
            if shortest_vec < 1e-6:
                shortest_vec = 1e-6
                
            score = -1.0 * np.log(shortest_vec)
            scores.append(score)
            
        except Exception:
            return -1000.0 # Code crashed

    # Return average score
    return float(np.mean(scores))

@funsearch.evolve
def reduce_basis(basis: np.ndarray) -> np.ndarray:
    """
    Heuristic to reduce a lattice basis.
    """
    import numpy as np
    n = basis.shape[0]
    
    # Simple pairwise reduction (Gram-Schmidt like)
    # The AI will try to improve this logic
    for i in range(n):
        for j in range(i + 1, n):
            # Projection coefficient
            dot_val = np.dot(basis[j], basis[i])
            norm_val = np.dot(basis[i], basis[i])
            
            if norm_val > 1e-9: # Avoid division by zero
                mu = round(dot_val / norm_val)
                if mu != 0:
                    basis[j] = basis[j] - mu * basis[i]
                
    # Sort rows by norm (shortest first)
    norms = np.linalg.norm(basis, axis=1)
    indices = np.argsort(norms)
    return basis[indices]