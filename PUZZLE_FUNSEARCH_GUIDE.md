# Running FunSearch on puzzle.py

## ✓ Validation Complete

puzzle.py has been **validated and is fully compatible** with FunSearch!

**Test Results:**
```
✓ Seed  42: score = 136.20
✓ Seed 123: score = 136.20
✓ Seed 456: score = 136.20
```

**Parsed Structure:**
- Total functions: 10
- @funsearch.evolve: `priority`
- @funsearch.run: `evaluate`
- Features computed: 165+

## Quick Start

### Option 1: Modify run_funsearch.py

Edit `run_funsearch.py` and change line 42 to:

```python
sys.argv = [
    'funsearch',
    'run',
    'examples/puzzle.py',  # <-- Change this line
    '8',
    '--model_name=Meta-Llama-3-8B-Instruct-Q5_K_M',  # Or your preferred model
    '--output_path=./data/',
    '--iterations=-1',  # Run indefinitely
    '--samplers=1',
    '--sandbox_type=ExternalProcessSandbox'
]
```

Then run:
```bash
python run_funsearch.py
```

### Option 2: Command Line (requires full setup)

```bash
python -m funsearch run examples/puzzle.py 8 \
    --model_name=Meta-Llama-3-8B-Instruct-Q5_K_M \
    --output_path=./data/ \
    --iterations=-1 \
    --samplers=1 \
    --sandbox_type=ExternalProcessSandbox
```

## What FunSearch Will Do

1. **Load puzzle.py** with all 66 solved Bitcoin puzzles
2. **Evolve the `priority` function** to discover pattern formulas
3. **Test predictions** against all solved puzzles (10-66)
4. **Reward accurate predictions** of position_ratio
5. **Save best formulas** to `data/backups/`

## Expected Behavior

FunSearch will iteratively:
- Generate new versions of the `priority` function
- Test them against the 56+ solved puzzles
- Keep the best performers
- Combine successful approaches

The goal: **Discover the mathematical formula that predicts where in each bit-range the private key falls**

## Features Available to LLM

The `priority` function receives a dictionary with 165+ features:

### Mathematical Methods (80+ features)
- **Fractal/Self-Similar**: pos_n_minus_1, pos_diff_1, fractal_coef_a, fibonacci_pred
- **Kolmogorov Complexity**: linear_n, quadratic_n, mod_2-13, lcg_simple, hash_mod
- **Hidden Markov Model**: pos_mean, pos_std, state_increasing, trend_strength
- **Wavelet/Frequency**: fft_dc, fft_fund, autocorr_1, autocorr_2
- **Topological**: embed_var, embed_mean_dist

### Exploitation Vectors (85+ features)
1. **PRNG State**: lcg_diff_ratio, key_xor_avg_popcount, mt_low_bit_variance
2. **HD Wallet**: hd_diff_consistency, hd_power_of_2_fraction
3. **Timestamp**: timestamp_correlation (2015-01-15)
4. **Float Artifacts**: float_rounding_error, exact_fraction_matches
5. **Hash Chains**: hash_chain_correlation
6. **Side Channels**: low_8_bit_entropy, low_bit_correlation
7. **Wallet Quirks**: wallet_batch_generation, mnemonic_entropy_pattern
8. **Human Patterns**: human_boundary_avoidance, human_round_percent, human_middle_bias
9. **Modular Arithmetic**: mod_97_diversity, mod_101_diversity, ..., coprime_to_small_primes

## Scoring System

The `evaluate` function scores based on:

1. **Prediction Accuracy** (exponential reward):
   - Perfect prediction (error < 0.01): +50 points per puzzle
   - Good prediction (error < 0.05): +20 points per puzzle
   - Decent prediction (error < 0.1): +10 points per puzzle

2. **Statistical Quality**:
   - Mean Absolute Error (MAE): Lower is better
   - Correlation: Higher is better
   - R-squared: Closer to 1.0 is better

3. **Puzzle 135 Bonus**:
   - Non-trivial prediction: +20 points

## Monitoring Progress

Watch these files:
- `data/last_eval.txt` - Latest evaluation results
- `data/backups/` - Saved program databases
- Console output - Real-time progress

## Expected Timeline

FunSearch is an evolutionary algorithm - it may take:
- **Hours** to find simple patterns
- **Days** to discover complex formulas
- **Weeks** to converge on optimal solutions

## What Success Looks Like

A successful discovery will:
1. **High scores** (>500) in evaluation
2. **Accurate predictions** across all puzzles (error < 0.05)
3. **Novel formula** in the `priority` function
4. **Puzzle 135 prediction** that's reasonable (0.0-1.0, not 0.5)

## Notes

- The baseline score is ~136.20 (simple recursive prediction)
- Any score >200 indicates pattern discovery
- The actual private key generation method is unknown
- This is an open research problem - breakthrough discoveries possible!

## Full Dependencies Required

If not already installed:
```bash
pip install -r requirements.txt
```

Key dependencies:
- numpy, scipy
- cloudpickle, absl-py
- LLM backend (llama-cpp-python or OpenAI API)

## Alternative: Run Without Full FunSearch

To just test the evaluation function:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
exec(open('examples/puzzle.py').read())
print(f'Score: {evaluate(42):.2f}')
"
```
