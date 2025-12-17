"""Standalone analyzer for FunSearch puzzle predictors.

This script loads a puzzle module (default: examples/puzzle.py),
executes its ``priority`` predictor over all eligible solved puzzles,
and writes a detailed text report with summary statistics and per-puzzle
results. It mirrors the evaluation logic from the puzzle module so the
reported score should match ``evaluate(seed)`` while surfacing richer
error metrics.
"""

import argparse
import importlib
import importlib.util
import math
import pathlib
from typing import Any, Dict, List

import numpy as np


def load_module(path_or_name: str):
  """Load a Python module by file path or importable name."""
  if path_or_name.endswith(".py"):
    module_path = pathlib.Path(path_or_name).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
  return importlib.import_module(path_or_name)


def compute_predictions(module: Any, min_puzzle: int, min_history: int):
  solved = module.SOLVED_PUZZLES
  predict_fn = module.priority
  feature_fn = module.compute_puzzle_features
  ratio_fn = module.get_position_ratio
  range_fn = module.get_puzzle_range

  predictions: List[float] = []
  actuals: List[float] = []
  per_puzzle: List[Dict[str, Any]] = []
  per_puzzle_scores: List[float] = []
  eligible = 0
  evaluated = 0

  for puzzle_num in sorted(solved):
    if puzzle_num < min_puzzle:
      continue

    history = {k: v for k, v in solved.items() if k < puzzle_num}
    if len(history) < min_history:
      continue

    eligible += 1

    features = feature_fn(puzzle_num, history)
    predicted_ratio = predict_fn(features)
    clipped = False

    if not np.isfinite(predicted_ratio):
      per_puzzle_scores.append(-20.0)
      per_puzzle.append({
          "puzzle": puzzle_num,
          "status": "non-finite",
          "actual_ratio": ratio_fn(puzzle_num, solved[puzzle_num]),
          "predicted_ratio": float("nan"),
          "error": float("nan"),
          "predicted_key": None,
          "range": range_fn(puzzle_num),
      })
      evaluated += 1
      continue

    if predicted_ratio < 0.0 or predicted_ratio > 1.0:
      clipped = True
    predicted_ratio = max(0.0, min(1.0, float(predicted_ratio)))

    actual_ratio = ratio_fn(puzzle_num, solved[puzzle_num])
    rmin, rmax = range_fn(puzzle_num)
    rsize = rmax - rmin
    predicted_key = round(rmin + predicted_ratio * rsize)
    actual_key = solved[puzzle_num]

    error = abs(predicted_ratio - actual_ratio)
    puzzle_score = math.exp(-10 * error) * 10.0
    if error < 0.01:
      puzzle_score += 50.0
    elif error < 0.05:
      puzzle_score += 20.0
    elif error < 0.1:
      puzzle_score += 10.0
    if clipped:
      puzzle_score -= 5.0

    per_puzzle_scores.append(puzzle_score)
    predictions.append(predicted_ratio)
    actuals.append(actual_ratio)
    evaluated += 1

    per_puzzle.append({
        "puzzle": puzzle_num,
        "status": "ok",
        "actual_ratio": actual_ratio,
        "predicted_ratio": predicted_ratio,
        "error": error,
        "predicted_key": predicted_key,
        "actual_key": actual_key,
        "range": (rmin, rmax),
        "clipped": clipped,
        "puzzle_score": puzzle_score,
    })

  coverage = evaluated / max(eligible, 1)

  if per_puzzle_scores:
    base_score = float(np.mean(per_puzzle_scores))
  else:
    base_score = -50.0

  metrics = {
      "eligible": eligible,
      "evaluated": evaluated,
      "coverage": coverage,
      "per_puzzle": per_puzzle,
      "base_score": base_score,
  }

  if predictions:
    preds = np.array(predictions)
    acts = np.array(actuals)

    mae = np.mean(np.abs(preds - acts))
    rmse = math.sqrt(np.mean((preds - acts) ** 2))
    corr = np.corrcoef(preds, acts)[0, 1] if np.std(preds) > 0 and np.std(acts) > 0 else float("nan")
    ss_res = np.sum((acts - preds) ** 2)
    ss_tot = np.sum((acts - np.mean(acts)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    bounded_mae = min(max(mae, 0.0), 1.0)
    score = base_score + max(0.0, 100.0 * (1.0 - bounded_mae) * coverage)
    if np.isfinite(corr):
      score += corr * (50.0 if corr >= 0 else 25.0) * coverage
    if np.isfinite(r2):
      r2 = float(np.clip(r2, -1.0, 1.0))
      score += r2 * (100.0 if r2 >= 0 else 50.0) * coverage

    metrics.update({
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "r2": r2,
        "score": score,
        "within_1pct": int(sum(1 for p in per_puzzle if p.get("error", 1) < 0.01)),
        "within_5pct": int(sum(1 for p in per_puzzle if p.get("error", 1) < 0.05)),
        "within_10pct": int(sum(1 for p in per_puzzle if p.get("error", 1) < 0.1)),
        "max_error": max((p.get("error", 0.0) for p in per_puzzle if math.isfinite(p.get("error", float("nan")))), default=float("nan")),
    })
  else:
    metrics.update({
        "mae": float("nan"),
        "rmse": float("nan"),
        "corr": float("nan"),
        "r2": float("nan"),
        "score": base_score,
        "within_1pct": 0,
        "within_5pct": 0,
        "within_10pct": 0,
        "max_error": float("nan"),
    })

  return metrics


def format_report(module: Any, metrics: Dict[str, Any]) -> str:
  lines: List[str] = []
  lines.append("Puzzle prediction analysis")
  lines.append("===========================")
  lines.append("")
  lines.append(f"Module           : {module.__file__}")
  lines.append(f"Priority func    : {module.priority.__name__}")
  lines.append(f"Solved puzzles   : {len(module.SOLVED_PUZZLES)}")
  lines.append(f"Eligible puzzles : {metrics['eligible']}")
  lines.append(f"Evaluated puzzles: {metrics['evaluated']}")
  lines.append(f"Coverage ratio   : {metrics['coverage']:.3f}")
  lines.append("")

  lines.append("Overall quality metrics")
  lines.append("------------------------")
  lines.append(f"Mean abs error      : {metrics['mae']:.4f}")
  lines.append(f"RMSE                : {metrics['rmse']:.4f}")
  lines.append(f"Correlation         : {metrics['corr']:.4f}")
  lines.append(f"R^2                 : {metrics['r2']:.4f}")
  lines.append(f"Max absolute error  : {metrics['max_error']:.4f}")
  lines.append("")

  lines.append("Threshold counts")
  lines.append("----------------")
  lines.append(f"<1% ratio error : {metrics['within_1pct']}")
  lines.append(f"<5% ratio error : {metrics['within_5pct']}")
  lines.append(f"<10% ratio error: {metrics['within_10pct']}")
  lines.append("")

  lines.append("Scoring (mirrors puzzle.evaluate)")
  lines.append("----------------------------------")
  lines.append(f"Base per-puzzle score mean : {metrics['base_score']:.4f}")
  lines.append(f"Composite evaluation score : {metrics['score']:.4f}")
  lines.append("")

  lines.append("Per-puzzle details (sorted by puzzle number)")
  lines.append("--------------------------------------------")
  header = (
      f"{'#':>4}  {'actual_ratio':>12}  {'pred_ratio':>12}  {'abs_err':>8}  "
      f"{'puzzle_score':>13}  {'pred_key':>10}  {'actual_key':>10}  {'range':>15}  status"
  )
  lines.append(header)
  lines.append("-" * len(header))

  for entry in sorted(metrics["per_puzzle"], key=lambda e: e["puzzle"]):
    rng = entry["range"]
    range_str = f"({rng[0]}, {rng[1]})"
    lines.append(
        f"{entry['puzzle']:>4}  "
        f"{entry['actual_ratio']:>12.6f}  "
        f"{entry['predicted_ratio']:>12.6f}  "
        f"{entry.get('error', float('nan')):>8.4f}  "
        f"{entry.get('puzzle_score', float('nan')):>13.6f}  "
        f"{str(entry.get('predicted_key')):>10}  "
        f"{str(entry.get('actual_key')):>10}  "
        f"{range_str:>15}  "
        f"{entry.get('status', '')}{' clipped' if entry.get('clipped') else ''}"
    )

  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description="Analyze puzzle predictor accuracy.")
  parser.add_argument("module", nargs="?", default="examples/puzzle.py",
                      help="Module path or import name containing SOLVED_PUZZLES and priority().")
  parser.add_argument("--output", "-o", default="puzzle_analysis.txt",
                      help="Path to write the analysis report.")
  parser.add_argument("--min-puzzle", type=int, default=10, help="Skip puzzles below this number.")
  parser.add_argument("--min-history", type=int, default=5, help="Require at least this many prior solved puzzles.")
  args = parser.parse_args()

  module = load_module(args.module)
  metrics = compute_predictions(module, min_puzzle=args.min_puzzle, min_history=args.min_history)
  report = format_report(module, metrics)
  pathlib.Path(args.output).write_text(report)
  print(f"Analysis written to {args.output}")

  if hasattr(module, "evaluate"):
    try:
      eval_score = module.evaluate(0)
      print(f"Module evaluate(0) score: {eval_score:.4f}")
    except Exception as exc:  # pragma: no cover - defensive output only
      print(f"evaluate(0) failed: {exc}")


if __name__ == "__main__":
  main()
