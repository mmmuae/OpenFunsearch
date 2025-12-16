import unittest

from examples import puzzle


class PuzzleFeatureTests(unittest.TestCase):
  def test_unsolved_puzzle_ranges_are_populated(self):
    features = puzzle.compute_puzzle_features(135, puzzle.SOLVED_PUZZLES)
    expected_min = 2 ** (135 - 1)
    expected_max = (2 ** 135) - 1

    self.assertEqual(features['range_min'], expected_min)
    self.assertEqual(features['range_max'], expected_max)
    self.assertIsInstance(features['range_min'], int)
    self.assertIsInstance(features['range_max'], int)

  def test_transcendental_digits_use_safe_lookup(self):
    features = puzzle.compute_puzzle_features(135, puzzle.SOLVED_PUZZLES)

    expected_pi_digit = int(puzzle.PI_DIGITS[135]) / 10.0
    expected_e_digit = int(puzzle.E_DIGITS[135]) / 10.0

    self.assertEqual(features['pi_digit'], expected_pi_digit)
    self.assertEqual(features['e_digit'], expected_e_digit)


if __name__ == '__main__':
  unittest.main()
