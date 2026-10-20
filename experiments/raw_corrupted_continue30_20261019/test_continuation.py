import unittest
from continuation import learning_rate


class Schedule(unittest.TestCase):
    def test_same_learning_rate_across_all_segments(self):
        for step in [45000, 45001, 67500, 90000, 112500, 134999, 135000]:
            self.assertEqual(learning_rate(step, 4500), 1e-5)

    def test_reject_wrong_counter(self):
        with self.assertRaises(AssertionError):
            learning_rate(0, 4500)


if __name__ == "__main__":
    unittest.main()
