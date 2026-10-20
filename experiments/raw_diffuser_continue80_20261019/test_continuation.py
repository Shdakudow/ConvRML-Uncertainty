import unittest, torch
from continuation import learning_rate


class Schedule(unittest.TestCase):
    def test_restart_and_final_rate(self):
        steps = 4500
        self.assertAlmostEqual(learning_rate(50 * steps, steps), 1e-5)
        self.assertAlmostEqual(learning_rate(80 * steps, steps), 1e-9)
        values = [learning_rate(n * steps, steps) for n in range(50, 81)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_resume_does_not_reset_schedule(self):
        self.assertEqual(
            learning_rate(55 * 4500 + 123, 4500), learning_rate(247623, 4500)
        )


if __name__ == "__main__":
    unittest.main()
