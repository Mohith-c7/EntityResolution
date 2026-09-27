"""Regression checks for a complete multi-epoch plan without training a model."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "research/final_2h/neural_epochs.py"
spec = importlib.util.spec_from_file_location("neural_epochs", SCRIPT)
epochs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(epochs)


class EpochPlanTests(unittest.TestCase):
    def test_each_epoch_has_a_new_reproducible_complete_permutation(self):
        orders = [epochs.epoch_order(1200, 42, epoch) for epoch in (1, 2, 3)]
        for number, order in enumerate(orders, 1):
            self.assertTrue(np.array_equal(order, epochs.epoch_order(1200, 42, number)))
            self.assertEqual(sorted(order.tolist()), list(range(1200)))
        self.assertFalse(np.array_equal(orders[0], orders[1]))
        self.assertFalse(np.array_equal(orders[1], orders[2]))

    def test_scheduler_decays_over_all_three_epochs(self):
        total, warmup = 11250, 675
        rates = [epochs.lr_multiplier(step, total, warmup) for step in range(total + 1)]
        self.assertTrue(all(0 <= value <= 1 for value in rates))
        self.assertGreater(rates[0], 0)
        self.assertEqual(rates[warmup - 1], 1)
        self.assertGreater(rates[3750], rates[7500])
        self.assertGreater(rates[7500], rates[total - 1])
        self.assertGreater(rates[total - 1], 0)
        self.assertEqual(rates[total], 0)
        self.assertTrue(all(a >= b for a, b in zip(rates[warmup:], rates[warmup + 1:])))

    def test_invalid_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            epochs.lr_multiplier(0, 3, 3)


if __name__ == "__main__":
    unittest.main()
