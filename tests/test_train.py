import unittest
import torch
from src.utils import accuracy

class TestTrain(unittest.TestCase):
    def test_accuracy(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        acc = accuracy(outputs, targets)
        self.assertAlmostEqual(acc, 1.0)

if __name__ == '__main__':
    unittest.main()
