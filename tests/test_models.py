import unittest
import torch
from models.cnn import CNNModel

class TestModels(unittest.TestCase):
    def test_cnn_forward(self):
        model = CNNModel()
        input_tensor = torch.randn(1, 1, 28, 28)
        output = model(input_tensor)
        self.assertEqual(output.shape[1], 10)  # num_classes

if __name__ == '__main__':
    unittest.main()
