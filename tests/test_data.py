import unittest
from data.dataset import get_train_loader

class TestData(unittest.TestCase):
    def test_train_loader(self):
        loader = get_train_loader(batch_size=32)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)  # inputs, labels

if __name__ == '__main__':
    unittest.main()
