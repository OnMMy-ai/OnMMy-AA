from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from .preprocess import get_transforms

def get_train_loader(batch_size):
    train_dataset = MNIST(root='./data', train=True, download=True, transform=get_transforms())
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size):
    test_dataset = MNIST(root='./data', train=False, download=True, transform=get_transforms())
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
