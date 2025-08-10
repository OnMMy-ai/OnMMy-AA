import torch
import torch.nn as nn
import yaml
from models.cnn import CNNModel
from data.dataset import get_test_loader
from src.utils import accuracy

def evaluate():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_test_loader(config['training']['batch_size'])
    model = CNNModel(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # Load saved model weights (assuming saved at 'model.pth')
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    avg_loss = total_loss / len(test_loader)
    avg_acc = total_acc / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

if __name__ == '__main__':
    evaluate()
