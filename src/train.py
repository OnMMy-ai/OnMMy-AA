import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models.cnn import CNNModel
from data.dataset import get_train_loader
from src.logger import setup_logger
from src.utils import accuracy

def train():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    setup_logger(config['logging']['log_dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_train_loader(config['training']['batch_size'])
    model = CNNModel(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    epochs = config['training']['epochs']

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_acc = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

if __name__ == '__main__':
    train()
