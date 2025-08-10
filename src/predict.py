import torch
import yaml
from models.cnn import CNNModel
from torchvision.transforms import ToTensor
from PIL import Image

def predict(image_path):
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNModel(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    image = Image.open(image_path).convert('L')
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
    print(f'Predicted class: {pred}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python predict.py <image_path>')
    else:
        predict(sys.argv[1])
