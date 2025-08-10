import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10, nhead=4, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.embedding(x).unsqueeze(1)  # (batch_size, seq_len=1, embed_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
