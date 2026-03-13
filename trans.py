
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

print("Loading features...")

features = np.load("features.npy")
labels = np.load("labels.npy")

print("Feature shape:", features.shape)

# -------------------------------------
# DATASET
# -------------------------------------

class HeartDataset(Dataset):

    def __len__(self):
        return len(features)

    def __getitem__(self, idx):

        x = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(labels[idx], dtype=torch.long)

        return x, y


# -------------------------------------
# TRANSFORMER MODEL
# -------------------------------------

class HeartTransformer(nn.Module):

    def __init__(self, input_dim=64, num_classes=4):

        super().__init__()

        self.embed = nn.Linear(input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.embed(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        out = self.classifier(x)

        return out


# -------------------------------------
# TRAINING
# -------------------------------------

dataset = HeartDataset()

loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = HeartTransformer()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("Training started...")

epochs = 3

for epoch in range(epochs):

    total_loss = 0

    for i, (x, y) in enumerate(loader):

        print(f"Epoch {epoch+1} Batch {i+1}/{len(loader)}")

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "heart_transformer_model.pth")

print("Model saved successfully")

