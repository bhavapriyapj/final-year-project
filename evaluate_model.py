
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# LOAD DATA
# ------------------------------

features = np.load("features.npy")
labels = np.load("labels.npy")

print("Dataset shape:", features.shape)

# ------------------------------
# DATASET
# ------------------------------

class HeartDataset(Dataset):

    def __len__(self):
        return len(features)

    def __getitem__(self, idx):

        x = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(labels[idx], dtype=torch.long)

        return x, y


dataset = HeartDataset()
loader = DataLoader(dataset, batch_size=128)

# ------------------------------
# MODEL
# ------------------------------

class HeartTransformer(nn.Module):

    def __init__(self,input_dim=64,num_classes=4):

        super().__init__()

        self.embed = nn.Linear(input_dim,64)

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

        self.classifier = nn.Linear(64,num_classes)

    def forward(self,x):

        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.classifier(x)


# ------------------------------
# LOAD TRAINED MODEL
# ------------------------------

model = HeartTransformer()
model.load_state_dict(torch.load("heart_transformer_model.pth"))
model.eval()

print("Model loaded")

# ------------------------------
# EVALUATION
# ------------------------------

all_preds = []
all_labels = []

with torch.no_grad():

    for x,y in loader:

        outputs = model(x)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

# ------------------------------
# METRICS
# ------------------------------

print("\nAccuracy:", accuracy_score(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

