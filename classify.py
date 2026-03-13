import torch
import torch.nn as nn
import librosa
import numpy as np
import os

# --------------------------------
# CONFIG
# --------------------------------

SAMPLE_RATE = 4000
MAX_LEN = 400

label_map = {
0: "Systolic Murmur",
1: "Diastolic Murmur",
2: "Continuous Murmur",
3: "No Murmur"
}

disease_hint = {
"Systolic Murmur": "Possible conditions: Mitral Regurgitation / Aortic Stenosis",
"Diastolic Murmur": "Possible conditions: Aortic Regurgitation",
"Continuous Murmur": "Possible conditions: Patent Ductus Arteriosus (PDA)",
"No Murmur": "Heart sound appears normal"
}

# --------------------------------
# MODEL
# --------------------------------

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
        out = self.classifier(x)

        return out


# --------------------------------
# LOAD MODEL
# --------------------------------

print("Loading trained model...")

model = HeartTransformer()

model.load_state_dict(torch.load("heart_transformer_model.pth", map_location="cpu"))

model.eval()

print("Model loaded successfully\n")


# --------------------------------
# FEATURE EXTRACTION
# --------------------------------

def extract_features(audio_path):

    signal,_ = librosa.load(audio_path,sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )

    mel = librosa.power_to_db(mel)

    mel = mel.T

    if mel.shape[0] < MAX_LEN:

        pad = MAX_LEN - mel.shape[0]

        mel = np.pad(mel,((0,pad),(0,0)))

    else:

        mel = mel[:MAX_LEN,:]

    mel = torch.tensor(mel,dtype=torch.float32).unsqueeze(0)

    return mel


# --------------------------------
# USER INPUT
# --------------------------------

audio = input("Enter heart sound file path: ")

if not os.path.exists(audio):

    print(" File not found")
    exit()


# --------------------------------
# FEATURE EXTRACTION
# --------------------------------

print("\nProcessing audio...")

features = extract_features(audio)

print("Feature extraction completed")


# --------------------------------
# PREDICTION
# --------------------------------

with torch.no_grad():

    output = model(features)

    prediction = torch.argmax(output,dim=1).item()

    probs = torch.softmax(output,dim=1)

    confidence = probs[0][prediction].item()


result = label_map[prediction]


# --------------------------------
# OUTPUT
# --------------------------------

print("\n==============================")
print(" HEART SOUND ANALYSIS")
print("==============================")

print("Detected Murmur Type:",result)

print("Confidence:",round(confidence*100,2),"%")

print("\nClinical Hint:")

print(disease_hint[result])

print("==============================")
