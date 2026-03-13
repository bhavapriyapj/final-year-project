import os
import pandas as pd
import librosa
import numpy as np

AUDIO_PATH = r"C:\Users\USER\OneDrive\Desktop\mumur_phase2\the-circor-digiscope-phonocardiogram-dataset-1.0.3\heart_murmurAI\model_training\training_data"

METADATA_PATH = r"C:\Users\USER\OneDrive\Desktop\mumur_phase2\the-circor-digiscope-phonocardiogram-dataset-1.0.3\heart_murmurAI\model_training\balanced_training_data.csv"

SAMPLE_RATE = 4000
MAX_LEN = 400

label_map = {
    "systolic":0,
    "diastolic":1,
    "continuous":2,
    "none":3
}

metadata = pd.read_csv(METADATA_PATH)

features = []
labels = []

for i,row in metadata.iterrows():

    patient = str(row["Patient ID"])
    locations = str(row["Recording locations:"]).split("+")
    label = label_map[row["chd types label"]]

    for loc in locations:

        file = f"{patient}_{loc}.wav"
        path = os.path.join(AUDIO_PATH,file)

        if os.path.exists(path):

            signal,_ = librosa.load(path,sr=SAMPLE_RATE)

            mel = librosa.feature.melspectrogram(
                y=signal,
                sr=SAMPLE_RATE,
                n_mels=64,
                n_fft=1024,
                hop_length=256
            )

            mel = librosa.power_to_db(mel)

            mel = mel.T   # (time , features)

            # PAD / TRIM
            if mel.shape[0] < MAX_LEN:
                pad = MAX_LEN - mel.shape[0]
                mel = np.pad(mel, ((0,pad),(0,0)))
            else:
                mel = mel[:MAX_LEN,:]

            features.append(mel)
            labels.append(label)

    if i % 50 == 0:
        print("Processed", i)

features = np.array(features)
labels = np.array(labels)

np.save("features.npy",features)
np.save("labels.npy",labels)

print("Feature extraction complete")
print("Features shape:",features.shape)
