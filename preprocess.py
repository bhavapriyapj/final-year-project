import os
import librosa
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

AUDIO_PATH = r"C:\Users\USER\OneDrive\Desktop\mumur_phase2\the-circor-digiscope-phonocardiogram-dataset-1.0.3\heart_murmurAI\model_training\training_data"

METADATA_PATH = r"C:\Users\USER\OneDrive\Desktop\mumur_phase2\the-circor-digiscope-phonocardiogram-dataset-1.0.3\heart_murmurAI\model_training\balanced_training_data.csv"

SAMPLE_RATE = 4000


# --------------------------------------------------
# AUDIO LOADER
# --------------------------------------------------

def load_audio(file_path):

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if np.max(np.abs(signal)) != 0:
        signal = signal / np.max(np.abs(signal))

    signal, _ = librosa.effects.trim(signal)

    return signal


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    metadata = pd.read_csv(METADATA_PATH)

    dataset = []

    for _, row in metadata.iterrows():

        patient_id = str(row["Patient ID"])

        locations = str(row["Recording locations:"]).split("+")

        label = row["chd types label"]

        for loc in locations:

            file_name = f"{patient_id}_{loc}.wav"

            file_path = os.path.join(AUDIO_PATH, file_name)

            if os.path.exists(file_path):

                try:

                    signal = load_audio(file_path)

                    dataset.append({
                        "patient_id": patient_id,
                        "location": loc,
                        "file": file_name,
                        "signal": signal,
                        "label": label
                    })

                except Exception as e:

                    print("Error loading:", file_name)

            else:

                print("Missing:", file_name)

    print("Total loaded recordings:", len(dataset))

    return dataset


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    data = load_dataset()

    print("\nExample:\n")

    print("Patient:", data[0]["patient_id"])
    print("File:", data[0]["file"])
    print("Signal length:", len(data[0]["signal"]))
    print("Label:", data[0]["label"])
