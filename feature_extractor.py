import librosa
import numpy as np

# Spectrogram parameters
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


def extract_mel_spectrogram(signal, sr=4000):

    print("\nExtracting Mel Spectrogram...")

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    print("Mel Spectrogram created")

    mel_spec_db = librosa.power_to_db(mel_spec)

    print("Converted to decibel scale")

    print("Spectrogram shape:", mel_spec_db.shape)

    return mel_spec_db


# --------------------------------------------------
# TEST FUNCTION
# --------------------------------------------------

if __name__ == "__main__":

    # test with example audio
    test_audio = r"C:\Users\USER\OneDrive\Desktop\mumur_phase2\the-circor-digiscope-phonocardiogram-dataset-1.0.3\heart_murmurAI\model_training\training_data\2530_AV.wav"

    print("Loading test audio...")

    signal, sr = librosa.load(test_audio, sr=4000)

    print("Audio loaded successfully")
    print("Signal length:", len(signal))

    mel = extract_mel_spectrogram(signal)

    print("\nFeature extraction completed")
