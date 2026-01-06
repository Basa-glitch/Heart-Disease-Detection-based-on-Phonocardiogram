import numpy as np
import librosa
import joblib

# Load your saved model
model = joblib.load("C:/Users/bryan/OneDrive/Desktop/Heart Sound using machine learning and stethoscope/Normal, Murmur & Extrastole/Second Set/svm-2-17 Nov(90 to 10).joblib")

# Define the mapping for potential output variations
predicted_label_map = {
    "Normal": "Normal",
    "Normal_2": "Normal",
    "Murmur": "Murmur",
    "Murmur_2": "Murmur",
    "Extrasystole": "Extrasystole",
    "Extrasystole_2": "Extrasystole"
}

# Function to load and preprocess the audio file
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

# Path to the audio file to test
audio_file_path = "C:/Users/bryan/OneDrive/Desktop/Heart Sound using machine learning and stethoscope/Data2/Data/Murmur_2/1.wav"

# Preprocess the audio
X_test = preprocess_audio(audio_file_path)

# Define the expected result
expected_class_name = "Murmur"

# Predict using the model
y_pred = model.predict(X_test)
print(f"Raw model output (y_pred): {y_pred}")  # Inspect the output

# Map prediction to standard class name
predicted_class_name = predicted_label_map.get(y_pred[0], y_pred[0])  # Maps if found; otherwise uses the raw output

# Display the predicted result and the expected result
print(f"Predicted Result: {predicted_class_name}")
print(f"Expected Result: {expected_class_name}")
