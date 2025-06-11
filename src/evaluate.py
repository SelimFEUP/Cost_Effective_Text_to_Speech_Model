import numpy as np
import librosa
from model import TTSModel
from preprocessing import text_vectorizer
import tensorflow as tf

def calculate_mcd(mel_true, mel_pred):
    mel_true = librosa.power_to_db(mel_true)
    mel_pred = librosa.power_to_db(mel_pred)
    diff = mel_true - mel_pred
    return np.mean(np.sqrt(np.sum(diff**2, axis=1)))

def calculate_rmse(mel_true, mel_pred):
    return np.sqrt(np.mean((mel_true - mel_pred)**2))

def evaluate_model(model_path, test_texts, test_mels):
    model = tf.keras.models.load_model(model_path)
    metrics = {'mcd': [], 'rmse': []}
    
    for text, mel_true in zip(test_texts, test_mels):
        text_vec = text_vectorizer([text]).numpy()
        mel_pred = model.predict([text_vec, np.zeros((1, MAX_MEL_LENGTH, MEL_BANDS))])[0]
        metrics['mcd'].append(calculate_mcd(mel_true, mel_pred))
        metrics['rmse'].append(calculate_rmse(mel_true, mel_pred))
    
    return {k: np.mean(v) for k, v in metrics.items()}
