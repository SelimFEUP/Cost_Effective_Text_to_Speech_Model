import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
from model import TTSModel
from preprocessing import text_vectorizer

def text_to_speech(model_path, text, output_path="generated_speech.wav"):
    model = tf.keras.models.load_model(model_path)
    text_vec = text_vectorizer([text]).numpy()
    dec_input = np.zeros((1, MAX_MEL_LENGTH, MEL_BANDS))
    mel_output = model.predict([text_vec, dec_input])[0]
    
    mel_db = librosa.db_to_power(mel_output.T)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_db, sr=22050, n_fft=1024, hop_length=256
    )
    
    sf.write(output_path, audio, 22050)
    print(f"Audio saved to {output_path}")
