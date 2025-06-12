import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
from src.model import TTSModel
from src.preprocessing import create_text_vectorizer

MAX_TEXT_LENGTH = 150
MAX_MEL_LENGTH = 900
VOCAB_SIZE = 1000
MEL_BANDS = 80

def text_to_speech(model, text, text_vectorizer, max_length=MAX_MEL_LENGTH):
    # Vectorize text
    text_vec = text_vectorizer([text]).numpy()
    
    # Initialize with zeros
    dec_input = np.zeros((1, max_length, MEL_BANDS))
    
    # Generate mel-spectrogram
    mel_output = model.predict([text_vec, dec_input])[0]
    
    # Convert mel to audio
    mel_db = librosa.db_to_power(mel_output.T)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_db, 
        sr=22050, 
        n_fft=1024, 
        hop_length=256
    )
    
    return audio, mel_output
