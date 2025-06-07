import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from preprocessing import *

# Hyperparameters (Tune it for better peformance)
EPOCHS = 50
BATCH_SIZE = 16
EMBEDDING_DIM = 16
ENCODER_LSTM_UNITS = 16
DECODER_LSTM_UNITS = 32
ATTENTION_DIM = 16
MEL_BANDS = 80
MAX_TEXT_LENGTH = 150
MAX_MEL_LENGTH = 900

# Encoder
class Encoder(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.Bidirectional(layers.LSTM(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        return output, state_h, state_c

# Simple Dot-Product Attention
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, query, values):
        # query: (batch, hidden)
        # values: (batch, time, hidden)
        query = tf.expand_dims(query, 1)  # (batch, 1, hidden)
        # Dot product attention
        score = tf.matmul(query, values, transpose_b=True)  # (batch, 1, time)
        attention_weights = tf.nn.softmax(score, axis=-1)   # (batch, 1, time)
        context_vector = tf.matmul(attention_weights, values)  # (batch, 1, hidden)
        context_vector = tf.squeeze(context_vector, axis=1)    # (batch, hidden)
        attention_weights = tf.squeeze(attention_weights, axis=1)  # (batch, time)
        return context_vector, attention_weights

# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, mel_dim, dec_units):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.attention = SimpleAttention()
        self.fc = tf.keras.layers.Dense(mel_dim)

    def call(self, x, enc_output, state_h, state_c):
        # x: (batch, 1, mel_dim)
        context_vector, attention_weights = self.attention(state_h, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, h, c = self.lstm(x, initial_state=[state_h, state_c])
        mel_output = self.fc(output)
        return mel_output, h, c, attention_weights

# Full Model
class TTSModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, mel_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units)
        self.decoder = Decoder(mel_dim, dec_units)
        self.mel_dim = mel_dim
        
    def call(self, inputs):
        # inputs should be a tuple of (text_input, mel_input)
        text_input, mel_input = inputs
        batch_size = tf.shape(text_input)[0]
        
        enc_output, state_h, state_c = self.encoder(text_input)
        
        # Initialize decoder with zeros for first time step
        dec_input = tf.zeros((batch_size, 1, self.mel_dim))
        outputs = []
        
        # We need to predict MAX_MEL_LENGTH steps
        for t in range(MAX_MEL_LENGTH):
            mel_output, state_h, state_c, _ = self.decoder(dec_input, enc_output, state_h, state_c)
            outputs.append(mel_output)
            # Teacher forcing - use ground truth as next input
            dec_input = tf.expand_dims(mel_input[:, t, :], 1) if mel_input is not None else mel_output
            
        outputs = tf.concat(outputs, axis=1)
        return outputs

# Usage:
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
ENC_UNITS = 16
DEC_UNITS = 32
MEL_BANDS = 80

tts_model = TTSModel(VOCAB_SIZE, EMBEDDING_DIM, ENC_UNITS, DEC_UNITS, MEL_BANDS)
tts_model.compile(optimizer='adam', loss='mse')

# Evaluation
from scipy.spatial.distance import cdist

def calculate_mcd(mel_true, mel_pred):
    """Calculate Mel-Cepstral Distortion"""
    # Convert power to dB
    mel_true = librosa.power_to_db(mel_true)
    mel_pred = librosa.power_to_db(mel_pred)
    
    # Calculate MCD
    diff = mel_true - mel_pred
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
    return mcd

def calculate_rmse(mel_true, mel_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((mel_true - mel_pred)**2))


def evaluate_model(model, test_texts, test_mels, text_vectorizer):
    metrics = {'mcd': [], 'rmse': []}
    
    for text, mel_true in zip(test_texts, test_mels):
        # Vectorize text
        text_vec = text_vectorizer([text]).numpy()
        
        # Predict mel-spectrogram
        mel_pred = model.predict([text_vec, np.zeros((1, MAX_MEL_LENGTH, MEL_BANDS))])[0]
        
        # Calculate metrics
        metrics['mcd'].append(calculate_mcd(mel_true, mel_pred))
        metrics['rmse'].append(calculate_rmse(mel_true, mel_pred))
    
    return {k: np.mean(v) for k, v in metrics.items()}

# Inference    
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

# 1. Train the model
history = tts_model.fit(
    x=(train_text_vec, train_mels),
    y=train_mels,
    validation_data=((test_text_vec, test_mels), test_mels),
    batch_size=16,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]
)

# 2. Evaluate
metrics = evaluate_model(tts_model, test_texts, test_mels, text_vectorizer)
print("Evaluation Metrics:")
print(f"MCD: {metrics['mcd']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# 3. Generate speech
text_to_test = "Hello world, this is a test of text to speech."
audio, mel = text_to_speech(tts_model, text_to_test, text_vectorizer)

# Save
import soundfile as sf
sf.write('generated_speech.wav', audio, 22050)


