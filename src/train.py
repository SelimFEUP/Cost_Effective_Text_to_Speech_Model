import tensorflow as tf
from src.model import TTSModel
from src.preprocessing import load_data
import numpy as np
import os

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
VOCAB_SIZE = 1000
MEL_BANDS = 80
EMBEDDING_DIM = 16 
ENCODER_LSTM_UNITS = 16 
DECODER_LSTM_UNITS = 32 

def train_model():
    # Load data
    train_mels, test_mels, train_text_vec, test_text_vec = load_data()
    #train_text_vec = text_vectorizer(train_texts).numpy()
    #test_text_vec = text_vectorizer(test_texts).numpy()

    # Initialize model
    tts_model = TTSModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        enc_units=ENCODER_LSTM_UNITS,
        dec_units=DECODER_LSTM_UNITS,
        mel_dim=MEL_BANDS
    )
    tts_model.compile(optimizer='adam', loss='mse')

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train
    history = tts_model.fit(
        x=(train_text_vec, train_mels),
        y=train_mels,
        validation_data=((test_text_vec, test_mels), test_mels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.keras',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )

    print("Training completed. Model saved to 'models/best_model.keras'")
    return history
