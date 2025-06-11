import tensorflow as tf
from model import TTSModel
from preprocessing import load_data, text_vectorizer
import numpy as np

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
VOCAB_SIZE = 1000
MEL_BANDS = 80

# Load data
train_texts, train_mels, test_texts, test_mels = load_data()
train_text_vec = text_vectorizer(train_texts).numpy()
test_text_vec = text_vectorizer(test_texts).numpy()

# Initialize model
tts_model = TTSModel(VOCAB_SIZE, EMBEDDING_DIM, ENCODER_LSTM_UNITS, DECODER_LSTM_UNITS, MEL_BANDS)
tts_model.compile(optimizer='adam', loss='mse')

# Train
history = tts_model.fit(
    x=(train_text_vec, train_mels),
    y=train_mels,
    validation_data=((test_text_vec, test_mels), test_mels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.keras', save_best_only=True)
    ]
)

print("Training completed. Model saved to 'models/best_model.keras'")
