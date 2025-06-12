import sys
import os
import tensorflow as tf
from tensorflow.keras import layers
from src.train import train_model
from src.evaluate import tts_model, evaluate_model
from src.inference import text_to_speech
from src.preprocessing import create_text_vectorizer, load_data, train_texts, test_texts

def main():
    # Step 1: Train the model
    train_model()
    
    # Step 2: Evaluate the model
    metrics = evaluate_model(tts_model, test_texts, test_mels, text_vectorizer)
    print(f"Evaluation Metrics: MCD={metrics['mcd']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Step 3: Generate speech from text
    text_to_test = "Hello world, this is a test of text to speech."
    audio, mel = text_to_speech(tts_model, text_to_test, text_vectorizer)

if __name__ == "__main__":
    text_vectorizer = create_text_vectorizer(train_texts + test_texts)
    _, test_mels, _, _ = load_data()
    main()
    
