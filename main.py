import sys
import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import text_to_speech

def main():
    # Step 1: Train the model
    train_model()
    
    # Step 2: Evaluate the model
    metrics = evaluate_model(
        model_path="tts_project/best_model.keras",
        test_texts=test_texts,  # Load your test data here
        test_mels=test_mels
    )
    print(f"Evaluation Metrics: MCD={metrics['mcd']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Step 3: Generate speech from text
    text_to_speech(
        model_path="tts_project/best_model.keras",
        text="Hello world, this is a text-to-speech example.",
        output_path="generated_speech.wav"
    )
    print("Done! Audio saved to 'generated_speech.wav'")

if __name__ == "__main__":
    # Add project directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import test data (modify as needed)
    from tts_project.preprocessing import load_data
    _, _, test_texts, test_mels = load_data()
    
    main()
