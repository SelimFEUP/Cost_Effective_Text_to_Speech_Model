import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

MEL_BANDS = 80
MAX_TEXT_LENGTH = 150
MAX_MEL_LENGTH = 900

# Data preparation functions
def download_ljspeech():
    data_dir = "LJSpeech-1.1"
    if not os.path.exists(data_dir):
        raise Exception("Please download LJSpeech dataset from https://keithito.com/LJ-Speech-Dataset/")
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata.columns = ["file", "text", "normalized_text"]
    return metadata

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c in [' ', "'", ',', '.', '?', '!']])
    return text

def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=MEL_BANDS)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
    return mel.T

def prepare_dataset(metadata, data_dir, max_samples=None):
    file_paths = []
    texts = []
    mel_lengths = []
    samples = metadata[:max_samples] if max_samples else metadata
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        file_path = os.path.join(data_dir, "wavs", row["file"] + ".wav")
        mel = load_and_preprocess_audio(file_path)
        if mel.shape[0] <= MAX_MEL_LENGTH and len(row["normalized_text"]) <= MAX_TEXT_LENGTH:
            file_paths.append(file_path)
            texts.append(preprocess_text(row["normalized_text"]))
            mel_lengths.append(mel.shape[0])
    return file_paths, texts, mel_lengths

def create_text_vectorizer(texts):
    text_vectorizer = layers.TextVectorization(
        max_tokens=None,
        output_sequence_length=MAX_TEXT_LENGTH,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int"
    )
    text_vectorizer.adapt(texts)
    return text_vectorizer

metadata = download_ljspeech()
file_paths, texts, mel_lengths = prepare_dataset(metadata, "LJSpeech-1.1", max_samples=1000) # Adjust accordingly
train_files, test_files, train_texts, test_texts = train_test_split(
    file_paths, texts, test_size=0.2, random_state=42
)

def load_mels(file_list):
    mels = []
    for file in file_list:
        mel = load_and_preprocess_audio(file)
        # Pad or truncate to exactly MAX_MEL_LENGTH
        if mel.shape[0] < MAX_MEL_LENGTH:
            pad_width = [(0, MAX_MEL_LENGTH - mel.shape[0]), (0, 0)]
            mel = np.pad(mel, pad_width, mode='constant')
        else:
            mel = mel[:MAX_MEL_LENGTH]
        mels.append(mel)
    return np.array(mels)
    
def load_data():
    train_mels = load_mels(train_files)
    test_mels = load_mels(test_files)
    text_vectorizer = create_text_vectorizer(train_texts + test_texts)  # Fit on all text data
    train_text_vec = text_vectorizer(train_texts).numpy()
    test_text_vec = text_vectorizer(test_texts).numpy()
    #train_x = (train_text_vec, train_mels)
    #train_y = train_mels
    #test_x = (test_text_vec, test_mels)
    #test_y = test_mels
    return train_mels, test_mels, train_text_vec, test_text_vec
