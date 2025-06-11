import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
EMBEDDING_DIM = 16
ENCODER_LSTM_UNITS = 16
DECODER_LSTM_UNITS = 32
MEL_BANDS = 80
MAX_TEXT_LENGTH = 150
MAX_MEL_LENGTH = 900

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

class SimpleAttention(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = tf.matmul(query, values, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        context_vector = tf.squeeze(context_vector, axis=1)
        attention_weights = tf.squeeze(attention_weights, axis=1)
        return context_vector, attention_weights

class Decoder(layers.Layer):
    def __init__(self, mel_dim, dec_units):
        super().__init__()
        self.lstm = layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.attention = SimpleAttention()
        self.fc = layers.Dense(mel_dim)

    def call(self, x, enc_output, state_h, state_c):
        context_vector, attention_weights = self.attention(state_h, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, h, c = self.lstm(x, initial_state=[state_h, state_c])
        mel_output = self.fc(output)
        return mel_output, h, c, attention_weights

class TTSModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, mel_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units)
        self.decoder = Decoder(mel_dim, dec_units)
        self.mel_dim = mel_dim
        
    def call(self, inputs):
        text_input, mel_input = inputs
        batch_size = tf.shape(text_input)[0]
        enc_output, state_h, state_c = self.encoder(text_input)
        dec_input = tf.zeros((batch_size, 1, self.mel_dim))
        outputs = []
        
        for t in range(MAX_MEL_LENGTH):
            mel_output, state_h, state_c, _ = self.decoder(dec_input, enc_output, state_h, state_c)
            outputs.append(mel_output)
            dec_input = tf.expand_dims(mel_input[:, t, :], 1) if mel_input is not None else mel_output
            
        return tf.concat(outputs, axis=1)
