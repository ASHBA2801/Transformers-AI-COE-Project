import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Sample dataset
text = "deep learning is interesting deep learning is powerful"

# 2. Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

vocab_size = len(tokenizer.word_index) + 1

# 3. Create input-output pairs
X, y = [], []

for i in range(1, len(sequences)):
    X.append(sequences[:i])
    y.append(sequences[i])

# Pad sequences
X = tf.keras.utils.pad_sequences(X, padding="pre")
y = np.array(y)

# 4. Define Transformer model
inputs = layers.Input(shape=(X.shape[1],))

# Word embeddings
embedding = layers.Embedding(vocab_size, 32)(inputs)

# Basic self-attention
attention = layers.MultiHeadAttention(
    num_heads=2,
    key_dim=16
)(
    embedding,
    embedding,
    use_causal_mask=True
)

# Residual connection and normalization
x = layers.Add()([embedding, attention])
x = layers.LayerNormalization()(x)

# Feed-forward network
ffn = layers.Dense(64, activation="relu")(x)
ffn = layers.Dense(32)(ffn)

x = layers.Add()([x, ffn])
x = layers.LayerNormalization()(x)

# Predict the next token using the final position
x = layers.Lambda(lambda t: t[:, -1, :])(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)

model = models.Model(inputs, outputs)

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train the model
model.fit(X, y, epochs=100, verbose=0)

# 7. Predict the next word
def predict_next_word(sentence):
    seq = tokenizer.texts_to_sequences([sentence])[0]

    if not seq:
        print("Please enter a sentence containing known words.")
        return

    seq = tf.keras.utils.pad_sequences(
        [seq],
        maxlen=X.shape[1],
        padding="pre",
        truncating="pre"
    )

    prediction = model.predict(seq, verbose=0)
    predicted_id = int(np.argmax(prediction))

    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            print("Input:", sentence)
            print("Predicted next word:", word)
            return

    print("No valid next word predicted.")

predict_next_word("deep learning is")