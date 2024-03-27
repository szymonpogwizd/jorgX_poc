import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import text, sequence

# Ładowanie danych
train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], batch_size=-1, as_supervised=True)
train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

# Dekodowanie danych z bajtów na stringi
decode = np.vectorize(lambda x: x.decode('utf-8'))
train_examples = decode(train_examples)
test_examples = decode(test_examples)

# Przygotowanie tokenizer'a i aktualizacja danych
tokenizer = text.Tokenizer(num_words=10000, oov_token="<UNK>")
tokenizer.fit_on_texts(train_examples)

# Konwersja tekstów na sekwencje numerów
train_sequences = tokenizer.texts_to_sequences(train_examples)
test_sequences = tokenizer.texts_to_sequences(test_examples)

# Padding sekwencji, aby miały równą długość
max_length = 256
train_padded = sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')


# Budowa modelu
model = tf.keras.Sequential([
    # Warstwa Embedding zamienia indeksy słów na wektory cech o określonej długości
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(), # Uśrednianie, aby zmniejszyć wymiarowość
    tf.keras.layers.Dense(16, activation='relu'), # Gęsta warstwa z aktywacją ReLU
    tf.keras.layers.Dense(1, activation='sigmoid') # Warstwa wyjściowa z aktywacją sigmoid, przewidująca sentyment
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels), verbose=2)

# Zapis modelu
model_save_path = "src/main/resources/model/saved_model"
model.save(model_save_path)

# Funkcja do przewidywania sentymentu recenzji
def predict_review_sentiment(review_text, tokenizer, max_length):
    # Przetworzenie recenzji przez tokenizer i konwersja na sekwencje
    seq = tokenizer.texts_to_sequences([review_text])  # Zmieniłem nazwę zmiennej, aby uniknąć pomyłek
    # Dopełnienie sekwencji, aby miała równą długość
    padded_sequence = sequence.pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    # Wykonanie predykcji
    prediction = model.predict(padded_sequence)
    # Użycie sigmoidy, aby uzyskać wartość prawdopodobieństwa między 0 a 1
    probability = tf.sigmoid(prediction).numpy()
    return "Pozytywna" if probability > 0.5 else "Negatywna", probability

# Użycie funkcji predict_review_sentiment z odpowiednimi argumentami
while True:
    user_input = input("Wprowadź recenzję filmu (lub 'quit' aby zakończyć): ")
    if user_input.lower() == 'quit':
        break
    sentiment, probability = predict_review_sentiment(user_input, tokenizer, max_length)
    print(f"Recenzja jest {sentiment} (prawdopodobieństwo: {probability[0][0]:.4f})")

