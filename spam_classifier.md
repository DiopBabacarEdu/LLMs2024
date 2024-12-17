
# Spam Classifier avec Keras et GloVe

## Configuration de l'environnement

```python
import tqdm
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision

# Paramètres pour la configuration
SEQUENCE_LENGTH = 100  # Longueur de chaque séquence (nombre de mots)
EMBEDDING_SIZE = 100   # Taille des vecteurs d'embedding GloVe
TEST_SIZE = 0.25       # Pourcentage des données de test
BATCH_SIZE = 64        # Taille du batch
EPOCHS = 20            # Nombre d'époques d'entraînement

# Mapping des labels : ham = 0, spam = 1
label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}
```

## Fonction pour Charger les Vecteurs d'Embeddings GloVe

```python
def get_embedding_vectors(tokenizer, dim=100):
    embedding_index = {}
    with open(f"data/glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
```

## Création du Modèle LSTM

```python
def get_model(tokenizer, lstm_units):
    embedding_matrix = get_embedding_vectors(tokenizer)
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,
                        EMBEDDING_SIZE,
                        weights=[embedding_matrix],
                        trainable=False,
                        input_length=SEQUENCE_LENGTH))
    model.add(LSTM(lstm_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy", Precision(), Recall()])
    model.summary()
    return model
```

## Chargement et Préparation des Données

```python
def load_data():
    texts, labels = [], []
    with open("data/SMSSpamCollection") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels
```

## Entraînement du Modèle

```python
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time

model = get_model(tokenizer=tokenizer, lstm_units=128)
model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}.h5",
                                   save_best_only=True,
                                   verbose=1)
tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[tensorboard, model_checkpoint],
          verbose=1)
```

## Test du Modèle

```python
tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))
model.load_weights("results/spam_classifier_0.06.h5")

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    prediction = model.predict(sequence)[0]
    return int2label[np.argmax(prediction)]

while True:
    text = input("Enter the mail:")
    print(get_predictions(text))
```
