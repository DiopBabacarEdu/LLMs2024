
# Classification de texte avec RNN (LSTM)

Ce projet utilise des **Réseaux de Neurones Récurrents (RNN)**, en particulier LSTM (Long Short-Term Memory), pour effectuer des tâches de classification de texte. Il couvre différentes étapes telles que la préparation des données, l'entraînement du modèle et les tests.

---

## 1. Paramètres globaux du projet

```python
from tensorflow.keras.layers import LSTM

# Longueur maximale des phrases (nombre de mots par phrase)
SEQUENCE_LENGTH = 300

# Taille des vecteurs d'embedding GloVe
EMBEDDING_SIZE = 300

# Nombre maximum de mots à utiliser
N_WORDS = 10000

# Token pour les mots hors vocabulaire
OOV_TOKEN = None

# Répartition des données : 30% pour le test et 70% pour l'entraînement
TEST_SIZE = 0.3

# Nombre de couches RNN (LSTM dans ce cas)
N_LAYERS = 1

# Cellule RNN utilisée
RNN_CELL = LSTM

# Utiliser un RNN bidirectionnel ou non
IS_BIDIRECTIONAL = False

# Nombre de neurones dans chaque couche LSTM
UNITS = 128

# Taux de dropout pour régularisation
DROPOUT = 0.4

# Paramètres d'entraînement
LOSS = "categorical_crossentropy"  # Fonction de perte
OPTIMIZER = "adam"  # Optimiseur
BATCH_SIZE = 64  # Taille des lots d'entraînement
EPOCHS = 6  # Nombre d'époques
```

---

## 2. Génération du nom unique du modèle

La fonction ci-dessous construit un nom unique basé sur les hyperparamètres.

```python
def get_model_name(dataset_name):
    # Construire le nom unique du modèle
    model_name = f"{dataset_name}-{RNN_CELL.__name__}-seq-{SEQUENCE_LENGTH}-em-{EMBEDDING_SIZE}-w-{N_WORDS}-layers-{N_LAYERS}-units-{UNITS}-opt-{OPTIMIZER}-BS-{BATCH_SIZE}-d-{DROPOUT}"
    if IS_BIDIRECTIONAL:
        model_name = "bid-" + model_name
    if OOV_TOKEN:
        model_name += "-oov"
    return model_name
```

---

## 3. Fonctions utilitaires

### 3.1 Chargement des vecteurs d'embedding GloVe

Les vecteurs GloVe sont utilisés pour représenter les mots sous forme vectorielle.

```python
from tqdm import tqdm
import numpy as np

def get_embedding_vectors(word_index, embedding_size=100):
    # Initialiser une matrice vide pour stocker les embeddings
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    with open(f"data/glove.6B.{embedding_size}d.txt", encoding="utf8") as f:
        for line in tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(values[1:], dtype="float32")
    return embedding_matrix
```

---

### 3.2 Création du modèle LSTM

Cette fonction crée et retourne un modèle séquentiel basé sur LSTM.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional

def create_model(word_index, units=128, n_layers=1, cell=LSTM, bidirectional=False,
                 embedding_size=100, sequence_length=100, dropout=0.3,
                 loss="categorical_crossentropy", optimizer="adam",
                 output_length=2):
    embedding_matrix = get_embedding_vectors(word_index, embedding_size)
    model = Sequential()

    # Ajouter une couche d'embedding pré-entrainée avec GloVe
    model.add(Embedding(len(word_index) + 1,
                        embedding_size,
                        weights=[embedding_matrix],
                        trainable=False,
                        input_length=sequence_length))
    
    for i in range(n_layers):
        if i == n_layers - 1:  # Dernière couche
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:  # Couches intermédiaires
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    
    # Couche de sortie
    model.add(Dense(output_length, activation="softmax"))
    
    # Compiler le modèle
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
```

---

## 4. Chargement des données IMDB

Cette fonction charge et prépare le dataset IMDB pour l'analyse de sentiment.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_imdb_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # Charger les avis et les étiquettes
    reviews = []
    with open("data/reviews.txt") as f:
        for review in f:
            reviews.append(review.strip())
    
    labels = []
    with open("data/labels.txt") as f:
        for label in f:
            labels.append(label.strip())
    
    # Tokenizer les avis textuels
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    X, y = np.array(X), np.array(labels)
    
    # Padding des séquences
    X = pad_sequences(X, maxlen=sequence_length)
    
    # Convertir les étiquettes en format one-hot
    y = to_categorical(y)
    
    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "tokenizer": tokenizer,
        "int2label": {0: "negative", 1: "positive"},
        "label2int": {"negative": 0, "positive": 1}
    }
    return data
```

---

## 5. Entraînement et évaluation du modèle

```python
from tensorflow.keras.callbacks import TensorBoard
import os

# Définir le nom du dataset
dataset_name = "imdb"

# Générer le nom unique du modèle
model_name = get_model_name(dataset_name)

# Charger les données IMDB
data = load_imdb_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

# Créer le modèle LSTM
model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS,
                     cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL,
                     embedding_size=EMBEDDING_SIZE,
                     sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT,
                     loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

# Résumé du modèle
model.summary()

# Initialiser TensorBoard pour suivre l'entraînement
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# Entraîner le modèle
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[tensorboard],
                    verbose=1)

# Sauvegarder le modèle
model.save(os.path.join("results", model_name) + ".h5")
```

---

## 6. Prédiction avec le modèle sauvegardé

```python
def get_predictions(text):
    sequence = data["tokenizer"].texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    prediction = model.predict(sequence)[0]
    return data["int2label"][np.argmax(prediction)]
```
