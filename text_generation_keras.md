
# **Réseau de Neurones Récurrents avec LSTM pour la Génération de Texte**

Ce projet utilise des réseaux de neurones récurrents (RNN) avec des couches **LSTM** pour entraîner un modèle sur un texte d'entrée et générer du texte similaire.

---

## **1. Importation des bibliothèques**
```python
import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from string import punctuation
```

---

## **2. Définition des hyperparamètres**
- **SEQUENCE_LENGTH** : Taille de la séquence d'entrée.
- **BATCH_SIZE** : Taille des lots pour l'entraînement.
- **EPOCHS** : Nombre d'époques d'entraînement.
```python
SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 3
```

---

## **3. Préparation des données**
### **3.1. Lecture et traitement du fichier texte**
```python
FILE_PATH = "data/wonderland.txt"
BASENAME = os.path.basename(FILE_PATH)

# Lecture du fichier texte
text = open(FILE_PATH, encoding="utf-8").read()

# Mise en minuscule et suppression de la ponctuation
text = text.lower()
text = text.translate(str.maketrans("", "", punctuation))

# Affichage des statistiques
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", len(vocab))
```

---

## **4. Conversion des caractères**
- **char2int** : Dictionnaire pour convertir les caractères en entiers.
- **int2char** : Dictionnaire pour reconvertir les entiers en caractères.
```python
char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

# Sauvegarde des dictionnaires
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))
```

---

## **5. Encodage du texte**
- Conversion de tout le texte en séquences d'entiers.
```python
encoded_text = np.array([char2int[c] for c in text])
```

---

## **6. Création des séquences d'entrée et cible**
### **6.1. Construction des séquences**
```python
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(2*SEQUENCE_LENGTH + 1, drop_remainder=True)
```

### **6.2. Fonction pour diviser les séquences**
```python
def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:SEQUENCE_LENGTH], sample[SEQUENCE_LENGTH]))
    for i in range(1, (len(sample)-1) // 2):
        input_ = sample[i: i+SEQUENCE_LENGTH]
        target = sample[i+SEQUENCE_LENGTH]
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds
```

---

## **7. Encodage One-Hot des entrées et sorties**
```python
def one_hot_samples(input_, target):
    return tf.one_hot(input_, len(vocab)), tf.one_hot(target, len(vocab))
dataset = sequences.flat_map(split_sample).map(one_hot_samples)
```

---

## **8. Préparation du Dataset**
- Shuffle, repeat et batch pour l'entraînement.
```python
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)
```

---

## **9. Construction du modèle**
### **Architecture du modèle LSTM**
- Deux couches LSTM avec un taux de **Dropout** pour la régularisation.
```python
model = Sequential([
    LSTM(256, input_shape=(SEQUENCE_LENGTH, len(vocab)), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(len(vocab), activation="softmax"),
])
```

### **Compilation du modèle**
```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
```

---

## **10. Entraînement du modèle**
```python
if not os.path.isdir("results"):
    os.mkdir("results")

model.fit(ds, steps_per_epoch=(len(encoded_text) - SEQUENCE_LENGTH) // BATCH_SIZE, epochs=EPOCHS)
model.save(f"results/{BASENAME}-{SEQUENCE_LENGTH}.h5")
```

---

## **11. Génération de texte**
### **Modèle de génération**
```python
import tqdm

seed = "alice is pretty"
s = seed
n_chars = 400
generated = ""

for i in tqdm.tqdm(range(n_chars), "Generating text"):
    X = np.zeros((1, SEQUENCE_LENGTH, len(vocab)))
    for t, char in enumerate(seed):
        X[0, (SEQUENCE_LENGTH - len(seed)) + t, char2int[char]] = 1
    predicted = model.predict(X, verbose=0)[0]
    next_index = np.argmax(predicted)
    next_char = int2char[next_index]
    generated += next_char
    seed = seed[1:] + next_char

print("Seed:", s)
print("Generated text:")
print(generated)
```

---

## **12. Résumé**
1. Lecture et nettoyage des données textuelles.
2. Encodage des caractères en entiers et en vecteurs one-hot.
3. Création et entraînement d'un modèle LSTM pour la prédiction de caractères.
4. Génération de texte basé sur un modèle entraîné.

---

## **Résultats attendus**
Après l'exécution du script :
- Le modèle LSTM entraînera un réseau capable de prédire et générer du texte similaire au corpus d'entrée.

---

## **Conclusion**
Ce projet démontre comment utiliser les couches LSTM pour effectuer de la **modélisation de texte** à partir d'un fichier texte. Vous pouvez tester différentes valeurs d'hyperparamètres pour observer l'impact sur les résultats.
