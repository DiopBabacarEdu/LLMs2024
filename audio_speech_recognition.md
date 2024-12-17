
# Tutoriel : Reconnaissance vocale avec Wav2Vec2 et PyTorch

Ce tutoriel explique comment utiliser le modèle **Wav2Vec2** de Facebook avec PyTorch pour réaliser une transcription automatique de fichiers audio.

---

## Installation des dépendances

### Étape 1 : Installation des bibliothèques nécessaires

Avant de commencer, assurez-vous d'installer les bibliothèques requises avec `pip` :

```bash
pip install transformers==4.11.2 datasets soundfile sentencepiece torchaudio pyaudio
```

---

## Chargement du modèle et configuration

### Étape 2 : Importation des bibliothèques

```python
# Importation des bibliothèques nécessaires
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import torchaudio
```

---

### Étape 3 : Chargement du modèle Wav2Vec2

On utilise le modèle pré-entraîné **`facebook/wav2vec2-large-960h-lv60-self`**, qui offre des performances optimales pour la transcription automatique.

```python
# Sélection et chargement du modèle et du processeur
model_name = "facebook/wav2vec2-large-960h-lv60-self"  # Taille ~1.18GB
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
```

---

## Préparation de l'audio

### Étape 4 : Téléchargement et chargement du fichier audio

On utilise un fichier audio existant au format WAV. Vous pouvez remplacer `audio_url` par un autre fichier si nécessaire.

```python
# URL d'exemple d'un fichier audio
audio_url = "https://github.com/x4nth055/pythoncode-tutorials/raw/master/machine-learning/speechrecognition/30-4447-0004.wav"

# Chargement du fichier audio avec torchaudio
speech, sr = torchaudio.load(audio_url)
speech = speech.squeeze()  # Suppression de la dimension supplémentaire
print(f"Fréquence d'échantillonnage : {sr}, Forme de l'audio : {speech.shape}")
```

---

### Étape 5 : Rééchantillonnage de l'audio

Le modèle Wav2Vec2 fonctionne avec des fichiers audio ayant une fréquence d'échantillonnage de **16 000 Hz**. On utilise `torchaudio.transforms.Resample` pour ajuster cette fréquence.

```python
# Rééchantillonnage vers 16 kHz
resampler = torchaudio.transforms.Resample(sr, 16000)
speech = resampler(speech)
print(f"Forme de l'audio après rééchantillonnage : {speech.shape}")
```

---

## Transcription du fichier audio

### Étape 6 : Tokenisation et traitement du fichier audio

On utilise le processeur Wav2Vec2 pour convertir l'audio en une représentation adaptée au modèle.

```python
# Conversion de l'audio en valeurs d'entrée pour le modèle
input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
print(f"Forme des valeurs d'entrée : {input_values.shape}")
```

---

### Étape 7 : Inférence pour la transcription

Le modèle effectue la prédiction et génère les logits. On utilise ensuite **`torch.argmax`** pour obtenir les ID des tokens prédits.

```python
# Prédiction des logits par le modèle
logits = model(input_values)["logits"]

# Sélection des ID prédits avec argmax
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Forme des ID prédits : {predicted_ids.shape}")
```

---

### Étape 8 : Décodage des ID en texte

On décode les tokens en texte lisible à l'aide du processeur.

```python
# Décodage des ID en texte
transcription = processor.decode(predicted_ids[0])
print(f"Transcription : {transcription.lower()}")
```

---

## Fonction de transcription

Pour réutiliser facilement les étapes précédentes, nous définissons une fonction `get_transcription`.

```python
def get_transcription(audio_path):
    # Chargement du fichier audio
    speech, sr = torchaudio.load(audio_path)
    speech = speech.squeeze()

    # Rééchantillonnage
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)

    # Tokenisation et inférence
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
    logits = model(input_values)["logits"]
    predicted_ids = torch.argmax(logits, dim=-1)

    # Décodage en texte
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# Exemple d'utilisation avec une URL audio
print(get_transcription(audio_url))
```

---

## Enregistrement et transcription en temps réel

### Étape 9 : Enregistrement d'un fichier audio

On utilise **PyAudio** pour enregistrer un fichier audio au format WAV en temps réel.

```python
import pyaudio
import wave

# Paramètres d'enregistrement
filename = "recorded.wav"
chunk = 1024  # Taille des blocs d'échantillons
FORMAT = pyaudio.paInt16
channels = 1
sample_rate = 16000
record_seconds = 10

# Initialisation de PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

print("Enregistrement en cours...")
frames = []

# Enregistrement des données audio
for _ in range(0, int(sample_rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("Enregistrement terminé.")

# Fermeture du flux
stream.stop_stream()
stream.close()
p.terminate()

# Sauvegarde du fichier audio
with wave.open(filename, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
```

---

### Étape 10 : Transcription du fichier enregistré

On utilise la fonction `get_transcription` pour transcrire le fichier audio enregistré.

```python
print("Transcription de l'audio enregistré :")
print(get_transcription("recorded.wav"))
```

---

## Conclusion

Dans ce tutoriel, nous avons appris à :

1. Utiliser le modèle **Wav2Vec2** pour transcrire des fichiers audio.
2. Réaliser un traitement des fichiers avec **Torchaudio**.
3. Enregistrer des fichiers audio en temps réel avec **PyAudio**.
4. Automatiser l'ensemble du processus dans une fonction réutilisable.

Vous pouvez désormais intégrer ces étapes dans vos projets de reconnaissance vocale ou d'analyse de fichiers audio.
