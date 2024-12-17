# Tutoriel : Pré-entrainement de BERT avec Transformers

Ce tutoriel explique étape par étape comment pré-entraîner un modèle **BERT** à l'aide de la bibliothèque **HuggingFace Transformers**. Il inclut la préparation des données, la création d'un tokenizer personnalisé, et l'entraînement sur un jeu de données.

---

## Installation des bibliothèques nécessaires

```bash
!pip install datasets transformers==4.18.0 sentencepiece
```

### Explications :
- **datasets** : Bibliothèque pour charger et gérer des jeux de données.
- **transformers** : Pour utiliser et entraîner des modèles comme BERT.
- **sentencepiece** : Utilisé pour tokenizer certains types de texte.

---

## Importation des bibliothèques

```python
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer
import os
import json
```

### Explications :
- **load_dataset** : Pour charger des jeux de données.
- **BertTokenizerFast** : Tokenizer rapide pour BERT.
- **BertForMaskedLM** : Modèle BERT pour la tâche **Masked Language Modeling (MLM)**.
- **Trainer** et **TrainingArguments** : Classes pour faciliter l'entraînement des modèles.
- **DataCollatorForLanguageModeling** : Prépare les données pour MLM.

---

## Téléchargement et préparation du jeu de données

### Chargement du jeu de données `cc_news`

```python
dataset = load_dataset("cc_news", split="train")
```

### Séparation des données en jeu d'entraînement et de test

```python
d = dataset.train_test_split(test_size=0.1)
d["train"], d["test"]
```

### Affichage des premières lignes du jeu d'entraînement

```python
for t in d["train"]["text"][:3]:
    print(t)
    print("="*50)
```

---

## Sauvegarde des données en fichiers texte

```python
def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)
dataset_to_text(d["train"], "train.txt")
dataset_to_text(d["test"], "test.txt")
```

### Explications :
- Cette fonction sauvegarde les textes des données dans des fichiers `.txt` pour les utiliser dans l'entraînement du tokenizer.

---

## Création et entraînement du tokenizer WordPiece

### Paramètres du tokenizer

```python
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
files = ["train.txt"]  # Fichier d'entraînement
vocab_size = 30522
max_length = 512
```

### Entraînement du tokenizer

```python
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)
```

### Sauvegarde du tokenizer

```python
model_path = "pretrained-bert"
if not os.path.isdir(model_path):
    os.mkdir(model_path)
tokenizer.save_model(model_path)
```

---

## Prétraitement des données avec le tokenizer

### Fonction pour encoder les données

```python
def encode_with_truncation(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

train_dataset = d["train"].map(encode_with_truncation, batched=True)
test_dataset = d["test"].map(encode_with_truncation, batched=True)
```

---

## Préparation des données pour l'entraînement

### Regroupement des textes en chunks de longueur maximale

```python
from itertools import chain
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

train_dataset = train_dataset.map(group_texts, batched=True)
test_dataset = test_dataset.map(group_texts, batched=True)
```

---

## Configuration et initialisation du modèle

```python
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
```

### Collateur de données pour MLM

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)
```

---

## Configuration des arguments d'entraînement

```python
training_args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8,
    logging_steps=1000,
    save_steps=1000,
)
```

---

## Entraînement du modèle

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()
```

---

## Prédiction avec le modèle pré-entraîné

### Chargement du modèle et du tokenizer

```python
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-6000"))
tokenizer = BertTokenizerFast.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
```

### Exemple de prédictions

```python
example = "It is known that [MASK] is the capital of Germany"
for prediction in fill_mask(example):
    print(prediction)
```

---

## Surveillance des ressources GPU

```bash
!nvidia-smi
```

---

## Résumé

Ce tutoriel couvre :
1. Téléchargement et préparation du jeu de données `cc_news`.
2. Création et entraînement d'un tokenizer WordPiece.
3. Prétraitement et encodage des données.
4. Configuration et entraînement du modèle BERT pour MLM.
5. Prédictions sur des exemples avec le modèle pré-entraîné.
6. Utilisation de l'API HuggingFace pour l'inférence.
