
# Chatbot interactif avec DialoGPT

Ce tutoriel vous montre comment utiliser le modèle **DialoGPT** de Microsoft pour créer un chatbot interactif. Le code permet plusieurs approches de génération de réponses, comme **greedy search**, **beam search** et différentes méthodes d'échantillonnage.

---

## Installation des dépendances

Avant de commencer, vous devez installer la bibliothèque `transformers` qui contient les modèles pré-entraînés de HuggingFace.

```bash
# Installation de transformers
!pip install transformers
```

---

## Importation des bibliothèques nécessaires

On importe les composants nécessaires pour le modèle DialoGPT ainsi que PyTorch pour la gestion des tensors.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
```

---

## Initialisation du modèle et du tokenizer

On utilise le modèle `DialoGPT` dans différentes tailles : `small`, `medium` ou `large`. Vous pouvez choisir celui que vous voulez en décommentant la ligne appropriée.

```python
# Choix du modèle DialoGPT
model_name = "microsoft/DialoGPT-medium"  # Taille moyenne par défaut
# model_name = "microsoft/DialoGPT-small"   # Petite taille
# model_name = "microsoft/DialoGPT-large"   # Grande taille

# Initialisation du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## Chat simple avec Greedy Search

On utilise **Greedy Search** pour générer une réponse simple du chatbot. Chaque étape prend l'entrée de l'utilisateur et génère une réponse.

```python
print("==== Greedy search chat ====")

# Historique des conversations
chat_history_ids = None

# Boucle de conversation sur 5 interactions
for step in range(5):
    # Prise d'entrée utilisateur
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")

    # Historique des conversations (concaténation)
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    # Génération de la réponse
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Décodage et affichage de la réponse
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"DialoGPT: {output}")
```

---

## Chat avec Beam Search

**Beam Search** permet d'explorer plusieurs chemins pour générer des réponses plus pertinentes.

```python
print("==== Beam search chat ====")

for step in range(5):
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        num_beams=3,  # Beam search avec 3 faisceaux
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"DialoGPT: {output}")
```

---

## Chat avec Sampling et Temperature

On utilise **sampling** (échantillonnage) avec un ajustement de la température pour rendre les réponses plus variées.

```python
print("==== Sampling chat with tweaking temperature ====")

for step in range(5):
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        temperature=0.75,  # Réglage de la température pour des réponses plus naturelles
        pad_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"DialoGPT: {output}")
```

---

## Chat avec Top-K Sampling et Temperature

**Top-K sampling** sélectionne les K mots les plus probables pour générer une réponse.

```python
print("==== Top-K sampling chat with tweaking temperature ====")

for step in range(5):
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_k=50,  # Top-K sampling pour restreindre les choix
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"DialoGPT: {output}")
```

---

## Conversation avec choix multiple

On génère plusieurs réponses et laisse l'utilisateur choisir celle qu'il préfère pour continuer la conversation.

```python
print("==== Nucleus sampling with multiple sentences ====")

for step in range(5):
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    # Génération de plusieurs réponses avec nucleus sampling
    chat_history_ids_list = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,  # Nucleus sampling
        top_k=50,
        temperature=0.75,
        num_return_sequences=5,  # Retourner 5 réponses
        pad_token_id=tokenizer.eos_token_id
    )

    # Afficher et choisir une réponse
    for i, response in enumerate(chat_history_ids_list):
        output = tokenizer.decode(response[bot_input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"DialoGPT {i}: {output}")

    choice_index = int(input("Choose the response you want for the next input: "))
    chat_history_ids = torch.unsqueeze(chat_history_ids_list[choice_index], dim=0)
```

---

## Conclusion

Ce tutoriel vous a permis de découvrir différentes techniques pour générer des réponses interactives avec **DialoGPT**. Vous pouvez personnaliser les méthodes de génération pour rendre les conversations plus naturelles et engageantes.

---
