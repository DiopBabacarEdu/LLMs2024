
# Tutoriel : Génération de Texte et de Code avec **Transformers**  

Ce tutoriel vous guidera étape par étape pour utiliser **GPT-2** et **GPT-J** afin de générer du texte, du code Python, des scripts Bash, du Java et même du LaTeX. Vous apprendrez à exploiter les modèles **Transformers** grâce à la bibliothèque **HuggingFace**.

---

## Prérequis  

Avant de commencer, assurez-vous d'avoir installé la bibliothèque **Transformers** qui est essentielle pour charger et exécuter les modèles GPT. Exécutez la commande suivante dans votre terminal ou dans un notebook Jupyter :  

```bash
pip install transformers
```

---

## Étape 1 : Importation et chargement du modèle GPT-2  

Le modèle **GPT-2** est l’un des modèles de génération de texte les plus connus. Il est capable de produire des phrases cohérentes à partir d’une amorce donnée.  

### Code Python :  

```python
# Importation de la fonction pipeline depuis HuggingFace Transformers
from transformers import pipeline

# Charger le modèle GPT-2 via la pipeline text-generation
gpt2_generator = pipeline('text-generation', model='gpt2')

# Générer plusieurs phrases basées sur une amorce donnée
sentences = gpt2_generator("To be honest, neural networks", 
                           do_sample=True, top_k=50, temperature=0.6, 
                           max_length=128, num_return_sequences=3)

# Afficher les résultats générés
for sentence in sentences:
    print(sentence["generated_text"])
    print("="*50)
```

---

## Étape 2 : Chargement et utilisation du modèle GPT-J  

Le modèle **GPT-J** est plus puissant et capable de générer des textes longs et des codes complexes.  

### Code Python :  

```python
# Charger le modèle GPT-J depuis HuggingFace
gpt_j_generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

# Générer du texte basé sur une amorce
sentences = gpt_j_generator("To be honest, robots will", 
                            do_sample=True, top_k=50, temperature=0.6, 
                            max_length=128, num_return_sequences=3)

# Afficher les résultats
for sentence in sentences:
    print(sentence["generated_text"])
    print("="*50)
```

---

## Étape 3 : Génération de code Python  

### Code Python :  

```python
# Exemple 1 : Générer une liste de pays africains
print(gpt_j_generator(
"""
import os
# make a list of all african countries
""", do_sample=True, top_k=10, temperature=0.05, max_length=256)[0]["generated_text"])

# Exemple 2 : Charger une image avec OpenCV et la retourner
print(gpt_j_generator(
"""
import cv2
image = "image.png"
# load the image and flip it
""", do_sample=True, top_k=10, temperature=0.05, max_length=256)[0]["generated_text"])
```

---

## Étape 4 : Génération de scripts Bash  

### Exemple :  

```python
print(gpt_j_generator(
"""
# get .py files in /opt directory
ls *.py /opt
# get public ip address
""", max_length=256, top_k=50, temperature=0.05, do_sample=True)[0]["generated_text"])
```

---

## Étape 5 : Génération de code Java  

### Exemple :  

```python
print(gpt_j_generator(
"""
public class Test {
public static void main(String[] args){
// printing the first 20 fibonacci numbers
""", max_length=128, top_k=50, temperature=0.1, do_sample=True)[0]["generated_text"])
```

---

## Étape 6 : Génération de code LaTeX  

### Exemple :  

```python
print(gpt_j_generator(
r"""
# % list of Asian countries
\begin{enumerate}
""", max_length=128, top_k=15, temperature=0.1, do_sample=True)[0]["generated_text"])
```

---

## Conclusion  

Nous avons exploré les capacités des modèles **GPT-2** et **GPT-J** à travers des exemples pratiques. Vous pouvez tester ces modèles pour :  

- Générer du texte  
- Automatiser du code  
- Créer des scripts Bash, Java et LaTeX  

Pour en savoir plus, consultez :  

- [Documentation officielle HuggingFace](https://huggingface.co/docs/transformers)  
- [Modèle GPT-2](https://huggingface.co/gpt2)  
- [Modèle GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)  
