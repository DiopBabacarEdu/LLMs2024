
# MachineTranslation-with-Transformers-PythonCode

Ce tutoriel présente un exemple complet de **traduction automatique** avec **Transformers** de Hugging Face. Vous y trouverez l'utilisation de modèles pré-entraînés pour traduire du texte dans différentes langues.

---

## 1. Installation des bibliothèques requises

```python
# Installation de la bibliothèque Transformers et SentencePiece (pour le traitement des sous-mots)
!pip install transformers==4.12.4 sentencepiece
```

**Commentaires :**  
- La version `4.12.4` est spécifiée pour assurer la compatibilité avec les modèles utilisés.  
- SentencePiece est requis pour la tokenisation des modèles basés sur des sous-mots.

---

## 2. Importation des bibliothèques nécessaires

```python
# Importation des composants nécessaires depuis Hugging Face Transformers
from transformers import *
```

**Commentaires :**  
- `transformers` est la bibliothèque centrale pour les modèles NLP.  
- `pipeline` facilite l'utilisation des modèles sans configurations complexes.

---

## 3. Traduction de l'anglais vers l'allemand

### 3.1 Initialisation du modèle de traduction

```python
# Langue source et cible
src = "en"  # Anglais
dst = "de"  # Allemand

# Construction du nom de la tâche et du modèle
task_name = f"translation_{src}_to_{dst}"
model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"

# Initialisation du pipeline de traduction
translator = pipeline(task_name, model=model_name, tokenizer=model_name)
```

**Commentaires :**  
- Le modèle `Helsinki-NLP` est utilisé pour traduire du texte entre l'anglais (`en`) et l'allemand (`de`).  
- Le `pipeline` simplifie l'intégration du modèle avec son tokenizer.

### 3.2 Traduction simple d'une phrase

```python
# Exemple de traduction d'une phrase
translator("You're a genius.")[0]["translation_text"]
```

**Commentaires :**  
- Cette étape traduit une phrase simple en utilisant le modèle.

---

## 4. Traduction d'un article complet

```python
# Exemple de texte en anglais (Albert Einstein)
article = """
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely
acknowledged to be one of the greatest physicists of all time.
...
"""

# Traduire l'article complet
translator(article)[0]["translation_text"]
```

**Commentaires :**  
- Le modèle traite un texte plus long pour la traduction.

---

## 5. Fonction pour obtenir un modèle et tokenizer

```python
def get_translation_model_and_tokenizer(src_lang, dst_lang):
    """
    Retourne le modèle et le tokenizer pour les langues spécifiées.
    Exemple de code : 
    model, tokenizer = get_translation_model_and_tokenizer("en", "zh")
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dst_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer
```

---

## 6. Traduction de l'anglais vers le chinois avec `AutoModelForSeq2SeqLM`

```python
# Définir les langues source et cible
src = "en"
dst = "zh"

# Récupérer le modèle et le tokenizer
model, tokenizer = get_translation_model_and_tokenizer(src, dst)

# Encoder le texte pour la traduction
inputs = tokenizer.encode(article, return_tensors="pt", max_length=512, truncation=True)

# Générer une traduction avec une recherche gloutonne (greedy search)
greedy_outputs = model.generate(inputs)
print(tokenizer.decode(greedy_outputs[0], skip_special_tokens=True))

# Générer une traduction avec beam search (3 beams)
beam_outputs = model.generate(inputs, num_beams=3)
print(tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
```

**Commentaires :**  
- `greedy search` génère une traduction simple en choisissant les meilleures sorties à chaque étape.  
- `beam search` explore plusieurs chemins pour obtenir des résultats plus optimisés.

---

## 7. Traduction de l'anglais vers l'arabe avec des comparaisons de résultats

```python
# Définir les langues source et cible
src = "en"
dst = "ar"

# Récupérer le modèle et tokenizer
model, tokenizer = get_translation_model_and_tokenizer(src, dst)

# Exemple de texte en anglais
text = "It can be severe, and has caused millions of deaths around the world as well as lasting health problems in some who have survived the illness."

# Tokeniser le texte
inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

# Générer des traductions avec 5 beams et comparer les résultats
beam_outputs = model.generate(inputs, num_beams=5, num_return_sequences=5, early_stopping=True)

# Afficher les traductions
for i, beam_output in enumerate(beam_outputs):
    print(tokenizer.decode(beam_output, skip_special_tokens=True))
    print("="*50)
```

**Commentaires :**  
- La traduction est effectuée avec 5 séquences (beam search).  
- Les différentes sorties montrent les variations possibles de traduction.

---

## Conclusion

Ce tutoriel illustre les étapes clés pour utiliser des modèles pré-entraînés **Transformers** afin d'effectuer des traductions entre plusieurs langues.  
Les techniques incluent :  
- Utilisation d'un pipeline de traduction.  
- Personnalisation du modèle et tokenizer.  
- Comparaison des résultats avec greedy search et beam search.  

Pour aller plus loin, explorez d'autres modèles et paramètres disponibles sur [Hugging Face Models](https://huggingface.co/models).
