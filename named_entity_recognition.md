
# Reconnaissance d'entités nommées (NER) avec Transformers et SpaCy

Ce tutoriel utilise **Transformers** (Hugging Face) et **SpaCy** pour effectuer la reconnaissance d'entités nommées (NER) sur un texte d'exemple.

---

## Installation des bibliothèques nécessaires

Assurez-vous que les bibliothèques nécessaires sont installées. Vous pouvez exécuter ces commandes pour les installer :

```bash
# Mise à jour de Transformers et installation de SentencePiece
!pip install --upgrade transformers sentencepiece

# Installation du modèle SpaCy pour l'anglais (en_core_web_trf)
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf3.2.0/en_core_web_trf-3.2.0-py3-none-any.whl

# Téléchargement d'un autre modèle SpaCy optimisé pour le CPU
!python -m spacy download en_core_web_sm
```

---

## Importation des bibliothèques nécessaires

On commence par importer les bibliothèques essentielles pour le projet.

```python
import spacy
from transformers import pipeline
```

---

## Texte d'exemple

Voici le texte qui sera analysé. Il provient de **Wikipedia** et décrit brièvement **Albert Einstein**.

```python
text = """
Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest
and most influential physicists of all time.
Einstein is best known for developing the theory of relativity, but he also made important contributions
to the development of the theory of quantum mechanics.
Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German
citizenship (as a subject of the Kingdom of Württemberg) the following year.
In 1897, at the age of 17, he enrolled in the mathematics and physics teaching diploma program at the
Swiss Federal polytechnic school in Zürich, graduating in 1900.
"""
```

---

## Chargement du modèle BERT pour la NER

On utilise le modèle **dslim/bert-base-NER** de Hugging Face, spécialisé pour la reconnaissance d'entités.

```python
# Chargement du modèle BERT fine-tuné pour la NER
ner = pipeline("ner", model="dslim/bert-base-NER")
```

---

## Analyse du texte avec BERT

On applique le modèle au texte et on affiche les entités trouvées.

```python
# Application du modèle sur le texte
doc_ner = ner(text)

# Résultat de l'analyse
print(doc_ner)
```

---

## Fonction pour visualiser les entités avec SpaCy

On utilise une fonction personnalisée pour afficher les entités en surbrillance dans un format visuel grâce à **SpaCy**.

```python
def get_entities_html(text, ner_result, title=None):
    """Retourne une version visuelle des entités NER grâce à SpaCy."""
    ents = []
    for ent in ner_result:
        e = {
            "start": ent["start"],
            "end": ent["end"],
            "label": ent["entity"]
        }
        if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
            ents[-1]["end"] = e["end"]
            continue
        ents.append(e)

    render_data = [{"text": text, "ents": ents, "title": title}]
    return spacy.displacy.render(render_data, style="ent", manual=True, jupyter=True)

# Affichage des entités avec SpaCy
get_entities_html(text, doc_ner)
```

---

## Chargement d'autres modèles pour comparaison

### Modèle XLM-RoBERTa (xlm-roberta-large)

```python
ner2 = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")
doc_ner2 = ner2(text)
get_entities_html(text, doc_ner2)
```

### Modèle RoBERTa large (Jean-Baptiste)

```python
ner3 = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")
doc_ner3 = ner3(text)
get_entities_html(text, doc_ner3)
```

---

## Utilisation de SpaCy pour la NER

### Chargement du modèle optimisé pour le CPU

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Affichage des résultats
spacy.displacy.render(doc, style="ent", jupyter=True)
```

---

### Chargement du modèle transformer intégré à SpaCy (en_core_web_trf)

```python
nlp_trf = spacy.load("en_core_web_trf")
doc_trf = nlp_trf(text)

# Affichage des résultats
spacy.displacy.render(doc_trf, style="ent", jupyter=True)
```

---

## Résumé

Ce tutoriel montre comment utiliser différents modèles **Transformers** et **SpaCy** pour effectuer la reconnaissance d'entités nommées sur un texte donné. Les résultats sont visualisés pour comparaison.

---

**Fin du tutoriel**.
