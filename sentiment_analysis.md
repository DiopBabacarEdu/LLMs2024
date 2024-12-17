
# Analyse de Sentiment avec VADER

Ce code utilise la bibliothèque **VADER SentimentIntensityAnalyzer** pour analyser les sentiments d'une série de phrases.

---

## Prérequis

Assurez-vous d'avoir installé la bibliothèque `vaderSentiment` avant d'exécuter le code :
```bash
pip install vaderSentiment
```

---

## Initialisation du modèle d'analyse de sentiment

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialisation de l'analyseur de sentiment VADER
sia = SentimentIntensityAnalyzer()
```

---

## Analyse d'un ensemble de phrases

### Liste des phrases
Définissons un ensemble de phrases à analyser :
```python
sentences = [
    "This food is amazing and tasty !",
    "Exoplanets are planets outside the solar system",
    "This is sad to see such bad behavior"
]
```

### Calcul du score de sentiment global

Pour chaque phrase, nous utilisons **`polarity_scores`** pour obtenir le score de sentiment **composé** :
```python
for sentence in sentences:
    score = sia.polarity_scores(sentence)["compound"]
    print(f'The sentiment value of the sentence :"{sentence}" is : {score}')
```

**Explications :**
- `polarity_scores` retourne un dictionnaire contenant :
    - `compound` : un score global du sentiment (positif ou négatif).
    - `pos` : proportion de mots positifs.
    - `neu` : proportion de mots neutres.
    - `neg` : proportion de mots négatifs.

Le résultat affiche le **score composé**, qui détermine si une phrase est globalement positive, négative ou neutre.

---

### Analyse détaillée des proportions des sentiments

Pour chaque phrase, nous obtenons les pourcentages des sentiments positifs, neutres et négatifs.

```python
for sentence in sentences:
    print(f'For the sentence "{sentence}"')
    polarity = sia.polarity_scores(sentence)
    pos = polarity["pos"]
    neu = polarity["neu"]
    neg = polarity["neg"]
    print(f'The percentage of positive sentiment in :"{sentence}" is : {round(pos*100,2)} %')
    print(f'The percentage of neutral sentiment in :"{sentence}" is : {round(neu*100,2)} %')
    print(f'The percentage of negative sentiment in :"{sentence}" is : {round(neg*100,2)} %')
    print("="*50)
```

**Explications :**
- `pos` : Score positif.
- `neu` : Score neutre.
- `neg` : Score négatif.
- Chaque score est multiplié par 100 pour afficher un pourcentage.

---

## Résultat attendu

Pour chaque phrase, le programme affichera :
1. Le **score composé** global de la phrase.
2. Les **pourcentages** des sentiments positifs, neutres et négatifs.

---

## Exemple de sortie

Voici un exemple de sortie possible :
```
The sentiment value of the sentence :"This food is amazing and tasty !" is : 0.85
The sentiment value of the sentence :"Exoplanets are planets outside the solar system" is : 0.0
The sentiment value of the sentence :"This is sad to see such bad behavior" is : -0.67

For the sentence "This food is amazing and tasty !"
The percentage of positive sentiment in :"This food is amazing and tasty !" is : 61.0 %
The percentage of neutral sentiment in :"This food is amazing and tasty !" is : 39.0 %
The percentage of negative sentiment in :"This food is amazing and tasty !" is : 0.0 %
==================================================

For the sentence "Exoplanets are planets outside the solar system"
The percentage of positive sentiment in :"Exoplanets are planets outside the solar system" is : 0.0 %
The percentage of neutral sentiment in :"Exoplanets are planets outside the solar system" is : 100.0 %
The percentage of negative sentiment in :"Exoplanets are planets outside the solar system" is : 0.0 %
==================================================

For the sentence "This is sad to see such bad behavior"
The percentage of positive sentiment in :"This is sad to see such bad behavior" is : 0.0 %
The percentage of neutral sentiment in :"This is sad to see such bad behavior" is : 68.0 %
The percentage of negative sentiment in :"This is sad to see such bad behavior" is : 32.0 %
==================================================
```

---

## Conclusion

Ce programme est une démonstration de l'utilisation de **VADER** pour analyser les sentiments dans des phrases en anglais. Il extrait des informations précises sur les proportions de sentiments positifs, neutres et négatifs.
