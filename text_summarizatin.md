
# **Tutoriel : Résumé de texte avec Transformers**

## **Introduction**
Ce tutoriel montre comment utiliser la bibliothèque **Transformers** de Hugging Face pour générer des résumés de texte. Nous utilisons deux approches :
1. **Pipeline API** pour simplifier la tâche.
2. Utilisation du modèle **T5** pour plus de contrôle.

---

## **Étape 1 : Installation des dépendances**

Assurez-vous d'avoir installé la bibliothèque `transformers` :
```bash
pip install transformers
```

---

## **Étape 2 : Résumé de texte avec Pipeline API**

La méthode la plus simple pour le résumé de texte utilise l'API **Pipeline**.

### **Code complet :**

```python
from transformers import pipeline

# Initialisation de l'API Pipeline pour la tâche de résumé
summarization = pipeline("summarization")

# Exemple 1 : Résumé du premier texte
original_text = '''
Paul Walker is hardly the first actor to die during a production.
But Walker's death in November 2013 at the age of 40 after a car crash was especially eerie given his
rise to fame in the "Fast and Furious" film franchise.
The release of "Furious 7" on Friday offers the opportunity for fans to remember -- and possibly grieve
again -- the man that so many have praised as one of the nicest guys in Hollywood.
...
'''
summary_text = summarization(original_text)[0]['summary_text']
print("Summary:", summary_text)

print("="*50)

# Exemple 2 : Résumé du deuxième texte
original_text = '''
For the first time in eight years, a TV legend returned to doing what he does best.
Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not
host Drew Carey but another familiar face in charge of the proceedings.
...
'''
summary_text = summarization(original_text)[0]['summary_text']
print("Summary:", summary_text)
```

---

## **Étape 3 : Résumé de texte avec le modèle T5**

Le modèle **T5** est plus flexible et puissant pour le résumé de texte.

### **Code complet :**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialisation du modèle et du tokenizer T5
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Texte original pour le résumé
article = '''
Justin Timberlake and Jessica Biel, welcome to parenthood.
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to
People.
...
'''

# Encodage du texte pour le modèle T5
inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)

# Génération du résumé
outputs = model.generate(
    inputs,
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Affichage du résumé
print(tokenizer.decode(outputs[0]))
```

---

## **Conclusion**

Nous avons utilisé deux approches pour générer des résumés :
1. **Pipeline API** pour sa simplicité.
2. **Modèle T5** pour des options plus avancées.

Ces outils permettent de traiter des articles volumineux et d'extraire les informations essentielles facilement.
