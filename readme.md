
# 📄 Attention Is All You Need

**Auteurs** : *Vaswani et al.*  
**Date de publication** : *2017*  
**Contexte** : Cet article a introduit le modèle **Transformer**, une architecture révolutionnaire pour les tâches de traitement du langage naturel (**NLP**).

---

## 🚀 Problème abordé

- Avant le **Transformer**, les modèles pour les tâches séquence-à-séquence (comme la traduction) utilisaient principalement des **Réseaux de Neurones Récurrents (RNN)** et des **LSTMs**.
- **Limites** de ces approches :
  - **Coûteuses** en calcul.
  - Difficultés de **parallélisation**.
  - Mauvaise gestion des **dépendances à long terme** dans les séquences.

---

## 💡 Contribution principale

Les auteurs proposent une **nouvelle architecture** : le **Transformer**.

- **Caractéristique clé** : Il n'utilise **ni récurrence** ni **convolution**.
- Il repose entièrement sur le mécanisme **self-attention** (attention par produit scalaire pondéré).

---

## 🔑 Innovation : Self-Attention

Le mécanisme **self-attention** permet de pondérer les relations entre les mots d'une séquence indépendamment de leur distance.

### 🧩 Fonctionnement :

1. **Calcul des représentations clés** :
   Chaque mot est représenté par 3 vecteurs :  
   - **Query** (Q)  
   - **Key** (K)  
   - **Value** (V)  

2. **Produit scalaire** :
   La similarité entre les mots est mesurée avec un produit scalaire entre **Q** et **K**, puis normalisée avec **Softmax**.

3. **Pondération des valeurs** :
   Le résultat est utilisé pour pondérer les **V** (valeurs).

---

## ⚙️ Architecture du Transformer

L'architecture est constituée de **6 couches** dans l'encodeur et le décodeur, avec des mécanismes d'attention multi-têtes.

### 📌 Encodeur

- **Self-Attention** : Chaque mot dans une séquence peut s'attarder sur les autres mots pour capturer leurs relations.
- **Feed-Forward Network** : Des couches denses pour le traitement des représentations.

### 📌 Décodeur

- **Masked Self-Attention** : Similaire à l'encodeur mais masque les mots futurs pour éviter les fuites d'informations.
- **Cross-Attention** : Relie les représentations du décodeur aux sorties de l'encodeur.

---

## 📊 Avantages du Transformer

1. **Parallélisation** :  
   Contrairement aux **RNN**, toutes les positions peuvent être calculées simultanément. Cela réduit considérablement le temps d'entraînement.

2. **Gestion des longues dépendances** :  
   Le mécanisme **self-attention** permet de relier directement tous les mots d'une séquence.

3. **Flexibilité** :  
   Le modèle s'applique à de nombreuses tâches NLP : traduction, résumé automatique, génération de texte, etc.

---

## 🌍 Applications et impact

Le Transformer a révolutionné le NLP et est à la base de modèles populaires tels que :  
- **BERT** : Pour l'analyse de texte et l'extraction d'informations.  
- **GPT** : Pour la génération de texte.  
- **T5** : Pour les tâches de traduction et de résumé.  

---

## 📝 Conclusion

L'architecture **Transformer** introduite par *"Attention Is All You Need"* a marqué un tournant dans le **traitement du langage naturel** grâce à son efficacité, sa parallélisation et son utilisation ingénieuse du mécanisme **self-attention**.

