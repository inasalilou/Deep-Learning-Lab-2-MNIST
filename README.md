# TP 2 — Deep Learning : Classification d'images avec MNIST

## Informations générales

- **Université :** Université Abdelmalek Essaâdi
- **Établissement :** Faculté des Sciences et Techniques de Tanger
- **Département :** Génie Informatique
- **Cycle :** LSI
- **Module :** Deep Learning
- **Enseignant :** Pr. Lotfi ELAACHAK
- **Réalisé par :** Inass Alilou
- **Année universitaire :** 2025–2026

---

## 1. Présentation du TP

L'objectif principal de ce travail pratique est de se familiariser avec la bibliothèque **PyTorch**, de construire différentes architectures de réseaux de neurones dédiées à la vision par ordinateur et de les entraîner en utilisant le GPU.

Ce TP porte sur la classification des chiffres manuscrits à partir de la base de données MNIST.

Nous étudions et comparons cinq architectures :

- CNN (Convolutional Neural Network)
- Faster R-CNN
- VGG16 (Transfer Learning)
- AlexNet (Transfer Learning)
- Vision Transformer (ViT) construit from scratch

## 2. Jeu de données : MNIST

La base de données MNIST contient des images de chiffres manuscrits allant de 0 à 9.

**Caractéristiques :**
- Images d'entraînement : 60 000
- Images de test : 10 000
- Dimensions des images : 28 × 28 pixels
- Nombre de classes : 10
- Type d'images : niveaux de gris

**Source :** [MNIST Dataset — Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## 3. Partie 1 — CNN Classifier

### 3.1 CNN

Nous avons construit une architecture CNN avec PyTorch comprenant :
- Des couches de convolution.
- Des fonctions d'activation ReLU.
- Des couches Max Pooling.
- Une couche Dropout pour la régularisation.
- Des couches entièrement connectées pour la classification.

Le modèle a été entraîné sur le GPU avec l'optimiseur Adam et la fonction de perte CrossEntropyLoss.

### 3.2 Faster R-CNN

Nous avons également implémenté Faster R-CNN avec une architecture ResNet50-FPN.

Le modèle a été adapté à MNIST en utilisant des images RGB et des boîtes englobantes couvrant l'image entière.

**Remarque :** les résultats présentés pour Faster R-CNN correspondent à une classification fondée sur la détection ayant le score le plus élevé. Il ne s'agit pas d'une évaluation complète de détection d'objets avec la métrique mAP.

### 3.3 Transfer Learning : VGG16 et AlexNet

Nous avons utilisé deux architectures pré-entraînées sur ImageNet :

- **VGG16**
- **AlexNet**

Les couches d'extraction de caractéristiques pré-entraînées ont été gelées, tandis que la dernière couche de classification a été remplacée par une couche adaptée aux 10 classes de MNIST et entraînée sur le nouveau jeu de données.

Cette approche correspond à l'utilisation du Transfer Learning avec extraction de caractéristiques.

### 3.4 Comparaison des modèles

Les quatre modèles ont été évalués sur les mêmes 2 000 premières images du jeu de test.

| Modèle | Accuracy | Precision | Recall | F1-score | Loss | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 98,40 % | 98,41 % | 98,40 % | 98,40 % | 0,0322 | 60,28 |
| Faster R-CNN | 90,30 % | 90,45 % | 90,30 % | 90,24 % | 0,0184 | 558,19 |
| VGG16 | 93,85 % | 94,06 % | 93,85 % | 93,86 % | 0,4246 | 1252,64 |
| AlexNet | 92,75 % | 93,26 % | 92,75 % | 92,75 % | 0,3333 | 456,72 |

*Les valeurs de Loss correspondent à la dernière époque d'entraînement.*

## 4. Partie 2 — Vision Transformer (ViT)

Dans cette deuxième partie, nous avons construit un Vision Transformer à partir de zéro avec PyTorch, sans utiliser de poids pré-entraînés.

L'architecture utilise les paramètres suivants :

| Paramètre | Valeur |
|---|---:|
| Taille de l'image | 28 × 28 |
| Taille des patches | 7 × 7 |
| Nombre de patches | 16 |
| Dimension des embeddings | 64 |
| Nombre de têtes d'attention | 4 |
| Nombre de couches Transformer | 4 |
| Dimension MLP | 128 |
| Nombre de classes | 10 |
| Nombre de paramètres | 139 018 |

Les principales étapes de construction sont :

1. Découpage des images en patches.
2. Création des Patch Embeddings.
3. Ajout du CLS Token.
4. Ajout des Positional Embeddings.
5. Construction du Transformer Encoder avec le mécanisme Self-Attention.
6. Ajout de la couche de classification.

Le modèle a été entraîné pendant 10 époques avec l'optimiseur Adam et CrossEntropyLoss.

### Résultats du ViT

| Métrique | Résultat |
|---|---:|
| Accuracy | 96,90 % |
| Precision | 96,92 % |
| Recall | 96,90 % |
| F1-score | 96,89 % |
| Loss finale | 0,0751 |
| Temps d'entraînement | 196,77 secondes |

## 5. Comparaison finale des cinq architectures

| Modèle | Accuracy | Precision | Recall | F1-score | Loss | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 98,40 % | 98,41 % | 98,40 % | 98,40 % | 0,0322 | 60,28 |
| Faster R-CNN | 90,30 % | 90,45 % | 90,30 % | 90,24 % | 0,0184 | 558,19 |
| VGG16 | 93,85 % | 94,06 % | 93,85 % | 93,86 % | 0,4246 | 1252,64 |
| AlexNet | 92,75 % | 93,26 % | 92,75 % | 92,75 % | 0,3333 | 456,72 |
| ViT | 96,90 % | 96,92 % | 96,90 % | 96,89 % | 0,0751 | 196,77 |

### Interprétation des résultats

Les résultats montrent que le CNN obtient la meilleure Accuracy avec 98,40 %, ainsi que le temps d'entraînement le plus faible (60,28 secondes).

Le Vision Transformer arrive en deuxième position avec 96,90 % d'Accuracy. Il démontre l'efficacité du mécanisme Self-Attention dans la classification des images.

VGG16 et AlexNet présentent également des performances satisfaisantes grâce à l'utilisation de modèles pré-entraînés.

Faster R-CNN obtient une Accuracy de 90,30 %. Cette architecture est principalement conçue pour la détection d'objets, ce qui la rend moins adaptée à une tâche de classification simple comme MNIST.

Il faut toutefois noter que les modèles n'ont pas tous été entraînés dans les mêmes conditions : les quantités de données d'entraînement et l'utilisation de poids pré-entraînés diffèrent.

## 6. Conclusion générale

Ce travail pratique nous a permis de découvrir et de mettre en œuvre plusieurs architectures de Deep Learning destinées à la vision par ordinateur.

Nous avons appris à construire des réseaux CNN, à utiliser Faster R-CNN, à appliquer le Transfer Learning avec VGG16 et AlexNet, et à développer un Vision Transformer à partir de zéro.

Les expérimentations montrent que le CNN représente le meilleur choix pour cette tâche de classification, grâce à sa précision élevée et à son temps d'entraînement réduit.

Le ViT constitue également une alternative intéressante, notamment grâce à son mécanisme d'attention qui permet de modéliser les relations entre les différentes parties d'une image.

## 7. Synthèse personnelle

Ce TP m'a permis d'améliorer mes connaissances pratiques en Deep Learning et en PyTorch.

J'ai appris à préparer un jeu de données, à construire différentes architectures neuronales, à configurer les hyperparamètres et à entraîner les modèles en utilisant le GPU.

J'ai également compris la différence entre la classification d'images et la détection d'objets, ainsi que l'intérêt du Transfer Learning pour exploiter des modèles pré-entraînés.

Enfin, la construction du Vision Transformer m'a permis de comprendre les concepts de patches, d'embeddings, de CLS Token, de Positional Embeddings et de Self-Attention.

Cette expérience m'a aidée à mieux comprendre les avantages et les limites des différentes architectures et à choisir un modèle adapté à un problème de vision par ordinateur.

## 8. Technologies utilisées

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Google Colab
- CUDA / GPU
- Git et GitHub

## 9. Exécution du projet

Le projet est réalisé dans Google Colab.

Pour exécuter le notebook :

1. Ouvrir le fichier `.ipynb` avec Google Colab.
2. Activer l'accélérateur GPU dans les paramètres d'exécution.
3. Exécuter les cellules dans l'ordre.
4. Le jeu de données MNIST sera téléchargé automatiquement.
5. Entraîner et évaluer les différents modèles.

---

**Réalisé par : Inass Alilou**  
**Encadré par : Pr. Lotfi ELAACHAK**  
**Université Abdelmalek Essaâdi — FST Tanger**  
**Année universitaire : 2025–2026**
