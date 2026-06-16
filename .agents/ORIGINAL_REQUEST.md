# Original User Request

## Initial Request — 2026-06-16T00:11:58+02:00

# Teamwork Project Prompt

Working directory: /home/adrien/cozmo_ia_project
Integrity mode: development

## Requirements

### R1. Calcul dynamique des poids de classes
Le système doit analyser la distribution des classes dans le jeu de données d'entraînement (et uniquement d'entraînement) pour calculer automatiquement des poids de compensation. 
La formule requise est : `weight_c = total_samples / (num_classes * samples_c)`. Vous pouvez ajouter cette logique dans `dataset.py` ou `train.py`.

### R2. Intégration dans la Loss Function
Les poids calculés doivent être convertis en tenseur PyTorch et passés en tant qu'argument `weight` à la fonction de perte `F.cross_entropy` dans `train.py` (fonctions `train_discrete_epoch` et `eval_discrete` si applicable).

### R3. Entraînement du modèle 2.1
Le script d'entraînement doit être lancé pour créer le nouveau modèle avec le nom `2_1`. (Commande suggérée : `./venv/bin/python train.py --mode discrete --name 2_1 --epochs 15`).

### R4. Génération de la Matrice de Confusion
À la fin de l'entraînement, générez une matrice de confusion sur le jeu de validation avec le nouveau modèle `models/cozmo_nn_discrete_2_1.pt`. Sauvegardez l'image (par exemple `confusion_matrix_2_1.png`).

## Acceptance Criteria

### Calcul et Apprentissage
- [ ] Le script calcule dynamiquement la distribution (sans valeurs "hardcodées").
- [ ] L'entraînement du modèle se termine sans erreur et génère les fichiers du modèle (`cozmo_nn_discrete_2_1.pt` etc).

### Vérification de la Matrice de Confusion
- [ ] La matrice de confusion générée sur le set de validation montre des prédictions hors de la ligne droite (les colonnes gauche/droite ne sont pas vides).
