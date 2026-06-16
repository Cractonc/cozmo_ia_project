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

## Follow-up — 2026-06-16T00:44:47+02:00

# Teamwork Project Prompt

Implémenter et entraîner le modèle CozmoPilotDiscrete 2.2 en appliquant la Méthode C (Symétrie Forcée + Écrêtage) pour stabiliser les poids de la fonction de perte.

Working directory: /home/adrien/cozmo_ia_project
Integrity mode: development

## Requirements

### R1. Symétrie Forcée des occurrences
Dans la logique de calcul des poids de classes (actuellement dans `train.py` autour de la ligne 153), modifiez les occurrences brutes avant d'appliquer la formule.
Moyennez les classes miroirs :
- `avg_2_3 = (counts[2] + counts[3]) / 2.0` (assignez cette valeur à counts[2] et counts[3])
- `avg_4_5 = (counts[4] + counts[5]) / 2.0` (assignez cette valeur à counts[4] et counts[5])

### R2. Calcul et Écrêtage (Clipping)
Calculez les poids bruts avec `raw_weights = total_samples / (7.0 * counts)`.
Appliquez ensuite un clipping : `clipped_weights = np.clip(raw_weights, 0.5, 3.0)`.
Assurez-vous que ce nouveau tenseur `clipped_weights` est bien passé à la CrossEntropyLoss.

### R3. Entraînement du modèle 2.2
Lancez l'entraînement du nouveau modèle avec l'argument `--name 2_2` (ex: `./venv/bin/python train.py --mode discrete --name 2_2 --epochs 15`).

### R4. Génération de la Matrice de Confusion
À la fin de l'entraînement, mettez à jour votre script d'évaluation ou utilisez `eval_confusion.py` en le modifiant pour pointer sur le modèle `cozmo_discrete_nn_2_2.pt` (ou `cozmo_nn_discrete_2_2.pt`) afin de générer une nouvelle matrice de confusion sur le set de validation et enregistrez-la sous `/home/adrien/.gemini/antigravity/brain/5418179d-c86f-4b55-9fc9-325fac478202/confusion_matrix_2_2.png`.

## Acceptance Criteria

### Évaluation du Code
- [ ] Le code applique explicitement la moyenne sur les indices 2/3 et 4/5.
- [ ] La fonction `np.clip` borne les poids entre 0.5 et 3.0 inclus.

### Vérification de la Matrice de Confusion
- [ ] L'entraînement génère les fichiers de modèle (`cozmo_discrete_nn_2_2.pt` ou équivalent).
- [ ] La nouvelle matrice de confusion montre un retour de la diagonale dominante (la majorité des images "Avant" sont prédites "Avant") tout en conservant des prédictions valides pour les autres classes.
