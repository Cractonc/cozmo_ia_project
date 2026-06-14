# 🤖 Cozmo IA — Assistant Robotique Multimodal

> **Apprentissage par imitation, navigation autonome NN + Gemini, et pilotage conversationnel pour le robot Anki Cozmo.**

---

## Table des matières

- [Présentation](#présentation)
- [Démonstration](#démonstration)
- [Architecture Globale](#architecture-globale)
- [Fonctionnalités](#fonctionnalités)
  - [Mode Conversationnel](#-mode-conversationnel)
  - [Mode Entraînement](#-mode-entraînement)
  - [Mode Pilote NN](#-mode-pilote-nn-autonome)
  - [Mode Exploration Hybride](#-mode-exploration-hybride-nn--gemini)
  - [Mode Navigation Waypoint](#-mode-navigation-waypoint-classique)
  - [Carte 3D SLAM](#-carte-3d-slam)
- [Prérequis](#prérequis)
  - [Matériel](#matériel)
  - [Logiciels](#logiciels)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Architecture du réseau de neurones](#architecture-du-réseau-de-neurones)
- [Structure du projet](#structure-du-projet)
- [Référence WebSocket](#référence-websocket)
- [Dépannage](#dépannage)
- [Idées d'amélioration](#idées-damélioration)

---

## Présentation

**Cozmo IA** est un système de contrôle avancé pour le robot **Anki Cozmo** qui combine intelligence artificielle multimodale et apprentissage automatique. Le projet permet de passer d'un robot téléguidé à un agent autonome capable de percevoir son environnement, de raisonner et d'agir de façon indépendante.

Le système se compose de quatre couches d'intelligence complémentaires :

| Couche | Technologie | Rôle |
|--------|-------------|------|
| **Perception** | Caméra 80×60 + 12 capteurs | Voir et sentir l'environnement |
| **Réflexe** | CNN PyTorch (20 Hz) | Pilotage réactif des roues |
| **Stratégie** | Gemini Vision AI (~5 s) | Navigation de haut niveau |
| **Conversation** | Gemini + Speech Recognition | Interaction naturelle |

Le tout est orchestré via un **dashboard web temps réel** (WebSocket + FastAPI) avec un rendu 3D de la carte explorée.

---

## Démonstration

```
http://localhost:8000/frontend/index.html
```

Le dashboard offre :
- **Flux vidéo live** de la caméra Cozmo (10 FPS)
- **Visualisation des décisions moteur** en temps réel (barres de vitesse gauche/droite + angle de tête)
- **Journaux Chain-of-Thought** : le raisonnement de l'IA étape par étape
- **Boussole** cap actuel vs cap cible (mode hybride)
- **Carte 3D** du parcours avec photos mémoire

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                        DASHBOARD WEB                            │
│          (HTML/CSS/JS + Three.js + WebSocket)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket ws://localhost:8000/ws
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI + uvicorn                             │
│                       main.py                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Mode Convers.│  │ Mode Entraîn.│  │  Mode NN / Hybride   │ │
│  │  Gemini + STT │  │ Capture 20Hz │  │  nn_inference_loop   │ │
│  └───────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ cozmo SDK
┌────────────────────────────▼────────────────────────────────────┐
│                       ROBOT COZMO                                │
│  Caméra 160×120 → 80×60 grayscale  │  Roues  │  Tête  │  Bras │
└─────────────────────────────────────────────────────────────────┘

          Cloud APIs                  Local ML
    ┌─────────────────┐         ┌───────────────────┐
    │  Google Gemini   │         │  CozmoNN (PyTorch)│
    │  gemini-3.1-     │         │  CNN + MLP Fusion │
    │  flash-lite      │         │  ~3 ms/inférence  │
    └─────────────────┘         └───────────────────┘
```

---

## Fonctionnalités

### 🗣️ Mode Conversationnel

Le mode par défaut. Cozmo répond aux commandes vocales ou texte avec :

- **Vision multimodale** : chaque question inclut automatiquement une capture caméra
- **Mémoire FIFO** : fenêtre de contexte de 3 tours (6 messages) pour des échanges cohérents
- **Actions physiques** : lever/baisser le bras, avancer, reculer, tourner gauche/droite
- **Animations faciales** : 4 émotions (happy, sad, surprised, angry)
- **Widget météo OLED** : affiche température et icône météo sur le visage du robot si une ville est mentionnée (via wttr.in)
- **Reconnaissance vocale** : via Google Speech Recognition (FR)

**Mots-clés de navigation** : `va vers`, `explore`, `trouve`, `dirige-toi vers`… → bascule automatiquement vers le mode hybride ou waypoint.

---

### 🎮 Mode Entraînement

Collecte de données pour l'apprentissage par imitation (**Behavioral Cloning**).

**Contrôles clavier** :

| Touche | Action |
|--------|--------|
| `Z` | Avancer |
| `S` | Reculer |
| `Q` | Tourner gauche |
| `D` | Tourner droite |
| `Q` + `Z` | Courbe gauche avant |
| `D` + `Z` | Courbe droite avant |
| `↑` / `↓` | Angle tête (−25° à +44.5°) |

**Données enregistrées à 20 Hz** :
- `frames` : image 80×60 grayscale (uint8)
- `sensors` : vecteur de 12 capteurs (float32)
- `actions` : [vitesse_roue_gauche, vitesse_roue_droite, angle_tête] (float32)

**Sécurité** : détection automatique de falaise pendant le pilotage manuel, arrêt immédiat si `cliff_detected` et si la commande n'est pas un recul ou une rotation.

---

### 🧠 Mode Pilote NN Autonome

Inférence du réseau de neurones entraîné à **20 Hz** (50 ms/cycle).

**Pipeline de traitement** :
1. Capture image → conversion grayscale 80×60 → normalisation `[0,1]`
2. Lecture vecteur capteurs → standardisation z-score (stats sauvegardées à l'entraînement)
3. Inférence `CozmoNN` → `[left_speed, right_speed, head_angle]`
4. Clip sécurisé : roues `[−150, 150]` mm/s, tête `[−25, +45]`°
5. Application des commandes + mise à jour dashboard (~4 Hz)

**Watchdog** : si 5 inférences consécutives dépassent 100 ms → arrêt d'urgence automatique.

**Visualisation en temps réel** :
- Barres de vitesse des roues gauche/droite (couleur verte = avant, rouge = arrière)
- Slider de l'angle de tête
- FPS, temps d'inférence, compteur d'alertes falaise

---

### 🚀 Mode Exploration Hybride NN + Gemini

Le mode le plus avancé : **deux boucles imbriquées** en parallèle.

```
┌─────────────────────────────────────────────────────┐
│           BOUCLE LENTE — Gemini (~5 s)              │
│  Image 320×240 → Analyse stratégique → cap_cible   │
│  "Tourne vers la porte, cap 45°, confiance 87%"    │
└────────────────────────┬────────────────────────────┘
                         │ shared_state["cap_cible"]
┌────────────────────────▼────────────────────────────┐
│           BOUCLE RAPIDE — NN (20 Hz)                │
│  Image 80×60 + capteurs + Δcap → roues             │
│  Correction PID du cap : gauche ↑ si cap > cible   │
└─────────────────────────────────────────────────────┘
```

**Option A** (`use_nn_heading_input: True`) : `CozmoNNv2` avec 13 entrées (12 capteurs + delta_heading normalisé).

**Option B** (`use_nn_heading_input: False`, défaut) : `CozmoNN` 12 entrées + **correction PID proportionnelle** appliquée en post-traitement :
```python
correction = kp * delta_heading_normalized * 50.0
left_speed  = nn_left  + correction
right_speed = nn_right - correction
```

**Intervalle Gemini adaptatif** : 5 s normalement, réduit à 2 s si changement de cap > 45°.

**Paramètres configurables** dans `hybrid_navigation.py` :

```python
HYBRID_CONFIG = {
    "use_nn_heading_input": False,
    "kp_correction": 1.0,
    "gemini_interval_s": 5.0,
    "gemini_fast_interval_s": 2.0,
    "cap_change_threshold": 45.0,
    "timeout_s": 300.0,
}
```

---

### 🗺️ Mode Navigation Waypoint Classique

Navigation purement Gemini sans NN, architecture **Stop → Look → Think → Act**.

**Stratégie de contournement Manhattan** (gérée par le prompt Gemini) :
1. **Étape A** : Pivot 90° (gauche ou droite)
2. **Étape B** : Avance latérale (15–60 cm selon la largeur de l'obstacle détectée visuellement)
3. **Étape C** : Pivot inverse 90° → retour au cap d'origine
4. **Étape D** : Avance 20–30 cm pour dépasser l'obstacle
5. **Étape E** : Réalignement fin vers l'objectif

**Sécurité anti-chute concurrent** : la surveillance du capteur IR tourne à 20 Hz *en parallèle* du mouvement via `asyncio.wait(FIRST_COMPLETED)`. En cas de falaise : recul d'urgence de 3 cm + injection de contexte d'urgence dans l'historique Gemini. Abandon après 5 détections.

---

### 🌍 Carte 3D SLAM

Visualisation Three.js en overlay fullscreen.

- **Tracé de trajectoire** par session avec couleur unique
- **Modèle 3D Cozmo** articulé (rouge et blanc, roues, tête, bras)
- **Photos mémoire Polaroid** : chaque décision IA ajoute une photo cliquable positionnée dans l'espace 3D
- **Marqueurs de falaise** : cônes ambrés aux points de danger détectés
- **Multi-session** : filtrage par session avec checkboxes
- **Caméra suiveuse** : option "Suivre Cozmo" (toggle)
- **Raycaster** : clic sur une photo → modal d'affichage

---

## Prérequis

### Matériel

- **Robot Anki Cozmo** (génération 1 ou 2)
- **Smartphone** iOS ou Android avec l'application **Cozmo** installée
- **Câble USB** (smartphone ↔ ordinateur)
- Ordinateur avec **Python 3.8+** (CPU suffisant, GPU optionnel)

> ⚠️ Le robot doit être connecté au smartphone via USB **avant** de lancer le programme.

### Logiciels

- Python ≥ 3.8 (strict — pas de walrus operator, pas d'`asyncio.to_thread`)
- [cozmo SDK](http://cozmosdk.anki.com) ≥ 1.4.10
- PyTorch ≥ 2.0 (CPU uniquement par défaut)
- Clé API **Google Gemini** (modèle `gemini-3.1-flash-lite`)
- Microphone (optionnel, pour la commande vocale)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/cozmo-ia.git
cd cozmo-ia
```

### 2. Créer l'environnement virtuel

```bash
python3.8 -m venv venv
source venv/bin/activate       # Linux/macOS
# ou
venv\Scripts\activate          # Windows
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip

# Framework web
pip install fastapi uvicorn[standard]

# SDK Cozmo
pip install cozmo[camera]

# Machine Learning
pip install torch torchvision    # CPU-only par défaut
# ou pour une installation légère CPU :
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Données & Images
pip install numpy pillow

# API & Reconnaissance vocale
pip install requests python-dotenv SpeechRecognition pyaudio

# Visualisation (optionnel, pour plot_training.py)
pip install matplotlib
```

> **Note macOS** : Si `pyaudio` échoue, installer d'abord `portaudio` :
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

> **Note Linux** : `sudo apt-get install python3-pyaudio portaudio19-dev`

### 4. Vérifier l'installation du SDK Cozmo

```bash
python -c "import cozmo; print('Cozmo SDK OK')"
```

Consultez la [documentation officielle du SDK](http://cozmosdk.anki.com/docs/initial.html) pour la configuration du smartphone.

---

## Configuration

### Fichier `.env`

Créez un fichier `.env` à la racine du projet :

```env
GEMINI_API_KEY=votre_clé_api_google_gemini_ici
```

Obtenez votre clé gratuite sur [Google AI Studio](https://aistudio.google.com/).

### Structure des dossiers à créer

```bash
mkdir -p training_data models
```

---

## Utilisation

### Démarrer le système

1. **Branchez** le smartphone au PC via USB
2. **Ouvrez** l'application Cozmo sur le smartphone
3. **Activez** le mode développeur dans l'app (Settings → SDK)
4. **Lancez** le programme :

```bash
python main.py
```

5. **Ouvrez** le dashboard dans votre navigateur :

```
http://localhost:8000/frontend/index.html
```

### Commandes texte disponibles

| Commande | Effet |
|----------|-------|
| `bonjour, que vois-tu ?` | Analyse visuelle conversationnelle |
| `lève le bras` | Action physique |
| `avance de 20 cm` | Mouvement précis |
| `quelle est la météo à Paris ?` | Widget OLED météo |
| `va vers la porte` | Déclenche la navigation autonome |
| `explore la pièce` | Déclenche le mode hybride |
| `stop` / `quitter` | Arrêt du programme |

---

## Pipeline d'entraînement

### Étape 1 — Collecter les données

```
Dashboard → 🎮 Mode Entraînement → ⏺ Démarrer Enregistrement
```
Pilotez Cozmo avec le clavier. Les sessions sont sauvegardées dans `training_data/session_YYYY-MM-DD_HH-MM-SS.npz`.

**Durée recommandée** : 10–20 minutes de conduite variée (avancer, courbes, marche arrière, différentes surfaces).

### Étape 2 — Vérifier les données

```bash
python verify_training_data.py training_data/session_2024-01-01_12-00-00.npz
```

Affiche : shapes, durée, fréquence, min/max par capteur, répartition avant/arrière/rotation/stop.

### Étape 3 — Entraîner le modèle

```bash
python train.py \
  --data_dir training_data/ \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --name mon_pilote_v1
```

Le meilleur modèle est sauvegardé dans `models/cozmo_nn_mon_pilote_v1.pt` avec ses stats de normalisation (`norm_stats_mon_pilote_v1.json`).

**Paramètres** :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--epochs` | 50 | Nombre d'époques max |
| `--batch_size` | 64 | Taille du batch |
| `--lr` | 1e-3 | Learning rate initial |
| `--data_dir` | `training_data/` | Dossier des données |
| `--output_dir` | `models/` | Dossier de sortie |
| `--name` | timestamp | Nom du run |

**Early stopping** : arrêt automatique si pas d'amélioration pendant 10 époques.

### Étape 4 — Visualiser la convergence

```bash
python plot_training.py models/history_mon_pilote_v1.json
```

### Étape 5 — Tester le pilote

```
Dashboard → 🧠 Mode NN → Sélectionner le modèle → ▶ Démarrer
```

### Données d'augmentation (automatique à l'entraînement)

- **Flip horizontal** (50%) : miroir de l'image + inversion des roues gauche/droite + inversion du cap
- **Bruit gaussien** (50%) : σ = 0.02 sur l'image normalisée
- **Jitter luminosité** (50%) : facteur aléatoire ×[0.8, 1.2]

---

## Architecture du réseau de neurones

```
Image (1, 60, 80)           Capteurs (12,)
       │                          │
   ┌───▼───────────┐         ┌────▼────────┐
   │  Branche CNN  │         │  Branche MLP│
   │  Conv2d 16    │         │  Linear 32  │
   │  Conv2d 32    │         │  Linear 32  │
   │  Conv2d 64    │         │  ReLU       │
   │  AvgPool 4×4  │         └─────────────┘
   │  Flatten 1024 │                │
   │  Linear 128   │                │
   │  ReLU, Drop30%│                │
   └───────────────┘                │
           │                        │
           └──────────┬─────────────┘
                      │ cat → (160,)
                 ┌────▼────────────────┐
                 │  Fusion MLP         │
                 │  Linear 160 → 64   │
                 │  ReLU, Dropout 20% │
                 │  Linear 64 → 32    │
                 │  ReLU              │
                 └────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Output (3) │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       tanh × 150    tanh × 150   tanh×35+10
      [-150,+150]   [-150,+150]   [-25,+45]
       Roue Gauche   Roue Droite   Tête (°)
```

### Vecteur sensoriel (12 valeurs)

| Index | Capteur | Unité |
|---|---|---|
| 0 | Position X (odométrie) | mm |
| 1 | Position Y (odométrie) | mm |
| 2 | Orientation cap (angle Z) | degrés |
| 3 | Inclinaison pitch | degrés |
| 4 | Vitesse roue gauche | mm/s |
| 5 | Vitesse roue droite | mm/s |
| 6 | Falaise détectée | 0 ou 1 |
| 7 | Hauteur bras élévateur | mm |
| 8 | Angle tête | degrés |
| 9 | Tension batterie | V |
| 10 | Quaternion q0 | — |
| 11 | Robot en mouvement | 0 ou 1 |

**Paramètres** : ~26 000 (très léger, optimisé pour CPU temps réel)

**Loss** : MSE pondérée — roues × 1.0, tête × 0.3 (les roues comptent plus que l'orientation de la tête)

---

## Structure du projet

```
cozmo-ia/
│
├── main.py                  # Orchestrateur principal + serveur FastAPI + WebSocket
├── model.py                 # Architectures CozmoNN et CozmoNNv2
├── dataset.py               # Dataset PyTorch + augmentation
├── train.py                 # Script d'entraînement
├── navigation.py            # Navigation Waypoint (Stop-Look-Think-Act)
├── hybrid_navigation.py     # Navigation Hybride NN + Gemini
│
├── plot_training.py         # Visualisation des courbes d'entraînement
├── verify_training_data.py  # Vérification des données .npz
├── test_gemini.py           # Test de connexion API Gemini
│
├── frontend/
│   ├── index.html           # Dashboard principal
│   ├── app.js               # Client WebSocket + logique UI
│   ├── style.css            # Design glassmorphism dark
│   └── map3d.js             # Carte 3D Three.js
│
├── training_data/           # Fichiers .npz (gitignore)
│   └── session_*.npz
│
├── models/                  # Modèles entraînés (gitignore)
│   ├── cozmo_nn_*.pt
│   ├── norm_stats_*.json
│   ├── history_*.json
│   └── cozmo_nn_*.json      # Métadonnées (displayName, version, paramètres)
│
├── .env                     # Clé API (gitignore)
├── .gitignore
└── README.md
```

---

## Référence WebSocket

Le dashboard communique avec le backend via `ws://localhost:8000/ws`.

### Messages Client → Serveur

| `type` | Champs | Description |
|---|---|---|
| `command` | `command: string` | Commande texte libre (conversationnel) |
| `training_mode` | `active: bool` | Active/désactive le mode entraînement |
| `training_start` | — | Démarre l'enregistrement |
| `training_stop` | — | Arrête et sauvegarde la session |
| `training_drive` | `left: float, right: float` | Commande roues (mm/s) |
| `training_head` | `angle: float` | Angle tête (degrés) |
| `nn_start` | `model: string` | Démarre l'inférence NN avec le modèle donné |
| `nn_stop` | — | Arrête le mode NN |
| `hybrid_start` | `model: string, objective: string` | Démarre la navigation hybride |
| `hybrid_stop` | — | Arrête le mode hybride (kill switch) |

### Messages Serveur → Client

| `type` | Champs principaux | Description |
|---|---|---|
| `video_frame` | `frame: base64` | Image JPEG encodée en base64 (~10 FPS) |
| `state_update` | `odometry, state, cliff_detected` | Télémetrie et état du robot |
| `training_status` | `recording, frame_count, duration_s, file_size_mb` | Statut de l'enregistrement |
| `nn_status` | `active, fps, inference_ms, left_speed, right_speed, head_angle, cliff_events` | Métriques NN |
| `hybrid_status` | `nn_active, gemini_active, cap_cible, cap_actuel, fps, inference_ms, ...` | Métriques hybride |
| `hybrid_gemini` | `cap_cible, description_scene, strategie, confiance` | Décision stratégique Gemini |
| `hybrid_stopped` | — | Confirmation d'arrêt du mode hybride |
| `ai_cot` | `etape, diagnostic, action, valeur, raw, odometry, frame` | Log Chain of Thought |
| `terminal_log` | `message, source` | Message terminal (robot / system / user) |

---

## Dépannage

### Le SDK Cozmo ne se connecte pas

- Vérifiez que le mode développeur est bien activé dans l'app Cozmo
- Essayez un autre câble USB (certains câbles ne supportent pas la communication de données)
- Sur macOS Monterey+ : autorisez la connexion USB dans les préférences de confidentialité
- Vérifiez que le smartphone est bien en mode **Android File Transfer** (Android) ou **Confiance** (iOS)

### `ModuleNotFoundError: No module named 'cozmo'`

```bash
source venv/bin/activate
pip install cozmo[camera]
```

### `pyaudio` introuvable (mode vocal désactivé)

Le mode texte fonctionne sans `pyaudio`. Pour activer la reconnaissance vocale :
```bash
# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio && pip install pyaudio
```

### Erreur 429 (Rate Limit Gemini)

Le système inclut un **retry avec backoff exponentiel** (3 tentatives, 2s → 4s → 8s). Si les erreurs persistent, vérifiez votre quota sur [Google AI Studio](https://aistudio.google.com/).

### Le modèle ne s'affiche pas dans le dashboard NN

Vérifiez que les deux fichiers existent et ont des noms cohérents :
```
models/cozmo_nn_MON_NOM.pt
models/norm_stats_MON_NOM.json
```

### Inférences trop lentes (> 100 ms)

- Le modèle tourne sur CPU. Fermez les autres applications.
- Le watchdog arrêtera le mode NN après 5 dépassements consécutifs.
- Si le problème persiste, réduisez la taille du batch à l'entraînement.

### Tester la connexion API Gemini

```bash
python test_gemini.py
```

---

## Idées d'amélioration

Voici des pistes d'évolution identifiées lors de l'analyse du projet :

### Court terme
- [ ] **Replay de session** : rejouer un fichier `.npz` dans l'interface pour visualiser les données collectées image par image
- [ ] **Statistiques d'entraînement live** : afficher la loss et la progression en temps réel pendant `train.py` via WebSocket
- [ ] **Gestion des modèles** : bouton de suppression et renommage des modèles directement depuis le dashboard
- [ ] **Export vidéo** : assembler les frames d'une session en `.mp4` pour analyse

### Moyen terme
- [ ] **DAGGER (Dataset Aggregation)** : pendant le mode NN, permettre à l'humain de reprendre le contrôle pour corriger des erreurs, et enregistrer automatiquement ces corrections comme nouvelles données d'entraînement
- [ ] **Mémoire longue terme Gemini** : sauvegarder l'historique des explorations passées dans un fichier JSON et l'injecter dans le contexte des nouvelles sessions
- [ ] **Mode multi-objectifs** : enchaîner des objectifs en séquence dans le mode hybride (`va à la porte, puis reviens au départ`)
- [ ] **Calibration automatique** : séquence d'auto-calibration des vitesses de roues au démarrage
- [ ] **CozmoNNv3 avec attention** : remplacer la branche CNN par un mini-ViT pour une meilleure compréhension spatiale

### Long terme
- [ ] **SLAM cartographique** : utiliser l'odométrie pour construire une carte 2D occupancy grid persistante entre les sessions
- [ ] **Interface mobile** : PWA responsive pour piloter depuis smartphone sans PC
- [ ] **Entraînement distribué** : pipeline de collecte de données multi-sessions avec serveur centralisé
- [ ] **Transfert de domaine** : adapter le modèle à différentes surfaces et conditions d'éclairage via fine-tuning rapide

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## Crédits

- **Anki Cozmo SDK** — [cozmosdk.anki.com](http://cozmosdk.anki.com)
- **Google Gemini** — Intelligence artificielle multimodale
- **Three.js** — Rendu 3D WebGL pour la carte
- **FastAPI** — Framework web Python asynchrone

---

*Projet développé avec ❤️ pour explorer les frontières entre robotique grand public et IA embarquée.*
