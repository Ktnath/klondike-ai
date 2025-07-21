# Klondike-AI

**Framework de recherche et d’entraînement pour résoudre le jeu Klondike Solitaire avec des intelligences artificielles.** Ce projet combine DQN, apprentissage par imitation et auto-jeu à la AlphaZero avec MCTS, le tout propulsé par un moteur de jeu Rust exposé en Python.

## 🌧 Introduction

Klondike-AI fournit un environnement complet pour l’étude et l’expérimentation de techniques avancées d’IA appliquées au Solitaire. Il permet l’entraînement de modèles DQN, l’apprentissage par imitation depuis des données humaines, la self-play avec MCTS à la AlphaZero ainsi que de nombreuses fonctionnalités de visualisation et de gestion de replays.

## ⚙️ Installation

Prérequis : **Python ≥ 3.8**, **Rust ≥ 1.60**, `maturin`, `pip`.

```bash
# Cloner le dépôt et créer un environnement virtuel
git clone <URL_vers_le_repo>
cd klondike-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances Python
pip install -r requirements.txt

# Compiler les bindings Rust
maturin develop --release
```

## 🚀 Lancement rapide

```bash
# Entraîner un agent DQN pendant 10 épisodes
python train/train_dqn.py --episodes 10

# Évaluer un modèle existant
python train/evaluate_dqn.py --model_path ./models/MODEL.pth

# Générer des données expertes humaines
python ai/generate_expert_data.py

# Lancer l’entraînement AlphaZero
python ai/train.py

# Vérifier la configuration du système
python check_system_ready.py

# Analyser les performances après entraînement
python train/analyze_training.py --logfile logs/training_log.csv --output_dir reports/
```

## 🤖 Fonctionnalités principales

- ✅ Environnement Gym propulsé par un moteur Rust
- ✅ Récompenses façonnées (shaped rewards)
- ✅ DQN et Dueling DQN
- ✅ Replay buffer avec priorités
- ✅ Self-play à la AlphaZero (Coach + MCTS)
- ✅ Génération de données expertes humaines
- ✅ Visualisation, Optuna et replays
- ✅ Imitation learning et collecte DAgger
- ✅ Génération automatique de jeux optimaux (dataset expert)
- ✅ Analyse post-entraînement des logs

## 📁 Structure du projet

```
klondike-ai/
├─ env/              # Environnement Gym
├─ train/            # DQN, Optuna, Imitation Learning
├─ ai/               # Coach, MCTS, apprentissage supervisé
├─ tools/            # Replays, visualisation
├─ utils/            # Configuration YAML
├─ core/             # Moteur Rust
├─ check_system_ready.py
```

## 🛠️ Configuration

Le fichier `config.yaml` définit les principaux hyperparamètres : taux d’apprentissage, architecture du réseau, nombre d’épisodes, etc. Il est chargé via `utils/config.py`.

## 💾 Sauvegardes et modèles

Les modèles sont sauvegardés avec un timestamp dans le dossier `/models`. Des checkpoints intermédiaires sont enregistrés avec `torch.save` et les résultats (scores, pertes) sont stockés dans `/logs`.

## 📋 TODO / Roadmap

- Ajout d’un MCTS plus avancé (exposition complète Rust)
- Pipeline hybride Imitation + DQN
- Interface graphique optionnelle

## 📄 Licence et contact

Ce projet est distribué sous licence **MIT**. Pour toute question, vous pouvez contacter l’équipe développement via le dépôt GitHub.
