# Klondike AI

Ce projet combine deux implémentations existantes du jeu Klondike Solitaire :
- L'implémentation performante en Rust de Lonelybot
- L'apprentissage par renforcement d'Alpha Zero General

## Structure du Projet

Le projet est divisé en trois crates principaux :

### Core (`klondike-core`)
- Moteur du jeu en Rust
- Génération des mouvements légaux
- Gestion de l'état du jeu
- Bindings Python pour l'intégration avec l'IA

### AI (`klondike-ai`)
- Implémentation de l'algorithme AlphaZero
- Réseau de neurones pour l'apprentissage
- Recherche arborescente Monte Carlo (MCTS)
- Entraînement et évaluation des modèles

### CLI (`klondike-cli`)
- Interface en ligne de commande
- Visualisation du jeu
- Interaction avec l'IA
- Outils de débogage et d'analyse

## Fonctionnalités

- Moteur de jeu haute performance en Rust
- Apprentissage par renforcement avec AlphaZero
- Support pour l'apprentissage par imitation
- Interface CLI interactive
- Bindings Python pour l'expérimentation

## Installation

```bash
# Construction du projet
cargo build --release

# Installation des dépendances Python pour l'IA
pip install -r ai/requirements.txt
```

## Utilisation

```bash
# Lancer l'interface CLI
cargo run --release -p klondike-cli

# Entraîner un nouveau modèle
python ai/train.py
```

## Licence

MIT