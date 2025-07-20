import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from klondike_core import Engine
from klondike_ai import Coach, TrainingConfig, NeuralNet

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Démarrage de l'entraînement du modèle Klondike AI...")

    # Configuration de l'entraînement
    config = TrainingConfig(
        num_iterations=50,      # Nombre d'itérations d'entraînement
        num_episodes=100,       # Épisodes par itération
        temp_threshold=10,      # Seuil de température pour l'exploration
        update_threshold=0.6,   # Seuil de mise à jour du meilleur modèle
        max_moves=200,          # Nombre maximum de coups par partie
        num_mcts_sims=25,      # Nombre de simulations MCTS par coup
        arena_compare=40,       # Nombre de parties pour la comparaison
        cpuct=1.0,             # Paramètre d'exploration MCTS
        checkpoint_interval=5,  # Fréquence de sauvegarde du modèle
        load_model=False,       # Charger un modèle existant
        train_examples_history=20  # Historique des exemples d'entraînement
    )

    # Création du dossier de modèles s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Initialisation du réseau de neurones
    input_shape = 156  # Taille de l'état du jeu encodé
    action_size = 96   # Nombre d'actions possibles
    net = NeuralNet(input_shape, action_size)

    if config.load_model and os.path.exists("models/best_model.pth"):
        logging.info("Chargement du modèle existant...")
        net.load_checkpoint("models", "best_model.pth")

    # Création du coach et démarrage de l'entraînement
    coach = Coach(net, config)

    try:
        logging.info("Démarrage de l'entraînement...")
        coach.learn()
        logging.info("Entraînement terminé avec succès !")

    except KeyboardInterrupt:
        logging.info("\nEntraînement interrompu par l'utilisateur.")
        # Sauvegarder le modèle avant de quitter
        logging.info("Sauvegarde du modèle...")
        net.save_checkpoint("models", "interrupted_model.pth")

    except Exception as e:
        logging.error("\nErreur pendant l'entraînement: %s", e)
        raise

def evaluate_model():
    logging.info("Évaluation du modèle...")
    net = NeuralNet(156, 96)
    net.load_checkpoint("models", "best_model.pth")

    engine = Engine()
    wins = 0
    total_games = 100

    for _ in tqdm(range(total_games)):
        engine = Engine()
        moves = 0
        max_moves = 200

        while not engine.get_state().is_won() and moves < max_moves:
            state = engine.get_state().encode_observation()
            pi, v = net.predict(state)
            
            # Sélectionner le meilleur coup
            available_moves = engine.get_available_moves()
            best_move = max(
                available_moves,
                key=lambda m: pi[m.get_move_index()]
            )
            
            engine.make_move(best_move)
            moves += 1

        if engine.get_state().is_won():
            wins += 1

    win_rate = wins / total_games
    logging.info("Taux de victoire: %.2f%%", 100 * win_rate)
    return win_rate

if __name__ == "__main__":
    main()
    evaluate_model()
