import json
import logging
import os
from typing import List

import numpy as np
from tqdm import tqdm
from klondike_core import Engine, Move


def record_game() -> List[dict]:
    """Enregistre une partie jouée par un humain."""
    engine = Engine()
    moves: List[dict] = []

    logging.info("\nEnregistrement d'une nouvelle partie...")
    logging.info("Pour chaque coup, entrez le numéro correspondant ou 'stop' pour terminer.")

    while not engine.get_state().is_won():
        display_game_state(engine)
        available_moves = engine.get_available_moves()

        logging.info("\nCoups disponibles:")
        for i, move in enumerate(available_moves):
            logging.info("%d: %s", i, move)

        choice = input("\nVotre choix (ou 'stop' pour terminer): ")
        if choice.lower() == "stop":
            break

        try:
            move_index = int(choice)
            if 0 <= move_index < len(available_moves):
                selected_move = available_moves[move_index]
                engine.make_move(selected_move)

                state = engine.get_state().encode_observation()
                moves.append(
                    {
                        "state": state.tolist(),
                        "move": selected_move.get_move_index(),
                        "result": 1.0 if engine.get_state().is_won() else 0.0,
                    }
                )
            else:
                logging.info("Index invalide !")
        except ValueError:
            logging.info("Entrée invalide !")

    return moves


def display_game_state(engine: Engine) -> None:
    """Affiche l'état actuel du jeu."""
    logging.info("\n=== État du jeu ===")
    logging.info("Score: %s", engine.get_state().get_score())


def save_games(games: List[List[dict]], filename: str = "expert_games.jsonl") -> None:
    """Sauvegarde les parties dans un fichier JSONL."""
    with open(filename, "a", encoding="utf-8") as f:
        for game in games:
            for move in game:
                f.write(json.dumps(move) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Générateur de données d'entraînement pour Klondike AI")
    logging.info("=============================================")

    games: List[List[dict]] = []
    while True:
        logging.info("\nOptions:")
        logging.info("1. Enregistrer une nouvelle partie")
        logging.info("2. Sauvegarder et quitter")

        choice = input("\nVotre choix: ")

        if choice == "1":
            game_moves = record_game()
            if game_moves:
                games.append(game_moves)
                logging.info("\nPartie enregistrée ! (%d coups)", len(game_moves))
            else:
                logging.info("\nPartie annulée.")

        elif choice == "2":
            if games:
                save_games(games)
                logging.info("\n%d parties sauvegardées dans expert_games.jsonl", len(games))
            break

        else:
            logging.info("\nOption invalide !")


def analyze_expert_data(filename: str = "expert_games.jsonl") -> None:
    """Analyse les données d'expert collectées."""
    moves_count = 0
    games_count = 0
    win_count = 0
    move_distribution = np.zeros(96)

    current_game_moves = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            move_data = json.loads(line)
            moves_count += 1
            move_distribution[move_data["move"]] += 1

            current_game_moves.append(move_data)
            if move_data["result"] > 0:
                win_count += 1
                games_count += 1
                current_game_moves = []

    logging.info("\nStatistiques des données d'expert:")
    logging.info("Nombre total de coups: %d", moves_count)
    logging.info("Nombre de parties: %d", games_count)
    logging.info("Nombre de victoires: %d", win_count)
    if games_count:
        logging.info("Longueur moyenne des parties: %.1f coups", moves_count / games_count)

    logging.info("\nDistribution des types de coups:")
    move_types = [
        ("DrawCard", 0, 1),
        ("WasteToTableau", 1, 8),
        ("WasteToFoundation", 8, 12),
        ("TableauToTableau", 12, 61),
        ("TableauToFoundation", 61, 89),
        ("FlipTableauCard", 89, 96),
    ]

    for name, start, end in move_types:
        count = move_distribution[start:end].sum()
        percentage = (count / moves_count) * 100 if moves_count else 0.0
        logging.info("%s: %.1f%%", name, percentage)


if __name__ == "__main__":
    main()
    if os.path.exists("expert_games.jsonl"):
        analyze_expert_data()
