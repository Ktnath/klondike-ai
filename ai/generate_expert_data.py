import json
import numpy as np
from tqdm import tqdm
from klondike_core import Engine, Move

def record_game():
    """Enregistre une partie jouée par un humain."""
    engine = Engine()
    moves = []

    print("\nEnregistrement d'une nouvelle partie...")
    print("Pour chaque coup, entrez le numéro correspondant ou 'stop' pour terminer.")

    while not engine.get_state().is_won():
        display_game_state(engine)
        available_moves = engine.get_available_moves()

        print("\nCoups disponibles:")
        for i, move in enumerate(available_moves):
            print(f"{i}: {move}")

        choice = input("\nVotre choix (ou 'stop' pour terminer): ")
        if choice.lower() == 'stop':
            break

        try:
            move_index = int(choice)
            if 0 <= move_index < len(available_moves):
                selected_move = available_moves[move_index]
                engine.make_move(selected_move)
                
                # Enregistrer l'état et le coup
                state = engine.get_state().encode_observation()
                moves.append({
                    'state': state.tolist(),
                    'move': selected_move.get_move_index(),
                    'result': 1.0 if engine.get_state().is_won() else 0.0
                })
            else:
                print("Index invalide !")
        except ValueError:
            print("Entrée invalide !")

    return moves

def display_game_state(engine):
    """Affiche l'état actuel du jeu."""
    print("\n=== État du jeu ===")
    print(f"Score: {engine.get_state().get_score()}")
    # Implémenter un affichage plus détaillé ici

def save_games(games, filename="expert_games.jsonl"):
    """Sauvegarde les parties dans un fichier JSONL."""
    with open(filename, 'a') as f:
        for game in games:
            for move in game:
                f.write(json.dumps(move) + '\n')

def main():
    print("Générateur de données d'entraînement pour Klondike AI")
    print("=============================================")

    games = []
    while True:
        print("\nOptions:")
        print("1. Enregistrer une nouvelle partie")
        print("2. Sauvegarder et quitter")

        choice = input("\nVotre choix: ")

        if choice == '1':
            game_moves = record_game()
            if game_moves:
                games.append(game_moves)
                print(f"\nPartie enregistrée ! ({len(game_moves)} coups)")
            else:
                print("\nPartie annulée.")

        elif choice == '2':
            if games:
                save_games(games)
                print(f"\n{len(games)} parties sauvegardées dans expert_games.jsonl")
            break

        else:
            print("\nOption invalide !")

def analyze_expert_data(filename="expert_games.jsonl"):
    """Analyse les données d'expert collectées."""
    moves_count = 0
    games_count = 0
    win_count = 0
    move_distribution = np.zeros(96)  # Nombre total d'actions possibles

    current_game_moves = []
    with open(filename, 'r') as f:
        for line in tqdm(f):
            move_data = json.loads(line)
            moves_count += 1
            move_distribution[move_data['move']] += 1

            current_game_moves.append(move_data)
            if move_data['result'] > 0:
                win_count += 1
                games_count += 1
                current_game_moves = []

    print(f"\nStatistiques des données d'expert:")
    print(f"Nombre total de coups: {moves_count}")
    print(f"Nombre de parties: {games_count}")
    print(f"Nombre de victoires: {win_count}")
    print(f"Longueur moyenne des parties: {moves_count/games_count:.1f} coups")
    
    # Afficher la distribution des types de coups
    print("\nDistribution des types de coups:")
    move_types = [
        ("DrawCard", 0, 1),
        ("WasteToTableau", 1, 8),
        ("WasteToFoundation", 8, 12),
        ("TableauToTableau", 12, 61),
        ("TableauToFoundation", 61, 89),
        ("FlipTableauCard", 89, 96)
    ]

    for name, start, end in move_types:
        count = move_distribution[start:end].sum()
        percentage = (count / moves_count) * 100
        print(f"{name}: {percentage:.1f}%")

if __name__ == "__main__":
    main()
    if os.path.exists("expert_games.jsonl"):
        analyze_expert_data()