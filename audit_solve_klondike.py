import argparse
from klondike_core import (
    solve_klondike,
    encode_observation,
    move_index,
    new_game,
    play_move,
)

N = 10  # Nombre de parties à tester


def audit_solve_klondike(limit: int | None = None) -> None:
    total_moves = 0
    all_actions = []
    all_intentions = []
    errors = 0

    print(f"🔎 Audit de {N} parties générées par `solve_klondike()`...")

    for i in range(N):
        try:
            state_json = new_game()
            result = solve_klondike(state_json)
            if not isinstance(result, list):
                print(f"❌ Partie {i+1}: format inattendu")
                errors += 1
                continue

            for move_idx, (move, intention) in enumerate(result):
                if limit is None or move_idx < limit:
                    print(f"Move: {move}, Intention: {intention}")

                obs = encode_observation(state_json)
                if not isinstance(obs, list) or len(obs) < 100:
                    print(f"⚠️ Observation invalide en partie {i+1}")
                    errors += 1
                    break

                idx = move_index(move)
                if idx == -1:
                    print(f"⚠️ move_index invalide pour le move : {move}")
                    errors += 1
                all_actions.append(idx)
                all_intentions.append(intention)
                total_moves += 1

                # Advance to next state using the move
                state_json, _ = play_move(state_json, move)

        except Exception as e:
            print(f"❌ Erreur partie {i+1}: {e}")
            errors += 1

    print("\n📊 Résumé :")
    print(f"- Total de coups analysés : {total_moves}")
    print(f"- Actions distinctes : {len(set(all_actions))}")
    print(f"- Intentions distinctes : {set(all_intentions)}")
    print(f"- Erreurs détectées : {errors}")

    if len(set(all_actions)) <= 1:
        print("❌ Pas de diversité dans les actions (dataset dégénéré).")
    else:
        print("✅ Diversité d’actions confirmée.")

    if any(i is None or i.strip() == "" for i in all_intentions):
        print("⚠️ Certaines intentions sont manquantes ou vides.")
    else:
        print("✅ Toutes les intentions sont renseignées.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit solve_klondike outputs")
    parser.add_argument(
        "--limit", type=int, default=None, help="Afficher seulement les N premiers coups"
    )
    args = parser.parse_args()
    audit_solve_klondike(limit=args.limit)
