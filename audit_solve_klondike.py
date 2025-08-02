import json
from collections import Counter
from klondike_core import solve_klondike, encode_observation, move_index, new_game

N = 10  # Nombre de parties à tester

def audit_solve_klondike():
    total_moves = 0
    all_actions = []
    all_intentions = []
    errors = 0

    print(f"🔎 Audit de {N} parties générées par `solve_klondike()`...")

    for i in range(N):
        try:
            state = new_game()
            result = solve_klondike(state)
            if not isinstance(result, list):
                print(f"❌ Partie {i+1}: format inattendu")
                errors += 1
                continue

            for state, move, intention in result:
                obs = encode_observation(state)
                if not isinstance(obs, list) or len(obs) < 100:
                    print(f"⚠️ Observation invalide en partie {i+1}")
                    errors += 1
                    break

                idx = move_index(move)
                all_actions.append(idx)
                all_intentions.append(intention)

                total_moves += 1

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
    audit_solve_klondike()
