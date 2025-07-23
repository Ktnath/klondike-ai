import json
from klondike_core import new_game, solve_klondike


def main() -> None:
    # Generate a new game state
    state_json = new_game()
    print("\n🂠 Nouvelle partie générée")
    print("État initial (JSON) :", state_json)
    print("Résolution en cours...\n")

    try:
        result = json.loads(solve_klondike(state_json))
    except Exception as exc:
        print(f"Erreur lors de la résolution : {exc}")
        return

    moves = result.get("result")
    if moves:
        print(f"✅ Solution trouvée en {len(moves)} coups :\n")
        for idx, mv in enumerate(moves, 1):
            print(f"{idx}. {mv}")
    else:
        print("❌ Aucune solution trouvée.")


if __name__ == "__main__":
    main()
