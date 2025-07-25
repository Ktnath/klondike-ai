import json

# New core module exposes the Rust bindings
try:  # pragma: no cover - demo script
    from core import new_game, solve_klondike  # type: ignore
except Exception:  # pragma: no cover - fallback for older builds
    from klondike_core import new_game, solve_klondike

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *


def main() -> None:
    # Generate a new game state
    state_json = new_game()
    print("\n🂠 Nouvelle partie générée")
    print("État initial (JSON) :", state_json)
    print("Résolution en cours...\n")

    try:
        parsed = json.loads(solve_klondike(state_json))
    except Exception as exc:
        print(f"Erreur lors de la résolution : {exc}")
        return

    moves = parsed
    if isinstance(parsed, dict):
        moves = parsed.get("result")

    if not moves:
        print("❌ Aucune solution trouvée.")
        return

    print("✅ Résolution terminée. Coups à jouer :")
    for item in moves:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            mv, intention = item[0], item[1]
            print(f" \u2192 {mv:<4}    \U0001F3AF Intention : {intention}")
        else:
            print(f" \u2192 {item}")


if __name__ == "__main__":
    main()
