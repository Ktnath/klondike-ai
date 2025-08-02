import json
from klondike_core import new_game, solve_klondike, encode_observation

print("🔍 Test de solve_klondike(state_json)...")

try:
    # 1. Génère une partie initiale
    state_json = new_game()
    print("✅ Partie générée.")

    # 2. Tente de résoudre cette partie
    solution = solve_klondike(state_json)
    print(f"🧠 {len(solution)} coups trouvés par le solveur.")

    # 3. Affiche les 5 premiers coups
    for i, (move, intention) in enumerate(solution[:5]):
        print(f"➡️ Coup {i+1}: {move} | Intention: {intention}")

    # 4. Affiche les dimensions de l'observation initiale
    obs = encode_observation(state_json)
    print(f"📐 Observation initiale: {len(obs)} dimensions")

except Exception as e:
    print("❌ Erreur:", e)
