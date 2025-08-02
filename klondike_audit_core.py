import sys
import inspect
import json


try:
    import klondike_core
except ImportError as e:
    print("❌ Erreur d'import du module Rust `klondike_core` :", e)
    sys.exit(1)

print("✅ Module `klondike_core` importé avec succès.\n")

# 🔍 Fonctions exposées
public_functions = {
    name: obj
    for name, obj in inspect.getmembers(klondike_core)
    if inspect.isbuiltin(obj) or inspect.isfunction(obj)
}

print("📦 Fonctions disponibles dans `klondike_core` :")
for name in sorted(public_functions):
    print(" -", name)

# ✅ Fonctions critiques attendues
expected = [
    "new_game",
    "legal_moves",
    "play_move",
    "encode_observation",
    "move_from_index",
    "move_index",
    "is_won",
    "is_lost",
]

print("\n🔧 Vérification des fonctions critiques :")
for func in expected:
    if func in public_functions:
        print(f"✅ {func}")
    else:
        print(f"❌ {func} manquante !")

# 🧪 Mini test si possible
print("\n🚀 Test minimal du moteur :")
try:
    state_json = klondike_core.new_game()
    encoded_state = json.loads(state_json)["encoded"]
    moves = klondike_core.legal_moves(encoded_state)
    print(f"🟢 Partie initialisée. {len(moves)} coups disponibles.")
except Exception as e:
    print("⚠️ Erreur lors de l'exécution minimale :", e)
