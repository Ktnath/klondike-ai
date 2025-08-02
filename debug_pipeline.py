import inspect
import types

from env.klondike_env import KlondikeEnv


def trace_env(debug_games=3):
    print("🔍 Initialisation de l’environnement Klondike...")

    env = KlondikeEnv(use_intentions=True)
    if not hasattr(env, "game") and not hasattr(env, "engine"):
        print("❌ Aucune instance .game ou .engine trouvée dans KlondikeEnv.")
        print("➡️ Vérifie si l’environnement instancie bien le moteur Rust.")
        return

    core_obj = getattr(env, "game", None) or getattr(env, "engine", None)

    print(f"✅ Moteur détecté : {type(core_obj)}")

    # Lister toutes les méthodes disponibles dans le moteur Rust exposé
    print("\n🧠 Méthodes disponibles depuis Python :")
    for name, func in inspect.getmembers(core_obj):
        if isinstance(func, types.MethodType) and not name.startswith("_"):
            print(f"  - {name}()")

    # Vérifie si get_best_move et encode_move sont appelables
    missing = []
    if not hasattr(core_obj, "get_best_move"):
        missing.append("get_best_move")
    if not hasattr(core_obj, "encode_move"):
        missing.append("encode_move")

    if missing:
        print(f"\n❌ Méthode(s) manquante(s) dans l’interface Rust → Python : {', '.join(missing)}")
        print("➡️ Tu dois vérifier que ces méthodes sont bien exportées avec #[pyfunction] ou #[pymethods] dans lib.rs.")
        return

    print("\n🔁 Test dynamique de quelques parties...\n")

    for i in range(debug_games):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < 50:
            move = core_obj.get_best_move()
            if move is None:
                print(f"[{i}-{step}] ❌ get_best_move() → None")
                break
            try:
                action = core_obj.encode_move(move)
            except Exception as e:
                print(f"[{i}-{step}] ❌ encode_move FAILED: {e}")
                break
            print(f"[{i}-{step}] ✅ move={move}, encoded={action}")
            try:
                obs, reward, done, info = env.step(move)
            except Exception as e:
                print(f"[{i}-{step}] ❌ step() error: {e}")
                break
            step += 1

    print("\n📤 Analyse terminée.")

if __name__ == "__main__":
    trace_env(debug_games=3)
