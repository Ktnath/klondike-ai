import logging
import os
import csv
import json  # PATCHED for 160-dim with intentions
import torch
import numpy as np
from tqdm import tqdm
from klondike_ai import Coach, TrainingConfig, NeuralNet
import argparse
import shutil
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import load_config
from self_play_generate import generate_self_play
from train.train_dqn import DQN, DuelingDQN, load_dataset
from utils.training import log_epoch_metrics


def _next_model_version(directory: str = "models") -> int:
    """Return the next available model version number."""
    os.makedirs(directory, exist_ok=True)
    versions = []
    for name in os.listdir(directory):
        if name.startswith("model_v") and name.endswith(".pt"):
            num = name[7:-3]
            if num.isdigit():
                versions.append(int(num))
    return max(versions) + 1 if versions else 1


def fine_tune_dqn(
    dataset_path: str,
    base_model: str | None,
    epochs: int,
    batch_size: int,
    output_path: str,
) -> str:
    """Fine tune a DQN on a dataset and save it to ``output_path``."""
    config = load_config()
    obs_arr, actions_arr, _ = load_dataset(dataset_path)

    X = torch.tensor(obs_arr, dtype=torch.float32)
    y = torch.tensor(actions_arr, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    action_dim = int(y.max().item()) + 1
    model_cls = DuelingDQN if getattr(config.model, "dueling", False) else DQN
    model = model_cls(input_dim, action_dim)
    if base_model and os.path.exists(base_model):
        try:
            model.load_state_dict(torch.load(base_model, map_location="cpu"))
        except Exception as exc:  # pragma: no cover - best effort load
            logging.warning("Could not load base model %s: %s", base_model, exc)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    metrics: list[tuple[int, float, float]] = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(1)
            correct += (preds == by).sum().item()
            total += bx.size(0)
        avg_loss = total_loss / total
        log_epoch_metrics(epoch, avg_loss, correct, total)
        acc = correct / total if total else 0.0
        metrics.append((epoch, avg_loss, acc))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    log_path = os.path.splitext(output_path)[0] + "_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy"])
        writer.writerows(metrics)
    logging.info("Model saved to %s", output_path)
    return output_path

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
    cfg = load_config()
    base_dim = getattr(cfg.env, "observation_dim", 156)
    use_int = getattr(cfg.env, "use_intentions", False)
    input_shape = base_dim + 4 if use_int else base_dim  # PATCHED for 160-dim with intentions
    action_size = getattr(cfg.env, "action_dim", 96)
    net = NeuralNet(input_shape, action_size)

    if config.load_model and os.path.exists("models/best_model.pth"):
        logging.info("Chargement du modèle existant...")
        net.load_checkpoint("models", "best_model.pth")

    # Création du coach et démarrage de l'entraînement
    coach = Coach(net, config, use_intentions=use_int)  # PATCHED for 160-dim with intentions
    if coach.env.observation_space.shape[0] != input_shape:  # PATCHED for 160-dim with intentions
        logging.warning(
            "Model input dim %d and env dim %d mismatch", input_shape, coach.env.observation_space.shape[0]
        )

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
    cfg = load_config()
    base_dim = getattr(cfg.env, "observation_dim", 156)
    use_int = getattr(cfg.env, "use_intentions", False)
    input_shape = base_dim + 4 if use_int else base_dim  # PATCHED for 160-dim with intentions
    action_size = getattr(cfg.env, "action_dim", 96)
    net = NeuralNet(input_shape, action_size)
    net.load_checkpoint("models", "best_model.pth")

    from env.klondike_env import KlondikeEnv
    env = KlondikeEnv(use_intentions=use_int)  # PATCHED for 160-dim with intentions

    wins = 0
    total_games = 100

    for _ in tqdm(range(total_games)):
        state, _ = env.reset()  # migrated from gym to gymnasium
        done = False
        moves = 0
        max_moves = 200

        while not done and moves < max_moves:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            pi, _ = net.predict(state.tolist())
            pi_valid = np.array([pi[a] for a in valid_actions])
            action = valid_actions[int(np.argmax(pi_valid))]
            next_state, _, terminated, truncated, _ = env.step(action)  # migrated from gym to gymnasium
            done = terminated or truncated
            state = next_state
            moves += 1

        if json.loads(env.state).get("is_won", False):
            wins += 1

    win_rate = wins / total_games
    logging.info("Taux de victoire: %.2f%%", 100 * win_rate)
    return win_rate


def run_self_play_cycle(
    episodes: int,
    base_model: str | None = None,
    fine_tune: bool = False,
) -> None:
    """Generate self-play data and optionally fine-tune the model."""
    config = load_config()

    new_data = config.self_play.save_path
    logging.info("Generating %d self-play episodes", episodes)
    generate_self_play(base_model, new_data, episodes, use_mcts=False)

    replay_path = config.dataset.replay_data
    merge_strategy = getattr(config.self_play, "merge_strategy", "append")

    if os.path.exists(replay_path) and merge_strategy == "append":
        logging.info("Merging new self-play data into %s", replay_path)
        old = np.load(replay_path, allow_pickle=True)
        new = np.load(new_data, allow_pickle=True)
        merged = {}
        for key in new.files:
            if key in old.files:
                merged[key] = np.concatenate([old[key], new[key]], axis=0)
            else:
                merged[key] = new[key]
        for key in old.files:
            if key not in merged:
                merged[key] = old[key]
        np.savez_compressed(replay_path, **merged)
    else:
        logging.info("Replacing replay buffer with new data")
        shutil.move(new_data, replay_path)

    if fine_tune:
        epochs = getattr(config.self_play, "fine_tune_epochs", 5)
        batch = config.training.batch_size
        version = _next_model_version()
        output_model = os.path.join("models", f"model_v{version}.pt")
        fine_tune_dqn(replay_path, base_model, epochs, batch, output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klondike AI Training")
    parser.add_argument("--self_play", action="store_true", help="Run self-play cycle")
    parser.add_argument("--episodes", type=int, default=1000, help="Self-play episodes")
    parser.add_argument("--fine_tune", action="store_true", help="Fine tune model after self-play")
    parser.add_argument("--base_model", type=str, help="Path to base DQN model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.self_play:
        run_self_play_cycle(args.episodes, args.base_model, args.fine_tune)
    else:
        main()
        evaluate_model()
