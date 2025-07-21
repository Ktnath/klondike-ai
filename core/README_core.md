# klondike_core

`klondike_core` est le coeur Rust du projet **Klondike-AI**. Ce crate implémente la logique
complète du solitaire Klondike et expose une API Python grâce à PyO3. Il peut être
utilisé pour la recherche d'IA, la simulation ou comme moteur autonome très rapide.

## Présentation
- **Moteur du jeu** écrit en Rust pour garantir performances et sûreté.
- **Interopérabilité Python** via PyO3 afin de s'intégrer dans les outils d'apprentissage
  automatique existants.
- Objectif principal : offrir un environnement stable et reproductible pour les
  agents de renforcement ou la recherche algorithmique.

## Architecture

### Structures principales

| Structure     | Rôle                                                                          |
|---------------|-------------------------------------------------------------------------------|
| `Card`        | Représente une carte (rang, couleur, face visible). Fournit les opérations de
                  base : retournement, vérification de couleur, placement valide, etc. |
| `GameState`   | Stocke l'état complet d'une partie : talon, défausse, piles du tableau,
                  fondations et score. Propose des méthodes pour piocher, encoder l'observation
                  ou vérifier la victoire. |
| `Move`        | Enum décrivant toutes les actions possibles (piocher, déplacer entre piles,
                  vers une fondation, retourner une carte…). Possède une conversion vers/depuis
                  un indice entier pour l'action-space discret. |
| `Engine`      | Gère l'application d'un `Move` sur un `GameState`. Maintient la liste des
                  coups légaux courants et vérifie la validité des mouvements. |

Les modules internes sont organisés comme suit :

```
core/
├─ src/
│  ├─ card.rs       # Définition des cartes
│  ├─ state.rs      # Structure GameState et encodage observation
│  ├─ moves.rs      # Enumération des coups
│  └─ engine.rs     # Moteur haut niveau
└─ lib.rs           # Expose l'API PyO3
```

`Engine` s'appuie sur `GameState` pour mettre à jour l'état du jeu et recalculer
les actions autorisées après chaque mouvement. `Move` fournit une indexation stable
(0‑95) utilisée par l'environnement Python.

## API PyO3
L'extension compilée expose les fonctions suivantes :

| Fonction                | Description | Entrées / Sorties |
|-------------------------|-------------|------------------|
| `new_game()`            | Crée une nouvelle partie et renvoie son état sérialisé en JSON. | `()` -> `str` |
| `legal_moves(state)`    | Renvoie la liste des coups légaux à partir d'un état JSON. | `str` -> `Vec[str]` |
| `play_move(state, mv)`  | Applique un coup à l'état fourni, renvoie le nouvel état JSON et un booléen indiquant la validité. | `(str, str)` -> `(str, bool)` |
| `encode_observation(state)` | Encode l'état sous forme de vecteur `f32` (format utilisé par l'apprentissage). | `str` -> `Vec[f32]` |
| `foundation_count(state)` | Nombre total de cartes déjà placées dans les fondations. | `str` -> `usize` |
| `is_won(state)`         | Indique si la partie est gagnée. | `str` -> `bool` |
| `move_index(mv)`        | Convertit un coup (JSON) en indice entier [0‑95]. | `str` -> `usize` |
| `move_from_index(idx)`  | Opération inverse, renvoie le coup correspondant à l'indice ou `None`. | `usize` -> `Option[str]` |

Tous les états de jeu et coups sont échangés sous forme de chaînes JSON. Ceci
permet d'interfacer facilement le moteur avec d'autres langages ou de stocker des
replays.

## Compilation et utilisation

```bash
# Lancer les tests unitaires du moteur
cargo test -p klondike_core

# Compiler l'extension Python avec maturin (mode développement)
maturin develop --release
```

Exemple d'appel depuis Python :

```python
from klondike_core import new_game, legal_moves, play_move, move_index

state = new_game()
print("Coups disponibles:", [move_index(m) for m in legal_moves(state)])
next_state, valid = play_move(state, legal_moves(state)[0])
```

## Limitations et TODO
- Seuls les coups légaux sont vérifiés, l'implémentation détaillée de chaque
  mouvement reste partielle dans ce prototype.
- L'encodage d'observation est minimaliste et pourrait être enrichi pour de
  meilleures performances d'apprentissage.
- Intégration plus poussée avec `klondike_ai` (MCTS, réseaux de neurones) prévue
  dans les prochaines versions.

