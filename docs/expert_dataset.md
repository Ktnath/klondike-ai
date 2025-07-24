# Expert Dataset

Ce document décrit le contenu du fichier `expert_dataset.npz` généré par `generate_expert_dataset.py`. Ce jeu de données rassemble des transitions optimales produites par le solveur du moteur Rust.

## Structure des fichiers .npz

Chaque fichier `.npz` regroupe plusieurs tableaux NumPy :

- `observations` : `np.ndarray` de forme `(N, 156)`, vecteurs normalisés décrivant l'état de jeu.
- `actions` : `np.ndarray` de forme `(N,)`, entiers compris entre `0` et `95`.
- `rewards` : `np.ndarray` de forme `(N,)`, valeurs flottantes.
- `dones` : `np.ndarray` de forme `(N,)`, booléens indiquant la fin de partie.
- `intentions` : `np.ndarray` de forme `(N,)`, chaînes UTF‑8 représentant l'intention.

## Contenu du fichier

`expert_dataset.npz` est un fichier NumPy compressé (`np.savez_compressed`) contenant cinq tableaux de même longueur :

| Clé | Forme | Type | Description |
| --- | --- | --- | --- |
| `observations` | `(N, 156)` | `float32` | Encodage de l'état de jeu avant chaque coup. Les 156 features sont organisées en trois segments de 52 valeurs représentant respectivement les cartes visibles du tableau, les fondations et les cartes hors tableau (stock, waste, face cachée). Une valeur vaut 1 si la carte est présente dans la zone, 0 sinon. Cet encodage correspond à la fonction `encode_observation` du moteur Rust. |
| `actions` | `(N,)` | `int64` | Indice de l'action jouée, obtenu via `move_index` dans `klondike_core`. Le mapping précis dépend de l'implémentation du moteur. |
| `rewards` | `(N,)` | `float32` | Récompense immédiate après le coup, calculée par `compute_base_reward_json`. Elle correspond au nombre de cartes placées dans les fondations, normalisé par 52. |
| `dones` | `(N,)` | `bool` | Indicateur de victoire de l'état résultant. |
| `intentions` | `(N,)` | `object` (str) | Intention associée au coup, déduite par la fonction `infer_intention`. Les valeurs possibles sont : `"Révéler une carte cachée"`, `"Monter à la fondation"`, `"Déplacer un roi sur colonne vide"` et `"Ranger carte sur une autre"`. |

Toutes les transitions des différentes parties sont concaténées les unes à la suite des autres ; il n'y a pas de séparation par épisode.

## Exemple d'utilisation

```python
import numpy as np

# Chargement du dataset
with np.load('data/expert_dataset.npz') as data:
    observations = data['observations']        # (N, 156)
    actions = data['actions']                  # (N,)
    rewards = data['rewards']                  # (N,)
    dones = data['dones']                      # (N,)
    intentions = data['intentions']            # (N,)

print('Première observation:', observations[0])
print('Intention associée:', intentions[0])
```

## Alignement avec le moteur Rust

L'encodage des observations s'appuie directement sur la fonction `encode_observation` définie dans `core/src/lib.rs`. Celle‑ci produit un vecteur de longueur 156 en combinant trois segments de cartes :

```rust
// Trois segments one-hot de 52 cartes chacun
let mut tableau_vec = vec![0.0f32; 52];
let mut foundation_vec = vec![0.0f32; 52];
let mut other_vec = vec![0.0f32; 52];
...
obs.extend(tableau_vec);
obs.extend(foundation_vec);
obs.extend(other_vec);
```

Les intentions enregistrées proviennent de la fonction `infer_intention` (fichier `core/src/intentions.rs`) :

```rust
if matches!(mv, Reveal(_)) {
    return "Révéler une carte cachée".to_string();
}
if after.get_stack().len() > before.get_stack().len() {
    return "Monter à la fondation".to_string();
}
match mv {
    DeckPile(c) | StackPile(c) => {
        let before_empty = count_empty_piles(before);
        let after_empty = count_empty_piles(after);
        if c.rank() == KING_RANK && after_empty < before_empty {
            "Déplacer un roi sur colonne vide".to_string()
        } else {
            "Ranger carte sur une autre".to_string()
        }
    }
    DeckStack(_) | PileStack(_) => "Monter à la fondation".to_string(),
    Reveal(_) => unreachable!(),
}
```

Ces chaînes textuelles sont stockées directement dans le tableau `intentions` du dataset. Elles peuvent servir pour des approches d'apprentissage supervisé ou pour l'analyse stratégique des solutions générées.

## Spécificité

- L'encodage des intentions peut être transformé en vecteurs via la clé `intention_embedding` du fichier `config.yaml`.
- Les indices d'action sont produits par les fonctions `move_to_index()` et `index_to_move()` exposées dans le moteur.

### Exemple minimal

```python
import numpy as np

data = np.load("data/expert_dataset.npz")
print(data["intentions"][:5])
```
