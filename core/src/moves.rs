use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    // Piocher une carte du talon
    DrawCard,
    
    // Déplacer une carte de la défausse vers une pile du tableau
    WasteToTableau {
        destination: usize,
    },
    
    // Déplacer une carte de la défausse vers une fondation
    WasteToFoundation {
        foundation: usize,
    },
    
    // Déplacer des cartes entre piles du tableau
    TableauToTableau {
        from: usize,
        to: usize,
        count: usize,
    },
    
    // Déplacer une carte d'une pile du tableau vers une fondation
    TableauToFoundation {
        from: usize,
        foundation: usize,
    },
    
    // Retourner une carte face cachée dans le tableau
    FlipTableauCard {
        pile: usize,
    },
}

impl Move {
    pub fn get_move_index(&self) -> usize {
        match self {
            Move::DrawCard => 0,
            Move::WasteToTableau { destination } => 1 + destination,
            Move::WasteToFoundation { foundation } => 8 + foundation,
            Move::TableauToTableau { from, to, .. } => 12 + from * 7 + to,
            Move::TableauToFoundation { from, foundation } => 61 + from * 4 + foundation,
            Move::FlipTableauCard { pile } => 89 + pile,
        }
    }

    pub fn from_move_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Move::DrawCard),
            1..=7 => Some(Move::WasteToTableau {
                destination: index - 1,
            }),
            8..=11 => Some(Move::WasteToFoundation {
                foundation: index - 8,
            }),
            12..=60 => {
                let idx = index - 12;
                let from = idx / 7;
                let to = idx % 7;
                if from == to {
                    None
                } else {
                    Some(Move::TableauToTableau {
                        from,
                        to,
                        count: 1, // La valeur sera ajustée lors de la validation
                    })
                }
            },
            61..=88 => {
                let idx = index - 61;
                Some(Move::TableauToFoundation {
                    from: idx / 4,
                    foundation: idx % 4,
                })
            },
            89..=95 => Some(Move::FlipTableauCard {
                pile: index - 89,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_index_conversion() {
        let moves = vec![
            Move::DrawCard,
            Move::WasteToTableau { destination: 3 },
            Move::WasteToFoundation { foundation: 2 },
            Move::TableauToTableau { from: 1, to: 4, count: 1 },
            Move::TableauToFoundation { from: 2, foundation: 1 },
            Move::FlipTableauCard { pile: 5 },
        ];

        for original_move in moves {
            let index = original_move.get_move_index();
            let reconstructed = Move::from_move_index(index).unwrap();
            assert_eq!(original_move, reconstructed);
        }
    }

    #[test]
    fn test_invalid_move_indices() {
        assert!(Move::from_move_index(96).is_none());
        
        // Test des mouvements tableau vers tableau invalides (même pile)
        let same_pile_index = 12 + 2 * 7 + 2; // from=2, to=2
        assert!(Move::from_move_index(same_pile_index).is_none());
    }
}