use crate::{card::KING_RANK, moves::Move, state::Solitaire};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabeledMove {
    pub mv: Move,
    pub intention: String,
}

fn count_empty_piles(st: &Solitaire) -> usize {
    st.compute_visible_piles()
        .iter()
        .filter(|p| p.is_empty())
        .count()
}

/// Infer the intention of a move by comparing the before and after states.
#[must_use]
pub fn infer_intention(before: &Solitaire, mv: &Move, after: &Solitaire) -> String {
    use Move::*;

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
}
