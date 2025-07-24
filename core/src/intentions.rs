use crate::{card::KING_RANK, moves::Move, state::Solitaire};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighLevelIntention {
    /// Reveal a hidden card
    Reveal,
    /// Move a card on tableau or waste
    StackPlay,
    /// Move a card to foundation
    FoundationPlay,
    /// Manage empty tableau columns with kings
    EmptyColumnManagement,
}

impl std::fmt::Display for HighLevelIntention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            HighLevelIntention::Reveal => "REVEAL",
            HighLevelIntention::StackPlay => "STACK_PLAY",
            HighLevelIntention::FoundationPlay => "FOUNDATION_PLAY",
            HighLevelIntention::EmptyColumnManagement => "EMPTY_COLUMN_MANAGEMENT",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabeledMove {
    pub mv: Move,
    pub intention: String,
    pub high_level: HighLevelIntention,
}

fn count_empty_piles(st: &Solitaire) -> usize {
    st.compute_visible_piles()
        .iter()
        .filter(|p| p.is_empty())
        .count()
}

/// Infer the intention of a move by comparing the before and after states.
#[must_use]
pub fn infer_intention(
    before: &Solitaire,
    mv: &Move,
    after: &Solitaire,
) -> (String, HighLevelIntention) {
    use Move::*;

    if matches!(mv, Reveal(_)) {
        let fine = "Révéler une carte cachée".to_string();
        return (fine, HighLevelIntention::Reveal);
    }

    if after.get_stack().len() > before.get_stack().len() {
        let fine = "Monter à la fondation".to_string();
        return (fine, HighLevelIntention::FoundationPlay);
    }

    match mv {
        DeckPile(c) | StackPile(c) => {
            let before_empty = count_empty_piles(before);
            let after_empty = count_empty_piles(after);
            if c.rank() == KING_RANK && after_empty < before_empty {
                (
                    "Déplacer un roi sur colonne vide".to_string(),
                    HighLevelIntention::EmptyColumnManagement,
                )
            } else {
                (
                    "Ranger carte sur une autre".to_string(),
                    HighLevelIntention::StackPlay,
                )
            }
        }
        DeckStack(_) | PileStack(_) => (
            "Monter à la fondation".to_string(),
            HighLevelIntention::FoundationPlay,
        ),
        Reveal(_) => unreachable!(),
    }
}

/// Map a labeled move to its high level intention category.
#[must_use]
pub fn map_to_high_level_intention(lm: &LabeledMove) -> HighLevelIntention {
    match lm.intention.as_str() {
        "Révéler une carte cachée" => HighLevelIntention::Reveal,
        "Monter à la fondation" => HighLevelIntention::FoundationPlay,
        "Déplacer un roi sur colonne vide" => HighLevelIntention::EmptyColumnManagement,
        "Ranger carte sur une autre" => HighLevelIntention::StackPlay,
        _ => HighLevelIntention::StackPlay,
    }
}
