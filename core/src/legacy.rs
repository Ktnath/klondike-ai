// LEGACY: Unused or CLI-only modules preserved for reference

pub mod dependencies {
    // LEGACY
    use core::mem::swap;

    use crate::{
        card::{Card, N_CARDS, N_SUITS},
        deck::N_PILES,
        moves::{Move, MoveMask},
        state::{ExtraInfo, Solitaire},
    };

    extern crate alloc;
    use alloc::vec::Vec;
    use arrayvec::ArrayVec;

    pub struct DependencyEngine {
        state: Solitaire,
        cards_from: [usize; N_CARDS as usize],
        cards_to: [usize; N_CARDS as usize],
        has_upper: [bool; N_CARDS as usize],
        emptying: ArrayVec<usize, { N_PILES as usize }>,
        last_draw: usize,
        n_moves: usize,
        dep: Vec<(usize, usize)>,
    }

    impl From<Solitaire> for DependencyEngine {
        fn from(value: Solitaire) -> Self {
            Self::new(value)
        }
    }

    impl DependencyEngine {
        #[must_use]
        pub fn new(state: Solitaire) -> Self {
            let mut cards = [usize::MAX; N_CARDS as usize];

            let vis = state.compute_visible_piles();
            let mut emptying = ArrayVec::<usize, { N_PILES as usize }>::new();

            let mut has_upper = [false; N_CARDS as usize];

            for pile in vis {
                if pile.is_empty() {
                    emptying.push(0);
                }

                let mut prev = false;
                for card in pile {
                    has_upper[card.mask_index() as usize] = prev;
                    cards[card.mask_index() as usize] = 0;
                    prev = true;
                }
            }
            for suit in 0..N_SUITS {
                let rank = state.get_stack().get(suit);
                if rank > 0 {
                    cards[Card::new(rank - 1, suit).mask_index() as usize] = 0;
                }
            }

            Self {
                state,
                cards_from: cards,
                cards_to: cards,
                has_upper,
                emptying,
                last_draw: 0,
                n_moves: 0,
                dep: Vec::default(),
            }
        }

        #[must_use]
        pub const fn state(&self) -> &Solitaire {
            &self.state
        }

        #[must_use]
        pub fn into_state(self) -> Solitaire {
            self.state
        }

        #[must_use]
        pub fn is_valid(&self, m: Move) -> bool {
            let moves = self.state.gen_moves::<false>();
            MoveMask::from(m).filter(&moves).is_empty()
        }

        pub fn add_dep(&mut self, from: usize) {
            self.dep.push((from, self.n_moves));
        }

        pub fn get_move_lock(&mut self, card: Card) -> usize {
            if self.has_upper[card.mask_index() as usize] {
                let mut upper = card.increase_rank_swap_color();
                let mut other_upper = upper.swap_suit();
                let mut m_upper = self.cards_to[upper.mask_index() as usize];
                let mut m_other_upper = self.cards_to[other_upper.mask_index() as usize];

                if m_upper < m_other_upper {
                    swap(&mut upper, &mut other_upper);
                    swap(&mut m_upper, &mut m_other_upper);
                }

                self.cards_to[upper.mask_index() as usize] = self.n_moves;
            }

            let val = self.cards_from[card.mask_index() as usize];
            if val == usize::MAX {
                let other = card.swap_suit();

                self.cards_from
                    .swap(card.mask_index() as usize, other.mask_index() as usize);
            }

            self.cards_from[card.mask_index() as usize]
        }

        pub fn get_move_lock_to(&mut self, card: Card) -> usize {
            if card.is_king() {
                self.emptying.pop_at(0).unwrap()
            } else {
                let mut upper = card.increase_rank_swap_color();
                let mut other_upper = upper.swap_suit();
                let mut m_upper = self.cards_to[upper.mask_index() as usize];
                let mut m_other_upper = self.cards_to[other_upper.mask_index() as usize];

                if m_upper > m_other_upper {
                    swap(&mut upper, &mut other_upper);
                    swap(&mut m_upper, &mut m_other_upper);
                }

                self.cards_to[upper.mask_index() as usize] = usize::MAX;
                self.has_upper[card.mask_index() as usize] = true;

                m_upper
            }
        }

        pub fn do_move(&mut self, m: Move) -> bool {
            if !self.is_valid(m) {
                return false;
            }

            self.n_moves += 1;

            let (_, (_, extra)) = self.state.do_move(m);

            match extra {
                ExtraInfo::Card(new) => {
                    self.cards_from[new.mask_index() as usize] = self.n_moves;
                    self.cards_to[new.mask_index() as usize] = self.n_moves;
                }
                ExtraInfo::RevealEmpty => {
                    self.emptying.push(self.n_moves);
                }
                ExtraInfo::None => {}
            }

            match m {
                Move::DeckStack(card) => {
                    self.add_dep(self.last_draw);
                    self.last_draw = self.n_moves;

                    if card.rank() > 0 {
                        let other = Card::new(card.rank() - 1, card.suit());
                        self.add_dep(self.cards_to[other.mask_index() as usize]);
                        self.cards_to[other.mask_index() as usize] = usize::MAX;
                        self.cards_from[other.mask_index() as usize] = usize::MAX;
                    }
                    self.cards_to[card.mask_index() as usize] = self.n_moves;
                    self.cards_from[card.mask_index() as usize] = self.n_moves;
                }
                Move::PileStack(card) => {
                    let from = self.get_move_lock(card);
                    self.add_dep(from);

                    if card.rank() > 0 {
                        let other = Card::new(card.rank() - 1, card.suit());
                        self.add_dep(self.cards_to[other.mask_index() as usize]);
                        self.cards_to[other.mask_index() as usize] = usize::MAX;
                        self.cards_from[other.mask_index() as usize] = usize::MAX;
                    }
                    self.cards_to[card.mask_index() as usize] = self.n_moves;
                    self.cards_from[card.mask_index() as usize] = self.n_moves;
                }
                Move::DeckPile(card) => {
                    self.add_dep(self.last_draw);
                    self.last_draw = self.n_moves;

                    let from = self.get_move_lock_to(card);
                    self.add_dep(from);

                    self.cards_to[card.mask_index() as usize] = self.n_moves;
                    self.cards_from[card.mask_index() as usize] = self.n_moves;
                }
                Move::StackPile(card) => {
                    if card.rank() > 0 {
                        let lower = Card::new(card.rank() - 1, card.suit());
                        self.cards_from[lower.mask_index() as usize] = self.n_moves;
                        self.cards_to[lower.mask_index() as usize] = self.n_moves;
                    }

                    // from stack
                    self.add_dep(self.cards_from[card.mask_index() as usize]);
                    let from = self.get_move_lock_to(card);
                    // has to have place to put
                    self.add_dep(from);

                    self.cards_from[card.mask_index() as usize] = self.n_moves;
                    self.cards_to[card.mask_index() as usize] = self.n_moves;
                }
                Move::Reveal(card) => {
                    let from = self.get_move_lock_to(card);
                    self.add_dep(from);

                    let from = self.get_move_lock(card);

                    self.add_dep(from);
                    // self.cards_from[card.mask_index() as usize] = self.n_moves;
                    // self.cards_from[card.mask_index() as usize] = self.n_moves;
                }
            }
            true
        }

        #[must_use]
        pub fn get(&self) -> &Vec<(usize, usize)> {
            &self.dep
        }
    }
}

pub mod graph {
    // LEGACY
    use crate::{
        card::Card,
        moves::{Move, MoveMask},
        pruning::FullPruner,
        state::{Encode, Solitaire},
        tracking::{DefaultTerminateSignal, EmptySearchStats, SearchStatistics, TerminateSignal},
        traverse::{traverse, Callback, Control, TpTable},
    };

    extern crate alloc;
    use alloc::vec::Vec;

    #[derive(Clone, Copy, Debug)]
    pub enum EdgeType {
        DeckPile,
        DeckStack,
        PileStack,
        PileStackReveal,
        StackPile,
        Reveal,
    }

    pub type Edge = (Encode, Encode, EdgeType);
    pub type Graph = Vec<Edge>;

    struct BuilderCallback<'a, S: SearchStatistics, T: TerminateSignal> {
        graph: Graph,
        stats: &'a S,
        sign: &'a T,
        depth: usize,
        prev_enc: Encode,
        last_move: Move,
        rev_move: Option<Move>,
    }

    const fn get_edge_type(m: Move, rm: Option<Move>) -> EdgeType {
        match m {
            Move::DeckStack(_) => EdgeType::DeckStack,
            Move::PileStack(_) if rm.is_some() => EdgeType::PileStack,
            Move::PileStack(_) => EdgeType::PileStackReveal,
            Move::DeckPile(_) => EdgeType::DeckPile,
            Move::StackPile(_) => EdgeType::StackPile,
            Move::Reveal(_) => EdgeType::Reveal,
        }
    }

    impl<'a, S: SearchStatistics, T: TerminateSignal> BuilderCallback<'a, S, T> {
        fn new(g: &Solitaire, stats: &'a S, sign: &'a T) -> Self {
            Self {
                graph: Graph::new(),
                stats,
                sign,
                depth: 0,
                prev_enc: g.encode(),
                last_move: Move::DeckPile(Card::DEFAULT),
                rev_move: None,
            }
        }
    }

    impl<S: SearchStatistics, T: TerminateSignal> Callback for BuilderCallback<'_, S, T> {
        type Pruner = FullPruner;

        fn on_win(&mut self, _: &Solitaire) -> Control {
            // win state
            self.graph.push((
                self.prev_enc,
                !0,
                get_edge_type(self.last_move, self.rev_move),
            ));
            Control::Ok
        }

        fn on_visit(&mut self, _: &Solitaire, e: Encode) -> Control {
            if self.sign.is_terminated() {
                return Control::Halt;
            }

            self.stats.hit_a_state(self.depth);
            self.graph.push((
                self.prev_enc,
                e,
                get_edge_type(self.last_move, self.rev_move),
            ));

            Control::Ok
        }

        fn on_move_gen(&mut self, m: &MoveMask, _: Encode) -> Control {
            self.stats.hit_unique_state(self.depth, m.len());
            Control::Ok
        }

        fn on_do_move(&mut self, _: &Solitaire, m: Move, e: Encode, prune: &FullPruner) -> Control {
            self.last_move = m;
            self.rev_move = prune.rev_move();
            self.prev_enc = e;
            self.depth += 1;
            Control::Ok
        }

        fn on_undo_move(&mut self, _: Move, _: Encode, _: &Control) {
            self.depth -= 1;
            self.stats.finish_move(self.depth);
        }
    }

    pub fn graph_with_tracking<S: SearchStatistics, T: TerminateSignal>(
        g: &mut Solitaire,
        stats: &S,
        sign: &T,
    ) -> (Control, Graph) {
        let mut tp = TpTable::default();
        let mut callback = BuilderCallback::new(g, stats, sign);

        let finished = traverse(g, FullPruner::default(), &mut tp, &mut callback);
        (finished, callback.graph)
    }

    pub fn graph(g: &mut Solitaire) -> (Control, Graph) {
        graph_with_tracking(g, &EmptySearchStats {}, &DefaultTerminateSignal {})
    }
}

pub mod hop_solver {
    // LEGACY
    use core::ops::{Add, AddAssign};

    use rand::RngCore;

    use crate::{
        legacy::solver::SearchResult,
        moves::Move,
        pruning::{FullPruner, Pruner},
        state::{Encode, Solitaire},
        tracking::TerminateSignal,
        traverse::{traverse, Callback, Control, TpTable},
    };

    struct HOPSolverCallback<'a, T: TerminateSignal> {
        sign: &'a T,
        result: SearchResult,
        limit: usize,
        n_visit: usize,
    }

    impl<T: TerminateSignal> Callback for HOPSolverCallback<'_, T> {
        type Pruner = FullPruner;

        fn on_win(&mut self, _: &Solitaire) -> Control {
            self.result = SearchResult::Solved;
            Control::Halt
        }

        fn on_visit(&mut self, g: &Solitaire, _: Encode) -> Control {
            if g.is_sure_win() {
                self.result = SearchResult::Solved;
                return Control::Halt;
            }

            if self.sign.is_terminated() {
                self.result = SearchResult::Terminated;
                return Control::Halt;
            }

            self.n_visit += 1;
            if self.n_visit > self.limit {
                self.result = SearchResult::Terminated;
                Control::Halt
            } else {
                Control::Ok
            }
        }
    }

    #[derive(Default, Clone, Copy)]
    pub struct HopResult {
        pub wins: usize,
        pub skips: usize,
        pub played: usize,
    }

    const SURE_WIN: HopResult = HopResult {
        wins: !0,
        skips: 0,
        played: !0,
    };

    const SURE_LOSE: HopResult = HopResult {
        wins: 0,
        skips: !0,
        played: !0,
    };

    const SKIPPED: HopResult = HopResult {
        wins: 0,
        skips: 1,
        played: 1,
    };

    impl Add for HopResult {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self {
                wins: self.wins + rhs.wins,
                skips: self.skips + rhs.skips,
                played: self.played + rhs.played,
            }
        }
    }

    impl AddAssign for HopResult {
        fn add_assign(&mut self, rhs: Self) {
            self.wins += rhs.wins;
            self.skips += rhs.skips;
            self.played += rhs.played;
        }
    }

    pub fn hop_solve_game<R: RngCore, T: TerminateSignal>(
        g: &Solitaire,
        m: Move,
        rng: &mut R,
        n_times: usize,
        limit: usize,
        sign: &T,
        prune_info: &FullPruner,
    ) -> HopResult {
        let mut total_wins = 0;
        let mut total_skips = 0;
        let mut total_played = 0;

        let mut tp = TpTable::default();

        // check if determinize
        let total_hidden: u8 = g.get_hidden().total_down_cards();
        if total_hidden <= 1 {
            // totally determinized
            let res = crate::legacy::solver::solve(&mut g.clone()).0;
            return if res == SearchResult::Solved {
                SURE_WIN
            } else if res == SearchResult::Unsolvable {
                SURE_LOSE
            } else {
                SKIPPED
            };
        }

        for _ in 0..n_times {
            let mut gg = g.clone();
            gg.hidden_shuffle(rng);
            let (rev_m, (_, extra)) = gg.do_move(m);
            let new_prune_info = FullPruner::update(prune_info, m, rev_m, extra);

            let mut callback = HOPSolverCallback {
                sign,
                result: SearchResult::Unsolvable,
                limit,
                n_visit: 0,
            };
            tp.clear();
            traverse(&mut gg, new_prune_info, &mut tp, &mut callback);
            if sign.is_terminated() {
                break;
            }
            total_played += 1;
            let result = callback.result;
            match result {
                SearchResult::Solved => total_wins += 1,
                SearchResult::Terminated => total_skips += 1,
                _ => {}
            }
        }
        HopResult {
            wins: total_wins,
            skips: total_skips,
            played: total_played,
        }
    }

    extern crate alloc;
    use alloc::vec::Vec;

    struct RevStatesCallback<'a, R: RngCore, T: TerminateSignal> {
        his: Vec<Move>,
        rng: &'a mut R,
        n_times: usize,
        limit: usize,
        sign: &'a T,
        res: Vec<(Vec<Move>, HopResult)>,
    }

    impl<R: RngCore, T: TerminateSignal> Callback for RevStatesCallback<'_, R, T> {
        type Pruner = FullPruner;

        fn on_win(&mut self, _: &Solitaire) -> Control {
            self.res.push((self.his.clone(), SURE_WIN));
            Control::Halt
        }

        fn on_do_move(
            &mut self,
            g: &Solitaire,
            m: Move,
            _: Encode,
            prune_info: &FullPruner,
        ) -> Control {
            self.his.push(m);
            let rev = prune_info.rev_move();
            // if rev.is_none() && (matches!(m, Move::Reveal(_)) || matches!(m, Move::PileStack(_))) {
            if rev.is_none() {
                self.res.push((
                    self.his.clone(),
                    hop_solve_game(
                        g,
                        m,
                        self.rng,
                        self.n_times,
                        self.limit,
                        self.sign,
                        prune_info,
                    ),
                ));
                Control::Skip
            } else {
                Control::Ok
            }
        }

        fn on_undo_move(&mut self, _: Move, _: Encode, _: &Control) {
            self.his.pop();
        }
    }

    pub fn list_moves<R: RngCore, T: TerminateSignal>(
        g: &mut Solitaire,
        rng: &mut R,
        n_times: usize,
        limit: usize,
        sign: &T,
    ) -> Vec<(Vec<Move>, HopResult)> {
        let mut callback = RevStatesCallback {
            his: Vec::default(),
            rng,
            n_times,
            limit,
            sign,
            res: Vec::default(),
        };

        let mut tp = TpTable::default();
        traverse(g, FullPruner::default(), &mut tp, &mut callback);
        callback.res
    }
}

pub mod solver {
    // LEGACY
    use crate::{
        moves::Move,
        pruning::FullPruner,
        state::{Encode, Solitaire},
        tracking::{DefaultTerminateSignal, EmptySearchStats, SearchStatistics, TerminateSignal},
        traverse::{traverse, Callback, Control, TpTable},
    };
    use arrayvec::ArrayVec;

    // before every progress you'd do at most 2*N_RANKS move
    // and there would only be N_FULL_DECK + N_HIDDEN progress step
    const N_PLY_MAX: usize = 1024;

    pub type HistoryVec = ArrayVec<Move, N_PLY_MAX>;

    #[derive(Debug, PartialEq, Eq)]
    pub enum SearchResult {
        Terminated,
        Solved,
        Unsolvable,
        Crashed,
    }

    use crate::intentions::{infer_intention, LabeledMove};

    struct SolverCallback<'a, S: SearchStatistics, T: TerminateSignal> {
        history: HistoryVec,
        annotated_moves: Vec<LabeledMove>,
        stats: &'a S,
        sign: &'a T,
        result: SearchResult,
    }

    impl<S: SearchStatistics, T: TerminateSignal> Callback for SolverCallback<'_, S, T> {
        type Pruner = FullPruner;
        fn on_win(&mut self, _: &Solitaire) -> Control {
            self.result = SearchResult::Solved;
            Control::Halt
        }

        fn on_visit(&mut self, _: &Solitaire, _: Encode) -> Control {
            if self.sign.is_terminated() {
                self.result = SearchResult::Terminated;
                return Control::Halt;
            }

            self.stats.hit_a_state(self.history.len());
            Control::Ok
        }

        fn on_move_gen(&mut self, m: &crate::moves::MoveMask, _: Encode) -> Control {
            self.stats.hit_unique_state(self.history.len(), m.len());
            Control::Ok
        }

        fn on_do_move(&mut self, g: &Solitaire, m: Move, _: Encode, _: &FullPruner) -> Control {
            let before = g.clone();
            let mut after = before.clone();
            after.do_move(m);
            let (intention, high) = infer_intention(&before, &m, &after);
            self.annotated_moves.push(LabeledMove { mv: m, intention, high_level: high });
            self.history.push(m);
            Control::Ok
        }

        fn on_undo_move(&mut self, _: Move, _: Encode, res: &Control) {
            if *res == Control::Ok {
                self.history.pop();
            }
            self.stats.finish_move(self.history.len());
        }
    }

    pub fn solve_with_tracking<S: SearchStatistics, T: TerminateSignal>(
        game: &mut Solitaire,
        stats: &S,
        sign: &T,
    ) -> (SearchResult, Option<(HistoryVec, Vec<LabeledMove>)>) {
        let mut tp = TpTable::default();

        let mut callback = SolverCallback {
            history: HistoryVec::new(),
            annotated_moves: Vec::new(),
            stats,
            sign,
            result: SearchResult::Unsolvable,
        };

        traverse(game, FullPruner::default(), &mut tp, &mut callback);

        let result = callback.result;

        if result == SearchResult::Solved {
            (result, Some((callback.history, callback.annotated_moves)))
        } else {
            (result, None)
        }
    }

    pub fn solve(game: &mut Solitaire) -> (SearchResult, Option<(HistoryVec, Vec<LabeledMove>)>) {
        solve_with_tracking(game, &EmptySearchStats {}, &DefaultTerminateSignal {})
    }
}
