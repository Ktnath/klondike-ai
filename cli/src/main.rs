use colored::*;
use klondike_ai::{Engine, NeuralNet, MCTS};
use std::io::{self, Write};

fn main() -> io::Result<()> {
    println!("{}", "Klondike AI - Interface de jeu".green().bold());
    println!("{}", "Commands disponibles:".yellow());
    println!("  play - Jouer manuellement");
    println!("  ai   - Laisser l'IA jouer");
    println!("  quit - Quitter");

    let mut engine = Engine::new();
    let neural_net = NeuralNet::new(156, 96).unwrap();
    let mut mcts = MCTS::new(1.0, neural_net);

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim() {
            "play" => play_manual(&mut engine)?,
            "ai" => play_ai(&mut engine, &mut mcts)?,
            "quit" => break,
            _ => println!("{}", "Commande invalide".red()),
        }
    }

    Ok(())
}

fn play_manual(engine: &mut Engine) -> io::Result<()> {
    loop {
        display_game_state(engine);

        let moves = engine.get_available_moves();
        println!("{}", "Coups disponibles:".yellow());
        for (i, mov) in moves.iter().enumerate() {
            println!("  {}: {:?}", i, mov);
        }

        print!("Entrez le numéro du coup (ou 'back' pour revenir): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();
        if input == "back" {
            break;
        }

        if let Ok(index) = input.parse::<usize>() {
            if index < moves.len() {
                engine.make_move(&moves[index]);
                if engine.get_state().is_won() {
                    display_game_state(engine);
                    println!("{}", "Félicitations ! Vous avez gagné !".green().bold());
                    break;
                }
            } else {
                println!("{}", "Index invalide".red());
            }
        } else {
            println!("{}", "Entrée invalide".red());
        }
    }

    Ok(())
}

fn play_ai(engine: &mut Engine, mcts: &mut MCTS) -> io::Result<()> {
    println!("{}", "L'IA joue...".blue());

    while !engine.get_state().is_won() {
        display_game_state(engine);

        let probs = mcts.search(engine, 0.0, 10); // Température 0 pour le meilleur coup
        let best_move = probs.iter()
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .map(|(m, _)| *m)
            .unwrap();

        println!("L'IA joue: {:?}", best_move);
        engine.make_move(&best_move);

        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    if engine.get_state().is_won() {
        display_game_state(engine);
        println!("{}", "L'IA a gagné !".green().bold());
    }

    Ok(())
}

fn display_game_state(engine: &Engine) {
    println!("{}", "=== État du jeu ===".blue());
    // Implémenter l'affichage du jeu ici
    println!("Score: {}", engine.get_state().get_score());
}
