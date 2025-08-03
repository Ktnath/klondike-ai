from klondike_core import new_game, solve_klondike, move_index

state = new_game()
solution = solve_klondike(state)

print(f"Nombre de coups résolus : {len(solution)}\n")

for i, (move, intention) in enumerate(solution[:30]):
    idx = move_index(move)
    print(f"{i+1:02d}. Move: {move:<10} → Index: {idx:<3} | Intention: {intention}")
