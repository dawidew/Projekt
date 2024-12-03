from deap import base, creator, tools, algorithms
import random

# Parametry algorytmu genetycznego
POPULATION_SIZE = 50
GENOTYPE_LENGTH = 5  # liczba cech strategii
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100

# Funkcja fitness
def fitness(strategy):
    zdobyte_pionki = random.randint(0, 12)
    przewaga_pozycyjna = random.uniform(0, 10)
    bezpieczne_pionki = random.randint(0, 12)
    kontrola_centrum = random.uniform(0, 5)
    wynik_gry = random.choice([0, 1])

    score = (
        strategy[0] * zdobyte_pionki +
        strategy[1] * przewaga_pozycyjna +
        strategy[2] * bezpieczne_pionki +
        strategy[3] * kontrola_centrum +
        strategy[4] * wynik_gry
    )
    return (score,)  # DEAP wymaga krotki

# Konfiguracja DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maksymalizacja fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=GENOTYPE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algorytm genetyczny
def genetic_algorithm():
    population = toolbox.population(n=POPULATION_SIZE)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(MAX_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

        # Zapis najlepszego osobnika
        best_ind = tools.selBest(population, k=1)[0]
        if best_ind.fitness.values[0] > best_fitness:
            best_fitness = best_ind.fitness.values[0]
            best_individual = best_ind

        print(f"Generacja {generation}: Najlepszy fitness: {best_fitness}")

    print(f"\nNajlepsza strategia: {best_individual}, fitness: {best_fitness}")
    return best_individual

# Inicjalizacja planszy
def initialize_board():
    board = [
        ['.', 'R', '.', 'R', '.', 'R', '.', 'R'],
        ['R', '.', 'R', '.', 'R', '.', 'R', '.'],
        ['.', 'R', '.', 'R', '.', 'R', '.', 'R'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', '.', 'P', '.', 'P', '.', 'P', '.'],
        ['.', 'P', '.', 'P', '.', 'P', '.', 'P'],
        ['P', '.', 'P', '.', 'P', '.', 'P', '.']
    ]
    return board

def print_board(board):
    print("  0 1 2 3 4 5 6 7")  # Kolumny
    for i, row in enumerate(board):
        print(f"{i} " + ' '.join(row))  # Wiersze
    print()

def get_possible_moves(board, player, only_captures=False):
    """Zwraca możliwe ruchy lub ruchy bicia dla pionków i króli gracza."""
    moves = []
    opponent = 'R' if player == 'P' else 'P'
    kings = {'P': 'D', 'R': 'T'}
    directions = [(-1, -1), (-1, 1)] if player == 'P' else [(1, -1), (1, 1)]

    for row in range(8):
        for col in range(8):
            if board[row][col] in {player, kings[player]}:
                for dr, dc in directions:
                    # Zwykły ruch
                    new_row, new_col = row + dr, col + dc
                    if (
                        0 <= new_row < 8 and 0 <= new_col < 8
                        and board[new_row][new_col] == '.'
                        and not only_captures
                    ):
                        moves.append(((row, col), (new_row, new_col)))

                    # Ruch bicia
                    capture_row, capture_col = row + 2 * dr, col + 2 * dc
                    if (
                        0 <= capture_row < 8 and 0 <= capture_col < 8
                        and board[row + dr][col + dc] in {opponent, kings[opponent]}
                        and board[capture_row][capture_col] == '.'
                    ):
                        moves.append(((row, col), (capture_row, capture_col)))

    return moves

def make_move(board, move):
    (start_row, start_col), (end_row, end_col) = move
    piece = board[start_row][start_col]
    board[end_row][end_col] = piece
    board[start_row][start_col] = '.'

    # Sprawdzenie bicia
    if abs(start_row - end_row) == 2:
        middle_row, middle_col = (start_row + end_row) // 2, (start_col + end_col) // 2
        board[middle_row][middle_col] = '.'  

    # Awans na króla
    if piece == 'P' and end_row == 0:
        board[end_row][end_col] = 'D'
    elif piece == 'R' and end_row == 7:
        board[end_row][end_col] = 'T'

def robot_move(board, strategy):
    """Ruch robota z uwzględnieniem strategii."""
    mandatory_captures = get_possible_moves(board, 'R', only_captures=True)
    if mandatory_captures:
        return random.choice(mandatory_captures)

    moves = get_possible_moves(board, 'R')
    if not moves:
        return None

    # Wybór najlepszego ruchu na podstawie strategii
    best_move = max(moves, key=lambda move: evaluate_move(board, move, strategy))
    return best_move

def evaluate_move(board, move, strategy):
    """Ocena ruchu dla robota na podstawie strategii."""
    simulated_board = [row[:] for row in board]
    make_move(simulated_board, move)

    zdobyte_pionki = sum(row.count('P') + row.count('D') for row in simulated_board)
    przewaga_pozycyjna = sum(row.count('R') * 0.5 + row.count('T') for row in simulated_board)
    bezpieczne_pionki = sum(1 for r in range(8) for c in range(8)
                            if simulated_board[r][c] in ['R', 'T'] and (r == 7 or r == 6))
    kontrola_centrum = sum(1 for r in range(2, 6) for c in range(2, 6)
                           if simulated_board[r][c] in ['R', 'T'])
    wynik_gry = len(get_possible_moves(simulated_board, 'R')) - len(get_possible_moves(simulated_board, 'P'))

    return (
        strategy[0] * zdobyte_pionki +
        strategy[1] * przewaga_pozycyjna +
        strategy[2] * bezpieczne_pionki +
        strategy[3] * kontrola_centrum +
        strategy[4] * wynik_gry
    )

def play_checkers_with_penalty(best_strategy):
    board = initialize_board()
    print("Gra w warcaby z zasadami obowiązkowego bicia!")
    print_board(board)

    while True:
        # Ruch gracza
        mandatory_captures = get_possible_moves(board, 'P', only_captures=True)
        possible_moves = get_possible_moves(board, 'P')
        if not possible_moves:
            print("Przegrałeś! Brak ruchów.")
            break

        print("Twoje możliwe ruchy:")
        for i, move in enumerate(possible_moves):
            print(f"{i + 1}. {move}")

        player_move = int(input("Wybierz numer ruchu: ")) - 1
        if mandatory_captures and possible_moves[player_move] not in mandatory_captures:
            print("Musisz wykonać bicie!")
            continue

        make_move(board, possible_moves[player_move])
        print_board(board)

        # Ruch robota
        robot_next_move = robot_move(board, best_strategy)
        if not robot_next_move:
            print("Robot przegrał! Wygrałeś!")
            break

        make_move(board, robot_next_move)
        print("Ruch robota:")
        print_board(board)

if __name__ == "__main__":
    best_strategy = genetic_algorithm()
    play_checkers_with_penalty(best_strategy)
