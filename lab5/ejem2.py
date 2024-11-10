import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Create a function to load the maze from a text file
def load_maze(file):
    maze = np.loadtxt(file, dtype=int)
    return maze

# Función para generar la población inicial sin movimientos que toquen una pared o retrocedan
def generate_population(size, genome_length, maze, start):
    population = []
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, Left, Up, Down

    for _ in range(size):
        x, y = start
        genome = []
        last_move = None  # Para evitar retroceder al último movimiento

        for _ in range(genome_length):
            valid_moves = []
            for i, (dx, dy) in enumerate(directions):
                if last_move is not None and (last_move == 0 and i == 1 or  # Right -> Left
                                              last_move == 1 and i == 0 or  # Left -> Right
                                              last_move == 2 and i == 3 or  # Up -> Down
                                              last_move == 3 and i == 2):  # Down -> Up
                    continue  # Evitar moverse en la dirección opuesta al último movimiento

                nx, ny = x + dx, y + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                    valid_moves.append(i)

            if not valid_moves:
                break  # Si no hay movimientos válidos, termina la creación del genoma

            move = random.choice(valid_moves)
            genome.append(move)
            last_move = move
            x, y = x + directions[move][0], y + directions[move][1]

        population.append(genome)
    return population

# Función de evaluación de la aptitud basada en la distancia a la salida
def reward(individual, maze, start, end):
    x, y = start

    for move in individual:
        if move == 0:  # Right
            y += 1
        elif move == 1:  # Left
            y -= 1
        elif move == 2:  # Up
            x -= 1
        elif move == 3:  # Down
            x += 1

        # Verifica los límites y si golpea una pared
        if not (0 <= x < maze.shape[0] and 0 <= y < maze.shape[1]) or maze[x, y] == 1:
            break  # Movimiento inválido

    # Calcular la distancia de Manhattan a la salida
    distance_to_goal = abs(end[0] - x) + abs(end[1] - y)
    # La aptitud es inversamente proporcional a la distancia
    fitness = max(0, 1000 - distance_to_goal)

    return fitness

# Función de selección
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)  # Selección aleatoria si todas las aptitudes son 0

    selection_probs = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=selection_probs)
    return population[selected_index]

# Función de cruce
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Función de mutación
def mutate(individual, mutation_rate, maze, start):
    x, y = start
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, Left, Up, Down

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            valid_moves = []
            for j, (dx, dy) in enumerate(directions):
                nx, ny = x + dx, y + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                    valid_moves.append(j)

            if valid_moves:
                individual[i] = random.choice(valid_moves)

        # Actualizar la posición actual
        move = individual[i]
        x, y = x + directions[move][0], y + directions[move][1]

    return individual

# Función de evolución
def evolve(population, maze, start, end, generations=100, mutation_rate=0.1):
    for generation in range(generations):
        fitnesses = [reward(individual, maze, start, end) for individual in population]

        # Seleccionar la nueva generación
        new_population = []
        for _ in range(len(population) // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1, mutation_rate, maze, start))
            new_population.append(mutate(offspring2, mutation_rate, maze, start))

        population = new_population

        # Imprime el mejor resultado de la generación actual
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        if best_fitness >= 1000:
            print("Goal reached!")
            break

    best_individual = population[fitnesses.index(max(fitnesses))]
    return best_individual

# Función para graficar la solución en el laberinto
def plot_solution(maze, path, start, end):
    x, y = start
    maze_copy = maze.copy()

    # Marcar el punto de inicio y fin
    maze_copy[start] = 3  # Marcar inicio
    maze_copy[end] = 4    # Marcar fin

    # Recorrer el camino y marcarlo en el laberinto
    for move in path:
        if move == 0:  # Right
            y += 1
        elif move == 1:  # Left
            y -= 1
        elif move == 2:  # Up
            x -= 1
        elif move == 3:  # Down
            x += 1

        # Verificar si el movimiento es válido
        if 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] == 0:
            maze_copy[x, y] = 2  # Marcar el camino en el laberinto
        else:
            break  # Detener si el movimiento es inválido

    # Definir el colormap
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])  # 0: camino, 1: pared, 2: recorrido, 3: inicio, 4: fin

    # Visualizar el laberinto con la solución
    plt.figure(figsize=(10, 10))
    plt.pcolor(maze_copy[::-1], cmap=cmap, edgecolors='k', linewidths=0.5)
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.title('Solución encontrada en el laberinto')
    plt.show()

# Cargar el laberinto desde un archivo de texto (asegúrate de que esté cargado correctamente)
maze = load_maze('maze.txt')
start = (1, 0)  # Coordenadas de inicio
end = (maze.shape[0] - 2, maze.shape[1] - 1)  # Coordenadas de fin

# Inicializar la población
population = generate_population(size=100, genome_length=50, maze=maze, start=start)

# Ejecutar la evolución y graficar la solución
best_path = evolve(population, maze, start, end)
plot_solution(maze, best_path, start, end)
