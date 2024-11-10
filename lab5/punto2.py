import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap


def create_maze(dim):
    # Create a grid filled with walls (1)
    # Set the seed to ensure reproducibility
    np.random.seed(2)
    maze = np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)

    # Define the starting point
    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0  # Mark start as open path

    # Initialize the stack with the starting point for DFS
    stack = [(x, y)]
    
    while stack:
        x, y = stack[-1]

        # Define possible directions (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)  # Randomize order for more organic mazes

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if next cell is within bounds and is a wall
            if 0 <= nx < dim and 0 <= ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                # Break wall to create path
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))  # Move to the next cell
                break
        else:
            stack.pop()  # Backtrack if no unvisited neighbors

    # Create entrance and exit points
    maze[1, 0] = 0  # Entrance
    maze[-2, -1] = 0  # Exit

    return maze

def display_maze(maze):
    cmap = ListedColormap(['white', 'black', 'green', 'black'])
    plt.figure(figsize=(6, 6))
    plt.pcolor(maze[::-1], cmap=cmap, edgecolors='k', linewidths=2)
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.title('Maze with Entrance and Exit')
    plt.show()

# Create a function to load the maze from a text file
def load_maze(file):
    maze = np.loadtxt(file, dtype=int)
    return maze



# Función para generar la población inicial
def generate_population(size, genome_length):
    population = []
    for _ in range(size):
        genome = [random.randint(0, 3) for _ in range(genome_length)]  # 0: Right, 1: Left, 2: Up, 3: Down
        population.append(genome)
    return population

# Función para evaluar la aptitud de un individuo
def reward(individual, maze, start, end):
    x, y = start
    visited = set()
    fitness = 0

    for move in individual:
        if move == 0:  # Right
            y += 1
        elif move == 1:  # Left
            y -= 1
        elif move == 2:  # Up
            x -= 1
        elif move == 3:  # Down
            x += 1

        # Verifica los límites del laberinto y si golpea una pared
        if x < 0 or x >= maze.shape[0] or y < 0 or y >= maze.shape[1] or maze[x, y] == 1:
            fitness -= 5  # Penalización por movimiento ilegal
            continue
        
        visited.add((x, y))

        # Recompensa por cada celda única visitada
        fitness += 1

        # Recompensa adicional por acercarse al objetivo
        if (x, y) == end:
            fitness += 100  # Gran recompensa por alcanzar el final
            break

    # Recompensa por proximidad al objetivo al final del recorrido
    distance_to_goal = abs(end[0] - x) + abs(end[1] - y)
    fitness += max(0, 50 - distance_to_goal)

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
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, 3)  # Cambia el gen aleatoriamente
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
            new_population.append(mutate(offspring1, mutation_rate))
            new_population.append(mutate(offspring2, mutation_rate))

        population = new_population

        # Imprime el mejor resultado de la generación actual
        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        if best_fitness >= 100:
            print("Goal reached!")
            break

    return best_individual


# Define the dimension of the maze (adjustable)
dim = 25  # This can be any positive integer
maze = create_maze(dim)
display_maze(maze)

# Save the maze to a text file
np.savetxt('maze.txt', maze, fmt='%d')

# Cargar el laberinto desde un archivo de texto
maze = load_maze('maze.txt')
#display_maze(maze)
start = (1, 0)  # Coordenadas de inicio
end = (maze.shape[0] - 2, maze.shape[1] - 1)  # Coordenadas de fin

# Inicializar la población y correr la evolución
population = generate_population(size=100, genome_length=50)
best_path = evolve(population, maze, start, end)

print("Best path found:", best_path)
