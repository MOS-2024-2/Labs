import random
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo
TAMANIO_POBLACION = 10
PORCENTAJE_SUPERVIVIENTES = 0.2
PROBABILIDAD_MUTACION = 0.01
NUMERO_GENERACIONES = 100
NIVEL_GRIS_FONDO = 128  # Puedes cambiar este valor para pruebas

# Función de aptitud
def calcular_aptitud(individuo, gris_fondo):
    return -abs(individuo - gris_fondo)

# Generar la población inicial
def inicializar_poblacion(tamanio):
    return [random.randint(0, 255) for _ in range(tamanio)]

# Selección de los mejores individuos (elitista)
def seleccion_elitista(poblacion, gris_fondo):
    evaluaciones = [(ind, calcular_aptitud(ind, gris_fondo)) for ind in poblacion]
    evaluaciones.sort(key=lambda x: x[1], reverse=True)
    num_supervivientes = int(len(poblacion) * PORCENTAJE_SUPERVIVIENTES)
    return [ind[0] for ind in evaluaciones[:num_supervivientes]]

# Cruce uniforme
def cruce(padre1, padre2):
    return (padre1 + padre2) // 2

# Aplicar mutación
def mutacion(individuo, prob_mutacion):
    if random.random() < prob_mutacion:
        return max(0, min(255, individuo + random.randint(-5, 5)))
    return individuo

# Crear una nueva generación
def nueva_generacion(supervivientes, tamanio_poblacion):
    nueva_poblacion = []
    
    # Asegurarse de que haya al menos dos individuos para seleccionar
    if len(supervivientes) < 2:
        # Duplicar individuos para mantener la diversidad mínima
        while len(supervivientes) < 2:
            supervivientes.append(random.randint(0, 255))
    
    while len(nueva_poblacion) < tamanio_poblacion:
        padre1, padre2 = random.sample(supervivientes, 2)
        hijo = cruce(padre1, padre2)
        hijo = mutacion(hijo, PROBABILIDAD_MUTACION)
        nueva_poblacion.append(hijo)
    
    return nueva_poblacion

# Algoritmo genético completo
def algoritmo_genetico():
    poblacion = inicializar_poblacion(TAMANIO_POBLACION)
    print(f"Población inicial: {poblacion}")
    mejores_aptitudes = []

    for generacion in range(NUMERO_GENERACIONES):
        aptitud_promedio = np.mean([calcular_aptitud(ind, NIVEL_GRIS_FONDO) for ind in poblacion])
        mejor_individuo = max(poblacion, key=lambda ind: calcular_aptitud(ind, NIVEL_GRIS_FONDO))
        mejores_aptitudes.append(mejor_individuo)

        print(f"Generación {generacion + 1}: Aptitud promedio = {aptitud_promedio}")
        print(f"Generación {generacion + 1}: Mejor aptitud = {calcular_aptitud(mejor_individuo, NIVEL_GRIS_FONDO)}")

        if calcular_aptitud(mejor_individuo, NIVEL_GRIS_FONDO) == 0:
            print("Se ha alcanzado el camuflaje óptimo.")
            break

        supervivientes = seleccion_elitista(poblacion, NIVEL_GRIS_FONDO)
        poblacion = nueva_generacion(supervivientes, TAMANIO_POBLACION)
        print(f'nueva_poblacion: {poblacion}')

    # Visualizar los resultados
    plt.plot(mejores_aptitudes)
    plt.xlabel("Generación")
    plt.ylabel("Mejor Nivel de Gris")
    plt.title("Progreso del Algoritmo Genético")
    plt.show()

# Ejecución del algoritmo
algoritmo_genetico()
