import numpy as np

# Definir el problema: maximizar 3x1 + 2x2
# Sujeto a las restricciones:
# 2x1 + x2 <= 100
# x1 + x2 <= 80
# x1 <= 40
# x1, x2 >= 0

# Construcción del tableau inicial
# Fila de la función objetivo: -Z = -3x1 - 2x2 (vamos a maximizar)
# Agregamos las variables de holgura (s1, s2, s3) para convertir las desigualdades en igualdades

def simplex(c, A, b):
    """
    Implementación del algoritmo Simplex tabular para maximización.
    
    Parámetros:
    - c: coeficientes de la función objetivo.
    - A: matriz de coeficientes de restricciones.
    - b: vector de términos independientes de las restricciones.
    """
    # Crear el tableau (A | b)
    tableau = np.hstack([A, np.eye(A.shape[0]), b.reshape(-1, 1)])
    
    # Añadir la fila de la función objetivo (-c | 0)
    tableau = np.vstack([tableau, np.hstack([-c, np.zeros(A.shape[0] + 1)])])
    
    # Mientras haya elementos negativos en la fila de la función objetivo, el algoritmo sigue.
    while np.any(tableau[-1, :-1] < 0):
        # Seleccionar la columna de pivote (el más negativo de la fila de la función objetivo)
        pivot_col = np.argmin(tableau[-1, :-1])
        
        # Seleccionar la fila de pivote (regla mínima razón positiva)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        pivot_row = np.where(ratios > 0, ratios, np.inf).argmin()
        
        # Realizar la operación de pivoteo
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]  # Hacer el pivote igual a 1
        
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    
    # Extraer la solución
    solution = np.zeros(c.shape[0])
    for i in range(A.shape[0]):
        if np.sum(tableau[i, :c.shape[0]] == 1) == 1:
            solution[np.argmax(tableau[i, :c.shape[0]])] = tableau[i, -1]
    
    # El valor óptimo de Z está en la última posición del tableau
    z_opt = -tableau[-1, -1]
    
    return solution, z_opt


# Coeficientes de la función objetivo
c = np.array([3, 2])

# Coeficientes de las restricciones
A = np.array([
    [2, 1],  # 2x1 + x2 <= 100
    [1, 1],  # x1 + x2 <= 80
    [1, 0]   # x1 <= 40
])

# Términos independientes de las restricciones
b = np.array([100, 80, 40])

# Ejecutar el algoritmo Simplex
solution, z_opt = simplex(c, A, b)

# Imprimir la solución
print(f"Solución óptima: x1 = {solution[0]}, x2 = {solution[1]}")
print(f"Valor óptimo de Z: {z_opt}")
