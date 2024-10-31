import numpy as np


# Queremos maximizar la expresión 3x1 + 2x2
# Las restricciones son:
#   2x1 + x2 <= 100
#   x1 + x2 <= 80
#   x1 <= 40
# Ademas, x1 y x2 no pueden ser negativas


def simplex(c, A, b):
    """   
    
    Parametros:
    - c: Este es el vector de coeficientes de la funcion objetivo (los "pesos" de x1 y x2 en la formula)
    - A: Matriz con los coeficientes de las restricciones (es decir, como x1 y x2 están limitados)
    - b: Vector que contiene los valores limite de cada restriccion
    """

    tableau = np.hstack([A, np.eye(A.shape[0]), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([-c, np.zeros(A.shape[0] + 1)])])
    
    while np.any(tableau[-1, :-1] < 0):

        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        pivot_row = np.where(ratios > 0, ratios, np.inf).argmin()
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    

    solution = np.zeros(c.shape[0])
    for i in range(A.shape[0]):
        if np.sum(tableau[i, :c.shape[0]] == 1) == 1:
            solution[np.argmax(tableau[i, :c.shape[0]])] = tableau[i, -1]
    
   
    z_opt = tableau[-1, -1]
    
    return solution, z_opt

# Coeficientes 
c = np.array([3, 2])

# Coeficientes de las restricciones
A = np.array([
    [2, 1],  # 2x1 + x2 <= 100
    [1, 1],  # x1 + x2 <= 80
    [1, 0]   #  x1 <= 40
])

# Limite superior 
b = np.array([100, 80, 40])
solution, z_opt = simplex(c, A, b)
print(f"Solucion optima: x1 = {solution[0]}, x2 = {solution[1]}")
print(f"Valor optimo de Z: {z_opt}")

