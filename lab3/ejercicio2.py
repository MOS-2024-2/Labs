# Código completo y estructurado para el Ejercicio 2

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Definir la función y sus derivadas usando Sympy
x = sp.symbols('x')
f = x**5 - 8*x**3 + 10*x + 6  # Función a analizar
f_prime = sp.diff(f, x)  # Primera derivada
f_double_prime = sp.diff(f_prime, x)  # Segunda derivada

# Funciones de Python para la evaluación numérica
f_func = sp.lambdify(x, f, 'numpy')
f_prime_func = sp.lambdify(x, f_prime, 'numpy')
f_double_prime_func = sp.lambdify(x, f_double_prime, 'numpy')

# Método de Newton-Raphson ajustado para encontrar extremos locales
def newton_raphson_local_extrema(x0, alpha=1.0):
    x_current = x0
    while abs(f_prime_func(x_current)) > 0.001:
        f_prime_x = f_prime_func(x_current)
        f_double_prime_x = f_double_prime_func(x_current)
        x_next = x_current - alpha * f_prime_x / f_double_prime_x
        x_current = x_next
    return x_current

# Intervalo de evaluación
x_vals = np.linspace(-3, 3, 400)
y_vals = f_func(x_vals)

# Inicialización de puntos de inicio para buscar extremos locales
starting_points = np.linspace(-3, 3, 10)  # Puntos iniciales para iterar
local_extrema = []

# Encontrar los extremos locales con Newton-Raphson
for x0 in starting_points:
    extremum = newton_raphson_local_extrema(x0)
    if extremum not in local_extrema:  # Evitar agregar duplicados
        local_extrema.append(extremum)

# Convertir los extremos locales a numpy para facilitar el filtrado
local_extrema = np.array(local_extrema)
local_extrema = np.round(local_extrema, decimals=5)  # Redondeo para evitar valores muy cercanos
local_extrema = np.unique(local_extrema)  # Filtrar valores únicos

# Evaluar los extremos locales en la función
y_extrema = f_func(local_extrema)

# Encontrar el máximo y mínimo globales
min_global = local_extrema[np.argmin(y_extrema)]
max_global = local_extrema[np.argmax(y_extrema)]

# Gráfica de la función y los extremos
plt.figure(figsize=(8, 8))

# Graficar la función
plt.plot(x_vals, y_vals, label="f(x)", color="blue")

# Marcar los mínimos locales en rojo
plt.scatter(local_extrema[y_extrema < 0], f_func(local_extrema[y_extrema < 0]), color="red", label="Min roots", zorder=5)

# Marcar los máximos locales en verde
plt.scatter(local_extrema[y_extrema > 0], f_func(local_extrema[y_extrema > 0]), color="red", label="Max roots", zorder=5)

# Marcar el mínimo global en negro
plt.scatter([min_global], [f_func(min_global)], color="black", label="Global minimum", zorder=6)

# Marcar el máximo global en negro
plt.scatter([max_global], [f_func(max_global)], color="black", label="Global maximum", zorder=6)

# Etiquetas y leyenda ajustadas
plt.title("Roots of the function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Mostrar la gráfica ajustada
plt.show()
