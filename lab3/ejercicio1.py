# Código completo y limpio para implementar el método de Newton-Raphson

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Definir la función y sus derivadas usando Sympy
x = sp.symbols('x')
f = 3*x**3 - 10*x**2 - 56*x + 50
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

# Funciones de Python para la evaluación numérica
f_func = sp.lambdify(x, f, 'numpy')
f_prime_func = sp.lambdify(x, f_prime, 'numpy')
f_double_prime_func = sp.lambdify(x, f_double_prime, 'numpy')

# Método de Newton-Raphson ajustado al pseudocódigo
def newton_raphson_pseudocode_exact(x0, alpha):
    x_current = x0
    iterations = [x_current]
    convergencia = 0.001
    while abs(f_prime_func(x_current)) > convergencia:
        f_prime_x = f_prime_func(x_current)
        f_double_prime_x = f_double_prime_func(x_current)
        x_next = x_current - alpha * f_prime_x / f_double_prime_x
        iterations.append(x_next)
        x_current = x_next
    return x_current, iterations

# Parámetros iniciales
x0_max = 4  # Punto de arranque para encontrar el máximo
x0_min = -5  # Punto de arranque para encontrar el mínimo
alpha_1 = 1.0  # Valor de alpha inicial
alpha_2 = 0.6  # Segundo valor de alpha

# Ejecutar el método para encontrar el máximo con los dos valores de alpha
optimum_alpha_1_max, iterations_alpha_1_max = newton_raphson_pseudocode_exact(x0_max, alpha_1)
optimum_alpha_2_max, iterations_alpha_2_max = newton_raphson_pseudocode_exact(x0_max, alpha_2)

# Ejecutar el método para encontrar el mínimo con los dos valores de alpha
optimum_alpha_1_min, iterations_alpha_1_min = newton_raphson_pseudocode_exact(x0_min, alpha_1)
optimum_alpha_2_min, iterations_alpha_2_min = newton_raphson_pseudocode_exact(x0_min, alpha_2)

# Generar los puntos para la gráfica
x_vals = np.linspace(-6, 6, 400)
y_vals = f_func(x_vals)

# Gráfica de los resultados ajustados para mostrar iteraciones y óptimos
plt.figure(figsize=(8, 10))

# Subplot 1: Raíces de la función (igual que antes, ya correcta)
plt.subplot(2, 1, 1)
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.scatter([optimum_alpha_1_max], [f_func(optimum_alpha_1_max)], color="red", label="Max roots", zorder=5)
plt.scatter([optimum_alpha_1_min], [f_func(optimum_alpha_1_min)], color="green", label="Min roots", zorder=5)
plt.legend()
plt.title("Roots of the function")
plt.xlabel("x")
plt.ylabel("f(x)")

# Subplot 2: Iteraciones con los puntos intermedios y los óptimos (ajustado a la salida esperada)
plt.subplot(2, 1, 2)
plt.plot(x_vals, y_vals, label="f(x)", color="blue")

# Iteraciones con Alpha = 1
plt.scatter(iterations_alpha_1_max, f_func(np.array(iterations_alpha_1_max)), color="purple", label="Alpha = 1", zorder=5, marker='o', edgecolor='red', facecolors='none')
plt.scatter(iterations_alpha_1_min, f_func(np.array(iterations_alpha_1_min)), color="purple", zorder=5, marker='o', edgecolor='red', facecolors='none')

# Iteraciones con Alpha = 0.6
plt.scatter(iterations_alpha_2_max, f_func(np.array(iterations_alpha_2_max)), color="green", label="Alpha = 0.6", zorder=5, marker='o', edgecolor='green', facecolors='none')
plt.scatter(iterations_alpha_2_min, f_func(np.array(iterations_alpha_2_min)), color="green", zorder=5, marker='o', edgecolor='green', facecolors='none')

# Marcar los puntos óptimos
plt.scatter([optimum_alpha_1_max], [f_func(optimum_alpha_1_max)], color="red", label="Optimum Alpha = 1", zorder=6, marker='o')
plt.scatter([optimum_alpha_2_max], [f_func(optimum_alpha_2_max)], color="red", zorder=6, marker='o')

plt.scatter([optimum_alpha_1_min], [f_func(optimum_alpha_1_min)], color="green", label="Optimum Alpha = 0.6", zorder=6, marker='o')
plt.scatter([optimum_alpha_2_min], [f_func(optimum_alpha_2_min)], color="green", zorder=6, marker='o')

# Etiquetas y leyendas

plt.legend()
plt.title("Iterations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.show()