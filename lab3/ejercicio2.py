import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# inicializacion de las funciones
x = sp.symbols('x')
f = x**5 - 8*x**3 + 10*x + 6 
f_prima = sp.diff(f, x)  # Derivada primera
f_segunda = sp.diff(f_prima, x)  # Derivada segunda

# Convertimos las funciones a versiones que se puedan evaluar numericamente
f_func = sp.lambdify(x, f, 'numpy')
f_prima_func = sp.lambdify(x, f_prima, 'numpy')
f_segunda_func = sp.lambdify(x, f_segunda, 'numpy')

# Encontramos extremos usando Newton-Raphson
def newton_raphson_extremos(x0, alpha=1.0):
    x_actual = x0
    while abs(f_prima_func(x_actual)) > 0.001:
        f_prima_val = f_prima_func(x_actual)
        f_segunda_val = f_segunda_func(x_actual)
        x_siguiente = x_actual - alpha * f_prima_val / f_segunda_val
        x_actual = x_siguiente
    return x_actual

# Definimos el rango para evaluar la funcion
x_vals = np.linspace(-3, 3, 400)
y_vals = f_func(x_vals)
puntos_inicio = np.linspace(-3, 3, 10)  
extremos_locales = []

# Buscamos extremos locales con Newton-Raphson
for x0 in puntos_inicio:
    extremo = newton_raphson_extremos(x0)
    if extremo not in extremos_locales: 
        extremos_locales.append(extremo)

# Convertimos a un array, eliminamos duplicados y redondeamos 
extremos_locales = np.array(extremos_locales)
extremos_locales = np.round(extremos_locales, decimals=5)  
extremos_locales = np.unique(extremos_locales)  

# Evaluamos la funcion
y_extremos = f_func(extremos_locales)
min_global = extremos_locales[np.argmin(y_extremos)]
max_global = extremos_locales[np.argmax(y_extremos)]

# Graficar
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.scatter(extremos_locales[y_extremos < 0], f_func(extremos_locales[y_extremos < 0]), color="red", label="Mínimos locales", zorder=5)
plt.scatter(extremos_locales[y_extremos > 0], f_func(extremos_locales[y_extremos > 0]), color="green", label="Máximos locales", zorder=5)
plt.scatter([min_global], [f_func(min_global)], color="black", label="Mínimo global", zorder=6)
plt.scatter([max_global], [f_func(max_global)], color="black", label="Máximo global", zorder=6)
plt.title("Extremos de la función")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

