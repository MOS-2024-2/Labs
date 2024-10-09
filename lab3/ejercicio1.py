import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Inicianilizacion de las funciones 
x = sp.symbols('x')
f = 3*x**3 - 10*x**2 - 56*x + 50
f_prima = sp.diff(f, x)
f_segunda = sp.diff(f_prima, x)

# Convertimos las funciones simbolicas a funciones numericas
f_func = sp.lambdify(x, f, 'numpy')
f_prima_func = sp.lambdify(x, f_prima, 'numpy')
f_segunda_func = sp.lambdify(x, f_segunda, 'numpy')

# Metodo de Newton-Raphson 
def newton_raphson(x0, alpha):
    x_actual = x0
    puntos = [x_actual]
    tolerancia = 0.001
    while abs(f_prima_func(x_actual)) > tolerancia:
        f_prima_val = f_prima_func(x_actual)
        f_segunda_val = f_segunda_func(x_actual)
        siguiente_x = x_actual - alpha * f_prima_val / f_segunda_val
        puntos.append(siguiente_x)
        x_actual = siguiente_x
    return x_actual, puntos

# Parametros iniciales
x_inicial_max = 4  # Punto inicial para el maximo
x_inicial_min = -5  # Punto inicial para el minimo
alpha1 = 1.0  # Primer valor de alpha
alpha2 = 0.6  # Segundo valor de alpha

# encontramos el máximo con los dos valores de alpha
optimo_max_1, iter_max_1 = newton_raphson(x_inicial_max, alpha1)
optimo_max_2, iter_max_2 = newton_raphson(x_inicial_max, alpha2)

#enncontramos el mínimo con los dos valores de alpha
optimo_min_1, iter_min_1 = newton_raphson(x_inicial_min, alpha1)
optimo_min_2, iter_min_2 = newton_raphson(x_inicial_min, alpha2)


x_vals = np.linspace(-6, 6, 400)
y_vals = f_func(x_vals)
plt.figure(figsize=(8, 10))

# Grafica 1
plt.subplot(2, 1, 1)
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.scatter([optimo_max_1], [f_func(optimo_max_1)], color="red", label="Raíces máximas")
plt.scatter([optimo_min_1], [f_func(optimo_min_1)], color="green", label="Raíces mínimas")
plt.legend()
plt.title("Raíces de la función")
plt.xlabel("x")
plt.ylabel("f(x)")

# Grafica 2
plt.subplot(2, 1, 2)
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.scatter(iter_max_1, f_func(np.array(iter_max_1)), color="purple", label="Alpha = 1", marker='o', edgecolor='red', facecolors='none')
plt.scatter(iter_min_1, f_func(np.array(iter_min_1)), color="purple", marker='o', edgecolor='red', facecolors='none')
plt.scatter(iter_max_2, f_func(np.array(iter_max_2)), color="green", label="Alpha = 0.6", marker='o', edgecolor='green', facecolors='none')
plt.scatter(iter_min_2, f_func(np.array(iter_min_2)), color="green", marker='o', edgecolor='green', facecolors='none')
plt.scatter([optimo_max_1], [f_func(optimo_max_1)], color="red", label="Óptimo Alpha = 1", marker='o')
plt.scatter([optimo_max_2], [f_func(optimo_max_2)], color="red", marker='o')
plt.scatter([optimo_min_1], [f_func(optimo_min_1)], color="green", label="Óptimo Alpha = 0.6", marker='o')
plt.scatter([optimo_min_2], [f_func(optimo_min_2)], color="green", marker='o')
plt.legend()
plt.title("Iteraciones")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.show()
