import numpy as np
import matplotlib.pyplot as plt

# Definir la función de Rosenbrock y sus derivadas para el pseudocódigo
def rosenbrock_function(x, y):
    return (x - 1)**2 + 100 * (y - x**2)**2

def gradient(x, y):
    df_dx = 2 * (x - 1) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def hessian(x, y):
    d2f_dx2 = 2 - 400 * (y - 3 * x**2)
    d2f_dxy = -400 * x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxy], [d2f_dxy, d2f_dy2]])

# Método de Newton-Raphson con el pseudocódigo ajustado
def newton_raphson_3d_exact(x0, y0, alpha=1.0):
    path = [(x0, y0)]
    x, y = x0, y0
    while True:
        grad = gradient(x, y)
        if np.linalg.norm(grad) < 0.001:
            break
        hess = hessian(x, y)
        step = np.linalg.solve(hess, grad)
        x, y = np.array([x, y]) - alpha * step
        path.append((x, y))
    return np.array(path)

# Parámetros iniciales (punto de arranque)
x0, y0 = 0, 10  # Punto inicial
alpha = 1.0  # Paso
path = newton_raphson_3d_exact(x0, y0, alpha)

# Graficar la superficie de la función de Rosenbrock
x_vals = np.linspace(-6, 6, 400)  # Ajustar los rangos en X
y_vals = np.linspace(-10, 10, 400)  # Ajustar los rangos en Y
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock_function(X, Y)

# Crear la gráfica en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Graficar la ruta de puntos encontrados
path_x = path[:, 0]
path_y = path[:, 1]
path_z = rosenbrock_function(path_x, path_y)
ax.scatter(path_x, path_y, path_z, color='red', label="Iterations", zorder=6)

# Marcar el óptimo y añadir una etiqueta con las coordenadas
opt_x, opt_y = path[-1]
opt_z = rosenbrock_function(opt_x, opt_y)
ax.scatter([opt_x], [opt_y], [opt_z], color='red', s=100, label="Optimum", zorder=7)
ax.text(opt_x, opt_y, opt_z, f'Optimum: ({opt_x:.2f}, {opt_y:.2f}, {opt_z:.2f})', color='black')

# Ajustar los límites de los ejes para que coincidan con la imagen esperada
ax.set_xlim([-6, 6])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 200000])

# Etiquetas y leyendas
ax.set_title("Newton-Raphson en 3D - Iterations and Optimum")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")
ax.legend()

plt.show()
