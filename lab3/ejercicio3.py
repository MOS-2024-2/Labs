import numpy as np
import matplotlib.pyplot as plt

# Funcion de rosenbrock y 
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

# Metodo de Newton-Raphson para el minimo
def newton_raphson_3d(x0, y0, alpha=1.0):
    camino = [(x0, y0)]
    x, y = x0, y0
    while True:
        grad = gradient(x, y)
        if np.linalg.norm(grad) < 0.001:
            break
        hess = hessian(x, y)
        step = np.linalg.solve(hess, grad)
        x, y = np.array([x, y]) - alpha * step
        camino.append((x, y))
    return np.array(camino)

# Parametros
x0, y0 = 0, 10  
alpha = 1.0  
camino = newton_raphson_3d(x0, y0, alpha)

# Grafica
x_vals = np.linspace(-6, 6, 400)
y_vals = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock_function(X, Y)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
# Camino de la ruta tomada por el metodo
camino_x = camino[:, 0]
camino_y = camino[:, 1]
camino_z = rosenbrock_function(camino_x, camino_y)
ax.scatter(camino_x, camino_y, camino_z, color='red', label="Iteraciones", zorder=6)
opt_x, opt_y = camino[-1]
opt_z = rosenbrock_function(opt_x, opt_y)
ax.scatter([opt_x], [opt_y], [opt_z], color='red', s=100, label="Óptimo", zorder=7)
ax.text(opt_x, opt_y, opt_z, f'Óptimo: ({opt_x:.2f}, {opt_y:.2f}, {opt_z:.2f})', color='black')
ax.set_xlim([-6, 6])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 200000])
ax.set_title("Newton-Raphson en 3D - Iteraciones y Óptimo")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")
ax.legend()
plt.show()

