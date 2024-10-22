from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# Función para eliminar componentes del modelo
def delete_component(Model, comp_name):
    list_del = [vr for vr in vars(Model)
                if comp_name == vr
                or vr.startswith(comp_name + '_index')
                or vr.startswith(comp_name + '_domain')]
    for kk in list_del:
        Model.del_component(kk)

# Crear el modelo
Model = ConcreteModel()

# Sets y parámetros
numNodes = 5
Model.N = RangeSet(1, numNodes)

# Parámetros de hops
Model.h = Param(Model.N, Model.N, mutable=True, initialize=999)
Model.h[1,2], Model.h[1,3] = 1, 1
Model.h[2,5], Model.h[3,4], Model.h[4,5] = 1, 1, 1

# Parámetros de costos
Model.c = Param(Model.N, Model.N, mutable=True, initialize=999)
Model.c[1,2], Model.c[1,3] = 10, 5
Model.c[2,5], Model.c[3,4], Model.c[4,5] = 10, 5, 5

# Variables binarias
Model.x = Var(Model.N, Model.N, domain=Binary)

# Función hops
def hops_rule(Model):
    return sum(Model.x[i,j] * Model.h[i,j] for i in Model.N for j in Model.N)
Model.hops = Expression(rule=hops_rule)

# Función costos
def cost_rule(Model):
    return sum(Model.x[i,j] * Model.c[i,j] for i in Model.N for j in Model.N)
Model.cost = Expression(rule=cost_rule)

# Restricciones
s, d = 1, 5  # Nodo origen y destino

def source_rule(Model, i):
    if i == s:
        return sum(Model.x[i, j] for j in Model.N) == 1
    return Constraint.Skip
Model.source = Constraint(Model.N, rule=source_rule)

def destination_rule(Model, j):
    if j == d:
        return sum(Model.x[i,j] for i in Model.N) == 1
    return Constraint.Skip
Model.destination = Constraint(Model.N, rule=destination_rule)

def intermediate_rule(Model, i):
    if i != s and i != d:
        return sum(Model.x[i,j] for j in Model.N) - sum(Model.x[j,i] for j in Model.N) == 0
    return Constraint.Skip
Model.intermediate = Constraint(Model.N, rule=intermediate_rule)

# Listas para almacenar los resultados del frente de Pareto
f1_vec = []
f2_vec = []

# Valores de epsilon (probaremos de 3 hacia abajo)
epsilon_values = [3, 2, 1]  # Reducimos epsilon progresivamente

# Iterar sobre los diferentes valores de epsilon
for epsilon in epsilon_values:
    # Definir la función objetivo (minimizar los costos)
    Model.obj = Objective(expr=Model.cost, sense=minimize)
    
    # Agregar restricción de hops (e-Constraint)
    Model.epsilon_constraint = Constraint(expr=Model.hops <= epsilon)
    
    # Resolver el modelo
    solver = SolverFactory('glpk')
    result = solver.solve(Model, tee=False)
    
    # Chequear si el solver encontró una solución óptima
    if (result.solver.termination_condition == TerminationCondition.optimal):
        f1_vec.append(value(Model.hops))  # Guardamos los hops
        f2_vec.append(value(Model.cost))  # Guardamos los costos
        print(f"Solución para epsilon={epsilon}: Costos={value(Model.cost)}, Hops={value(Model.hops)}")
        
        # Imprimir las variables x[i,j] para mostrar la ruta seleccionada
        for i in Model.N:
            for j in Model.N:
                if value(Model.x[i,j]) > 0.5:
                    print(f"Enlace seleccionado: {i} -> {j}")
    else:
        print(f"No se encontró solución para epsilon={epsilon}")
    
    # Eliminar componentes para la siguiente iteración
    delete_component(Model, 'obj')
    delete_component(Model, 'epsilon_constraint')

# Graficar el frente óptimo de Pareto con los ejes invertidos
plt.plot(f1_vec , f2_vec,'o-.')  # Invertimos los ejes (hops en x, costos en y)
plt.title('Frente Óptimo de Pareto (método e-Constraint)')
plt.xlabel('Hops (F1)')  # Ahora los hops estarán en el eje x
plt.ylabel('Costos (F2)')  # Los costos estarán en el eje y
plt.grid(True)
plt.show()

