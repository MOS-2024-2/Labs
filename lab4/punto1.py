from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt


def eliminar_componente(Modelo, nombre_comp):
    componentes_a_eliminar = [var for var in vars(Modelo)
                              if nombre_comp == var
                              or var.startswith(nombre_comp + '_index')
                              or var.startswith(nombre_comp + '_domain')]
    for comp in componentes_a_eliminar:
        Modelo.del_component(comp)


modelo = ConcreteModel()

# Conjuntos
num_nodos = 5
modelo.N = RangeSet(1, num_nodos)

# Parámetros de saltos
modelo.h = Param(modelo.N, modelo.N, mutable=True, initialize=999)
modelo.h[1, 2], modelo.h[1, 3] = 1, 1
modelo.h[2, 5], modelo.h[3, 4], modelo.h[4, 5] = 1, 1, 1

# Parámetros de costos
modelo.c = Param(modelo.N, modelo.N, mutable=True, initialize=999)
modelo.c[1, 2], modelo.c[1, 3] = 10, 5
modelo.c[2, 5], modelo.c[3, 4], modelo.c[4, 5] = 10, 5, 5

# Variables binarias
modelo.x = Var(modelo.N, modelo.N, domain=Binary)

# Función para calcular los saltos totales
def regla_saltos(modelo):
    return sum(modelo.x[i, j] * modelo.h[i, j] for i in modelo.N for j in modelo.N)
modelo.saltos = Expression(rule=regla_saltos)

# Función para calcular el costo total
def regla_costo(modelo):
    return sum(modelo.x[i, j] * modelo.c[i, j] for i in modelo.N for j in modelo.N)
modelo.costo_total = Expression(rule=regla_costo)

# Restricciones
origen, destino = 1, 5  # Nodo de origen y nodo de destino

def restriccion_origen(modelo, i):
    if i == origen:
        return sum(modelo.x[i, j] for j in modelo.N) == 1
    return Constraint.Skip
modelo.restriccion_origen = Constraint(modelo.N, rule=restriccion_origen)

def restriccion_destino(modelo, j):
    if j == destino:
        return sum(modelo.x[i, j] for i in modelo.N) == 1
    return Constraint.Skip
modelo.restriccion_destino = Constraint(modelo.N, rule=restriccion_destino)

def restriccion_intermedia(modelo, i):
    if i != origen and i != destino:
        return sum(modelo.x[i, j] for j in modelo.N) - sum(modelo.x[j, i] for j in modelo.N) == 0
    return Constraint.Skip
modelo.restriccion_intermedia = Constraint(modelo.N, rule=restriccion_intermedia)


lista_saltos = []
lista_costos = []


valores_epsilon = [3, 2, 1]


for epsilon in valores_epsilon:
    # Funcion objetivo
    modelo.objetivo = Objective(expr=modelo.costo_total, sense=minimize)
    
    # Restriccion
    modelo.restriccion_epsilon = Constraint(expr=modelo.saltos <= epsilon)
   
    solver = SolverFactory('glpk')
    resultado = solver.solve(modelo, tee=False)
    
    # Verificamos si el solver encontró una solución óptima
    if (resultado.solver.termination_condition == TerminationCondition.optimal):
        lista_saltos.append(value(modelo.saltos))    # Guardamos los saltos
        lista_costos.append(value(modelo.costo_total))  # Guardamos los costos
        print(f"Solución para epsilon={epsilon}: Costos={value(modelo.costo_total)}, Saltos={value(modelo.saltos)}")
        
        # Imprimimos las variables x[i,j] para mostrar la ruta seleccionada
        for i in modelo.N:
            for j in modelo.N:
                if value(modelo.x[i, j]) > 0.5:
                    print(f"Enlace seleccionado: {i} -> {j}")
    else:
        print(f"No se encontró solución para epsilon={epsilon}")
    
  
    eliminar_componente(modelo, 'objetivo')
    eliminar_componente(modelo, 'restriccion_epsilon')


plt.plot(lista_saltos, lista_costos, 'o-.')
plt.title('Frente Óptimo de Pareto (método e-Constraint)')
plt.xlabel('Saltos (F1)')
plt.ylabel('Costos (F2)')
plt.grid(True)
plt.show()


