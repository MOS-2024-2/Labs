from pyomo.environ import *

# Modelo
model = ConcreteModel()

# Conjunto de ciudades (por ejemplo, 1, 2, 3, 4)
model.N = Set(initialize=[1, 2, 3, 4])

# Variables binarias x[i,j] que indican si se viaja de la ciudad i a la ciudad j
model.x = Var(model.N, model.N, domain=Binary)

# Restricción 1: Cada ciudad debe ser visitada exactamente una vez (sale una vez)
def leave_city_rule(model, i):
    return sum(model.x[i, j] for j in model.N if j != i) == 1

model.leave_city = Constraint(model.N, rule=leave_city_rule)

# Restricción 2: Cada ciudad debe recibir exactamente una visita (llega una vez)
def arrive_city_rule(model, j):
    return sum(model.x[i, j] for i in model.N if i != j) == 1

model.arrive_city = Constraint(model.N, rule=arrive_city_rule)

# Solucionador
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)
model.display()
