
from __future__ import division
from pyomo.environ import *

from pyomo.opt import SolverFactory

Model = ConcreteModel()

# Data de entrada
numTrabajos = 5
numEmpleados = 3

p = RangeSet(1, numTrabajos)
j = RangeSet(1, numEmpleados)

horasEmpleados = {1:8, 2:10, 3:6}
gananciaTrabajo = {1:50, 2:60, 3:40, 4:70, 5:30}
horasTrabajo = {1:4, 2:5, 3:3, 4:6, 5:2}

# Variable de decisión
Model.x = Var(p, j, domain=Binary)

# Función objetivo
Model.obj = Objective(expr = sum(Model.x[tb,ep]*gananciaTrabajo[tb] for tb in p for ep in j), sense=maximize)

# Restricciones
Model.tiempo_empleado = ConstraintList()
Model.trabajoUsado = ConstraintList()

# Restriccion de horas de trabajo
for ep in j:
    Model.tiempo_empleado.add(expr = sum(Model.x[tb, ep]*horasTrabajo[tb] for tb in p) <= horasEmpleados[ep])

# Restriccion para que cada trabajo se asigne maximo una vez
for tb in p:
    Model.trabajoUsado.add(expr = sum(Model.x[tb, ep] for ep in j) <= 1)


# Especificación del solver
SolverFactory('glpk').solve(Model)

Model.display()

# Presentar la información de asignacion
for tb in p:
    for ep in j:
        if Model.x[tb, ep].value == 1:
            print(f"El trabajo {tb} fue asignado al empleado {ep}")






