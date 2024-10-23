import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

# Crear el solver
solver = SolverFactory('ipopt')

# Definir el modelo general para maximizar la función de beneficio y unidades vendidas
def create_model():
    model = ConcreteModel()
    model.a = Var(bounds=(1e-6, None), initialize=1, within=NonNegativeReals)  # Gasto en publicidad
    # Función de unidades vendidas
    model.q = Expression(expr=890 - 3.8 * model.a + 20 * sqrt(model.a)) # Expression(expr=1000 - 10 * model.price + 20 * sqrt(model.a))
    # Función de beneficio
    model.profit = Expression(expr=-1.444 * (model.a ** 2) + 7.6 * model.a * sqrt(model.a) + 80 * sqrt(model.a) + 322 * model.a + 3560) #Expression(expr=(model.q * (model.price - 6)) - model.a)
    return model

# Función para maximizar el beneficio
def maximize_profit():
    model = create_model()
    model.obj_profit = Objective(expr=model.profit, sense=maximize)
    results = solver.solve(model)
    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        profit_max = model.profit()
        a_profit_max = model.a()
        q_at_profit_max = model.q()
        return profit_max, a_profit_max, q_at_profit_max
    else:
        raise ValueError('El solver no encontró una solución óptima para el beneficio.')

# Función para maximizar las unidades vendidas
def maximize_units_sold():
    model = create_model()
    model.obj_q = Objective(expr=model.q, sense=maximize)
    results = solver.solve(model)
    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        q_max = model.q()
        a_q_max = model.a()
        profit_at_q_max = model.profit()
        return q_max, a_q_max, profit_at_q_max
    else:
        raise ValueError('El solver no encontró una solución óptima para las unidades vendidas.')

# Función para encontrar la frontera de Pareto usando el método epsilon-constraint
def calculate_pareto_front(profit_max, profit_min, w1_values):
    profit_list = []
    q_list = []
    
    for w1 in w1_values:
        profit_limit = profit_max - (profit_max - profit_min) * w1
        model = create_model()
        model.obj_q = Objective(expr=model.q, sense=maximize)  # Maximizar unidades vendidas
        model.profit_limit = Constraint(expr=model.profit >= profit_limit)  # Restricción del beneficio
        
        results = solver.solve(model)
        
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            profit = model.profit()
            q = model.q()
            profit_list.append(profit)
            q_list.append(q)
        else:
            print(f'Solver no encontró solución óptima para w1 = {w1}')
            profit_list.append(None)
            q_list.append(None)

    # Limpiar las listas de None antes de graficar
    profit_list = [p for p in profit_list if p is not None]
    q_list = [q for q in q_list if q is not None]
    
    return profit_list, q_list

# Función para graficar la frontera de Pareto
def plot_pareto_front(profit_list, q_list):
    plt.figure(figsize=(10,6))
    plt.plot(profit_list, q_list, 'o-', label='Soluciones')
    plt.xlabel('Beneficio')
    plt.ylabel('Unidades Vendidas')
    plt.title('Frontera de Pareto: Beneficio vs Unidades Vendidas')
    plt.grid(True)
    plt.legend()
    plt.show()

# Ejecución del código
try:
    # Maximizar el beneficio
    profit_max, a_profit_max, q_at_profit_max = maximize_profit()
    print(f'Máximo beneficio: {profit_max}, Unidades vendidas: {q_at_profit_max}, Gasto en publicidad: {a_profit_max}')

    # Maximizar las unidades vendidas
    q_max, a_q_max, profit_at_q_max = maximize_units_sold()
    print(f'Máximas unidades vendidas: {q_max}, Beneficio: {profit_at_q_max}, Gasto en publicidad: {a_q_max}')

    # Obtener la frontera de Pareto
    w1_values = np.linspace(0, 1, 10)
    profit_list, q_list = calculate_pareto_front(profit_max, profit_at_q_max, w1_values)
    
    # Graficar la frontera de Pareto
    plot_pareto_front(profit_list, q_list)

except ValueError as e:
    print(e)
