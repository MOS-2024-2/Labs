import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

solver = SolverFactory('ipopt')


def create_model():
    model = ConcreteModel()
    model.a = Var(bounds=(1e-6, None), initialize=1, within=NonNegativeReals)  

    # Calculamos las unidades vendidas 
    model.q = Expression(expr=890 - 3.8 * model.a + 20 * sqrt(model.a))

    # Definimos la función de beneficio usando el gasto en publicidad
    model.profit = Expression(expr=-1.444 * (model.a ** 2) + 7.6 * model.a * sqrt(model.a) + 80 * sqrt(model.a) + 322 * model.a + 3560)
    
    return model

def maximize_profit():
    model = create_model()
    model.obj_profit = Objective(expr=model.profit, sense=maximize)
    results = solver.solve(model)
    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        return model.profit(), model.a(), model.q()
    else:
        raise ValueError('El solver no pudo encontrar una solucion optima para el beneficio')

def maximize_units_sold():
    model = create_model()
    model.obj_q = Objective(expr=model.q, sense=maximize)
    results = solver.solve(model)
    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        return model.q(), model.a(), model.profit()
    else:
        raise ValueError('El solver no pudo encontrar una solucion optima para las unidades vendidas')


def calculate_pareto_front(profit_max, profit_min, w1_values):
    profit_list = []
    q_list = []
    
    for w1 in w1_values:
        profit_limit = profit_max - (profit_max - profit_min) * w1
        model = create_model()
        model.obj_q = Objective(expr=model.q, sense=maximize) 
        model.profit_limit = Constraint(expr=model.profit >= profit_limit) 
        
        results = solver.solve(model)
        
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            profit_list.append(model.profit())
            q_list.append(model.q())
        else:
            print(f'Solver no pudo encontrar solucion optima para w1 = {w1}')
            profit_list.append(None)
            q_list.append(None)


    profit_list = [p for p in profit_list if p is not None]
    q_list = [q for q in q_list if q is not None]
    

def plot_pareto_front(profit_list, q_list):
    plt.figure(figsize=(10,6))
    plt.plot(profit_list, q_list, 'o-', label='Soluciones')
    plt.xlabel('Beneficio')
    plt.ylabel('Unidades Vendidas')
    plt.title('Frontera de Pareto: Beneficio vs Unidades Vendidas')
    plt.grid(True)
    plt.legend()
    plt.show()


try:

    profit_max, a_profit_max, q_at_profit_max = maximize_profit()
    print(f'Maximo beneficio: {profit_max}, Unidades vendidas: {q_at_profit_max}, Gasto en publicidad: {a_profit_max}')

    q_max, a_q_max, profit_at_q_max = maximize_units_sold()
    print(f'Maximas unidades vendidas: {q_max}, Beneficio: {profit_at_q_max}, Gasto en publicidad: {a_q_max}')


    w1_values = np.linspace(0, 1, 10)
    profit_list, q_list = calculate_pareto_front(profit_max, profit_at_q_max, w1_values)
    

    plot_pareto_front(profit_list, q_list)

except ValueError as e:
    print(e)
