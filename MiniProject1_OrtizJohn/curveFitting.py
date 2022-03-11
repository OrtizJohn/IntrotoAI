from geneticAlgParams import GAParams
from geneticSearchAlgorithms import curve_fit_using_genetic_algorithm
from random import random
import math 
from matplotlib import pyplot as plt
from simulatedAnnealing import run_simulated_annealing 
def one_dimensional_curve_fitting_test(lambda_fun, x_limits, n_data_points, pop_size = 1000, num_iters = 100, n_test_points = 100, method='ga'):
    params = GAParams()
    (a, b) = x_limits
    assert a < b
    # first generate data 
    data = [] 
    for i in range(n_data_points):
        x_value = a + random() * (b-a)
        data.append( ([x_value], lambda_fun(x_value)) )
    
    delta = (b-a)/(n_test_points)
    test_points = []
    for j in range(n_test_points+1):
        test_points.append([a + j * delta])
    
    params.test_points = test_points
    params.regression_training_data = data 
    if method == 'ga':
        (best_expr, best_fitness, stats) = curve_fit_using_genetic_algorithm(params, ['x'], pop_size, num_iters)
        best_expr = best_expr.simplify()
        print(f'GA Returned Solution: {best_expr} with fitness {best_fitness}')
    else: 
        params.temperature = params.simulated_annealing_start_temp
        (best_expr, best_fitness, stats) = run_simulated_annealing(20000, ['x'], params)
    plt.figure(1)
    x_values = [x_value for ([x_value], _) in data]
    plt.plot(x_values, [y for (_,y) in data],'x')
    test_xvalues = sorted([x for [x] in test_points])
    result = [best_expr.eval({'x':x_value}) for x_value in test_xvalues ]
    gTruth = [lambda_fun(x_value) for x_value in test_xvalues ]
    plt.plot(test_xvalues, result, 'r-',label='ga_fit')
    plt.plot(test_xvalues, gTruth, 'g-', label='ground-truth')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure(2)
    plt.plot(range(len(stats)), [st for st in stats], 'b-')
    plt.xlabel('Iters')
    plt.ylabel('Max Fitness')
    #plt.plot(range(len(stats)), [(st[1] if st[1] > -100 else -100) for st in stats], 'r--')
    plt.show()

if __name__ == '__main__':
    one_dimensional_curve_fitting_test(lambda x: 0.2*math.exp(x/4.0) -  math.sin(2*x)  , (-10.0, 10.0), 25, method='sa')
    