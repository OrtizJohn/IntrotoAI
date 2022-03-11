from symbolicExpressions import * 
import math 
debug = False

def make_env(lst_of_identifiers, test_pt):
    env = {}
    for (id, v) in zip(lst_of_identifiers, test_pt):
        env[id] = v
    return env

def checkFunctionValidity(fun_expr, lst_of_identifiers, test_point_list):
    for test_pt in test_point_list:
        env = make_env(lst_of_identifiers, test_pt) 
        try:
            fun_expr.eval(env)
        except:
            if debug:
                print(f'Failed expression {fun_expr}')
            return False 
    return True 


def is_viable_expr(fun_expr, lst_of_identifiers, params):
    return checkFunctionValidity(fun_expr, lst_of_identifiers, params.test_points)


def compute_fitness(fun_expr, lst_of_identifiers, params):
    regression_training_data = params.regression_training_data 
    fitness = 0.0
    for (test_pt, y) in regression_training_data:
        env = make_env(lst_of_identifiers, test_pt) 
        try:
            yHat = fun_expr.eval(env)
            fitness = fitness - (yHat - y)**2
        except:
            fitness = -float('inf') 
            if debug:
                print(f'Warning: Expression evaluation failed: {fun_expr} @ {test_pt}')
            return fitness 
    return (fitness)
    
               

