from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices, random 
import math 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams


# Implement simulated annealing: this is not compulsory but
# one of the ways you can go "above and beyond" is to compare GA
# to simulated annealing.
# This is not the main assignment however -- please attempt this only
#  if you have solved GA with plenty of time left over.
# Function: run_simulated_annealing
# Inputs: n_steps -- number of steps of SA to run.
#        lst_of_identifiers -- list of identifiers
#        params -- parameters for the problem (same as geneticAlgParams)
#        Useful params for simulated annealing are
#        params.simulated_annealing_cool_steps=100
#        params.simulated_annealing_cool_frac = 0.8
#        params.simulated_annealing_start_temp = 100
#  Feel free to reuse ideas from assignment 4.
def run_simulated_annealing(n_steps, lst_of_identifiers, params):
    raise NotImplementedError('Simulated Annealing not implemented')
    return (best_so_far, best_fitness_so_far, stats)


