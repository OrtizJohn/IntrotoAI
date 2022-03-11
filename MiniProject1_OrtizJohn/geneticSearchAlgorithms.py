from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices
import math
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
from matplotlib import pyplot as plt

class GASolver:
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        if(self.N <=0):
            self.pop = []
        else:
            self.pop = []

            while( len(self.pop) != self.N):
                expr = generate_random_expr(self.params.depth,lst_of_identifiers,self.params)
                if(is_viable_expr(expr,lst_of_identifiers,params)):
                    self.pop.append(expr)
            #for i in range(self.N):

        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')

    # Please add whatever helper functions you wish.
    def sorterHelper(self,population):
        testList = [compute_fitness(x,self.identifiers,self.params)  for x in population]
        testDict = dict(zip(population,testList))
        testDict_sorted = sorted(testDict.items(), key=lambda item: item[1],reverse=True) #return list of tuples of sorted items
        sorted_dict = {k: v for k, v in testDict_sorted} #puts the tups back in dictionary form
        newSortedExpr = sorted_dict.keys()
        return list(newSortedExpr)

    def explicitPrint(self,population): #for sanity checks
        for i in range(len(population)):
            fit = compute_fitness(population[i],self.identifiers,self.params)
            print(str(i)+", Fitness: "+str(fit)+ ": "+str(population[i]))
    def getWeights(self): #return list of weights based on current pop
        weights = [0]*self.N
        for i in range(len(self.pop)):
            fit = compute_fitness(self.pop[i],self.identifiers,self.params)
            w = math.exp( fit/self.params.temperature)
            weights[i] = w
        #print(len(weights))
        return weights
    def get_nMinusK_expr(self,weights):

        e1_list = choices(self.pop, weights = weights,k=1) #if we end up breakin things just do it without replacement _______here_____________
        e1 = e1_list[0]

        weightsE2 = weights.copy() #create separate weights for the second one where e1 index in the weights should now = 0
        weightsE2[self.pop.index(e1)] = 0
        e2_list = choices(self.pop, weights = weightsE2,k=1)
        e2 = e2_list[0]

        #cross over expressions
        e1_cross,e2_cross =random_subtree_crossover(e1, e2, copy = True)

        #mutate each expression
        e1_cross_mutate = random_expression_mutation(e1_cross,self.identifiers, self.params, copy=True)
        e2_cross_mutate = random_expression_mutation(e2_cross,self.identifiers, self.params, copy=True)


        return (e1_cross_mutate,e2_cross_mutate) #not checked if viable yet

    def run_one_ga_iteration(self):
        k = int(self.params.elitism_fraction * self.N)
        #print("N is: "+ str(self.N) +"---Our k is: "+str(k))

        newSortedExpr = self.sorterHelper(self.pop)


        #________________________________grab k elites for top 20% percent of list__________________________________________
        eliteExpr = [-1]*k
        for i in range(k):
            eliteExpr[i]= newSortedExpr[i]
        #sanity check for length
        #print(len(eliteExpr))


        #_________________________we need to produce n-k members_________________________________________________
        n_kMembers = []
        weightsList = self.getWeights()
        nKMembCountGoal = self.N-k
        nKMembCount= 0
        while(nKMembCount < nKMembCountGoal): #5
            e1,e2 = self.get_nMinusK_expr(weightsList)
            if(is_viable_expr(e1,self.identifiers,self.params) and ((nKMembCount+1) <= nKMembCountGoal) ):
                n_kMembers.append(e1)
                nKMembCount+=1


            if(is_viable_expr(e2,self.identifiers,self.params) and ((nKMembCount+1) <= nKMembCountGoal) ):
                n_kMembers.append(e2)
                nKMembCount+=1


        #sanity check
        #-is there n-k n_kMembers

        newPopulation = eliteExpr + n_kMembers
        #sanity check
        #self.explicitPrint(newPopulation)
        self.pop = self.sorterHelper(newPopulation)#we need to update self.pop for other iterations to progress
        #self.explicitPrint(self.pop)






    # TODO: Implement the genetic algorithm as described in the
    # project instructions.
    # This function need not return anything. However, it should
    # update the fields best_solution_so_far, best_fitness_so_far and
    # population_stats
    def run_ga_iterations(self, n_iter=1000):
        sortedPop = self.sorterHelper(self.pop)
        #update fields before starting
        self.best_solution_so_far = sortedPop[0]
        fit = compute_fitness(sortedPop[0],self.identifiers,self.params)
        self.best_fitness_so_far = fit
        self.population_stats.append(fit)
        for i in range(15):
            self.run_one_ga_iteration()
            competitorExpr = self.pop[0]
            f1 =compute_fitness(competitorExpr,self.identifiers,self.params)
            if(f1 > self.best_fitness_so_far):
                self.best_fitness_so_far = f1
                self.best_solution_so_far = competitorExpr
                #print("newBest Fitness: "+str(self.best_fitness_so_far))
            #update population stats
            self.population_stats.append(f1)
        #print(self.population_stats)

## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)


# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
       ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
    solver = GASolver(params,['x'],500)
    solver.run_ga_iterations(100)

    print('Done!')
    print(f'Best solution found: {solver.best_solution_so_far.simplify()}, fitness = {solver.best_fitness_so_far}')
    stats = solver.population_stats
    niters = len(stats)
    plt.plot(range(niters), [st[0] for st in stats] , 'b-')
    plt.show()
