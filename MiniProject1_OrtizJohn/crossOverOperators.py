# Implement a crossover operator between two expressions
from symbolicExpressions import *
from copy import deepcopy   
from random import choice 
from random import choice, random 
from symbolicExpressions import * 
from geneticAlgParams import GAParams
from makeRandomExpressions import generate_random_expr

class CollectSubExprsVisitorForCrossOver(ExpressionVisitorPattern):
    def __init__(self, ret_list): 
        self.ret_list = ret_list 

    def visitPlus(self, e):
        self.ret_list.append(e)
        super().visitPlus(e)
    


    def visitMult(self, e):
        self.ret_list.append(e)
        super().visitMult(e)

    def visitMinus(self, e):
        self.ret_list.append(e)
        super().visitMinus(e)
        

    def visitDiv(self, e):
        self.ret_list.append(e)
        super().visitDiv(e)

    def visitUnaryFnApplication(self, e):
         self.ret_list.append(e)
         super().visitUnaryFnApplication(e)


def collect_all_subexpressions(e):
    e_subexpr_list = []
    e_subexpr_collector = CollectSubExprsVisitorForCrossOver(e_subexpr_list)
    e_subexpr_collector.visitExpr(e)
    return e_subexpr_list

def random_subtree_crossover(e1, e2, copy = True): 
    # Crossover operator must take two expressions e1 and e2
    # Return a tuple of expresions (e3, e4)..
    if e1.is_leaf_expr() or e2.is_leaf_expr():
        return (e1, e2)
    e_a = deepcopy(e1) if copy else e1 
    e_b = deepcopy(e2) if copy else e2
    
    ea_subexpr_list = collect_all_subexpressions(e_a)
    eb_subexpr_list = collect_all_subexpressions(e_b)
    # Now choose a random subexpression 
    e_subst1 = choice(ea_subexpr_list)
    e_subst2 = choice(eb_subexpr_list)
    # Now choose a random child from each subexpression
    sub1 = choice(range(e_subst1.num_children()))
    sub2 = choice(range(e_subst2.num_children()))
    e_child1 = e_subst1.get_child(sub1)
    e_child2 = e_subst2.get_child(sub2)
    # Implement the crossover
    e_subst1.set_child(sub1, e_child2)
    e_subst2.set_child(sub2, e_child1)
    # Return the results
    return (e_a, e_b)


def situate_expression_into_random_expr(e_orig, lst_of_identifiers, params):
    u = random()
    if u <= 0.8:
        e1 = generate_random_expr(params.depth, lst_of_identifiers, params)
        if u <= 0.2:
            return Plus([e_orig, e1])
        elif u <= 0.4:
            return Minus(e_orig, e1)
        elif u <= 0.6:
            return Div(e_orig, e1)
        else: 
            return Mult([e_orig, e1])
    else:
        fn = choice(params.allowed_unary_funs)
        return UnaryFnApplication(fn, e_orig)


def random_expression_mutation(e_orig, lst_of_identifiers, params, copy=True):
    e_copy = deepcopy(e_orig) if copy else e_orig 
    e_subexprs = collect_all_subexpressions(e_copy)
    e_random_subexpr = choice(e_subexprs)
    if random() <= params.replace_by_subexpr:
        return e_random_subexpr
    elif random() <= params.grow_subexpr:
        return situate_expression_into_random_expr(e_random_subexpr, lst_of_identifiers, params)
    else: 
        child_id = choice(range(e_random_subexpr.num_children()))
        rexpr = generate_random_expr(e_random_subexpr.depth()-1, lst_of_identifiers, params)
        e_random_subexpr.set_child(child_id, rexpr)
        return e_random_subexpr




if __name__ == '__main__':
    params = GAParams()
    params.allowed_unary_funs = ['sin', 'sin', 'sin', 'cos', 'cos', 'cos', 
                                'log','exp','exp','exp','atan','tanh','sinh',
                                'cosh','sqrt','sqrt' ]
    params.subexpr_cardinality_list=[2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,5]
    params.prob_of_early_cutoff_random_expr = 0.1
    params.prob_leaf_constant = 0.3
    params.replace_by_subexpr = 0.1
    params.lst_of_random_constants = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, None]
    lst_of_identifiers = ['x','y']
    e1 = generate_random_expr(3, lst_of_identifiers, params)
    e2 = generate_random_expr(3, lst_of_identifiers, params)
    print(f'e1 = {e1} and e2 = {e2}\n')
    (ea, eb) = random_subtree_crossover(e1, e2)
    print(f'ea = {ea} and eb = {eb}')
    ec = random_expression_mutation(ea, lst_of_identifiers, params)
    print(f'ec = {ec}')