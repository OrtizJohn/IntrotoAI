from random import choice, random 
from symbolicExpressions import * 
from geneticAlgParams import GAParams


def generate_random_constant(params):
    f = choice(params.lst_of_random_constants)
    if f == None: 
        f = -10.0 + 20.0 * random()
    return Const(f) 

def generate_random_identifier(lst_of_identifiers):
    return Ident(choice(lst_of_identifiers))

def generate_random_expr(depth, lst_of_identifiers, params):
    if depth == 0: #or random() <= params.prob_of_early_cutoff_random_expr:
        if random() <= params.prob_leaf_constant:
            return generate_random_constant(params)
        else:
            return generate_random_identifier(lst_of_identifiers)
    else: 
        # First choose what type of Expressions we wish to see.
        expr_types = ['plus','mult','div', 'minus', 'unaryFunApp'] 
        # should they be equiprobable, fix these.
        expr_choice = choice(expr_types)
        if expr_choice == 'plus':
            num_subexprs = choice(params.subexpr_cardinality_list)
            e_list = [generate_random_expr(depth -1 , lst_of_identifiers, params) for j in range(num_subexprs)]
            return Plus(e_list)
        elif expr_choice== 'mult':
            num_subexprs = choice(params.subexpr_cardinality_list)
            e_list = [generate_random_expr(depth -1 , lst_of_identifiers, params) for j in range(num_subexprs)]
            return Mult(e_list)
        elif expr_choice == 'minus':
            e1 = generate_random_expr(depth-1 , lst_of_identifiers, params)
            e2 = generate_random_expr(depth-1, lst_of_identifiers, params)
            return Minus(e1, e2)
        elif expr_choice == 'div':
            e1 = generate_random_expr(depth-1 , lst_of_identifiers, params)
            e2 = generate_random_expr(depth-1, lst_of_identifiers, params)
            return Div(e1, e2)
        elif expr_choice == 'unaryFunApp':
            e = generate_random_expr(depth-1, lst_of_identifiers, params)
            fun_name = choice(params.allowed_unary_funs)
            return UnaryFnApplication(fun_name, e)
        else: 
            assert False , f'Unknown function type {expr_choice}'

if __name__ == '__main__':
    params = GAParams()
    params.allowed_unary_funs = ['sin', 'sin', 'sin', 'cos', 'cos', 'cos', 
                                'log','exp','exp','exp','atan','tanh','sinh',
                                'cosh','sqrt','sqrt' ]
    params.subexpr_cardinality_list=[2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,5]
    params.prob_of_early_cutoff_random_expr = 0.1
    params.prob_leaf_constant = 0.8
    params.lst_of_random_constants = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, None]
    lst_of_identifiers = ['x','y']
    for i in range(100):
        print(f'--- Expr # {i} ----')
        print(f'{generate_random_expr(4, lst_of_identifiers, params)}')
    

