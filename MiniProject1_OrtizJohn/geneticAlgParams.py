class GAParams:
    def __init__(self):
        self.allowed_unary_funs = ['sin', 'sin', 'sin', 'cos', 'cos', 'cos', 
                                'log','exp','exp','exp','atan','sqrt','sqrt','tanh','sinh','cosh' ]
        self.subexpr_cardinality_list=[2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,5]
        self.prob_of_early_cutoff_random_expr = 0.25
        self.prob_leaf_constant = 0.3
        self.replace_by_subexpr = 0.3
        self.grow_subexpr = 0.2
        self.lst_of_random_constants = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, None]
        self.regression_training_data = []
        self.test_points = [] 
        self.depth = 3
        self.elitism_fraction = 0.2
        self.temperature = 10
        self.simulated_annealing_cool_steps=100
        self.simulated_annealing_cool_frac = 0.8
        self.simulated_annealing_start_temp = 100
    # Do not forget to set the training data and test_points for viability
