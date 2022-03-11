from functools import reduce 
import math 
from copy import deepcopy

# Abstract Syntax Tree Implementation in Python
# Author: Sriram Sankaranarayanan (srirams@colorado)

# Exception: EvaluationFailedException
# We will raise this exception in case there is a problem with 
# evaluating an expression such as taking square root of -ve number,
# log of -ve number, division by zero etc.. 
class EvaluationFailedException(Exception):
    def __init__(self,msg):
        self.message = msg 
    
    def  __repr__(self):
        return f'Evaluation failed : {self.message}'


# Class Expr
# This is a base class for all Expressions
# Representing a generic expression. We will 
# extend from this  base class.
class Expr: 
    # Evaluate the expression using env to lookup values for identifier
    def eval(self, env):
        raise EvaluationFailedException('What do you want me to eval? I am just a parentless class here. Boo hoo!')

    # Get the number of children
    def num_children(self): 
        raise NotImplementedError
    
    # Get a particular child 
    def get_child(self, idx):
        raise NotImplementedError

    # Replace a subtree by e
    def set_child(self, idx, e):
        raise NotImplementedError

    # Leaf expressions are Const/Ident
    def is_leaf_expr(self):
        return False

    # Compute tree depth
    def depth(self):
        return 0

    # Do a simplification to be able to do some "constant folding"
    def simplify(self):
        return deepcopy(self)


# Visitor Pattern for an expression
# This is useful in implementing functionality outside the Expr class and its derived class.
# We will make use of visitors to manipulate expressions in other functions.
class ExpressionVisitorPattern:
    def __init__(): 
        pass 

    def visitExpr(self, e): 
        if isinstance(e, Const):
            self.visitConst(e)
        elif isinstance(e, Ident): 
            self.visitIdent(e)
        elif isinstance(e, Plus):
            self.visitPlus(e)
        elif isinstance(e, Mult):
            self.visitMult(e)
        elif isinstance(e, Minus): 
            self.visitMinus(e)
        elif isinstance(e, Div):
            self.visitDiv(e)
        elif isinstance(e, UnaryFnApplication):
            self.visitUnaryFnApplication(e)
        else: 
            raise NotImplementedError(f'Unimplemented visitor for type {e.__class__}')

    def visitConst(self, e):
        pass 

    def visitIdent(self, e):
        pass 

    def visitPlus(self, e):
        for ej in e.e_list:
            self.visitExpr(ej)

    def visitMult(self, e):
        for ej in e.e_list:
            self.visitExpr(ej)

    def visitMinus(self, e):
        self.visitExpr(e.args[0])
        self.visitExpr(e.args[1]) 

    def visitDiv(self, e):
        self.visitExpr(e.args[0])
        self.visitExpr(e.args[1])

    def visitUnaryFnApplication(self, e):
        self.visitExpr(e.arg)
   


# Class Const: 
#  Reprents a constant expression with f as the constant (double precision) number 

class Const(Expr):
    def __init__(self, f):
        self.f = f

    def __repr__(self):
        return str(self.f)

    def eval(self, env):
        return  self.f

    def is_leaf_expr(self):
        return True

    def get_constant(self):
        return self.f

# Class: Ident
# Represents a variable with symb (string) as the name of the variable.

class Ident(Expr): 
    def __init__(self, symb):
        self.symb = symb

    def __repr__(self):
        return self.symb

    def eval(self, env):
        if self.symb in env :
            return env[self.symb]
        else:
            raise EvaluationFailedException()

    def is_leaf_expr(self):
        return True


# Class: Plus
# Sum of sub-expressions. The children expression are stored in a list e_list
# It is generally assumed that e_list has 2 or more elements in it though we do not
# enforce it.

class Plus(Expr): 

    def __init__(self, e_list):
        self.e_list = e_list

    def __repr__(self):
        return '('+ (' + '.join([str(ei) for ei in self.e_list])) + ')'

    def eval(self, env):
        flist = [ei.eval(env) for ei in self.e_list]
        return sum(flist)

    def num_children(self): 
        return len(self.e_list)

    def get_child(self, idx):
        assert 0 <= idx < len(self.e_list)
        return self.e_list[idx]

    def set_child(self, idx, e_new):
        assert 0 <= idx < len(self.e_list)
        self.e_list[idx] = e_new

    def depth(self):
        return 1 + max([ej.depth() for ej in self.e_list])

    def simplify(self):
        new_list = [e.simplify() for e in self.e_list]
        const_portion= sum([e.get_constant() for e in new_list if isinstance(e, Const)])
        non_const_list = [e for e in new_list if not isinstance(e, Const)]
        if (len(non_const_list) < len(new_list)):
            non_const_list.append(Const(const_portion))
        return Plus(non_const_list)




# Class: Mult
# Product of sub-expressions. The children expression are stored in a list e_list
# It is generally assumed that e_list has 2 or more elements in it though we do not
# enforce it.

class Mult(Expr):
    def __init__(self, e_list):
        self.e_list = e_list 

    def __repr__(self):
        return '('+ (' * '.join([str(ei) for ei in self.e_list])) + ')'

    def eval(self, env):
        flist = [ei.eval(env) for ei in self.e_list]
        return reduce(lambda a,b:a*b, flist, 1.0 )
    
    def simplify(self):
        new_list = [e.simplify() for e in self.e_list]
        const_portion= reduce(lambda a,b: a*b, [e.get_constant() for e in new_list if isinstance(e, Const)], 1.0)
        non_const_list = [e for e in new_list if not isinstance(e, Const)]
        if (len(non_const_list) < len(new_list)):
            non_const_list.append(Const(const_portion))
        return Mult(non_const_list)

    def num_children(self): 
        return len(self.e_list)

    def get_child(self, idx):
        assert 0 <= idx < len(self.e_list)
        return self.e_list[idx]

    def set_child(self, idx, e_new):
        assert 0 <= idx < len(self.e_list)
        self.e_list[idx] = e_new

    def depth(self):
        return 1 + max([ej.depth() for ej in self.e_list])


# Class: Mult
#  Minus of two expressions e1 - e2

class Minus(Expr):
    def __init__(self, e1, e2):
        self.args = (e1, e2)
    def __repr__(self):
        return str(self.args[0])+' - '+str(self.args[1])

    def eval(self, env):
        f1 = self.args[0].eval(env)
        f2 = self.args[1].eval(env)
        if abs(f2) <= 1E-10:
            raise EvaluationFailedException(f'division by {f2}')
        return f1 - f2

    def simplify(self):
        e1 = self.args[0].simplify()
        e2 = self.args[1].simplify()
        if isinstance(e1, Const) and isinstance(e2, Const):
            f = e1.get_constant() - e2.get_constant()
            return Const(f)
        return Minus(e1, e2)

    def num_children(self): 
        return 2

    def get_child(self, idx):
        assert 0 <= idx < 2
        return self.args[idx]

    def set_child(self, idx, e_new):
        assert 0 <= idx < 2
        if idx == 0:
            self.args = (e_new, self.args[1]) 
        else: 
            self.args = (self.args[0], e_new)
    
    def depth(self):
        return 1 + max(self.args[0].depth(), self.args[1].depth())

# Class: Div
#  Division of two expressions e1 - e2

class Div(Expr):
    def __init__(self, e1, e2):
        self.args = (e1, e2)
    def __repr__(self):
        return '('+ str(self.args[0])+'/'+str(self.args[1])+')'

    def eval(self, env):
        f1 = self.args[0].eval(env)
        f2 = self.args[1].eval(env)
        if abs(f2) <= 1E-10:
            raise EvaluationFailedException(f'division by {f2}')
        return f1/f2

    def depth(self):
        return 1 + max(self.args[0].depth(), self.args[1].depth())

    def num_children(self): 
        return 2

    def get_child(self, idx):
        assert 0 <= idx < 2
        return self.args[idx] 

    def set_child(self, idx, e_new):
        assert 0 <= idx < 2
        if idx == 0:
            self.args = (e_new, self.args[1]) 
        else: 
            self.args = (self.args[0], e_new)
    
    def simplify(self):
        e1 = self.args[0].simplify()
        e2 = self.args[1].simplify()
        if isinstance(e1, Const) and isinstance(e2, Const):
            f = e1.get_constant()/ e2.get_constant()
            return Const(f)
        return Div(e1, e2)
    
# Class: UnaryFnApplication
#  Application of a unary function to a subexpression given by arg.
#  See self.allowed_fun_list for the functions supported by our current implementation.

class UnaryFnApplication(Expr):
    def __init__(self, fn_name, arg):
        self.allowed_fun_list = ['sin','cos','log','exp','atan','tanh','sinh','cosh','sqrt']
        self.funs = {'sin': lambda f: math.sin(f),
        'cos': lambda f: math.cos(f),
        'tan': lambda f: math.tan(f),
        'exp': lambda f: math.exp(f),
        'atan': lambda f: math.atan(f),
        'tanh': lambda f: math.tanh(f),
        'log': lambda f: math.log(f) if f > 0 else None,
        'tanh': lambda f: math.tanh(f),
        'sinh': lambda f: math.sinh(f),
        'cosh': lambda f: math.cosh(f),
        'sqrt': lambda f: math.sqrt(f) if f >= 0.0 else None        
        }
        self.fn_name = fn_name 
        self.arg = arg 
        assert (fn_name in self.allowed_fun_list)
    
    def __repr__(self):
        return f'{self.fn_name}({str(self.arg)})'
    
    def eval(self,env):
        f = self.arg.eval(env)
        assert self.fn_name in self.funs 
        fhandle = self.funs[self.fn_name]
        r  = fhandle(f)
        if r == None: 
            raise EvaluationFailedException(f'function {self.fn_name} raised exception')
        else:
            return r 

    def simplify(self):
        e = self.arg.simplify()
        if isinstance(e, Const):
            f = e.get_constant()
            assert self.fn_name in self.funs 
            fhandle = self.funs[self.fn_name]
            r  = fhandle(f)
            return Const(r)
        return UnaryFnApplication(self.fn_name, e)

    def num_children(self): 
        return 1

    def get_child(self, idx):
        assert idx == 0 
        return self.arg

    def set_child(self, idx, e_new):
        assert idx == 0
        self.arg = e_new 

    def depth(self):
        return 1 + self.arg.depth()
    

