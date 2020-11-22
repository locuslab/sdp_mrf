from .solvers import find_solver

class PottsModel(object):
    def __init__(self, A=None, h=None, k=None):
        self.set_model_parameters(A, h, k)
    
    def set_model_parameters(self, A, h, k):
        self.A = A
        self.h = h
        self.k = k
    
    def solve_map(self, solver='M4', **kwargs):
        if isinstance(solver, str):
            solver = find_solver(solver)
        return solver.solve_map(self.A, self.h, self.k, **kwargs)
    
    def solve_partition_function(self, solver='M4', **kwargs):
        if isinstance(solver, str):
            solver = find_solver(solver)
        return solver.solve_partition_function(self.A, self.h, self.k, **kwargs)
    
    def solve_probability(self, condition=None):
        pass
