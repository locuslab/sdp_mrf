from .solvers import *

class PottsModel(object):
    def __init__(self, A=None, h=None, k=None):
        self.A = A
        self.h = h
        self.k = k
        
        # initialize solvers
        self.m4_solver = M4Solver()
        self.m4plus_solver = M4PlusSolver()
        self.exact_solver = ExactSolver()
        self.ais_solver = AISSolver()
    
    def set_model_parameters(self, A, h, k):
        self.A = A
        self.h = h
        self.k = k
    
    def solve_map(self, solver='M4', **kwargs):
        assert solver in ['M4', 'M4+', 'AIS', 'Exact']
        if solver == 'M4':
            rounding_iters = 500
            if 'rounding_iters' in kwargs:
                rounding_iters = kwargs['rounding_iters']
            max_iter = 100
            if 'max_iter' in kwargs:
                max_iter = kwargs['max_iter']
            eps = 0
            if 'eps' in kwargs:
                eps = kwargs['eps']
            return self.m4_solver.solve_map(self.A, self.h, self.k, rounding_iters=rounding_iters,
                                            max_iter=max_iter, eps=eps)
        elif solver == 'M4+':
            rounding_iters = 500
            if 'rounding_iters' in kwargs:
                rounding_iters = kwargs['rounding_iters']
            max_iter = 100
            if 'max_iter' in kwargs:
                max_iter = kwargs['max_iter']
            eps = 0
            if 'eps' in kwargs:
                eps = kwargs['eps']
            return self.m4plus_solver.solve_map(self.A, self.h, self.k, rounding_iters=rounding_iters,
                                                max_iter=max_iter, eps=eps)
        elif solver == 'Exact':
            return self.exact_solver.solve_map(self.A, self.h, self.k)
        elif solver == 'AIS':
            num_samples = 500
            if 'num_samples' in kwargs:
                num_samples = kwargs['num_samples']
            T = 3
            if 'T' in kwargs:
                T = kwargs['T']
            num_cycles = 1
            if 'num_cycles' in kwargs:
                num_cycles = kwargs['num_cycles']
            return self.ais_solver.solve_map(self.A, self.h, self.k, num_samples=num_samples, T=T,
                                             num_cycles=num_cycles)
    
    def solve_partition_function(self, solver='M4', **kwargs):
        assert solver in ['M4', 'M4+', 'AIS', 'Exact']
        if solver == 'M4':
            rounding_iters = 500
            if 'rounding_iters' in kwargs:
                rounding_iters = kwargs['rounding_iters']
            max_iter = 100
            if 'max_iter' in kwargs:
                max_iter = kwargs['max_iter']
            eps = 0
            if 'eps' in kwargs:
                eps = kwargs['eps']
            return self.m4_solver.solve_partition_function(self.A, self.h, self.k, rounding_iters=rounding_iters,
                                                           max_iter=max_iter, eps=eps)
        elif solver == 'M4+':
            rounding_iters = 500
            if 'rounding_iters' in kwargs:
                rounding_iters = kwargs['rounding_iters']
            max_iter = 100
            if 'max_iter' in kwargs:
                max_iter = kwargs['max_iter']
            eps = 0
            if 'eps' in kwargs:
                eps = kwargs['eps']
            return self.m4plus_solver.solve_partition_function(self.A, self.h, self.k, rounding_iters=rounding_iters,
                                                               max_iter=max_iter, eps=eps)
        elif solver == 'Exact':
            return self.exact_solver.solve_partition_function(self.A, self.h, self.k)
        elif solver == 'AIS':
            num_samples = 500
            if 'num_samples' in kwargs:
                num_samples = kwargs['num_samples']
            T = 3
            if 'T' in kwargs:
                T = kwargs['T']
            num_cycles = 1
            if 'num_cycles' in kwargs:
                num_cycles = kwargs['num_cycles']
            return self.ais_solver.solve_partition_function(self.A, self.h, self.k, num_samples=num_samples, T=T,
                                                            num_cycles=num_cycles)
    
    def solve_probability(self, condition=None):
        pass
    
