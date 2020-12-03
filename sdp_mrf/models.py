from .solvers import find_solver

class PottsModel(object):
    def __init__(self, A=None, h=None, k=None):
        '''
        Instantiates a k-class Potts Model, with given coupling matrix A, biases h
        :param A: The coupling matrix. A symmetric numpy array of dimension (n, n)
        :param h: The unary biases. A numpy array of dimension (n, k)
        :param k: The number of classes in the Potts model
        Example usage on a 4-class Potts model on 10 variables:
            n, k = 10, 4
            A, h = np.random.rand(n, n), np.random.rand(n, k)
            A = (A + A.T) / 2
            p = PottsModel(A, h, k)
        '''
        self.set_model_parameters(A, h, k)
    
    def set_model_parameters(self, A, h, k):
        '''
        Setter for the model parameters.
        :param A: The coupling matrix. A symmetric numpy array of dimension (n, n)
        :param h: The unary biases. A numpy array of dimension (n, k)
        :param k: The number of classes in the Potts model
        '''
        self.A = A
        self.h = h
        self.k = k
    
    def solve_map(self, solver='M4', **kwargs):
        '''
        Solves the MAP estimation problem.
        :param solver: The solver to solve the MAP estimation problem
                       Can be one of ['M4', 'M4+', 'Exact', 'AIS']. Default: 'M4'.
        :param kwargs: Can contain additional parameters for the solvers.
                       e.g. number of iterations in M4/M4+, gibbs sampling parameters in AIS, etc.
        Returns: mode_x (MAP configuration), mode_f(f value at mode_x)
        '''
        if isinstance(solver, str):
            solver = find_solver(solver)
        return solver.solve_map(self.A, self.h, self.k, **kwargs)
    
    def solve_partition_function(self, solver='M4', **kwargs):
        '''
        Solves the MAP estimation problem.
        :param solver: The solver to solve the MAP estimation problem
                       Can be one of ['M4', 'M4+', 'Exact', 'AIS']. Default: 'M4'.
        :param kwargs: Can contain additional parameters for the solvers.
                       e.g. number of iterations in M4/M4+, gibbs sampling parameters in AIS, etc.
        Returns: mode_x (MAP configuration), mode_f(f value at mode_x)
        '''
        if isinstance(solver, str):
            solver = find_solver(solver)
        return solver.solve_partition_function(self.A, self.h, self.k, **kwargs)
    
    def solve_probability(self, condition=None):
        pass
