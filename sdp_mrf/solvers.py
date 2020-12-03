import numpy as np
import time
from abc import ABC, abstractmethod
import sys, os
from . import _solvers

def get_string_from_vector(v):
    '''
    Converts vector to string
    :param v: a numpy array with digits in 0-9
    Returns: the string form of v
    '''
    ret = ""
    for digit in v:
        assert digit >=0 and digit < 10
        ret += str(digit)
    return ret

def get_vector_from_string(s):
    '''
    Converts string to vector
    :param s: a string with digits in 0-9
    Returns: the numpy array form of s
    '''
    ret = []
    for c in s:
        assert int(c) >= 0 and int(c) < 10
        ret.append(int(c))
    return np.array(ret)

def get_f(A, h, s):
    """
    Computes f = \sum_{ij}Aij\delta(si, sj)/2 + \sum_i\sum_l h_il\delta(si, l)
    :param A: the coupling matrix, numpy array of dim (n, n)
    :param h: the unary biases, numpy array of dim (n, k)
    :param s: the argument at which f is to be computed, string/numpy array
    Returns: f value at s
    """
    k = h.shape[1]
    n = A.shape[0]
    if type(s) == str:
        s = np.array(list(s), dtype=int)
    delta = np.zeros((k, n))
    delta[s, np.arange(n)] = 1
    sm = np.sum((delta.T @ delta) * A) - np.sum(A) / 2
    truth = np.eye(k)
    sm += 2 * np.sum((delta.T @ truth) * h) - np.sum(h)
    return sm

def rand_unit_vector(k, d):
    '''
    Samples k uniformly random unit vectors of dimension d
    :param k: number of unit vectors to sample, int
    :param d: dimension of unit vectors, int
    Returns: numpy array of dim (k, d) with rows as unit vectors
    '''
    r = np.random.normal(0, 1, size=(k, d))
    return  r / np.linalg.norm(r, axis=1, keepdims=True)
    
def obtain_rounded_v(V, B):
    '''
    Obtain rounded solution from SDP solution V and simplex B
    :param V: SDP solution, numpy array of dim (n, d)
    :param B: simplex, numpy array of dim (k, d)
    Returns: the rounded configuration, numpy array of dim (n,)
    '''
    n = V.shape[0]
    d = V.shape[1]
    k = B.shape[0]
    r = rand_unit_vector(k, d)

    rounded_v = np.argmax(V @ r.T, axis=1)
    rounded_v_one_hot = np.zeros((n, k))
    rounded_v_one_hot[np.arange(n), rounded_v] = 1

    # shape(num_classes): saying that r:i maps to S:j
    r_to_B = np.argmax(r @ B.T, axis=1)
    transformation_matrix = np.zeros((k, k))
    transformation_matrix[np.arange(k), r_to_B] = 1
    rounded_v_one_hot = rounded_v_one_hot @ transformation_matrix
    rounded_v = np.argmax(rounded_v_one_hot, axis=1)

    return rounded_v

def ensure_C(x):
    '''ensure input matrix x is C-compatible'''
    return np.ascontiguousarray(x, dtype=np.float32)

def LSE(y):
    '''Log-sum-exp funciton'''
    max_y = np.max(y)
    return np.log(np.sum(np.exp(np.array(y) - max_y))) + max_y

# get regular or probabilistic simplex
def get_simplex(k, d, is_prob=False):
    '''if is_prob: return a probabilistic k-simplex in R^d
       else:       return a regular k-simplex (centered a origin) in R^d'''
    B = np.zeros((k, d))
    B[np.arange(k), np.arange(k)] = 1

    if not is_prob:
        r0 = np.sum(B, axis=0) / k
        c = np.sqrt((k - 1) / k)
        B = (B - r0[np.newaxis, :]) / c

    return B

class Solver(ABC):
    """
    Abstract base class for Solvers
    """
    @abstractmethod
    def solve_map(self):
        pass
    
    @abstractmethod
    def solve_partition_function(self):
        pass

class M4Solver(Solver):
    def __init__(self):
        '''
        Instantiates a M4 Solver.
        Example usage: solver = M4Solver()
                       solver.solve_map(A, h, k)
        '''
        pass

    def solve(self, A, h, V_init, max_iter, eps):
        '''
        Solve the SDP using M4
        :param A: the coupling matrix, numpy array of dim (n,n)
        :param h: biases, numpy array of dim (n,k)
        :param V_init: initialization of unit vectors, dim (n, d)
        :param max_iter: max number of iterations to run M4, int
        :param eps: tolerance to stop M4, float
        Returns: Solution to SDP
        '''
        n = A.shape[0]
        d = V_init.shape[1]
        assert h.shape[1] == d

        A, h, V = map(ensure_C, [A, h, V_init])
        diff = _solvers.M4(A, h, V, eps, max_iter)
        return diff, V
    
    def solve_map(self, A, h, k, rounding_iters=500, max_iter=100, eps=0, returnVB=False, returnTime=False):
        '''
        Solve the MAP estimation problem.
        :param rounding_iters: number of rounding iterations, int
        :param returnVB: boolean value used for partition function estimation
        :param returnTime: boolean value used for timing expts
        '''
        n = A.shape[0]
        k = h.shape[1]
        d = int(np.ceil(np.sqrt(2*(n+k*(k+1)/2)) + 1))
        V = rand_unit_vector(n, d)
        B = get_simplex(k, d)
        diff, V = self.solve(A, h @ B, V, max_iter, eps)

        mode_x, mode_f = None, -np.inf
        t_list = []
        f_list = []
        for _ in range(rounding_iters):
            x = obtain_rounded_v(V, B)
            f = get_f(A, h, x)
            t_list.append(time.time())
            f_list.append(f)
            if f > mode_f:
                mode_x = x
                mode_f = f
        
        if returnVB:
            return mode_x, mode_f, V, B
        
        if returnTime:
            return mode_x, mode_f, t_list, f_list
        return mode_x, mode_f
    
    def solve_partition_function(self, A, h, k, rounding_iters=500, max_iter=100, eps=0):
        '''
        Compute the partition function from the SDP solution.
        '''
        n = A.shape[0]
        _, _, V, B = self.solve_map(A, h, k, rounding_iters=rounding_iters, max_iter=max_iter,
                                    eps=eps, returnVB=True)
        s_list = {}
        f_list = []
        for _ in range(rounding_iters):
            x = obtain_rounded_v(V, B)
            f = get_f(A, h, x)
            s = get_string_from_vector(x)
            if s not in s_list:
                s_list[s] = 1
                f_list.append(f)

        rem = np.log(1-np.exp(np.log(len(f_list))-n*np.log(k)))

        y_list = []
        while True:
            if len(y_list) >= rounding_iters: break
            x = np.random.choice(k, n, replace=True)
            s = get_string_from_vector(x)
            if s in s_list: continue
            f = get_f(A, h, x)
            log_q = -n * np.log(k) - rem
            f_minus_log_q = f - log_q
            y_list.append(f_minus_log_q)
        sm = LSE(y_list) - np.log(len(y_list))
        sm = LSE([sm]+f_list)

        return sm
    
class M4PlusSolver(Solver):
    def __init__(self):
        '''
        Instantiates a M4+ Solver.
        Example usage: solver = M4PlusSolver()
                       solver.solve_map(A, h, k)
        '''
        pass

    def solve(self, A, h, Z_init, k, max_iter, eps):
        '''
        Solve the SDP using M4+
        :param A: the coupling matrix, numpy array of dim (n,n)
        :param h: biases, numpy array of dim (n,k)
        :param V_init: initialization of unit vectors, dim (n, d)
        :param max_iter: max number of iterations to run M4, int
        :param eps: tolerance to stop M4, float
        Returns: Solution to SDP
        '''
        n = A.shape[0]
        d = Z_init.shape[1]
        m = d // k
        assert A.shape[1] == n and h.shape[0] == n and h.shape[1] == d
        assert d % k == 0
        h, Z = [x.reshape((n, m, k)).transpose(0, 2, 1).reshape(n, d) for x in [h, Z_init]]
        A, h, Z = map(ensure_C, [A, h, Z])
        
        diff = _solvers.M4_plus(A, h, Z, k, eps, max_iter)
        Z = Z.reshape((n, k, m)).transpose(0, 2, 1).reshape((n, d))
        return diff, Z
        
    def __mul_S(self, s, V):
        '''
        Efficiently multiply V with s using block structure
        '''
        d = V.shape[1]
        k = s.shape[0]
        assert d % k == 0
        return (V.reshape(-1, d // k, k) @ s).reshape(V.shape)
        
    def solve_map(self, A, h, k, rounding_iters=500, max_iter=100, eps=0, returnVB=False, returnTime=False):
        '''
        Solve the MAP estimation problem.
        :param rounding_iters: number of rounding iterations, int
        :param returnVB: boolean value used for partition function estimation
        :param returnTime: boolean value used for timing expts
        '''
        n = len(A)
        k = h.shape[1]
        d = int(np.ceil(k * np.sqrt(2*n) + 1))

        while(d % k != 0):
            d += 1
        assert d >= k

        C_hat = (k/(k-1))*np.eye(k) - (1/(k-1))*np.full((k, k), 1)
        U, Sigma, Ut = np.linalg.svd(C_hat)
        s = (np.diag(Sigma) ** 0.5) @ Ut

        Z = np.abs(rand_unit_vector(n, d))
        B = get_simplex(k, d, is_prob=True)

        diff, Z = self.solve(A, h @ B, Z, k, max_iter, eps)

        V, B = self.__mul_S(s.T, Z), self.__mul_S(s.T, B)
        
        mode_x, mode_f = None, -np.inf
        t_list = []
        f_list = []
        for _ in range(rounding_iters):
            x = obtain_rounded_v(V, B)
            f = get_f(A, h, x)
            t_list.append(time.time())
            f_list.append(f)
            if f > mode_f:
                mode_x = x
                mode_f = f

        if returnVB:
            return mode_x, mode_f, V, B
        
        if returnTime:
            return mode_x, mode_f, t_list, f_list
        return mode_x, mode_f
    
    def solve_partition_function(self, A, h, k, rounding_iters=500, max_iter=100, eps=0):
        '''
        Compute the partition function from the SDP solution.
        '''
        n = A.shape[0]
        _, _, V, B = self.solve_map(A, h, k, rounding_iters=rounding_iters, max_iter=max_iter,
                                    eps=eps, returnVB=True)
        s_list = {}
        f_list = []
        for _ in range(rounding_iters):
            x = obtain_rounded_v(V, B)
            f = get_f(A, h, x)
            s = get_string_from_vector(x)
            if s not in s_list:
                s_list[s] = 1
                f_list.append(f)

        rem = np.log(1-np.exp(np.log(len(f_list))-n*np.log(k)))

        y_list = []
        while True:
            if len(y_list) >= rounding_iters: break
            x = np.random.choice(k, n, replace=True)
            s = get_string_from_vector(x)
            if s in s_list: continue
            f = get_f(A, h, x)
            log_q = -n * np.log(k) - rem
            f_minus_log_q = f - log_q
            y_list.append(f_minus_log_q)
        sm = LSE(y_list) - np.log(len(y_list))
        sm = LSE([sm]+f_list)

        return sm
    
class AISSolver(Solver):
    def __init__(self):
        '''
        Instantiates a AIS Solver.
        Example usage: solver = AISSolver()
                       solver.solve_map(A, h, k)
        '''
        pass
    
    # p(x) \propto \exp(\sum_{ij}Aij\delta(i, j)/2 + \sum_i\sum_k b_ik\delta(i, k))
    # x is a vector in [0, k-1]^{n}
    def __gibbs_sampling(self, A, h, x, temp, num_cycles=10):
        '''
        Run a Gibbs Sampling chain on x
        :param A: the coupling matrix, numpy array of dim (n,n)
        :param h: biases, numpy array of dim (n,k)
        :param x: the seed for gibbs sampling, numpy array/string of dim (n,)
        :param temp: temperature of sampling, int
        :param num_cycles: number of cycles of gibbs sampling, int
        Returns: The sample after doing num_cycles cycles of gibbs sampling
        '''
        n = len(x)
        k = h.shape[1]
        for cycle in range(num_cycles):
            for i in range(n):
                mx = -np.inf
                for j in range(k):
                    x[i] = j
                    f_j = get_f(A, h, x) / temp
                    mx = max(mx, f_j)

                denominator = 0
                for j in range(k):
                    x[i] = j
                    denominator += np.exp(get_f(A, h, x) / temp - mx)

                sm_p = 0
                p = np.random.rand()
                for j in range(k):
                    x[i] = j
                    p_j = np.exp(get_f(A, h, x) / temp - mx) / denominator
                    sm_p += p_j
                    if p < sm_p:
                        break
        return x
    
    def __log_f_t(self, x, t, inv_temps, A, h):
        '''
        Compute function value at t^th step of annealing in AIS
        '''
        n = len(x)
        k = h.shape[1]
        weight_on_uniform = (inv_temps[t] - 1) * n * np.log(k)
        f = get_f(A, h, x)
        weight_on_true = inv_temps[t] * (f)
        return weight_on_uniform + weight_on_true
    
    def solve_map(self, A, h, k, num_samples=500, T=100, num_cycles=10, returnTime=False):
        '''
        Solve the MAP estimation problem.
        :param num_samples: number of annealed samples
        :param T: number of temperatures used in annealing
        :param num_cycles: number of cycles of gibbs sampling
        :param returnTime: boolean value used for timing expts
        '''
        n = len(A)
        inv_temps = np.linspace(0, 1, T)
        mode_x, mode_f = None, -np.inf
        t_list = []
        f_list = []
        for i in range(num_samples):  
            x = np.random.choice(k, size=n, replace=True)
            for t in range(1, T):
                x = self.__gibbs_sampling(A, h, x, 1 / inv_temps[t], num_cycles=num_cycles)
                f = get_f(A, h, x)
                t_list.append(time.time())
                f_list.append(f)
                if f > mode_f:
                    mode_x = x
                    mode_f = f
        if returnTime:
            return mode_x, mode_f, t_list, f_list
        return mode_x, mode_f
        
    def solve_partition_function(self, A, h, k, num_samples=500, T=100, num_cycles=10):
        '''
        Compute the partition function via AIS.
        '''
        n = len(A)
        inv_temps = np.linspace(0, 1, T)
        log_w_list = []
        mx = -np.inf
        for i in range(num_samples):  
            x = np.random.choice(k, size=n, replace=True)
            w = 0
            for t in range(1, T):
                w = w + self.__log_f_t(x, t, inv_temps, A, h) - self.__log_f_t(x, t-1, inv_temps, A, h)
                x = self.__gibbs_sampling(A, h, x, 1 / inv_temps[t], num_cycles=num_cycles)
            log_w_list.append(w)
            mx = max(mx, w)
        log_w_list = [elem - mx for elem in log_w_list]
        logZ = mx + np.log(np.sum(np.exp(log_w_list))) - np.log(num_samples)
        return logZ
    
class ExactSolver(Solver):
    def __init__(self):
        '''
        Instantiates a Exact Solver.
        Example usage: solver = ExactSolver()
                       solver.solve_map(A, h, k)
        '''
        pass
    
    def __generate_strings(self, n, k):
        '''
        Generate all k^n strings in the support
        '''
        assert k >= 2 and k < 10
        if n == 1:
            return [str(i) for i in range(k)]
        ret = []
        all_smaller_strings = self.__generate_strings(n-1, k)
        for i in range(k):
            ret.extend([str(i) + s for s in all_smaller_strings])
        return ret

    def solve_map(self, A, h, k):
        '''
        Compute the map estimate exactly via brute force
        '''
        n = len(A)
        all_strings = self.__generate_strings(n, k)
        mode_x, mode_f = None, -np.inf
        for s in all_strings:
            f = get_f(A, h, s)
            if f > mode_f:
                mode_x = get_vector_from_string(s)
                mode_f = f
         
        return mode_x, mode_f
    
    def solve_partition_function(self, A, h, k):
        '''
        Compute the partition function exactly via brute force
        '''
        n = len(A)
        all_strings = self.__generate_strings(n, k)
        mx = -np.inf
        sm_list = []
        for s in all_strings:
            sm = get_f(A, h, s)    
            mx = max(mx, sm)
            sm_list.append(sm)
        sm_list = [elem - mx for elem in sm_list]
        logZ = np.log(np.sum(np.exp(sm_list))) + mx

        return logZ


solver_registry = {  
        'M4':           M4Solver,
        'M4+': M4PlusSolver,
        'AIS':          AISSolver,
        'Exact':        ExactSolver,
    }

def find_solver(name):
    '''
    Map solver names to solver objects
    '''
    try:
        solver = solver_registry[name]()
    except KeyError as e:
        raise KeyError('Must be one of '+', '.join(solver_registry.keys())+'.')
        
    return solver
