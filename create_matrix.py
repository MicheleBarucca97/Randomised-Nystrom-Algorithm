import numpy as np
from scipy.stats import wishart
from math import exp
from numpy.linalg import norm

class create_matrix:
    
    def __init__(self, n, R = 0):
        self.n = n
        self.R = R
    
    ######## Low-Rank and PSD Noise.
    def fun_LowRank(self, epsilon):
        '''
        Build matrix A such that
        A = diag(1,..., 1, 0, ..., 0) + xi/n W \in R^{n x n}
        where W is the wishart ditribution WISHART(n, n; R) distribution

        R: initial digonal values equal to 1
        epsilon: controls the signal-to-noise ratio
        '''
        A = np.zeros((self.n, self.n))
        A[:self.R, :self.R] = np.eye(self.R)         
        
        # Create a symmetric positive definite matrix G
        G = np.random.normal(loc=0, scale=1.0, size=(self.n, self.n))
        G = np.dot(G, G.T)  # Ensure symmetry
        G += np.eye(self.n)  # Add a multiple of the identity matrix to make it positive definite
        
        W = wishart.rvs(df = 2*self.n, scale=G, size = 1)
        
        A += (epsilon*self.n**(-1))*W
        
        U, S, V = np.linalg.svd(A)
        
        sigmas = S
        
        return A, sigmas

    ######## Low-Rank and PSD Noise. II version.
    def psd_Noise(self, xi, seed = 166297):
        '''
        Second way to build A such that
        A = diag(1, 1, 0, ..., 0) + xi/n W \in R^{n x n}

        R: initial digonal values equal to 1
        epsilon: controls the signal-to-noise ratio
        '''
        np.random.seed(seed)
        W = np.random.normal(loc= 0.0, scale = 1.0, size = [self.n, self.n])
        W = W@np.transpose(W)
        diagA = np.zeros((self.n))

        diagA[0:self.R] = 1
        A = np.diag(diagA) + (xi/self.n)*W
    
        return A
    
    ####### Polynomial Decay.
    def fun_PolyDecay(self, p):
        '''
        diag(1,..., 1, 2^{−p}, 3^{−p},..., (n − R + 1)^{−p}) \in R^{n x n}

        R: initial digonal values equal to 1
        epsilon: controls the signal-to-noise ratio
        '''
        # Create a vector for the diagonal values
        diagonal_values = np.array([1 if i <= self.R else (i - self.R + 1)**(-p) for i in range(self.n)])
        
        A = np.diag(diagonal_values)

        return A
    
    ####### Exponential Decay.
    def fun_expoDecay(self, q):
        '''
        diag(1,..., 1, 10^{−q}, 10^{−2q},..., 10^{−(n−R)q}) \in R^{n x n}

        R: initial digonal values equal to 1
        epsilon: controls the signal-to-noise ratio
        '''
        # Create a vector for the diagonal values
        diagonal_values = np.array([1 if i <= self.R else 10**(-(i-self.R)*q) for i in range(self.n)])
        
        A = np.diag(diagonal_values)
        
        return A

    ####### MNIST matrix.
    def buildA_sequential(self, data, c, normalize_term, save=True):
        '''
        Function to build A out of a database
        using the RBF exp( − | | x i − x j | | / c)

        normalize_term: equal to a number if you want to normalize your dataset (e.g. 255 for MNIST)
        '''
        A = np.zeros((self.n, self.n))
        for j in range(self.n):
            for i in range(j):
                A[i, j] = exp(-norm(data[i, :]/normalize_term - data[j, :]/normalize_term) ** 2 / c)
        A = A + np.transpose(A)
        np.fill_diagonal(A, 1.0)
        if save:
            A.tofile('./A.csv', sep=',', format='%10.f')
        return A
