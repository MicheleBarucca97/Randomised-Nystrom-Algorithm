import numpy as np
import math
from numba import njit, prange

class MatrixGenerator:
    """
    Class for generating different types of matrices with specified properties.
    """
    
    def __init__(self, n, R = 0):
        """
        Initialize the MatrixGenerator.

        Parameters:
        n (int): Dimension of the square matrix.
        R (int): Number of initial diagonal values equal to 1.
        """
        self.n = n
        self.R = R

    def psd_noise_matrix(self, xi, seed=166297):
        """
        Generate a matrix with PSD noise. A = diag(1, 1, 0, ..., 0) + xi/n W in R^{n x n}

        Parameters:
        xi (float): Controls the signal-to-noise ratio.
        seed (int): Random seed for reproducibility.

        Returns:
        ndarray: Generated matrix A.
        """
        np.random.seed(seed)
        W = np.random.normal(loc=0.0, scale=1.0, size=(self.n, self.n))
        W = W @ W.T

        diagA = np.zeros(self.n)
        diagA[:self.R] = 1
        A = np.diag(diagA) + (xi / self.n) * W

        return A

    def polynomial_decay_matrix(self, p):
        """
        Generate a matrix with polynomial decay on the diagonal.
        A = diag(1,..., 1, 2^{−p}, 3^{−p},..., (n − R + 1)^{−p}) in R^{n x n}

        Parameters:
        p (float): Exponent for polynomial decay.

        Returns:
        ndarray: Generated matrix A.
        """
        diagonal_values = np.array([1 if i < self.R else (i - self.R + 1) ** -p for i in range(self.n)])
        A = np.diag(diagonal_values)
        return A

    def exponential_decay_matrix(self, q):
        """
        Generate a matrix with exponential decay on the diagonal.
        A = diag(1,..., 1, 10^{−q}, 10^{−2q},..., 10^{−(n−R)q}) in R^{n x n}

        Parameters:
        q (float): Exponent for exponential decay.

        Returns:
        ndarray: Generated matrix A.
        """
        diagonal_values = np.array([1 if i < self.R else 10 ** (-(i - self.R) * q) for i in range(self.n)])
        A = np.diag(diagonal_values)
        return A

    def build_sequential_matrix(self, data, c, normalize_term, save=True):
        """
        Build a matrix from a dataset using RBF.

        Parameters:
        data (ndarray): Dataset to build the matrix from.
        c (float): Variance parameter for RBF.
        normalize_term (float): Normalization term for the dataset.
        save (bool): Whether to save the matrix to a file.

        Returns:
        ndarray: Generated matrix A.
        """
        A = np.zeros((self.n, self.n))
        for j in range(self.n):
            for i in range(j):
                A[i, j] = np.exp(-np.linalg.norm(data[i, :] / normalize_term - data[j, :] / normalize_term) ** 2 / c)
        A = A + np.transpose(A)
        np.fill_diagonal(A, 1.0)
        if save:
            A.tofile('./A.csv', sep=',', format='%10.f')
        return A

    def build_numba_matrix(self, data, c, normalize_term, save=True):
        """
        Build a matrix from a dataset using RBF.

        Parameters:
        data (ndarray): Dataset to build the matrix from.
        c (float): Variance parameter for RBF.
        normalize_term (float): Normalization term for the dataset.
        save (bool): Whether to save the matrix to a file.

        Returns:
        ndarray: Generated matrix A.
        """
        data_normalized = data / normalize_term
        A = compute_rbf_matrix(data_normalized, c, self.n)
        np.fill_diagonal(A, 1.0)
        if save:
            np.savetxt('A.csv', A, delimiter=',', fmt='%10.5f')
        return A

@njit(cache=True, fastmath=True)
def compute_rbf_matrix(data, c, n):
    """
    Compute the RBF matrix.

    Parameters:
    data (ndarray): Normalized dataset.
    c (float): Variance parameter for RBF.
    n (int): Dimension of the square matrix.

    Returns:
    ndarray: RBF matrix.
    """
    A = np.zeros((n, n))
    for j in prange(n):
        for i in range(j):
            diff = data[i] - data[j]
            norm_squared = np.dot(diff, diff)
            A[i, j] = math.exp(-norm_squared / c)
    A += A.T

    return A
