import numpy as np
from math import sqrt, log2
import torch
from hadamard_transform import hadamard_transform


class sketching_matrices:

    def __init__(self):
        pass  # This is an empty default constructor

    def Gaussian(self, m, n, l, seed):
        '''
        Build Omega1
        IN :
        m : number of rows of starting matrix A
        n : number of columns of starting matrix A
        l : sketch dimension
        seed: for reproducibility
        OUT :
        Omega1 : SRHT sketching matrix in R^{m,l}
        '''
        np.random.seed(seed)  # Set a seed for reproducibility
        Omega1 = np.random.normal(loc=0.0, scale=1.0, size=[m, l])
        return Omega1

    # Sketch matrix subsampled random Hadamard transform (SRHT).
    def SRHT(self, m, n, l, seed):
        '''
        Build Omega1
        IN :
            m : number of rows of starting matrix A
            n : number of columns of starting matrix A
            l : sketch dimension
            seed: for reproducibility
        OUT :
            Omega1 : SRHT sketching matrix in R^{m,l}
        '''
        # Check that n=2^q otherwise lunch an asser
        number_str = str(log2(m))
        decimal_index = number_str.find('.')
        assert all(char == '0' for char in number_str[decimal_index + 1]), "n must be a power of 2"

        assert l < m, f'The value of l= {l} n= {m}'

        np.random.seed(seed)  # Set a seed for reproducibility
        d = np.array([1 if np.random.rand() < 0.5 else -1 for i in range(m)])
        D = np.diag(sqrt(m / l) * d)
        P = np.random.choice(range(m), l, replace=False)
        Omega1 = D
        Omega1 = np.array([hadamard_transform(torch.from_numpy(Omega1[:, i])).numpy() for i in range(m)])
        Omega1 = np.transpose(Omega1)
        Omega1 = Omega1[P, :]  # dim(l x n)

        return np.transpose(Omega1)

    def block_SRHT_serial(self, m, n, l, seed):
        '''
        Build Omega1
        IN :
        m : number of rows of starting matrix A
        n : number of columns of starting matrix A
        l : sketch dimension
        seed: for reproducibility
        OUT :
        Omega1 : SRHT sketching matrix in R^{m,l}
        '''
        # Check that n=2^q otherwise lunch an asser
        number_str = str(log2(m))
        decimal_index = number_str.find('.')
        assert all(char == '0' for char in number_str[decimal_index + 1]), "n must be a power of 2"

        assert l < m, f'The value of l= {l} n= {m}'

        np.random.seed(seed)  # Set a seed for reproducibility
        dr = np.array([1 if np.random.rand() < 0.5 else -1 for i in range(m)])
        Dr = np.diag(sqrt(m / l) * dr)
        P = np.random.choice(range(m), l, replace=False)
        Omega = Dr
        Omega = np.array([hadamard_transform(torch.from_numpy(Omega[:, i])).numpy() for i in range(m)])
        Omega = np.transpose(Omega)
        Omega = Omega[P, :]  # dim(l x n)
        dl = np.array([1 if np.random.rand() < 0.5 else -1 for i in range(l)])
        Dl = np.diag(dl)
        Omega1 = Dl@Omega  # dim(l x n)

        return np.transpose(Omega1)

    def block_SRHT_parallel(self, comm, m, n, l, seed):
        '''
        Build Omega1
        IN :
        comm, rank, size : for parallelism
        m : number of rows of starting matrix A
        n : number of columns of starting matrix A
        l : sketch dimension
        seed: for reproducibility
        OUT :
        Omega1_local : SRHT sketching matrix in R^{m/size,l}
        '''
        rank = comm.Get_rank()
        size = comm.Get_size()
        np.random.seed(seed+rank)  # Set a seed for reproducibility

        local_size = int(m / size)
        assert l < local_size, f'The value of local_size= {local_size} n= {m}'

        dr = np.zeros(m, dtype='int')
        dr_local = np.zeros(local_size, dtype='int')
        dl = np.array([1 if np.random.rand() < 0.5 else -1 for i in range(l)])
        Dl = np.diag(np.sqrt(m / (size * l)) * dl)

        P = np.zeros(l, dtype='int')

        # Begin with the compressed problem
        if rank == 0:
            dr = np.array([1 if np.random.rand() < 0.5 else -1 for i in range(m)])
            P = np.random.choice(range(local_size), l, replace=False)

        comm.Scatterv(dr, dr_local, root=0)
        Dr_local = np.diag(dr_local)
        P = comm.bcast(P, root=0)

        omega = Dr_local
        omega = np.array([hadamard_transform(torch.from_numpy(omega[:, i])).numpy() for i in
                          range(local_size)])  # dim(m/p x m/p)
        omega = np.transpose(omega)
        omega = omega[P, :]  # dim(l x m/p)

        Omega1_local = Dl @ omega  # dim(l x m/p)

        return np.transpose(Omega1_local)
