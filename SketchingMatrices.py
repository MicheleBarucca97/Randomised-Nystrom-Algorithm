import numpy as np
from math import sqrt, log2
import torch
from hadamard_transform import hadamard_transform
import functools

def seed_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)  # Set a seed for reproducibility
        return func(self, *args, **kwargs)
    return wrapper
class SketchingMatrices:

    def __init__(self, m, l):
        self.m = m
        self.l = l

    @seed_decorator
    def Gaussian(self, seed=None):
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
        return np.random.normal(loc=0.0, scale=1.0, size=[self.m, self.l])

    # Sketch matrix subsampled random Hadamard transform (SRHT).
    @seed_decorator
    def SRHT(self, seed=None):
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
        assert log2(self.m).is_integer(), "n must be a power of 2"
        assert self.l < self.m, f'The value of l= {self.l} n= {self.m}'

        d = np.random.choice([1, -1], size=self.m)
        D = np.diag(sqrt(self.m / self.l) * d)

        P = np.random.choice(range(self.m), self.l, replace=False)
        Omega1 = D @ hadamard_transform(torch.from_numpy(D)).numpy()
        Omega1 = Omega1.transpose()
        Omega1 = Omega1[P, :]

        return Omega1.transpose()

    @seed_decorator
    def block_SRHT_serial(self, seed=None):
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
        assert log2(self.m).is_integer(), "m must be a power of 2"
        assert self.l < self.m, f'The value of l= {self.l} must be less than m= {self.m}'

        dr = np.random.choice([1, -1], size=self.m)
        Dr = np.diag(sqrt(self.m / self.l) * dr)

        P = np.random.choice(range(self.m), self.l, replace=False)
        Omega = np.array([hadamard_transform(torch.from_numpy(Dr[:, i])).numpy() for i in range(self.m)])
        Omega = Omega.transpose()
        Omega = Omega[P, :]  # dim(l x n)

        dl = np.random.choice([1, -1], size=self.l)
        Dl = np.diag(dl)
        Omega1 = Dl @ Omega  # dim(l x n)

        return np.transpose(Omega1)

    @seed_decorator
    def block_SRHT_parallel(self, comm, seed=None):
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
        np.random.seed(seed + rank)

        local_size = int(self.m / size)
        assert self.l < local_size, f'The value of local_size= {local_size} n= {self.m}'

        dr = np.zeros(self.m, dtype='int')
        dr_local = np.zeros(local_size, dtype='int')
        dl = np.random.choice([1, -1], size=self.l)
        Dl = np.diag(sqrt(self.m / (size * self.l)) * dl)

        P = np.zeros(self.l, dtype='int')

        if rank == 0:
            dr = np.random.choice([1, -1], size=self.m)
            P = np.random.choice(range(local_size), self.l, replace=False)

        comm.Scatterv(dr, dr_local, root=0)
        Dr_local = np.diag(dr_local)
        P = comm.bcast(P, root=0)

        omega = Dr_local @ hadamard_transform(torch.from_numpy(Dr_local)).numpy()
        omega = omega.transpose()
        omega = omega[P, :]

        Omega1_local = Dl @ omega

        return Omega1_local.transpose()