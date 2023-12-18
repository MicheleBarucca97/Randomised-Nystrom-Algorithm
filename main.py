import numpy as np
from numpy.linalg import norm, cholesky, solve, qr, svd, lstsq, inv
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import time

# My classes
from create_matrix import create_matrix
from sketching_matrices import sketching_matrices
from randomized_nystrom import randomized_nystrom
from TSQR import TSQR

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")

def plot_singularValue(A, name_matrix):
    U, S, V = np.linalg.svd(A)

    singular_values = S

    n = A.shape[0]

    x_axis = range(n)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_axis, singular_values, c="#003aff", marker='o', label="$\sigma $")
    plt.legend()
    plt.title(r'Singular value of ' + name_matrix)
    plt.xlabel("n")
    plt.ylabel(r'$\sigma$')
    Name = r'singular values ' + name_matrix
    plt.tight_layout()
    plt.savefig(Name, dpi=80)

def readData(filename, non_homogeneous=0, size=784):
    '''
    INPUT
    filename: nome file
    inhomogeneous: if =1 then for non-homogeneous data, =0 otherwise
    size: dimension of the feature (MNIST = 784, YearPrediction = 90)
    '''
    if non_homogeneous:
        dataR = pd.read_csv(filename, sep=',', header=None)
        n = len(dataR)
        data = np.zeros((n, size))
        labels = np.zeros((n, 1))
        # Format accordingly
        for i in range(n):
            l = dataR.iloc[i, 0]
            labels[i] = int(l[0])  # We know that the first digit is the label
            l = l[2:]
            indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()]
            # Separate indices and values
            indices, values = zip(*indices_values)
            indices = [int(i) for i in indices]
            # Fill in the values at the specified indices
            data[i, indices] = values

        return data, labels
    else:
        features = []
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read the file line by line
            for line in file:
                # Split the line by space
                parts = line.split()
                # Exclude the first element (term)
                remaining_terms = parts[1:]
                # Split each remaining term by ':', and collect the second values
                second_values = [float(term.split(':')[1]) for term in remaining_terms]
                # Append the second values to the all_rows list
                features.append(second_values)
        data = np.array(features)

        return data

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ls = [600, 1000, 2000]
ks = [100, 200, 300, 400, 500, 600]
err_cN2 = np.zeros((len(ks), 1))

A = None
Omega1 = None
Omega1T = None
m = n = 0

if rank == 0:
    # YearPredictionMSD
    c = 10 ** 8
    filename = "dataset/YearPredictionMSD_8192"
    data = readData(filename)

    # # MNIST
    # c = 10**4
    # filename = "dataset/mnist_not_scaled_4096"
    # data, labels = readData(filename, non_homogeneous=1)

    m = n = data.shape[0]
    fun = create_matrix(data.shape[0])
    # YearPredictionMSD
    A = fun.buildA_sequential(data, c, 1, save = False)
    # # MNIST
    # A = fun.buildA_sequential(data, c, 255, save = False)

m = comm.bcast(m, root=0)
n = comm.bcast(n, root=0)
# Initialize comm_raw and comm_col in order to distribute by blocks A
color_raw = rank // int(sqrt(size))
key_raw = rank % int(sqrt(size))

color_col = rank % int(sqrt(size))
key_col = rank // int(sqrt(size))

comm_raw = comm.Split(color=color_raw, key=key_raw)
rank_raw = comm_raw.Get_rank()
size_raw = comm_raw.Get_size()

comm_col = comm.Split(color=color_col, key=key_col)
rank_col = comm_col.Get_rank()
size_col = comm_col.Get_size()

n_blocks = int(n / int(sqrt(size)))

# Scatter matrix along the raws
submatrix = np.empty((n_blocks, n), dtype='d')

if color_col == 0:
    comm_col.Scatterv(A, submatrix, root=0)

# Scatter matrix along the columns
# initialize the transpose of the local matrix of dim(n/p x n/p)
blockMatrix = np.empty((n_blocks, n_blocks), dtype='d')
receiveMat = np.empty((n_blocks * n_blocks), dtype='d')
# do the transpose of the matrix previously calculated dim(n/p x n)
arrs = np.split(submatrix, n, axis=1)
raveled = [np.ravel(arr) for arr in arrs]
submatrixT = np.concatenate(raveled)
comm_raw.Scatterv(submatrixT, receiveMat, root=0)
# recompose the matrix
subArrs = np.split(receiveMat, n_blocks)
blockMatrix = np.transpose(np.array([np.ravel(arr, order='F') for arr in subArrs]))
# print("rank_raw: ", rank_raw, "blockMatrix", blockMatrix)

# Create a figure and axis
# if rank == 0:
#     fig, ax = plt.subplots()

sketch = sketching_matrices()
for i in range(len(ls)):
    l = ls[i]
    if size > 1:
        Omega1_local = np.empty((n_blocks, l), dtype='d')
        Omega1T_local = np.empty((n_blocks, l), dtype='d')
        wt = MPI.Wtime()  # We are going to time this
        # Retrive sketching matrix
        # if rank == 0:
        #     Omega1 = sketch.Gaussian(m, n, l, seed=123445)
        #     arrs = np.split(Omega1, n, axis=0)
        #     raveled = [np.ravel(arr) for arr in arrs]
        #     Omega1T = np.concatenate(raveled)
        #
        # # Scatter Omega1 along rows
        # if color_raw == 0:
        #     comm_raw.Scatterv(Omega1T, Omega1_local, root=0)

        if color_raw == 0:

            Omega1_local = sketch.block_SRHT_parallel(comm_raw, m, n, l, seed=123445)

        Omega1_local = comm_col.bcast(Omega1_local, root=0)

        if rank == 0:
            wt2 = MPI.Wtime() - wt
            print("Time to compute B and C: ", wt2)

        local_mult = blockMatrix @ Omega1_local  # dim(n/p x l)

        C_local = np.empty((n_blocks, l), dtype='d')
        comm_raw.Reduce(local_mult, C_local, op=MPI.SUM, root=0)

        root_to_consider = 0
        if rank_raw == 0:
            root_to_consider = int(rank / int(sqrt(size)))

        root_to_consider = comm_raw.bcast(root_to_consider, root=0)

        Omega1_local = comm_raw.bcast(Omega1_local, root=root_to_consider)
        local_mult2 = np.empty((l, l), dtype='d')
        if rank_raw == 0:  # so e.g. for proc 0 and 2 if we are working with 4 processors
            local_mult2 = np.transpose(Omega1_local) @ C_local

            B = np.empty((l, l), dtype='d')
            comm_col.Allreduce(local_mult2, B, op=MPI.SUM)



            # STEP 3
            try:
                L = cholesky(B)  # dim(l x l)
                # STEP 4: since we have an overdetermined system, we cannot use the solve command,
                #       but we need the least square method
                Z_local = lstsq(L, np.transpose(C_local))[0]  # dim(l x m/sqrt(p))
                Z_local = np.transpose(Z_local)  # dim(m/sqrt(p) x l)
            # Do SVD
            except np.linalg.LinAlgError as err:
                U_tilda, S_tilda, V_tilda = svd(B)
                L = U_tilda @ inv(np.diag(S_tilda))
                Z_local = lstsq(L, np.transpose(C_local))[0]  # dim(l x m/sqrt(p))
                Z_local = np.transpose(Z_local)  # dim(m/sqrt(p) x l)
            # STEP 5
            tsqr = TSQR(comm_col, Z_local)
            # _, R = tsqr.subdivide_matrix_between_processors()
            # Z = comm_col.gather(Z_local, root=0)
            # if rank == 0:
            #     Z = np.concatenate(Z)
            #     # STEP 6
            #     _, _, VT = svd(R)
            Q, R = tsqr.getQexplicitly()
            if rank == 0:
                # STEP 6
                U, S, _ = svd(R)
                wt3 = MPI.Wtime() - wt
                for j in range(len(ks)):
                    wt_cycle = MPI.Wtime()
                    k = ks[j]
                    # VTk = VT[:k, :]
                    # # STEP 7
                    # Uk_hat = Z @ np.transpose(VTk)
                    # wt4 = MPI.Wtime() - wt_cycle
                    # print(f"Time to run Nystrom algorithm in parallel for k= {k}: {wt3+wt4}")
                    # err_cN2[j] = norm(A - Uk_hat @ np.transpose(Uk_hat), 'nuc') / norm(A, 'nuc')
                    Uk = U[:, :k]
                    Sk = np.diag(S[:k])
                    # STEP 7
                    Uk_hat = np.dot(Q, Uk)
                    Sk_square = Sk@Sk
                    wt4 = MPI.Wtime() - wt_cycle
                    print(f"Time to run Nystrom algorithm in parallel for k= {k}: {wt3 + wt4}")
                    # err_cN2[j] = norm(A - Uk_hat @ Sk_square @ np.transpose(Uk_hat), 'nuc') / norm(A, 'nuc')
                # ax.loglog(ks, err_cN2, '*-', label=f'l= {l}')
    else:
        # get the start time
        st = time.time()
        sketch = sketching_matrices()
        Omega1 = sketch.block_SRHT_serial(m, n, l, seed=123445)
        nyst_algo = randomized_nystrom()
        et = time.time()
        for j in range(len(ks)):
            st_cycle = time.time()
            k = ks[j]
            Uk_hat, Sk_square = nyst_algo.nystrom_serial_accurate(A, Omega1, k)
            # Uk_hat = nyst_algo.nystrom_serial_faster(A, Omega1, k)
            et_cycle = time.time()
            elapsed_time = (et - st) + (et_cycle - st_cycle)
            print(f"Time to run Nystrom algorithm in series for k= {k}: {elapsed_time}")
            # err_cN2[j] = norm(A - Uk_hat @ np.transpose(Uk_hat), 'nuc') / norm(A, 'nuc')

        # ax.loglog(ks, err_cN2, '*-', label=f'l= {l}')

# if rank == 0:
#     ax.set_xlabel('k')
#     ax.set_ylabel('Trace relative error')
#     ax.legend()
#
#     plt.show()
