import numpy as np
from numpy.linalg import norm, cholesky, solve, qr, svd, lstsq, inv
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import time

# My classes
from MatrixGenerator import MatrixGenerator
from SketchingMatrices import SketchingMatrices
from randomized_nystrom import randomized_nystrom
from TSQR import TSQR

# To ignore all warnings
import warnings
warnings.filterwarnings("ignore")

def plot_singularValue(A, name_matrix):
    '''
    INPUT
    A: matrix of which you want to plot the singular values
    name_matrix: string containing the name of the matrix
    '''
    _, S, _ = np.linalg.svd(A)
    x_axis = range(A.shape[0])

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_axis, S, c="#003aff", marker='o', label="$\\sigma$")
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
            # We know that the first digit is the label
            labels[i] = int(l[0])
            indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()[2:]]
            # Separate indices and values
            indices, values = zip(*[(int(j), v) for j, v in indices_values])
            # Fill in the values at the specified indices
            data[i, indices] = values

        return data, labels
    else:
        # Open the file in read mode
        with open(filename, 'r') as file:
            features = [
                [float(term.split(':')[1]) for term in line.split()[1:]] for line in file]
        data = np.array(features)

        return data

def load_and_process_data(dataset_name):
    # Configuration for different datasets
    datasets = {
        "YearPredictionMSD": {
            "c": 10 ** 8,
            "filename": "dataset/YearPredictionMSD_4096",
            "non_homogeneous": None,
            "build_method": "build_numba_matrix",
            "additional_params": [1]
        },
        "MNIST": {
            "c": 10 ** 4,
            "filename": "dataset/mnist_not_scaled_4096",
            "non_homogeneous": 1,
            "build_method": "build_sequential_matrix",
            "additional_params": [255]
        }
    }

    config = datasets[dataset_name]
    filename = config["filename"]
    c = config["c"]

    # Read data with or without non_homogeneous parameter
    if config["non_homogeneous"] is not None:
        data = readData(filename, non_homogeneous=config["non_homogeneous"])
    else:
        data = readData(filename)

    m = n = data.shape[0]
    fun = MatrixGenerator(n)

    # Dynamically call the appropriate build method
    build_method = getattr(fun, config["build_method"])
    A = build_method(data, c, *config["additional_params"], save=False)

    return m, n, A


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ls : sketching matrix dimension (embedding of a high dimensional subspace into a low dimensional one, while
# preserving some geometry, with high probability)
ls = [600, 1000, 2000]
# ks : low rank-k approximation
ks = [100, 200, 300, 400, 500, 600]
# err_cN2 : contains the error made by considering the different combinations of ls and ks
err_cN2 = np.zeros((len(ks), 1))

A = None
Omega1 = None
Omega1T = None
m = n = 0

if rank == 0:
    dataset_name = "YearPredictionMSD"  # or "MNIST"
    m, n, A = load_and_process_data(dataset_name)

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

# Scatter matrix along the rows
sub_matrix = np.empty((n_blocks, n), dtype='d')

if color_col == 0:
    comm_col.Scatterv(A, sub_matrix, root=0)

# Scatter matrix along the columns
# Initialize the transpose of the local matrix of dim(n/p x n/p)
receiveMat = np.empty((n_blocks * n_blocks), dtype='d')

# Transpose and ravel the local sub-matrix
submatrixT = np.ravel(sub_matrix.T)
comm_raw.Scatterv(submatrixT, receiveMat, root=0)

# recompose the matrix
receiveMat = np.split(receiveMat, n_blocks)
blockMatrix = np.array([np.ravel(arr, order='F') for arr in receiveMat]).T

# Create a figure and axis
if rank == 0:
    fig, ax = plt.subplots()

# Flag to test parallel version of the sketching matrix
use_parallel_sketching = 1
for i in range(len(ls)):
    l = ls[i]
    sketch = SketchingMatrices(m, l)
    # Parallel
    if size > 1:
        Omega1_local = np.empty((n_blocks, l), dtype='d')
        Omega1T_local = np.empty((n_blocks, l), dtype='d')
        wt = MPI.Wtime()  # We are going to time this

        # Retrieve sketching matrix
        if use_parallel_sketching:
            if color_raw == 0:
                Omega1_local = sketch.block_SRHT_parallel(comm_raw, m, n, l, seed=123445)
        else:
            if rank == 0:
                Omega1 = sketch.Gaussian(m, n, l, seed=123445)
                arrs = np.split(Omega1, n, axis=0)
                raveled = [np.ravel(arr) for arr in arrs]
                Omega1T = np.concatenate(raveled)

            # Scatter Omega1 along rows
            if color_raw == 0:
                comm_raw.Scatterv(Omega1T, Omega1_local, root=0)

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
            Q, R = tsqr.getQexplicitly()
            if rank == 0:
                # STEP 6
                U, S, _ = svd(R)
                wt3 = MPI.Wtime() - wt
                for j in range(len(ks)):
                    wt_cycle = MPI.Wtime()
                    k = ks[j]
                    Uk = U[:, :k]
                    Sk = np.diag(S[:k])
                    # STEP 7
                    Uk_hat = np.dot(Q, Uk)
                    Sk_square = Sk@Sk
                    wt4 = MPI.Wtime() - wt_cycle
                    print(f"Time to run Nystrom algorithm in parallel for k= {k}: {wt3 + wt4}")
                    err_cN2[j] = norm(A - Uk_hat @ Sk_square @ np.transpose(Uk_hat), 'nuc') / norm(A, 'nuc')
                ax.loglog(ks, err_cN2, '*-', label=f'l= {l}')
    # Serial
    else:
        # get the start time
        st = time.time()
        Omega1 = sketch.Gaussian(m, n, l, seed=123445)
        nyst_algo = randomized_nystrom()
        et = time.time()
        for j in range(len(ks)):
            st_cycle = time.time()
            k = ks[j]
            Uk_hat, Sk_square = nyst_algo.nystrom_serial_accurate(A, Omega1, k)
            et_cycle = time.time()
            elapsed_time = (et - st) + (et_cycle - st_cycle)
            print(f"Time to run Nystrom algorithm in series for k= {k}: {elapsed_time}")
            err_cN2[j] = norm(A - Uk_hat @ np.transpose(Uk_hat), 'nuc') / norm(A, 'nuc')

        ax.loglog(ks, err_cN2, '*-', label=f'l= {l}')

if rank == 0:
    ax.set_xlabel('k')
    ax.set_ylabel('Trace relative error')
    ax.legend()
    plt.show()
