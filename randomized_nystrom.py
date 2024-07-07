import numpy as np
from numpy.linalg import cholesky, qr, svd, lstsq, inv

class randomized_nystrom:

    def __init__(self, comm='comm', rank='rank', size='size'):
        self.comm = comm
        self.rank = rank
        self.size = size

    def nystrom_serial_accurate(self, A, Omega1, k):
        '''
        IN :
        A : matrix to be factorized R^{m x m}
        Omega1 : sketch matrix R^{m x l}
        k : order of approximation
        OUT :
        U_hatk : approximated product between Q and approximated left
                singular vectors
        Sk : approximated diagonal matrix of singular values
        '''

        # STEP 1
        C = np.dot(A, Omega1)  # dim(m x l)
        # STEP 2
        B = np.dot(np.transpose(Omega1), C)  # dim(l x l)
        # STEP 3
        try:
            L = cholesky(B)  # dim(l x l)
            # STEP 4: since we have an overdetermined system, we cannot use the solve command,
            #       but we need the least square method
            Z = lstsq(L, np.transpose(C))[0]  # dim(l x m)
            Z = np.transpose(Z)  # dim(m x l)
        # Do SVD
        except np.linalg.LinAlgError as err:
            U_tilda, S_tilda, V_tilda = svd(B)
            L = U_tilda @ inv(np.diag(S_tilda))
            Z = lstsq(L, np.transpose(C))[0]  # dim(l x m)
            Z = np.transpose(Z)  # dim(m x l)
        # STEP 5
        Q, R = qr(Z, mode='reduced')
        # STEP 6
        U, S, _ = svd(R)
        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        # STEP 7
        Uk_hat = np.dot(Q, Uk)

        return Uk_hat, Sk @ Sk

    def nystrom_serial_faster(self, A, Omega1, k):
        '''
        IN :
        A : matrix to be factorized R^{m x m}
        Omega1 : sketch matrix R^{m x l}
        k : order of approximation
        OUT :
        U_hatk : approximated product between Z and V_k
        '''

        # STEP 1
        C = np.dot(A, Omega1)  # dim(m x l)
        # STEP 2
        B = np.dot(np.transpose(Omega1), C)  # dim(l x l)
        # STEP 3
        try:
            L = cholesky(B)  # dim(l x l)
            # STEP 4: since we have an overdetermined system, we cannot use the solve command,
            #       but we need the least square method
            Z = lstsq(L, np.transpose(C))[0]  # dim(l x m)
            Z = np.transpose(Z)  # dim(m x l)
        # Do SVD
        except np.linalg.LinAlgError as err:
            U_tilda, S_tilda, V_tilda = svd(B)
            L = U_tilda @ inv(np.diag(S_tilda))
            Z = lstsq(L, np.transpose(C))[0]  # dim(l x m)
            Z = np.transpose(Z)  # dim(m x l)

        # STEP 5
        _, R = qr(Z, mode='reduced')
        # STEP 6
        _, _, VT = svd(R)

        VTk = VT[:k, :]
        Uk_hat = Z @ np.transpose(VTk)

        return Uk_hat
