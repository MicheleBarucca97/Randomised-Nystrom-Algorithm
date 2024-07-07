from mpi4py import MPI
import numpy as np
import math


class TSQR:
    
    # Constructor    
    def __init__(self, comm, A_local):
        self.comm = comm
        self.A_local = A_local

    def subdivide_matrix_between_processors(self):
        '''
        Function to create the list of Qs for each processor.
        '''
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        self.comm.Barrier()

        local_size = self.A_local.shape[0]
        n = self.A_local.shape[1]

        assert n < local_size, f'The value of l= {n} local_size= {local_size}'
        
        # Vector which will contain all the matrices Q
        Q_list_local = []

        Q_local, R_local = np.linalg.qr(self.A_local, mode='reduced')
        Q_list_local.append(Q_local)

        # If you want to run with 16 processors is range(1, math.ceil(math.log(self.size)) + 2)
        end = math.ceil(math.log(size)) + (2 if size >= 16 else 1)

        for k in range(1, end):
            sqrt_size = int(size / (math.ceil(size / 2) / 2 ** (k - 1)))

            # Synchronize all processes at this point
            self.comm.Barrier()
            color = rank // sqrt_size
            key = rank % sqrt_size
            comm_branch = self.comm.Split(color=color, key=key)
            rank_branch = comm_branch.Get_rank()

            R_local_receive = comm_branch.bcast(R_local, root=sqrt_size // 2)

            if rank_branch == 0:
                R_local = np.vstack((R_local, R_local_receive))
                Q_local, R_local = np.linalg.qr(R_local, mode='reduced')
                Q_list_local.append(Q_local)

            comm_branch.Free()

        return Q_list_local, R_local

    def getQexplicitly(self):
        '''
        Get the matrix Q explicitly
        '''
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        wt = MPI.Wtime()  # We are going to time this
        localQks, R = self.subdivide_matrix_between_processors()
        self.comm.Barrier()
        n = localQks[0].shape[1]
        Q = None
        if rank == 0:
            Q = np.eye(n, n)
            Q = localQks[-1] @ Q  # dim(2n * n)
            localQks.pop()  # remove last element of the list

        # Start iterating through the tree backwards
        # If you want to run with 16 processors is range(math.ceil(math.log(self.size)), -1, -1)
        end = math.ceil(math.log(size)) - (2 if size >= 16 else 1)
        for k in range(end, -1, -1):
            color = rank % (2 ** k)
            key = rank // (2 ** k)
            comm_branch = self.comm.Split(color=color, key=key)

            # print("k: ", k, "Rank: ", rank, " color: ", color, " new rank: ", rank_branch)
            # I enter only with color 0 because e.g. the rank 0 in color 1 doesn't know Q
            if color == 0:
                # We scatter the columns of the Q we have
                Qrows = np.empty((n, n), dtype='d')
                comm_branch.Scatterv(Q, Qrows, root=0)
                # Local multiplication
                Qlocal = localQks[-1] @ Qrows
                localQks.pop()
                # Gather
                Q = comm_branch.gather(Qlocal, root=0)
                if rank == 0:
                    Q = np.concatenate(Q, axis=0)

            comm_branch.Free()

        if rank == 0:
            wt = MPI.Wtime() - wt
            print("Time to do the QR: ", wt)
            return Q, R
        else:
            return None, None
     