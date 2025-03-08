import pytest
import os
import numpy as np
from numba import cuda

from matmul import matmul, matmul_numba_gpu
from matmul.utils import create_block

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI


try:
    mpi_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
except KeyError:
    mpi_size = 0


@pytest.mark.skipif(mpi_size < 2,reason='Not running MPI or running just one task')
def test_parallel_cpu():
    SIZE = 50

    # Initialise MPI with multithreading enabled and share work among processes
    status = MPI.Init_thread(MPI.THREAD_FUNNELED)
    if status != MPI.THREAD_FUNNELED:
        print("Unable to provide required thread level")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    rest = SIZE%npes
    n_loc = SIZE//npes + (rank < rest)

    workloads = np.array([SIZE//npes + (i<rest) for i in range(npes)], dtype=int)

    row_offset = np.cumsum(workloads)[rank-1] if rank > 0 else 0

    # initialise matrices somehow
    A = np.arange(1, SIZE*n_loc + 1, dtype=np.float64).reshape((n_loc,SIZE)) + (row_offset * SIZE)
    B = np.zeros((n_loc,SIZE), dtype=np.float64)
    C = np.zeros((n_loc,SIZE), dtype=np.float64)
    for i in range(n_loc):
        B[i, i+row_offset] = 1

    # Compute quantities for Allgatherv and allocate required memory
    ncols = workloads[0]
    rcvcounts = workloads*ncols
    displacements = np.cumsum(rcvcounts) - rcvcounts

    B_block = np.empty((n_loc,ncols), dtype=np.float64)
    B_col = np.empty((SIZE,ncols), dtype=np.float64)

    start = 0
    for i in range(npes):
        # Recompute stuff for Algatherv at some point if needed (because of different workloads)
        if i == rest:
            ncols = workloads[i]
            rcvcounts = workloads*ncols
            displacements = np.cumsum(rcvcounts) - rcvcounts

            B_block = np.empty((n_loc,ncols), dtype=np.float64)
            B_col = np.empty((SIZE,ncols), dtype=np.float64)

        # create a contiguous block from B to communicate
        create_block(B, B_block, start, ncols)
        # gather all pieces of B from other processes
        comm.Allgatherv([B_block, MPI.DOUBLE], [B_col, rcvcounts,displacements, MPI.DOUBLE])

        # multiply
        matmul(A,B_col,C[:,start:start+ncols],None)

        start += ncols



    rcvcounts = workloads*SIZE
    displacements = np.cumsum(rcvcounts) - rcvcounts
    if rank == 0:
        A_tot = np.arange(1, SIZE*SIZE + 1, dtype=np.float64).reshape((SIZE,SIZE))
    C_tot = np.zeros((SIZE,SIZE))
    comm.Gatherv([C, MPI.DOUBLE], [C_tot, rcvcounts, displacements, MPI.DOUBLE])

    if rank == 0:
        assert np.allclose(A_tot,C_tot)

    comm.Barrier()

    if (not cuda.is_available()):
        MPI.Finalize()


@pytest.mark.skipif(mpi_size < 2,reason='Not running MPI or running just one task')
@pytest.mark.skipif((not cuda.is_available()), reason='Could not find any CUDA GPU')
def test_parallel_gpu():
    SIZE = 50

    if (not MPI.Is_initialized()) and (not MPI.Is_finalized()):
        # Initialise MPI with multithreading enabled and share work among processes
        status = MPI.Init_thread(MPI.THREAD_FUNNELED)
        if status != MPI.THREAD_FUNNELED:
            print("Unable to provide required thread level")


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    rest = SIZE%npes
    n_loc = SIZE//npes + (rank < rest)

    workloads = np.array([SIZE//npes + (i<rest) for i in range(npes)], dtype=int)

    row_offset = np.cumsum(workloads)[rank-1] if rank > 0 else 0

    # initialise matrices somehow
    A = np.arange(1, SIZE*n_loc + 1, dtype=np.float64).reshape((n_loc,SIZE)) + (row_offset * SIZE)
    B = np.zeros((n_loc,SIZE), dtype=np.float64)
    C = np.zeros((n_loc,SIZE), dtype=np.float64)
    for i in range(n_loc):
        B[i, i+row_offset] = 1

    # Compute quantities for Allgatherv and allocate required memory
    ncols = workloads[0]
    rcvcounts = workloads*ncols
    displacements = np.cumsum(rcvcounts) - rcvcounts

    B_block = np.empty((n_loc,ncols), dtype=np.float64)
    B_col = np.empty((SIZE,ncols), dtype=np.float64)

    # Select a GPU and move arrays to device
    num_devices = len(cuda.gpus)
    cuda.select_device(rank%num_devices)
    a_d = cuda.to_device(A)
    c_d = cuda.to_device(C)

    nthreads = 16
    blocks_per_grid = ((n_loc + nthreads-1)//nthreads,(ncols + nthreads-1)//nthreads)
    threads_per_block = (nthreads, nthreads)

    start = 0
    for i in range(npes):
        # Recompute stuff for Algatherv at some point if needed (because of different workloads)
        if i == rest:
            ncols = workloads[i]
            rcvcounts = workloads*ncols
            displacements = np.cumsum(rcvcounts) - rcvcounts

            B_block = np.empty((n_loc,ncols), dtype=np.float64)
            B_col = np.empty((SIZE,ncols), dtype=np.float64)

            blocks_per_grid = ((n_loc + nthreads-1)//nthreads,(ncols + nthreads-1)//nthreads)

        # create a contiguous block from B to communicate
        create_block(B, B_block, start, ncols)
        # gather all pieces of B from other processes
        comm.Allgatherv([B_block, MPI.DOUBLE], [B_col, rcvcounts,displacements, MPI.DOUBLE])

        # move slice of B to device
        b_d = cuda.to_device(B_col)

        # multiply
        matmul_numba_gpu[blocks_per_grid, threads_per_block](a_d,b_d,c_d[:,start:start+ncols])

        start += ncols

    C = c_d.copy_to_host()

    rcvcounts = workloads*SIZE
    displacements = np.cumsum(rcvcounts) - rcvcounts
    if rank == 0:
        A_tot = np.arange(1, SIZE*SIZE + 1, dtype=np.float64).reshape((SIZE,SIZE))
    C_tot = np.zeros((SIZE,SIZE))
    comm.Gatherv([C, MPI.DOUBLE], [C_tot, rcvcounts, displacements, MPI.DOUBLE])

    if rank == 0:
        assert np.allclose(A_tot,C_tot)

    comm.Barrier()

    MPI.Finalize()
