from functools import wraps
import warnings
import numpy as np
from numba import cuda

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

from matmul.utils import create_block, read_config, custom_warning
import argparse
import importlib

try:
    from line_profiler import profile
except ModuleNotFoundError:
    warnings.formatwarning = custom_warning
    warnings.warn("Did not find line_profiler. Please install it to access profiling information.")
    def profile(f,*args,**kwargs):
        def wrapper(*args,**kwargs):
            f(*args,**kwargs)
        return wrapper

@profile
def main_cpu(params: dict):
    SIZE = params["size"]
    md = importlib.import_module("matmul")
    routine = getattr(md, params["function"]["routine"])
    bs = params["function"]["block_size"]

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
    A = np.arange(1, SIZE*n_loc + 1, dtype=np.float64).reshape((n_loc,SIZE),order='C') + (row_offset * SIZE)
    B = np.zeros((n_loc,SIZE), dtype=np.float64,order='C')
    C = np.zeros((n_loc,SIZE), dtype=np.float64,order='C')
    for i in range(n_loc):
        B[i, i+row_offset] = 1

    # Compute quantities for Allgatherv and allocate required memory
    ncols = workloads[0]
    rcvcounts = workloads*ncols
    displacements = np.cumsum(rcvcounts) - rcvcounts

    B_block = np.empty((n_loc,ncols), dtype=np.float64,order='C')
    B_col = np.empty((SIZE,ncols), dtype=np.float64,order='C')

    t_tot = 0
    start = 0
    for i in range(npes):
        # Recompute stuff for Algatherv at some point if needed (because of different workloads)
        if i == rest:
            ncols = workloads[i]
            rcvcounts = workloads*ncols
            displacements = np.cumsum(rcvcounts) - rcvcounts

            B_block = np.empty((n_loc,ncols), dtype=np.float64,order='C')
            B_col = np.empty((SIZE,ncols), dtype=np.float64,order='C')

        # create a contiguous block from B to communicate
        create_block(B, B_block, start, ncols)
        # gather all pieces of B from other processes
        comm.Allgatherv([B_block, MPI.DOUBLE], [B_col, rcvcounts,displacements, MPI.DOUBLE])

        t1 = MPI.Wtime()
        # multiply
        routine(A,B_col,C[:,start:start+ncols],bs)
        t2 = MPI.Wtime()
        t_tot += (t2-t1)

        start += ncols

    print(t_tot)

    if params["print"]:
        if rank == 0:
            print(C)
            for i in range(1,npes):
                block = np.zeros((workloads[i], SIZE))
                block = comm.recv(source=i,tag=i)
                print(block)
        else:
            comm.send(C,dest=0,tag=rank)


    MPI.Finalize()

@profile
def main_gpu(params: dict):
    SIZE = params["size"]
    md = importlib.import_module("matmul")
    routine = getattr(md, params["function"]["routine"])
    bs = params["function"]["block_size"]

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

    # each process at each step computes a block of C of size n_loc x ncols
    # we set parameters for the kernel accordingly
    nthreads = bs
    blocks_per_grid = ((n_loc + nthreads-1)//nthreads,(ncols + nthreads-1)//nthreads)
    threads_per_block = (nthreads, nthreads)

    t_tot = 0
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

        t1 = cuda.event(timing=True)
        t2 = cuda.event(timing=True)
        t1.record()
        # multiply
        routine[blocks_per_grid, threads_per_block](a_d,b_d,c_d[:,start:start+ncols])
        t2.record()
        t2.synchronize()

        t_tot += (cuda.event_elapsed_time(t1,t2)/1000)

        start += ncols
    # move final result back to host
    C = c_d.copy_to_host()

    print(t_tot)

    if params["print"]:
        if rank == 0:
            print(C)
            for i in range(1,npes):
                block = np.zeros((workloads[i], SIZE))
                block = comm.recv(source=i,tag=i)
                print(block)
        else:
            comm.send(C,dest=0,tag=rank)


    MPI.Finalize()


cpu_routines = ['matmul',
                'matmul_numba_serial',
                'matmul_numba_cpu',
                'matmul_numba_block_serial',
                'matmul_numba_block_cpu']

gpu_routines = ['matmul_numba_gpu',
                'matmul_numba_block_gpu']

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config yaml file")
    parser = parser.parse_args()

    if not parser.config:
        raise RuntimeError("Please specify a yaml config file with `--config <filename>`.")
    params = read_config(parser.config)
    routine = params["function"]["routine"]

    if params["device"] == "cpu" :
        if not routine in cpu_routines:
            raise ValueError(f"Specified routine '{routine}' is incompatible with device 'cpu'. Compatible routines are {cpu_routines}.")
        main_cpu(params)
    elif params["device"] == "gpu" :
        if not cuda.is_available():
            raise RuntimeError("Trying to run on GPU but CUDA is not available")
        if not routine in gpu_routines:
            raise ValueError(f"Specified routine '{routine}' is incompatible with device 'gpu'. Compatible routines are {gpu_routines}.")
        main_gpu(params)
    else:
        raise ValueError(f"Parameter `device` can be either 'cpu' or 'gpu', instead got {params['device']}.")
