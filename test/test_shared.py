import pytest
import numpy as np
import numba

from matmul import matmul, matmul_numba_cpu, matmul_numba_serial, matmul_numba_block_cpu, matmul_numba_block_serial, matmul_numba_gpu, matmul_numba_block_gpu

def test_matmul():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    matmul(A,B,C,None)

    assert np.allclose(np.eye(size),C)

    with pytest.raises(AssertionError):
        A = np.empty((size,size+1))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul(A,B,C,None)
        A = np.empty((size+1,size))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul(A,B,C,None)
        A = np.empty((size,size))
        B = np.empty((size,size+1))
        C = np.empty((size,size))
        matmul(A,B,C,None)



def test_matmul_numba_cpu():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    matmul_numba_cpu(A,B,C,None)

    assert np.allclose(np.eye(size),C)

    with pytest.raises(AssertionError):
        A = np.empty((size,size+1),dtype=np.float64)
        B = np.empty((size,size),dtype=np.float64)
        C = np.empty((size,size),dtype=np.float64)
        matmul_numba_cpu(A,B,C,None)
        A = np.empty((size+1,size),dtype=np.float64)
        B = np.empty((size,size),dtype=np.float64)
        C = np.empty((size,size),dtype=np.float64)
        matmul_numba_cpu(A,B,C,None)
        A = np.empty((size,size),dtype=np.float64)
        B = np.empty((size,size+1),dtype=np.float64)
        C = np.empty((size,size),dtype=np.float64)
        matmul_numba_cpu(A,B,C,None)


def test_matmul_numba_serial():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    matmul_numba_serial(A,B,C,None)

    assert np.allclose(np.eye(size),C)

    with pytest.raises(AssertionError):
        A = np.empty((size,size+1))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_serial(A,B,C,None)
        A = np.empty((size+1,size))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_serial(A,B,C,None)
        A = np.empty((size,size))
        B = np.empty((size,size+1))
        C = np.empty((size,size))
        matmul_numba_serial(A,B,C,None)

def test_matmul_numba_block_cpu():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    matmul_numba_block_cpu(A,B,C,6)

    assert np.allclose(np.eye(size),C)

    with pytest.raises(AssertionError):
        A = np.empty((size,size+1))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_block_cpu(A,B,C,6)
        A = np.empty((size+1,size))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_block_cpu(A,B,C,6)
        A = np.empty((size,size))
        B = np.empty((size,size+1))
        C = np.empty((size,size))
        matmul_numba_block_cpu(A,B,C,6)

def test_matmul_numba_block_serial():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    matmul_numba_block_serial(A,B,C,6)

    assert np.allclose(np.eye(size),C)

    with pytest.raises(AssertionError):
        A = np.empty((size,size+1))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_block_serial(A,B,C,6)
        A = np.empty((size+1,size))
        B = np.empty((size,size))
        C = np.empty((size,size))
        matmul_numba_block_serial(A,B,C,6)
        A = np.empty((size,size))
        B = np.empty((size,size+1))
        C = np.empty((size,size))
        matmul_numba_block_serial(A,B,C,6)

@pytest.mark.skipif((not numba.cuda.is_available()), reason='Could not find any CUDA GPU')
def test_matmul_numba_gpu():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    a_d = numba.cuda.to_device(A)
    b_d = numba.cuda.to_device(B)
    c_d = numba.cuda.to_device(C)

    nthreads = 16
    blocks_per_grid = ((size + nthreads-1)//nthreads,(size + nthreads-1)//nthreads)
    threads_per_block = (nthreads, nthreads)

    matmul_numba_gpu[blocks_per_grid,threads_per_block](a_d,b_d,c_d)

    C = c_d.copy_to_host()

    assert np.allclose(np.eye(size),C)

    # No tests for assertion errors on matrix shape since they are only available
    # for debug=True

@pytest.mark.skipif((not numba.cuda.is_available()), reason='Could not find any CUDA GPU')
def test_matmul_numba_block_gpu():
    size = 20
    np.random.seed(0)
    A = np.random.rand(size,size)
    B = np.linalg.inv(A)
    C = np.zeros((size,size),dtype=np.float64)

    a_d = numba.cuda.to_device(A)
    b_d = numba.cuda.to_device(B)
    c_d = numba.cuda.to_device(C)

    nthreads = 16
    blocks_per_grid = ((size + nthreads-1)//nthreads,(size + nthreads-1)//nthreads)
    threads_per_block = (nthreads, nthreads)

    matmul_numba_block_gpu[blocks_per_grid,threads_per_block](a_d,b_d,c_d,nthreads)

    C = c_d.copy_to_host()

    assert np.allclose(np.eye(size),C)

    # No tests for assertion errors on matrix shape since they are only available
    # for debug=True
