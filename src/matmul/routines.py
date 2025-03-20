from numba import void, float64, int32, njit, prange, cuda
import numba

def matmul(A,B,C,_):
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), f"Matrices have incompatible shapes: {A.shape}, {B.shape}, {C.shape}"
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i,k] * B[k,j]
            C[i,j] = tmp

@njit(void(float64[:,::1],float64[:,::1],float64[:,:],numba.optional(int32)), cache=True)
def matmul_numba_serial(A,B,C,_):
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), f"Matrices have incompatible shapes: {A.shape}, {B.shape}, {C.shape}"
    for i in range(A.shape[0]):
        for k in range(A.shape[-1]):
            for j in range(B.shape[1]):
                C[i,j] += A[i,k] * B[k,j]

@njit(void(float64[:,::1],float64[:,::1],float64[:,:],numba.optional(int32)), parallel=True, nogil=True, cache=True)
def matmul_numba_cpu(A,B,C,_):
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), f"Matrices have incompatible shapes: {A.shape}, {B.shape}, {C.shape}"
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                C[i,j] += A[i,k] * B[k,j]



@njit(void(float64[:,::1],float64[:,::1],float64[:,:],int32), parallel=True, nogil=True, cache=True)
def matmul_numba_block_cpu(A,B,C, bs=64):
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), f"Matrices have incompatible shapes: {A.shape}, {B.shape}, {C.shape}"
    N = A.shape[0]
    M = B.shape[1]
    K = A.shape[1]
    niblocks = (N//bs) + ((N % bs) > 0)
    for ii in prange(0,niblocks):
        i0 = ii*bs
        imax = min(i0+bs,N)
        for kk in range(0,K,bs):
            kmax = min(kk+bs,K)
            for jj in range(0,M,bs):
                jmax = min(jj+bs,M)
                for i in range(i0,imax):
                    for k in range(kk,kmax):
                        for j in range(jj,jmax):
                            C[i,j] += A[i,k] * B[k,j]

@njit(void(float64[:,::1],float64[:,::1],float64[:,:],int32), parallel=False, nogil=True, cache=True)
def matmul_numba_block_serial(A,B,C, bs=64):
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), f"Matrices have incompatible shapes: {A.shape}, {B.shape}, {C.shape}"
    N = A.shape[0]
    M = B.shape[1]
    K = A.shape[1]
    niblocks = (N//bs) + ((N % bs) > 0)
    for ii in range(0,niblocks):
        i0 = ii*bs
        imax = min(i0+bs,N)
        for kk in range(0,K,bs):
            kmax = min(kk+bs,K)
            for jj in range(0,M,bs):
                jmax = min(jj+bs,M)
                for i in range(i0,imax):
                    for k in range(kk,kmax):
                        for j in range(jj,jmax):
                            C[i,j] += A[i,k] * B[k,j]

@cuda.jit(void(float64[:,::1],float64[:,::1],float64[:,:]), cache=True, debug=False)
def matmul_numba_gpu(A,B,C):
    # this only has effect if function is compiled with debug = True
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), "Matrices have incompatible shapes"
    i, j = cuda.grid(ndim=2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(B.shape[0]):
            tmp += A[i,k] * B[k,j]
        C[i,j] = tmp

BLOCK_SIZE = 16
@cuda.jit(void(float64[:,::1],float64[:,::1],float64[:,:]), cache=True, debug=False)
def matmul_numba_block_gpu(A,B,C):
    # this only has effect if function is compiled with debug = True
    assert (A.shape[0] == C.shape[0]) and (A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[1]), "Matrices have incompatible shapes"

    bi = cuda.blockIdx.y
    bj = cuda.blockIdx.x
    ti = cuda.threadIdx.y
    tj = cuda.threadIdx.x
    bh = cuda.blockDim.y
    bw = cuda.blockDim.x
    gi = bi*bh + ti
    gj = bj*bw + tj
    nblocks = (A.shape[1] + BLOCK_SIZE - 1)//BLOCK_SIZE

    Ashared = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE),dtype=float64)
    Bshared = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE),dtype=float64)
    tmp = 0.
    for b in range(nblocks):

        Ashared[ti,tj] = 0
        Bshared[ti,tj] = 0
        
        if gi < A.shape[0] and (tj + b*BLOCK_SIZE) < A.shape[1]:
            Ashared[ti,tj] = A[gi,tj + b*BLOCK_SIZE]
        if (ti + b*BLOCK_SIZE) < B.shape[0] and gj < B.shape[1]:
            Bshared[ti,tj] = B[ti + b*BLOCK_SIZE,gj]
        cuda.syncthreads()

        for k in range(BLOCK_SIZE):
            tmp += Ashared[ti,k] * Bshared[k,tj]
        cuda.syncthreads()

    if gi < C.shape[0] and gj < C.shape[1]:
        C[gi,gj] = tmp

