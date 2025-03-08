__all__ = [
        'matmul',
        'matmul_numba_serial',
        'matmul_numba_cpu',
        'matmul_numba_gpu',
        'matmul_numba_block_serial',
        'matmul_numba_block_cpu',
        'matmul_numba_block_gpu']


from .routines import matmul, matmul_numba_serial, matmul_numba_cpu, matmul_numba_gpu, matmul_numba_block_serial, matmul_numba_block_cpu, matmul_numba_block_gpu
