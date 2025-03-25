import warnings

__all__ = [
        'matmul',
        'matmul_numba_serial',
        'matmul_numba_cpu',
        'matmul_numba_block_serial',
        'matmul_numba_block_cpu']
from .utils import custom_warning
from .routines import matmul, matmul_numba_serial, matmul_numba_cpu, matmul_numba_block_serial, matmul_numba_block_cpu 
try:
    from .routines import matmul_numba_gpu, matmul_numba_block_gpu
    __all__.append('matmul_numba_gpu')
    __all__.append('matmul_numba_block_gpu')
except ImportError:
    warnings.formatwarning = custom_warning
    warnings.warn("CUDA not found: GPU functions won't be available.")
