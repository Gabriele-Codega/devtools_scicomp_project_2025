from numba.pycc import CC
from numba import prange
import numpy as np

cc = CC('_compiled')
cc.verbose = True
cc.target_cpu = "host"

@cc.export('_distance_numba_compiled', 'f8(f8[:],f8[:])')
def distance_numba(point1: np.ndarray, point2: np.ndarray) -> float:
    distance = 0
    for i in prange(len(point2)):
        distance += (point1[i]-point2[i])*(point1[i]-point2[i])

    return distance

if __name__=='__main__':
    cc.compile()
