from pyclassify.utils import distance_numpy, distance_numba
import numpy as np
import time

max_d = 2000
n = 1000

times = np.zeros((max_d-1,3))
for d in range(1,max_d):
    a = np.random.random(d)
    b = np.random.random(d)

    tnp = 0
    tnb = 0
    for _ in range(n):
        t1np = time.time()
        distance_numpy(a,b)
        t2np = time.time()
        tnp += (t2np - t1np)

        t1nb = time.time()
        distance_numba(a,b)
        t2nb = time.time()
        tnb += (t2nb - t1nb)

    times[d-1,:] = [d, tnp/n, tnb/n]

np.savetxt("./data/times.dat", times)
