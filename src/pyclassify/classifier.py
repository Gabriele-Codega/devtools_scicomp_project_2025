from .utils import distance, majority_vote, distance_numpy, distance_numba
from functools import partial
from typing import Union
import numpy as np
from line_profiler import profile

class kNN:
    def __init__(self, k: int, backend: str = 'plain'):
        if not isinstance(k,int):
            raise TypeError
        if not isinstance(backend,str):
            raise TypeError

        self.n_neigh = k
        self.backend = backend
        
        if backend == 'plain':
            self.distance = distance
        elif backend == 'numpy':
            self.distance = distance_numpy
        elif backend == 'numba':
            self.distance = distance_numba
        else:
            raise ValueError(f"Attribute `backend` must be 'plain', 'numpy' or 'numba'. Got {backend} instead")

    @profile
    def _get_k_nearest_neighbors(self, X: Union[list[list[float]], np.ndarray], y: Union[list[int], np.ndarray], x: Union[list[float], np.ndarray]) -> list[int]:
        distances = list(map(partial(self.distance,x),X))

        nn = sorted(range(len(distances)), key= lambda i: distances[i])

        labels = []
        for idx in nn[:self.n_neigh]:
            labels.append(y[idx])
        return labels

    @profile
    def __call__(self, data: tuple[list[list[float]],list[int]], new_points: list[list[float]]):
        if self.backend == 'numpy' or self.backend == 'numba':
            data = (np.array(data[0]), np.array(data[1]))
            new_points = np.array(new_points)
        nn_labels = list(map(partial(self._get_k_nearest_neighbors,*data), new_points))
        prediction = list(map(majority_vote,nn_labels))
        return prediction
