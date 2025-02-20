from .utils import distance, majority_vote
from functools import partial

class kNN:
    def __init__(self, k: int):
        if not isinstance(k,int):
            raise TypeError
        self.n_neigh = k

    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x: list[float]) -> list[int]:
        distances = list(map(partial(distance,x),X))

        nn = sorted(range(len(distances)), key= lambda i: distances[i])

        labels = []
        for idx in nn[:self.n_neigh]:
            labels.append(y[idx])
        return labels

    def __call__(self, data: tuple[list[list[float]],list[int]], new_points):
        nn_labels = list(map(partial(self._get_k_nearest_neighbors,*data), new_points))
        prediction = list(map(majority_vote,nn_labels))
        return prediction
