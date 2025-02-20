import os
import yaml

def distance(point1: list[float], point2: list[float]) -> float:
    distance = 0
    for x1,x2 in zip(point1,point2):
        distance += (x1-x2)*(x1-x2)

    return distance


def majority_vote(neighbors: list[int]) -> int:
    labels = range(max(neighbors)+1)
    counts = [ 0 for _ in labels ]
    votes = dict(zip(labels,counts)) 

    for label in neighbors:
        votes[label] += 1

    return max(votes, key=lambda key: votes[key])

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(file: str) -> tuple[list[list[float]],list[int]]:
    y = []
    X = []
    with open(file, 'r') as stream:
        for line in stream:
            line = line.strip().split(',')
            label = 0 if line[-1] == 'b' else 1
            y.append(label)
            X.append([float(x) for x in line[:-1]])
    return X,y
