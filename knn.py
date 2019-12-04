import numpy as np
import operator

data = open("iris.data").readlines()

dataset = []
for line in data:
    linesplit = line.split(",")
    dataset.append(linesplit)


def euclidean(sample1, sample2):
    x, y, z, w = [(float(sample1[x]) - float(sample2[x])) ** 2 for x in range(4)]
    return np.sqrt((x + y + z + w))


def knn(sample, k=3):
    dists = []
    for input in dataset:
        dists.append([euclidean(sample, input), input[4]])
    dists.sort()
    res = {}
    for d in dists[:k]:
        res[d[1]] = res.get(d[1]) + 1 if res.get(d[1]) is not None else 1

    return max(res.items(), key=operator.itemgetter(1))[0]


print(knn([7, 3, 6, 2]))
