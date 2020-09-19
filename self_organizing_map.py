import itertools
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class SOMTopology(ABC):
    def __init__(self, nnodes: int,
                 starting_neighbor_d: int = 5,
                 neighborhood_decay_fn: Callable[[int, int, int], int] =
                 lambda x: x - 1):
        self._nnodes = nnodes
        self._d0 = starting_neighbor_d
        self._d = starting_neighbor_d
        self._decay_fn = neighborhood_decay_fn

    @abstractmethod
    def find_neighbors(self, node_idx: int) -> Tuple[int, ...]:
        pass

    def shrink_neighborhood(self, epoch: int):
        new_range = int(self._decay_fn(self._d0, self._d, epoch))
        self._d = max(new_range, 0)

    @property
    def node_count(self) -> int:
        return self._nnodes

    @abstractmethod
    def map(self, node_labels: Dict[int, str]) -> np.ndarray:
        pass


class LinearSOMTopology(SOMTopology):
    def __init__(self, nnodes: int,
                 starting_neighbor_d: int = 5,
                 neighborhood_decay_fn:
                 Callable[[int, int, int], int] = lambda x: x - 1):
        super(LinearSOMTopology, self).__init__(nnodes,
                                                starting_neighbor_d,
                                                neighborhood_decay_fn)

    def find_neighbors(self, node_idx: int):
        # don't overshoot the number of nodes
        # i.e. don't "wrap around"
        left_range = max(node_idx - self._d, 0)
        right_range = min(node_idx + self._d + 1, self._nnodes)

        left_neighbors = list(range(left_range, node_idx))
        right_neighbors = list(range(node_idx + 1, right_range))

        return tuple(left_neighbors + right_neighbors)

    def map(self, node_labels: Dict[int, str]) -> np.ndarray:
        output = np.empty(self._nnodes, dtype='U32')

        indices = list(node_labels.keys())
        labels = list(node_labels.values())

        output[indices] = labels
        return output


class CircularSOMTopology(LinearSOMTopology):
    # exactly like a linear topology, but "wraps" around itself

    def find_neighbors(self, node_idx: int):
        left_range = node_idx - self._d
        right_range = node_idx + self._d + 1

        if left_range < 0:
            left_neighbors = list(range(0, node_idx)) + \
                             list(range(left_range % self._nnodes,
                                        self._nnodes))
        else:
            left_neighbors = list(range(left_range, node_idx))

        if right_range > self._nnodes:
            right_neighbors = list(range(node_idx + 1, self._nnodes)) + \
                              list(range(0, right_range % self._nnodes))
        else:
            right_neighbors = list(range(node_idx + 1, right_range))

        # use a set to remove possible duplicates
        return tuple(set(left_neighbors + right_neighbors))


class GridSOMTopology(SOMTopology):
    def __init__(self, nrows: int, ncols: int,
                 starting_neighbor_d: int = 5,
                 neighborhood_decay_fn:
                 Callable[[int, int, int], int] = lambda x: x - 1):
        super(GridSOMTopology, self).__init__(nrows * ncols,
                                              starting_neighbor_d,
                                              neighborhood_decay_fn)
        self._grid = np.arange(self._nnodes).reshape((nrows, ncols))

    def find_neighbors(self, node_idx: int) -> Tuple[int, ...]:
        row = node_idx // self._grid.shape[0]
        col = node_idx % self._grid.shape[1]

        neighbors = []
        for r, c in itertools.product(range(0, self._grid.shape[0]),
                                      range(0, self._grid.shape[1])):
            if np.abs(r - row) + np.abs(c - col) <= self._d:
                neighbors.append(self._grid[r, c])

        return tuple(neighbors)

    def map(self, node_labels: Dict[int, str]) -> np.ndarray:
        output = np.empty(self._nnodes, dtype='U32')

        indices = list(node_labels.keys())
        labels = list(node_labels.values())

        output[indices] = labels
        return output.reshape(self._grid.shape)


class SelfOrganizingMap:
    def __init__(self, x_dims: int, topology: SOMTopology):
        super(SelfOrganizingMap, self).__init__()
        self._dims = x_dims
        self._topo = topology
        self._W = rand_gen.uniform(low=0.0, high=1.0,
                                   size=(topology.node_count, x_dims))

    def train(self, X: np.ndarray, eta: float = 0.2, n_epochs: int = 20):
        for epoch in range(n_epochs):
            # shuffle in each epoch
            X = X.copy()
            rand_gen.shuffle(X)

            for x_sample in X:
                distances = np.apply_along_axis(
                    func1d=lambda w: np.dot((x_sample - w).T, (x_sample - w)),
                    axis=1,
                    arr=self._W
                )

                min_d_idx = np.argmin(distances).item()

                # update weights of winning node
                self._W[min_d_idx] += eta * (x_sample - self._W[min_d_idx])

                # update the neighbors
                for neighbor in self._topo.find_neighbors(min_d_idx):
                    self._W[neighbor] += eta * (x_sample - self._W[neighbor])

            # epoch done!
            self._topo.shrink_neighborhood(epoch)

    def map(self, X: np.ndarray, labels: Iterable[str]) -> np.ndarray:
        node_label_map = dict()
        label_d_map = dict()

        for x_sample, label in zip(X, labels):
            distances = np.apply_along_axis(
                func1d=lambda w: np.dot((x_sample - w).T, (x_sample - w)),
                axis=1,
                arr=self._W
            )

            label_d_map[label] = distances
            min_d_idx = np.argmin(distances).item()

            if min_d_idx in node_label_map:
                print(f'Conflict in mapping, '
                      f'node {min_d_idx}: {label} | '
                      f'{node_label_map[min_d_idx]}')

            node_label_map[min_d_idx] = label

        return self._topo.map(node_label_map)


def decay_fn(d0: int, d: int, epoch: int) -> int:
    return d0 - int(2.5 * epoch)


if __name__ == '__main__':
    # load data
    animals_fts = np.fromfile('data/animals.dat', sep=',')
    animals_fts = animals_fts.reshape((32, 84))
    labels = []
    with open('data/animalnames.txt', 'r') as fp:
        for line in fp.readlines():
            labels.append(line.strip()[1:-1])
            # [1: -1] is to remove extra quotation marks

    som = SelfOrganizingMap(
        x_dims=84,
        topology=LinearSOMTopology(nnodes=100,
                                   starting_neighbor_d=50,
                                   neighborhood_decay_fn=decay_fn))
    som.train(animals_fts, n_epochs=25)
    results = som.map(animals_fts, labels)

    print(results)
    print(np.count_nonzero(results[results != '']), results.size)
