import itertools
from abc import ABC, abstractmethod
from typing import Callable, Collection, Tuple, Union

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class SOMTopology(ABC):
    def __init__(self, nnodes: int,
                 starting_neighbor_d: int = 5,
                 neighborhood_decay_fn: Callable[[int, int, int], int] =
                 lambda x: x - 1):
        """
        Self Organizing Map topology class.

        :param nnodes: Number of nodes in the map.
        :param starting_neighbor_d:  Starting neighborhood distance.
        :param neighborhood_decay_fn: Decay function governing how the
        neighborhood distance evolves over time. Should take as parameters
        the starting neighborhood distance, the current neighborhood distance
        and the current epoch.
        """

        self._nnodes = nnodes
        self._d0 = starting_neighbor_d
        self._d = starting_neighbor_d
        self._decay_fn = neighborhood_decay_fn

    @abstractmethod
    def find_neighbors(self, node_idx: int) -> Tuple[int, ...]:
        """
        Finds the neighbors of the node with the given index.

        :param node_idx: Index of the node.
        :return: A tuple containing the neighbors of node node_idx.
        """
        pass

    def shrink_neighborhood(self, epoch: int):
        """
        Shrinks the neighborhood using the stored neighborhood decay function.

        :param epoch: Current training epoch.
        """
        new_range = int(self._decay_fn(self._d0, self._d, epoch))
        self._d = max(new_range, 0)

    @property
    def node_count(self) -> int:
        """
        :return: Node count of this topology.
        """
        return self._nnodes

    @abstractmethod
    def map(self, label_per_node: np.ndarray) -> np.ndarray:
        """
        Maps a set collection of labels to this topology.

        :param label_per_node: A np.ndarray of length equal to the number of
        nodes in this topology, containing a label for each node.
        :return: Labels arranged in accordance to this internal topology.
        """
        pass

    @abstractmethod
    def get_coordinate_for_node(self, node_id: int) \
            -> Tuple[Union[int, float], ...]:
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

    def map(self, label_per_node: np.ndarray) -> np.ndarray:
        return label_per_node

    def get_coordinate_for_node(self, node_id: int) \
            -> Tuple[Union[int, float], ...]:
        return node_id,


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
        """
        Grid topology for Self Organizing Maps. The number of nodes of this 
        topology corresponds to the number of columns times the number of rows.
        
        :param nrows: Number of rows in the grid.
        :param ncols: Number of columns in the grid.
        :param starting_neighbor_d: Starting neighborhood size.
        :param neighborhood_decay_fn: Neighborhood decay function.
        """
        super(GridSOMTopology, self).__init__(nrows * ncols,
                                              starting_neighbor_d,
                                              neighborhood_decay_fn)
        self._grid = np.arange(self._nnodes).reshape((nrows, ncols))

    def find_neighbors(self, node_idx: int) -> Tuple[int, ...]:
        row = node_idx // self._grid.shape[0]
        col = node_idx % self._grid.shape[1]

        neighbors = []
        row_range = range(max(row - self._d, 0),
                          min(row + self._d + 1, self._grid.shape[0]))
        col_range = range(max(col - self._d, 0),
                          min(col + self._d + 1, self._grid.shape[1]))
        for r, c in itertools.product(row_range, col_range):
            # skip 0 since a node is not its own neighbor
            if 0 < np.abs(r - row) + np.abs(c - col) <= self._d:
                neighbors.append(self._grid[r, c])

        return tuple(neighbors)

    def map(self, label_per_node: np.ndarray) -> np.ndarray:
        return label_per_node.reshape(self._grid.shape)

    def get_coordinate_for_node(self, node_id: int) \
            -> Tuple[Union[int, float], ...]:
        row = node_id // self._grid.shape[0]
        col = node_id % self._grid.shape[1]

        return row, col


class SelfOrganizingMap:
    def __init__(self, topology: SOMTopology):
        """
        Implementation of a self organizing map.

        :param topology: The topology of this map.
        """
        super(SelfOrganizingMap, self).__init__()
        self._topo = topology
        self._W = np.empty(0)

    def train(self, X: np.ndarray, eta: float = 0.2, n_epochs: int = 20):
        """
        Train this map on the given input data.

        :param X: Matrix of size MxN, where M is the number of samples and N
        is the number of attributes.
        :param eta: Learning factor.
        :param n_epochs: Epochs to train for.
        """

        self._W = rand_gen.uniform(low=0.0, high=1.0,
                                   size=(self._topo.node_count, X.shape[1]))

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

    def map_labels_to_output_space(self,
                                   X: np.ndarray,
                                   labels: Collection[str]) -> np.ndarray:
        """
        Maps a set of labeled inputs to the output space of this SOM.
        Returns a representation of the output space where the index of each
        element corresponds to a node, and the value of each element
        corresponds to the label of the input for which said node reacted
        strongest.

        :param X: Matrix of size MxN, where M is the number of samples and N
        is the number of attributes.
        :param labels: Array of labels of length M
        :return: A representation of the labeled output space.
        """

        assert X.shape[0] == len(labels)

        output = np.empty(self._topo.node_count, dtype='U32')
        output[:] = ''
        min_distances = np.empty(self._topo.node_count)
        min_distances[:] = np.inf

        for x_sample, label in zip(X, labels):
            distances = np.apply_along_axis(
                func1d=lambda w: np.dot((x_sample - w).T, (x_sample - w)),
                axis=1,
                arr=self._W
            )

            # for all output distances larger than the distances to this
            # sample, set their label to this label and set their distances
            # to their distances to this sample
            # this way we get a "continuous" output in the nodes, where even
            # if a node is not the minimum for a specific input, it is still
            # marked if that input is the closest to it anyway.
            indices = np.nonzero(distances < min_distances)
            output[indices] = [label] * len(indices)
            min_distances[indices] = distances[indices]

        return self._topo.map(output)

    def map(self, X: np.ndarray, coordinates: bool = False) -> np.ndarray:
        """
        Map a series of inputs to the output space. For each input, returns
        the index of the node closest to it.

        :param X: Matrix of size MxN, where M is the number of samples and N
        is the number of attributes.
        :return: An array of length M, where each index corresponds to an
        input, and each value to the node closest to that input.
        """

        output = np.empty(X.shape[0], dtype=np.int)
        output[:] = np.nan

        for i, x_sample in enumerate(X):
            distances = np.apply_along_axis(
                func1d=lambda w: np.dot((x_sample - w).T, (x_sample - w)),
                axis=1,
                arr=self._W
            )

            output[i] = np.argmin(distances).item()

        return output if not coordinates \
            else np.array([self._topo.get_coordinate_for_node(o)
                           for o in output])
