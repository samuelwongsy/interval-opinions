from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import numpy.typing as npt
import networkx as nx

from .exceptions import PointsDimensionError, InvalidParameterError
from .helpers import ResultSerializer


class IntervalOpinion(ABC):
    """
    IntervalOpinion is the abstract base class for all interval opinions.

    Attributes
    ----------
    n : int
        Number of opinion pairs.
    d : int
        Dimension of opinion pairs.
    dynamic_matrix : npt.NDArray
        Dynamic Matrix of the interval opinions stored in a numpy array.
    opinions: npt.NDArray
        Opinions stored in a numpy array. Format is [c1, c2, c3, p1, p2, p3]
        where castor is c and pollux is p and [c1, p1] are a pair of opinions.
    save_results: bool
        Save the simulation at every step into a file.
    result_serializer: ResultSerializer
        Module to serialize the IntervalOpinion.
    """
    def __init__(self, n: int, dimension: int, *args, **kwargs):
        super().__init__()
        self.n = n
        self.d = dimension
        self.dynamic_matrix = self.create_dynamic_matrix(n)
        self.opinions = self.create_opinions(n, dimension)

        self.save_results = kwargs.get('save_results', False)
        self.result_serializer = ResultSerializer(
            base_file_name=kwargs.get('file_name', 'opinion'),
            base_folder=kwargs.get('path', 'results'))

    @staticmethod
    def create_dynamic_matrix(n: int) -> npt.NDArray[np.float64]:
        return np.empty(shape=(2*n, 2*n))

    @staticmethod
    def create_opinions(n: int, d: int) -> npt.NDArray[np.float64]:
        return np.empty(shape=(d, 2*n))

    @staticmethod
    def normalize_rows(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
        matrix = matrix.T
        matrix /= matrix.sum(axis=0)
        matrix = matrix.T
        return matrix

    @staticmethod
    def normalize_cols(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
        matrix /= matrix.sum(axis=0)
        return matrix

    @staticmethod
    def distance(vector1: npt.ArrayLike, vector2: npt.ArrayLike) -> np.float64:
        return np.linalg.norm(vector1-vector2)

    @staticmethod
    def get_dynamic_matrix(castor_to_castor: npt.ArrayLike,
                           castor_to_pollux: npt.ArrayLike,
                           pollux_to_castor: npt.ArrayLike,
                           pollux_to_pollux: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Concatenate 4 different matrices into the dynamic matrix and normalise across the columns.

        Returns
        -------
        npt.NDArray[np.float64]
            Dynamic Matrix that describes the relationship between the opinions.
        """
        row1 = np.concatenate((castor_to_castor, castor_to_pollux), axis=1)
        row2 = np.concatenate((pollux_to_castor, pollux_to_pollux), axis=1)
        matrix = np.concatenate((row1, row2), axis=0)
        matrix = IntervalOpinion.normalize_cols(matrix)
        return matrix

    def init_opinions(self) -> None:
        """Initialize the opinions randomly with the shape [d, 2n]."""
        # randomize initial values
        self.opinions = np.random.rand(self.d, 2*self.n)

    def update_opinions(self, dynamic_matrix: npt.ArrayLike, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Update opinions by doing a matrix multiplication with dynamic matrix.

        Returns
        -------
        npt.NDArray[np.float64]
            New opinions.

        Raises
        ------
        PointsDimensionError
            If the opinion matrix shape changed during the matrix multiplication.
        """
        new_opinions = np.matmul(opinions, dynamic_matrix)
        if new_opinions.shape != opinions.shape:
            raise PointsDimensionError(
                f"New Opinions shape {new_opinions.shape}!= Opinions matrix shape {opinions.shape}")

        return new_opinions

    def update_dynamic_matrix(self) -> npt.NDArray[np.float64]:
        """
        Updates dynamic matrix based on the opinions.

        Returns
        -------
        npt.NDArray[np.float64]
            Updated dynamic matrix.
        """
        opinions = self.opinions

        castor_to_castor = self.get_castor_to_castor(opinions)
        castor_to_pollux = self.get_castor_to_pollux(opinions)
        pollux_to_castor = self.get_pollux_to_castor(opinions)
        pollux_to_pollux = self.get_pollux_to_pollux(opinions)

        dynamic_matrix = self.get_dynamic_matrix(castor_to_castor, castor_to_pollux, pollux_to_castor,
                                                 pollux_to_pollux)
        return dynamic_matrix

    def print_opinions(self) -> None:
        print(self.opinions)

    def print_dynamic_matrix(self) -> None:
        print(self.dynamic_matrix)

    def _update(self) -> None:
        """
        Update 1 time step.
        """
        self.opinions = self.update_opinions(self.dynamic_matrix, self.opinions)
        self.dynamic_matrix = self.update_dynamic_matrix()

    def run_simulation(self, max_steps: int = 5000) -> None:
        """
        Run the simulation till the opinions do not change or the max number of steps.

        Params
        ------
        max_steps: int
            Max number of steps to run the simulation to.
        """
        self.init_opinions()
        self.init_dynamic_matrix()
        step = 0

        print("Initial Opinions:")
        self.print_opinions()
        print()
        print("Initial Dynamic Matrix:")
        self.print_dynamic_matrix()

        if self.save_results:
            self._save_matrix(self.opinions, self.dynamic_matrix)

        while step < max_steps:
            old_opinions = self.opinions
            self._update()

            # Compare old opinions to new opinions and break if the same
            comparison = self.opinions == old_opinions
            if comparison.all():
                break

            step += 1
            if self.save_results:
                self._save_matrix(self.opinions, self.dynamic_matrix)

        print()
        print(f"Finished {step} steps:")
        print("Final Opinions:")
        self.print_opinions()
        print()
        print("Final Dynamic Matrix:")
        self.print_dynamic_matrix()

    def _save_matrix(self, opinions: npt.ArrayLike, dynamic_matrix: npt.ArrayLike) -> None:
        self.result_serializer.save_results(opinions, dynamic_matrix)

    @abstractmethod
    def init_dynamic_matrix(self) -> None:
        """Initialise dynamic matrix based on opinions."""
        pass

    @abstractmethod
    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Initialise the matrix for how castor points affect other castor points."""
        pass

    @abstractmethod
    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Initialise the edges for how castor points affect other pollux points."""
        pass

    @abstractmethod
    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Initialise the edges for how pollux points affect other castor points."""
        pass

    @abstractmethod
    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Initialise the edges for how pollux points affect other pollux points."""
        pass


class GraphType(Enum):
    """Enum class for different graph types that can be initialised."""

    @classmethod
    def list(cls):
        return list(map(lambda g: g.name, cls))

    RANDOM = 1
    COMPLETE = 2
    CYCLE = 3


class NetworkIntervalOpinion(IntervalOpinion, ABC):
    """
    NetworkIntervalOpinion is the abstract base class for interval opinions
    that require adjacency matrices for the edges of the directed castor and pollux graphs.

    Attributes
    ----------
    castor_graph_type : GraphType
        Type of graph for the castor adjacency matrix.
    pollux_graph_type : GraphType
        Type of graph for the pollux adjacency matrix.
    castor_graph : npt.NDArray[np.int_]
        Adjacency matrix for castor points.
    pollux_graph : npt.NDArray[np.int_]
        Adjacency matrix for pollux points.
    self_edges : bool
        Allow edges to self for each node.
    """

    def __init__(self,
                 n: int,
                 dimension: int,
                 *args, **kwargs):
        castor_graph_type = self.parse_graph_type(kwargs.pop("castor_graph_type", GraphType.RANDOM))
        pollux_graph_type = self.parse_graph_type(kwargs.pop("pollux_graph_type", GraphType.RANDOM))
        self.self_edges = kwargs.pop("self_edges", True)
        super().__init__(n, dimension, *args, **kwargs)

        self.castor_graph_type = castor_graph_type
        self.pollux_graph_type = pollux_graph_type

    @staticmethod
    def parse_graph_type(graph_type) -> GraphType:
        """
        Parse graph type from string or GraphType.

        Params
        ------
        graph_type : str or GraphType
            Takes the graph type as a case insensitive string or GraphType enum.

        Returns
        -------
        GraphType
            GraphType Enum

        Raises
        ------
        InvalidParameterError
            When graph_type is not a valid string or GraphType
        """
        if isinstance(graph_type, GraphType):
            return graph_type
        elif isinstance(graph_type, str):
            graph_type = graph_type.upper()
            try:
                return GraphType[graph_type]
            except KeyError as err:
                raise InvalidParameterError(
                    f"Graph Type: '{graph_type}' not in allowed graph types: {GraphType.list()}", err)

        raise InvalidParameterError(
            f"Invalid type for graph_type, only accept str or GraphType enum")

    def initialize_graph(self, graph_type: GraphType) -> npt.NDArray[np.int_]:
        if graph_type == GraphType.RANDOM:
            return self.init_random_graph()
        elif graph_type == GraphType.COMPLETE:
            return self.init_complete_graph()
        elif graph_type == GraphType.CYCLE:
            return self.init_cycle_graph()
        raise NotImplementedError(f"Graph Type: '{graph_type}' allowed but method not implemented")

    def init_random_graph(self) -> npt.NDArray[np.int_]:
        n = self.n
        numpy_graph = np.random.randint(0, 2, size=(n, n))
        if self.self_edges:
            self.add_self_edges(numpy_graph)
        return numpy_graph

    def init_complete_graph(self) -> npt.NDArray[np.int_]:
        n = self.n
        network_graph = nx.complete_graph(n)
        numpy_graph = nx.to_numpy_array(network_graph, dtype=np.int_)
        if self.self_edges:
            self.add_self_edges(numpy_graph)
        return numpy_graph

    def init_cycle_graph(self) -> npt.NDArray[np.int_]:
        n = self.n
        network_graph = nx.cycle_graph(n)
        numpy_graph = nx.to_numpy_array(network_graph, dtype=np.int_)
        if self.self_edges:
            self.add_self_edges(numpy_graph)
        return numpy_graph

    def add_self_edges(self, graph: npt.NDArray) -> npt.NDArray[np.int_]:
        for i in range(self.n):
            graph[i][i] = 1
        return graph

    def run_simulation(self, max_steps: int = 5000) -> None:
        """
        Run the simulation till the opinions do not change or the max number of steps.
        Overwritten to initialize adjacency matrices first.

        Params
        ------
        max_steps: int
            Max number of steps to run the simulation to.
        """
        self.castor_graph = self.initialize_graph(self.castor_graph_type)
        self.pollux_graph = self.initialize_graph(self.pollux_graph_type)
        print("Castor Graph:")
        print(self.castor_graph)
        print("Pollux Graph:")
        print(self.pollux_graph)
        IntervalOpinion.run_simulation(self, max_steps)


class IndependentCastorAndPollux(IntervalOpinion):
    """
    IndependentCastorAndPollux represents two independent sets of opinions.
    The two different sets of opinions do not influence each other, but each point in the set
    affects all other points in the set.
    """

    def __init__(self, n: int, dimension: int, *args, **kwargs):
        super().__init__(n, dimension, *args, **kwargs)

    def init_dynamic_matrix(self) -> None:
        self.dynamic_matrix = self.update_dynamic_matrix()

    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        castor_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                if i == j:
                    castor_to_castor[i][j] = (1 + 1 * inertia) / denominator
                else:
                    castor_to_castor[i][j] = 1 / denominator

        return castor_to_castor

    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n = self.n
        return np.zeros(shape=(n, n))

    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n = self.n
        return np.zeros(shape=(n, n))

    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        pollux_to_pollux = np.zeros(shape=(n, n))

        for i in range(n, 2*n):
            for j in range(n, 2*n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                if i == j:
                    pollux_to_pollux[i-n][j-n] = (1 + 1 * inertia) / denominator
                else:
                    pollux_to_pollux[i-n][j-n] = 1 / denominator

        return pollux_to_pollux


class IndependentNetworkCastorAndPollux(NetworkIntervalOpinion):
    """
    IndependentNetworkCastorAndPollux represents two independent sets of opinions,
    but the influence within a set is defined by an adjacency matrix.
    """

    def __init__(self, n: int, dimension: int, edge_ratio: float = 0.5, *args, **kwargs):
        super().__init__(n, dimension, *args, **kwargs)

    def init_dynamic_matrix(self) -> None:
        self.dynamic_matrix = self.update_dynamic_matrix()

    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        castor_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                inertia_value = inertia if i == j else 0
                castor_to_castor[i][j] = (self.castor_graph[i][j] + inertia_value) / denominator

        return castor_to_castor

    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n = self.n
        return np.zeros(shape=(n, n))

    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n = self.n
        return np.zeros(shape=(n, n))

    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        pollux_to_pollux = np.zeros(shape=(n, n))

        for i in range(n, 2 * n):
            for j in range(n, 2 * n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                inertia_value = inertia if i == j else 0
                pollux_to_pollux[i - n][j - n] = (self.pollux_graph[i - n][j - n] + inertia_value) / denominator

        return pollux_to_pollux


class CoupledNetworkCastorAndPollux(NetworkIntervalOpinion):
    """
    CoupledNetworkCastorAndPollux (CoNCaP) allows castors and pollux to be co-dependent.
    Castors and Polluxes will influence one another.

    Persistent CoNCaP will have a fixed Castor-Pollux influence.
    Dynamic CoNCaP will have varying Castor-Pollux influence strength according to the distance of points.

    Attributes
    ----------
    influence_type : str
        Determines the type of CoNCaP interval opinion.
        Only allowed 'persistent' or 'dynamic'.
    value : float
        Represents the parameter for influence.
    """

    def __init__(self, n: int, dimension: int, influence_type: str = 'persistent', value: float = 0.5, *args, **kwargs):
        super().__init__(n, dimension, *args, **kwargs)
        if influence_type not in {'persistent', 'dynamic'}:
            raise InvalidParameterError(f"{influence_type} is not in allowed types [persistent, dynamic]!")
        self.influence_type = influence_type
        self.value = value

    def init_dynamic_matrix(self) -> None:
        self.dynamic_matrix = self.update_dynamic_matrix()

    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        castor_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                inertia_value = inertia if i == j else 0
                castor_to_castor[i][j] = ((1 - self.value) * (self.castor_graph[i][j] + inertia_value)) / denominator

        return castor_to_castor

    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon = self.n, 0.1
        castor_to_pollux = np.zeros(shape=(n, n))

        for i in range(n):
            j = i + n
            denominator = 1 if self.influence_type == 'persistent' else epsilon + self.distance(
                opinions[:, i], opinions[:, j])
            castor_to_pollux[i][i] = self.value / denominator

        return castor_to_pollux

    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon = self.n, 0.1
        pollux_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            j = i + n
            denominator = 1 if self.influence_type == 'persistent' else epsilon + self.distance(
                opinions[:, j], opinions[:, i])
            pollux_to_castor[i][i] = self.value / denominator

        return pollux_to_castor

    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        pollux_to_pollux = np.zeros(shape=(n, n))

        for i in range(n, 2*n):
            for j in range(n, 2*n):
                denominator = epsilon + self.distance(opinions[:, i], opinions[:, j])
                inertia_value = inertia if i == j else 0
                pollux_to_pollux[i - n][j - n] = ((1 - self.value) * (
                        self.pollux_graph[i - n][j - n] + inertia_value)) / denominator

        return pollux_to_pollux


class FullyCoupledNetworkCastorAndPollux(NetworkIntervalOpinion):
    """
    FullyCoupledNetworkCastorAndPollux treats the Castor - Pollux pair as an interval.
    Each edge in the influence graph will be adjusted towards the entire interval of opinions, not just as points.

    Attributes
    ----------
    value : float
        Represents the parameter for influence.
    """

    def __init__(self, n: int, dimension: int, value: float = 0.5, *args, **kwargs):
        super().__init__(n, dimension, *args, **kwargs)
        self.value = value

    def init_dynamic_matrix(self) -> None:
        self.dynamic_matrix = self.update_dynamic_matrix()

    @staticmethod
    def projection(point: npt.ArrayLike,
                   position_vector: npt.ArrayLike,
                   direction_vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
        constant = np.dot(direction_vector, point-position_vector) / np.dot(direction_vector, direction_vector)
        return position_vector + constant * direction_vector

    def alpha_castor(self, castor_i: npt.ArrayLike,
                     castor_j: npt.ArrayLike,
                     pollux_i: npt.ArrayLike) -> float:
        projection_vector = self.projection(castor_j, castor_i, pollux_i-castor_i)
        alpha = float('inf')
        for i in range(len(castor_i)):
            if castor_i[i] - pollux_i[i] != 0:
                alpha = (projection_vector[i] - pollux_i[i]) / (castor_i[i] - pollux_i[i])
                break
        return alpha if 0 <= alpha <= 1 else 1 if alpha > 1 else 0

    def alpha_pollux(self, castor_i: npt.ArrayLike,
                     pollux_i: npt.ArrayLike,
                     pollux_j: npt.ArrayLike) -> float:
        projection_vector = self.projection(pollux_j, castor_i, pollux_i-castor_i)
        alpha = float('inf')
        for i in range(len(castor_i)):
            if pollux_i[i] - castor_i[i] != 0:
                alpha = (projection_vector[i] - castor_i[i]) / (pollux_i[i] - castor_i[i])
                break
        return alpha if 0 <= alpha <= 1 else 1 if alpha > 1 else 0

    def proximity_opinion_point_castor(self, alpha_castor: float, castor_i: npt.ArrayLike,
                                       pollux_i: npt.ArrayLike) -> npt.NDArray[np.float64]:
        proximity_opinion_point = alpha_castor * castor_i + ((1 - alpha_castor) * pollux_i)
        return proximity_opinion_point

    def proximity_opinion_point_pollux(self, alpha_pollux: float, castor_i: npt.ArrayLike,
                                       pollux_i: npt.ArrayLike) -> npt.NDArray[np.float64]:
        proximity_opinion_point = alpha_pollux * pollux_i + ((1 - alpha_pollux) * castor_i)
        return proximity_opinion_point

    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        castor_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            castor_i, pollux_i = opinions[:, i], opinions[:, i+n]
            for j in range(n):
                castor_j = opinions[:, j]
                inertia_value = inertia if i == j else 0

                alpha_castor = self.alpha_castor(castor_i, castor_j, pollux_i)
                proximity_opinion_point = self.proximity_opinion_point_castor(alpha_castor, castor_i, pollux_i)
                denominator = epsilon + self.distance(proximity_opinion_point, castor_j)

                castor_to_castor[i][j] = ((1 - self.value) * (
                        self.castor_graph[i][j] + inertia_value) * alpha_castor) / denominator

        return castor_to_castor

    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon = self.n, 0.1
        castor_to_pollux = np.zeros(shape=(n, n))

        for i in range(n):
            castor_i, pollux_i = opinions[:, i], opinions[:, i+n]
            for j in range(n, 2*n):
                pollux_j = opinions[:, j]
                if i == j-n:
                    castor_to_pollux[i][j-n] = self.value / (epsilon + self.distance(castor_i, pollux_j))
                else:
                    alpha_pollux = self.alpha_pollux(castor_i, pollux_i, pollux_j)
                    proximity_point_pollux = self.proximity_opinion_point_pollux(alpha_pollux, castor_i, pollux_i)
                    castor_to_pollux[i][j - n] = ((1 - self.value) * self.pollux_graph[i][j-n] * (1 - alpha_pollux)) / (
                        epsilon + self.distance(proximity_point_pollux, pollux_j))

        return castor_to_pollux

    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon = self.n, 0.1
        pollux_to_castor = np.zeros(shape=(n, n))

        for i in range(n, 2*n):
            castor_i, pollux_i = opinions[:, i-n], opinions[:, i]
            for j in range(n):
                castor_j = opinions[:, j]
                if i-n == j:
                    pollux_to_castor[i - n][j] = self.value / (epsilon + self.distance(pollux_i, castor_j))
                else:
                    alpha_castor = self.alpha_castor(castor_i, castor_j, pollux_i)
                    proximity_point_castor = self.proximity_opinion_point_castor(alpha_castor, castor_i, pollux_i)
                    pollux_to_castor[i - n][j] = ((1 - self.value) * self.castor_graph[i-n][j] * (1 - alpha_castor)) / (
                        epsilon + self.distance(proximity_point_castor, castor_j))

        return pollux_to_castor

    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon, inertia = self.n, 0.1, 0.5
        pollux_to_pollux = np.zeros(shape=(n, n))

        for i in range(n, 2*n):
            castor_i, pollux_i = opinions[:, i-n], opinions[:, i]
            for j in range(n, 2*n):
                pollux_j = opinions[:, j]
                inertia_value = inertia if i == j else 0

                alpha_pollux = self.alpha_pollux(castor_i, pollux_i, pollux_j)
                proximity_opinion_point = self.proximity_opinion_point_pollux(alpha_pollux, castor_i, pollux_i)
                denominator = epsilon + self.distance(proximity_opinion_point, pollux_j)

                pollux_to_pollux[i - n][j - n] = ((1 - self.value) * (
                            self.pollux_graph[i-n][j-n] + inertia_value) * alpha_pollux) / denominator

        return pollux_to_pollux
