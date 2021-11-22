from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from .exceptions import PointsDimensionError, PointsValueError, InvalidParameterError


class IntervalOpinion(ABC):

    def __init__(self, n: int, dimension: int):
        super().__init__()
        self.n = n
        self.d = dimension
        self.dynamic_matrix = self.create_dynamic_matrix(n)
        self.opinions = self.create_opinions(n, dimension)

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
    def combine_dynamic_matrix(castor_to_castor: npt.ArrayLike,
                               castor_to_pollux: npt.ArrayLike,
                               pollux_to_castor: npt.ArrayLike,
                               pollux_to_pollux: npt.ArrayLike) -> npt.NDArray[np.float64]:

        row1 = np.concatenate((castor_to_castor, castor_to_pollux), axis=1)
        row2 = np.concatenate((pollux_to_castor, pollux_to_pollux), axis=1)
        matrix = np.concatenate((row1, row2), axis=0)

        return matrix

    @staticmethod
    def init_random_graph(n: int, self_edges: bool = True) -> npt.NDArray[np.int_]:
        graph = np.random.randint(0, 2, size=(n, n))
        if self_edges:
            for i in range(n):
                graph[i][i] = 1

        return graph

    def init_opinions(self) -> None:
        # randomize initial values and normalize rows such that the sum of the row = 1.0
        opinions = np.random.rand(self.d, 2*self.n)
        self.opinions = self.normalize_cols(opinions)

    def update_opinions(self, dynamic_matrix: npt.ArrayLike, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        new_opinions = np.matmul(opinions, dynamic_matrix)
        if new_opinions.shape != opinions.shape:
            raise PointsDimensionError(
                f"New Opinions shape {new_opinions.shape}!= Opinions matrix shape {opinions.shape}")

        return self.normalize_cols(new_opinions)

    def update_dynamic_matrix(self) -> npt.NDArray[np.float64]:
        opinions = self.opinions

        castor_to_castor = self.get_castor_to_castor(opinions)
        castor_to_pollux = self.get_castor_to_pollux(opinions)
        pollux_to_castor = self.get_pollux_to_castor(opinions)
        pollux_to_pollux = self.get_pollux_to_pollux(opinions)

        dynamic_matrix = self.combine_dynamic_matrix(castor_to_castor, castor_to_pollux, pollux_to_castor,
                                                     pollux_to_pollux)
        dynamic_matrix = self.normalize_cols(dynamic_matrix)
        return dynamic_matrix

    def print_opinions(self) -> None:
        print(self.opinions)

    def print_dynamic_matrix(self) -> None:
        print(self.dynamic_matrix)

    def update(self, max_steps: int) -> None:
        step = 0
        print("Initial Opinions:")
        self.print_opinions()
        print()
        print("Initial Dynamic Matrix:")
        self.print_dynamic_matrix()

        while step < max_steps:
            new_opinions = self.update_opinions(self.dynamic_matrix, self.opinions)

            # Compare old opinions to new opinions and break if the same
            comparison = self.opinions == new_opinions
            if comparison.all():
                break
            self.opinions = new_opinions

            self.dynamic_matrix = self.update_dynamic_matrix()
            step += 1

        print()
        print(f"Finished {step} steps:")
        print("Final Opinions:")
        self.print_opinions()
        print()
        print("Final Dynamic Matrix:")
        self.print_dynamic_matrix()

    def run_simulation(self, num_runs: int, max_steps: int = 5000) -> None:
        run = 1

        while run <= num_runs:
            self.init_opinions()
            self.init_dynamic_matrix()
            print(f"Run {run}:")
            self.update(max_steps=max_steps)
            run += 1

    @abstractmethod
    def init_dynamic_matrix(self) -> None:
        pass

    @abstractmethod
    def get_castor_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_castor_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_pollux_to_pollux(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass


class IndependentCastorAndPollux(IntervalOpinion):

    def __init__(self, n: int, dimension: int):
        super().__init__(n, dimension)

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


class IndependentNetworkCastorAndPollux(IntervalOpinion):

    def __init__(self, n: int, dimension: int, edge_ratio: float = 0.5):
        super().__init__(n, dimension)
        self.castor_graph = self.init_random_graph(n)
        self.pollux_graph = self.init_random_graph(n)

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


class CoupledNetworkCastorAndPollux(IntervalOpinion):

    def __init__(self, n: int, dimension: int, type: str = 'persistent', value: float = 0.5):
        super().__init__(n, dimension)
        if type not in {'persistent', 'dynamic'}:
            raise InvalidParameterError(f"{type} is not in allowed types [persistent, dynamic]!")
        self.type = type
        self.value = value
        self.castor_graph = self.init_random_graph(n)
        self.pollux_graph = self.init_random_graph(n)

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
            denominator = 1 if self.type == 'persistent' else epsilon + self.distance(opinions[:, i], opinions[:, j])
            castor_to_pollux[i][i] = self.value / denominator

        return castor_to_pollux

    def get_pollux_to_castor(self, opinions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        n, epsilon = self.n, 0.1
        pollux_to_castor = np.zeros(shape=(n, n))

        for i in range(n):
            j = i + n
            denominator = 1 if self.type == 'persistent' else epsilon + self.distance(opinions[:, j], opinions[:, i])
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


class FullyCoupledNetworkCastorAndPollux(IntervalOpinion):

    def __init__(self, n: int, dimension: int, value: float = 0.5):
        super().__init__(n, dimension)
        self.value = value
        self.castor_graph = self.init_random_graph(n)
        self.pollux_graph = self.init_random_graph(n)

    def init_dynamic_matrix(self) -> None:
        self.dynamic_matrix = self.update_dynamic_matrix()

    @staticmethod
    def projection(point: npt.ArrayLike, line: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return (np.dot(line, point) / np.dot(line, line)) * line

    def alpha_castor(self, castor_i: npt.ArrayLike,
                     castor_j: npt.ArrayLike, pollux_i: npt.ArrayLike) -> float:
        projection_vector = self.projection(castor_j, pollux_i-castor_i)
        alpha = (projection_vector[0] - pollux_i[0]) / (castor_i[0] - pollux_i[0]) if castor_i[0] - pollux_i[0] != 0 else 1
        return alpha if 0 <= alpha <= 1 else 1 if alpha > 1 else 0

    def alpha_pollux(self, castor_i: npt.ArrayLike,
                     pollux_i: npt.ArrayLike, pollux_j: npt.ArrayLike) -> float:
        projection_vector = self.projection(pollux_j, pollux_i-castor_i)
        alpha = (projection_vector[0] - castor_i[0]) / (pollux_i[0] - castor_i[0]) if pollux_i[0] - castor_i[0] != 0 else 1
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
