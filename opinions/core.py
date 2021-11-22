from .exceptions import PointsDimensionError

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Point:

    dimension: int
    points: np.array
    name: str
    in_degree: int = 0
    out_degree: int = 0
    point_id: int = -1


@dataclass
class Interval:

    castor: Point
    pollux: Point
    interval_id: int = -1

@dataclass
class Edge:

    dst: int
    weight: float


@dataclass
class PointAdjacencyDict:

    edges: Dict[int, Edge]


@dataclass
class IntervalAdjacencyDict:

    castor_edges: PointAdjacencyDict
    pollux_edges: PointAdjacencyDict


class PointOperations:

    @staticmethod
    def dot_product(p1: Point, p2: Point) -> float:
        if p1.dimension != p2.dimension:
            raise PointsDimensionError(f"p1 dimension: {p1.dimension} != p2 dimension: {p2.dimension}")

        if p1.dimension == 0:
            return 0

        return np.dot(p1.points, p2.points)

    @staticmethod
    def cross_product(p1: Point, p2: Point) -> np.array:
        try:
            return np.cross(p1, p2)
        except ValueError:
            raise PointsDimensionError(f"p1 dimension: {p1.dimension} and p2 dimension: {p2.dimension} is not 2 or 3")
