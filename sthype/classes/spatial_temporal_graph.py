import networkx as nx
import numpy as np
from scipy.ndimage import median_filter

from ..utils import to_monotonic

Edge = tuple[int, int]


class SpatialTemporalGraph(nx.Graph):
    def __init__(
        self, incoming_graph_data: nx.Graph = None, smoothing: int = 11, **attr
    ):
        """SpatialTemporalGraph init

        Parameters
        ----------
        incoming_graph_data : nx.Graph, optional
            A graph with attribute edge and center and activation to each of its edge,
            and position to each of its node, by default None
        smoothing : int, optional
            A smoothing to correct activation,
            more precisely, activation step of length < smoothing // 2, by default 11
        """
        super().__init__(incoming_graph_data, **attr)
        self._initial_edges_edges_gathered = False
        self.initial_edges_edges: dict[Edge, list[Edge]] = (
            self.get_initial_edges_edges()
        )
        self.correct_activations(smoothing)

    def get_initial_edges_edges(self) -> dict[Edge, list[Edge]]:
        """Return a dict with initial_edge as attribute and list of edge as value

        Returns
        -------
        dict[Edge, list[Edge]]
            dict with initial_edge as attribute and an ordered list of edge as value
        """
        if self._initial_edges_edges_gathered:
            return self.initial_edges_edges
        initial_edges_edges: dict[Edge, list[Edge]] = {}
        for node1, node2, edge_data in self.edges(data=True):
            node_initial_edge1 = min(edge_data["initial_edge"])
            node_initial_edge2 = max(edge_data["initial_edge"])
            if initial_edges_edges.get((node_initial_edge1, node_initial_edge2)):
                initial_edges_edges[node_initial_edge1, node_initial_edge2].append(
                    (node1, node2)
                )
            else:
                initial_edges_edges[node_initial_edge1, node_initial_edge2] = [
                    (node1, node2)
                ]

        ordered_initial_edges_edges: dict[Edge, list[Edge]] = {}
        for initial_edge in initial_edges_edges:
            initial_edge_node1, _ = initial_edge
            edges = initial_edges_edges[initial_edge].copy()

            searched_node = initial_edge_node1
            ordered_initial_edge_edges = []
            while edges:
                next_edge = [edge for edge in edges if searched_node in edge][0]
                edges.remove(next_edge)
                if next_edge[0] != searched_node:
                    next_edge = next_edge[::-1]
                ordered_initial_edge_edges.append(next_edge)
                searched_node = next_edge[1]

            ordered_initial_edges_edges[initial_edge] = ordered_initial_edge_edges

        self._initial_edges_edges_gathered = True
        return ordered_initial_edges_edges

    def correct_activations(self, smoothing: int = 11):
        """Add a corrected_activation attribute to the edges

        Parameters
        ----------
        smoothing : int, optional
            smoothing used in median_filter to remove errors, by default 11
        """
        for edges in self.initial_edges_edges.values():
            activations = np.zeros(len(edges))
            for index, (node1, node2) in enumerate(edges):
                activations[index] = self[node1][node2]["activation"]

            if len(edges) >= 5:
                activations[1] = np.median(activations[:5])
                activations[-2] = np.median(activations[-5:])
            if len(edges) >= 3:
                activations[0] = np.median(activations[:3])
                activations[-1] = np.median(activations[-3:])
            corrected_activations = median_filter(
                activations,
                size=smoothing,
                mode="nearest",
            )
            corrected_activations = to_monotonic(corrected_activations)

            for corrected_activation, (node1, node2) in zip(
                corrected_activations, edges
            ):
                self[node1][node2]["corrected_activation"] = corrected_activation

    def get_initial_edge_edges(self, node1: int, node2: int) -> list[Edge]:
        """Return the edges of an initial_edge from node1 to node2

        Parameters
        ----------
        node1 : int
            starting node
        node2 : int
            ending node

        Returns
        -------
        list[Edge]
            list of the edges of an initial_edge from node1 to node2
        """
        if node1 < node2:
            return self.initial_edges_edges[node1, node2]
        return [edge[::-1] for edge in reversed(self.initial_edges_edges[node2, node1])]
