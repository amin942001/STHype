import networkx as nx
import numpy as np
from scipy.ndimage import median_filter

from ..utils import to_monotonic
from .utils.hyperedge import edge_matches_time_angle

Edge = tuple[int, int]
InitialEdge = tuple[int, int]  # The first node is smaller than the second one
HyperEdge = int


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
        self.smoothing = smoothing
        self.positions = nx.get_node_attributes(incoming_graph_data, "position")

        super().__init__(incoming_graph_data, **attr)

        self._initial_edges_edges_gathered = False
        self.initial_edges_edges: dict[InitialEdge, list[Edge]] = (
            self.get_initial_edges_edges()
        )
        self.initial_edges_time_interval: dict[InitialEdge, tuple[int, int]] = {}
        self.correct_activations(smoothing)

        self._initial_graph_gathered = False
        self.initial_graph = self.get_initial_graph()

        self._hyperedges_initial_edges_gathered = False
        self.hyperedges_initial_edges: dict[HyperEdge, list[Edge]] = (
            self.get_hyperedges_initial_edges()
        )

    def get_initial_edges_edges(self) -> dict[InitialEdge, list[Edge]]:
        """Return a dict with initial_edge as attribute and list of edge as value

        Returns
        -------
        dict[InitialEdge, list[Edge]]
            dict with initial_edge as attribute and an ordered list of edge as value
            The first node of an InitialEdge in smaller than the second one
        """
        if self._initial_edges_edges_gathered:
            return self.initial_edges_edges
        initial_edges_edges: dict[InitialEdge, list[Edge]] = {}
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

        ordered_initial_edges_edges: dict[InitialEdge, list[Edge]] = {}
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

        self.initial_edges_edges = ordered_initial_edges_edges
        self._initial_edges_edges_gathered = True
        return ordered_initial_edges_edges

    def correct_activations(self, smoothing: int = 11):
        """Add a corrected_activation attribute to the edges

        Parameters
        ----------
        smoothing : int, optional
            smoothing used in median_filter to remove errors, by default 11
        """
        initial_edges_time_interval: dict[InitialEdge, tuple[int, int]] = {}
        for initial_edge, edges in self.initial_edges_edges.items():
            activations = np.zeros(len(edges), dtype=int)
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
                size=np.max((len(edges), smoothing)),
                mode="nearest",
            )
            corrected_activations = to_monotonic(corrected_activations)

            initial_edges_time_interval[initial_edge] = (
                corrected_activations[0],
                corrected_activations[-1],
            )

            for corrected_activation, (node1, node2) in zip(
                corrected_activations, edges
            ):
                self[node1][node2]["corrected_activation"] = corrected_activation

        self.initial_edges_time_interval = initial_edges_time_interval
        self._initial_edges_edges_gathered = False
        self._hyperedges_initial_edges_gathered = False

    def get_initial_graph(self):
        if self._initial_graph_gathered:
            return self.initial_graph
        initial_graph = nx.Graph()
        for initial_edge in self.initial_edges_edges:
            first_edge_node1, first_edge_node2 = self.initial_edges_edges[initial_edge][
                0
            ]
            attributes = self[first_edge_node1][first_edge_node2]
            initial_graph.add_edge(
                *initial_edge,
                edges=self.initial_edges_edges[initial_edge],
                time_interval=self.initial_edges_time_interval[initial_edge],
                **attributes
            )
        nx.set_node_attributes(initial_graph, self.positions, "position")

        self.initial_graph = initial_graph
        self._initial_graph_gathered = True
        return initial_graph

    def get_hyperedges_initial_edges(self):
        if self._hyperedges_initial_edges_gathered:
            return self.hyperedges_initial_edges
        Cor: dict[InitialEdge, list[InitialEdge]] = {
            initial_edge: [(0, 0)] * 10 for initial_edge in self.initial_edges_edges
        }
        for node_intersection in self.initial_graph:
            nodes_times = {}
            for node in self.initial_graph[node_intersection]:
                if node < node_intersection:
                    nodes_times[node] = self.initial_edges_time_interval[
                        node, node_intersection
                    ][1]
                else:
                    nodes_times[node] = self.initial_edges_time_interval[
                        node_intersection, node
                    ][0]
            matches = edge_matches_time_angle(
                node_intersection, nodes_times, self.positions
            )
            for match in matches:
                edge1, edge2 = match[0], match[1]
                Cor[edge1][Cor[edge1].index((0, 0))] = edge2
                Cor[edge2][Cor[edge2].index((0, 0))] = edge1

        initial_edges_hyperedge: dict[InitialEdge, HyperEdge] = {
            initial_edge: 0 for initial_edge in self.initial_edges_edges
        }
        CurrentMark = 1
        for initial_edge in self.initial_edges_edges:
            if initial_edges_hyperedge[initial_edge] == 0:
                stack = [initial_edge]
                visited = set()
                while stack:
                    current = stack.pop()
                    initial_edges_hyperedge[current] = CurrentMark
                    related_edges = [
                        cor
                        for cor in Cor[current]
                        if cor != (0, 0)
                        and initial_edges_hyperedge[cor] == 0
                        and cor not in visited
                    ]
                    stack.extend(related_edges)
                    visited.update(related_edges)
                CurrentMark += 1

        hyperedges_initial_edges: dict[HyperEdge, list[InitialEdge]] = {}
        for initial_edge, hyperedge in initial_edges_hyperedge.items():
            node1, node2 = initial_edge
            self.initial_graph[node1][node2]["hyperedge"] = hyperedge
            if hyperedges_initial_edges.get(hyperedge):
                hyperedges_initial_edges[hyperedge].append(initial_edge)
            else:
                hyperedges_initial_edges[hyperedge] = [initial_edge]

        ordered_hyperedges_initial_edges: dict[HyperEdge, list[InitialEdge]] = {}
        for hyperedge, initial_edges in hyperedges_initial_edges.items():
            nodes_hyperedge = np.array(initial_edges).flatten()
            nodes, count = np.unique(nodes_hyperedge, return_counts=True)
            first_node = nodes[count < 2][0]
            if nodes[count > 2].size > 0:
                continue
            initial_edges_to_search_in = initial_edges.copy()

            searched_node = first_node
            ordered_hyperedge_initial_edges = []
            while initial_edges_to_search_in:
                next_initial_edge = [
                    initial_edge
                    for initial_edge in initial_edges_to_search_in
                    if searched_node in initial_edge
                ][0]
                initial_edges_to_search_in.remove(next_initial_edge)
                if next_initial_edge[0] != searched_node:
                    next_initial_edge = next_initial_edge[::-1]
                ordered_hyperedge_initial_edges.append(next_initial_edge)
                searched_node = next_initial_edge[1]

            ordered_hyperedges_initial_edges[hyperedge] = (
                ordered_hyperedge_initial_edges
            )

        self.hyperedges_initial_edges = ordered_hyperedges_initial_edges
        self._hyperedges_initial_edges_gathered = True
        return ordered_hyperedges_initial_edges

    def get_hyperedges_edges(self):
        pass

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
