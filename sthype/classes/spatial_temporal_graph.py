import networkx as nx
import numpy as np
from scipy.ndimage import median_filter
from shapely import Point

from ..utils import breakable_into_two_monotonic, is_monotonic, to_monotonic
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
        self.positions: dict[int, Point] = nx.get_node_attributes(
            incoming_graph_data, "position"
        )

        super().__init__(incoming_graph_data, **attr)

        self.max_age = self.get_max_age()
        self._initial_edges_edges_gathered = False
        self.initial_edges_edges: dict[InitialEdge, list[Edge]] = (
            self.get_initial_edges_edges()
        )
        self.initial_edges_time_interval: dict[InitialEdge, tuple[int, int]] = {}
        self.correct_activations(smoothing)

        self._initial_graph_gathered = False
        self.initial_graph = self.get_initial_graph()

        self._hyperedges_initial_edges_gathered = False
        self.hyperedges_initial_edges: dict[HyperEdge, list[InitialEdge]] = (
            self.get_hyperedges_initial_edges()
        )
        self.correct_activations_post_hyperedge(smoothing=2 * smoothing - 1)
        self._directed_graph_gathered = False
        self.directed_graph = self.get_directed_graph()

    def get_max_age(self) -> int:
        """Return the maximum activation in edges

        Returns
        -------
        int
            The maximum
        """
        max_age = 0
        for _, _, edge_data in self.edges(data=True):
            max_age = np.max((max_age, edge_data["activation"]))
        return max_age

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

    def get_initial_graph(self) -> nx.Graph:
        """Return the graph formed by initial edges

        Returns
        -------
        nx.Graph
            initial edges graph
        """
        if self._initial_graph_gathered:
            return self.initial_graph
        initial_graph = nx.Graph()
        for initial_edge in self.initial_edges_edges:
            first_edge_node1, first_edge_node2 = self.initial_edges_edges[initial_edge][
                len(self.initial_edges_edges[initial_edge]) // 2
            ]
            attributes = self[first_edge_node1][first_edge_node2]
            initial_graph.add_edge(
                *initial_edge,
                edges=self.initial_edges_edges[initial_edge],
                time_interval=self.initial_edges_time_interval[initial_edge],
                **attributes,
            )
        nx.set_node_attributes(initial_graph, self.positions, "position")

        self.initial_graph = initial_graph
        self._initial_graph_gathered = True
        return initial_graph

    def get_hyperedges_initial_edges(self) -> dict[HyperEdge, list[InitialEdge]]:
        """Return a dict with hyperedges as key
        and the list of initial edge forming them as value

        Returns
        -------
        dict[HyperEdge, list[InitialEdge]]
            The dict
        """
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
            for node_begin, node_end in self.get_initial_edge_edges(node1, node2):
                self[node_begin][node_end]["hyperedge"] = hyperedge
            if hyperedges_initial_edges.get(hyperedge):
                hyperedges_initial_edges[hyperedge].append(initial_edge)
            else:
                hyperedges_initial_edges[hyperedge] = [initial_edge]

        ordered_hyperedges_initial_edges: dict[HyperEdge, list[InitialEdge]] = {}
        for hyperedge, initial_edges in hyperedges_initial_edges.items():
            nodes_hyperedge = np.array(initial_edges).flatten()
            nodes, count = np.unique(nodes_hyperedge, return_counts=True)

            if nodes[count % 2 == 1].size != 2:
                first_node, last_node = nodes[0], nodes[0]
            else:
                first_node, last_node = nodes[count % 2 == 1]
            initial_edges_to_search_in = initial_edges.copy()

            searched_node = first_node
            ordered_hyperedge_initial_edges = []
            while searched_node != last_node or not ordered_hyperedge_initial_edges:
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

    def correct_activations_post_hyperedge(self, smoothing: int = 21):
        """Correct the activation and hyperedges

        Parameters
        ----------
        smoothing : int, optional
            The smoothing for median filter, by default 21
        """
        new_hyperedge = max(self.hyperedges_initial_edges) + 1
        new_hyperedges_initial_edges = self.hyperedges_initial_edges.copy()
        for hyperedge, initial_edges in self.hyperedges_initial_edges.items():
            edges: list[Edge] = []

            indexes_initial_edges: list[int] = []
            activs: list[int] = []
            for initial_edge in initial_edges:
                for node1, node2 in self.get_initial_edge_edges(*initial_edge):
                    edges.append((node1, node2))
                    activs.append(self[node1][node2]["corrected_activation"])
                indexes_initial_edges.append(len(activs))
            indexes_initial_edges.pop()
            activations = np.array(activs)

            if is_monotonic(activations):
                for activation, (node1, node2) in zip(activations, edges):
                    self[node1][node2]["post_hyperedge_activation"] = activation
                continue

            activations_med_fil = median_filter(activations, smoothing)
            if is_monotonic(activations_med_fil):
                if (activations_med_fil == activations_med_fil[0]).all():
                    break_point = breakable_into_two_monotonic(
                        activations, indexes_initial_edges
                    )
                    if break_point >= 0:
                        new_hyperedges_initial_edges[hyperedge] = []
                        new_hyperedges_initial_edges[new_hyperedge] = []
                        for index_initial_edge, initial_edge in enumerate(
                            initial_edges
                        ):
                            if index_initial_edge <= break_point:
                                new_hyperedges_initial_edges[new_hyperedge].append(
                                    initial_edge
                                )
                                self.initial_graph[initial_edge[0]][initial_edge[1]][
                                    "hyperedge"
                                ] = new_hyperedge
                                for node1, node2 in self.get_initial_edge_edges(
                                    *initial_edge
                                ):
                                    self[node1][node2]["hyperedge"] = new_hyperedge
                            else:
                                new_hyperedges_initial_edges[hyperedge].append(
                                    initial_edge
                                )
                            for node1, node2 in self.get_initial_edge_edges(
                                *initial_edge
                            ):
                                self[node1][node2]["post_hyperedge_activation"] = self[
                                    node1
                                ][node2]["corrected_activation"]
                        new_hyperedge += 1
                        continue

                for activation, (node1, node2) in zip(activations_med_fil, edges):
                    self[node1][node2]["post_hyperedge_activation"] = activation
                continue

            break_point = breakable_into_two_monotonic(
                activations, indexes_initial_edges
            )
            if break_point >= 0:
                new_hyperedges_initial_edges[hyperedge] = []
                new_hyperedges_initial_edges[new_hyperedge] = []
                for index_initial_edge, initial_edge in enumerate(initial_edges):
                    if index_initial_edge <= break_point:
                        new_hyperedges_initial_edges[new_hyperedge].append(initial_edge)
                        self.initial_graph[initial_edge[0]][initial_edge[1]][
                            "hyperedge"
                        ] = new_hyperedge
                        for node1, node2 in self.get_initial_edge_edges(*initial_edge):
                            self[node1][node2]["hyperedge"] = new_hyperedge
                    else:
                        new_hyperedges_initial_edges[hyperedge].append(initial_edge)
                    for node1, node2 in self.get_initial_edge_edges(*initial_edge):
                        self[node1][node2]["post_hyperedge_activation"] = self[node1][
                            node2
                        ]["corrected_activation"]
                new_hyperedge += 1
                continue

            new_smoothing = smoothing + 2
            while not (is_monotonic(activations_med_fil)):
                activations_med_fil = median_filter(activations, new_smoothing)
                new_smoothing += 2

            for activation, (node1, node2) in zip(activations_med_fil, edges):
                self[node1][node2]["post_hyperedge_activation"] = activation
            continue

        for node1, node2, edge_data in self.edges(data=True):
            if "post_hyperedge_activation" not in edge_data:
                self[node1][node2]["post_hyperedge_activation"] = self[node1][node2][
                    "corrected_activation"
                ]

        self.hyperedges_initial_edges = new_hyperedges_initial_edges

    def get_directed_graph(self) -> nx.DiGraph:
        """Get directed graph

        Returns
        -------
        nx.DiGraph
            The directed graph
        """
        if self._directed_graph_gathered:
            return self.directed_graph

        di_graph = nx.DiGraph()
        for initial_edges in self.hyperedges_initial_edges.values():
            left_edge = self.get_initial_edge_edges(*initial_edges[0])[0]
            right_edge = self.get_initial_edge_edges(*initial_edges[-1])[-1]
            left_time = self[left_edge[0]][left_edge[1]]["post_hyperedge_activation"]
            right_time = self[right_edge[0]][right_edge[1]]["post_hyperedge_activation"]
            if left_time < right_time or (
                left_time == right_time
                and self.degree[self.get_initial_edge_edges(*initial_edges[-1])[-1][1]]
                <= self.degree[self.get_initial_edge_edges(*initial_edges[0])[0][0]]
            ):
                for node1, node2 in initial_edges:
                    di_graph.add_edge(node1, node2, **self.initial_graph[node1][node2])
            else:
                for node1, node2 in initial_edges:
                    di_graph.add_edge(node2, node1, **self.initial_graph[node1][node2])
        nx.set_node_attributes(di_graph, self.positions, "position")

        return di_graph

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

    def get_hyperedge_initial_edges(self, hyperedge: HyperEdge) -> list[InitialEdge]:
        """Get the initial edges of an hyperedge ordered

        Parameters
        ----------
        hyperedge : HyperEdge
            The hyperedge

        Returns
        -------
        list[InitialEdge]
            the list of initial edges forming the hyperedge
        """
        return self.get_hyperedges_initial_edges()[hyperedge]

    def get_hyperedge_edges(self, hyperedge: HyperEdge) -> list[Edge]:
        """Get the edges of an hyperedge ordered

        Parameters
        ----------
        hyperedge : HyperEdge
            The hyperedge

        Returns
        -------
        list[Edge]
            the list of edges forming the hyperedge
        """
        hyperedge_initial_edges = self.get_hyperedge_initial_edges(hyperedge)
        hyperedge_edges: list[Edge] = []
        for initial_edge in hyperedge_initial_edges:
            hyperedge_edges.extend(self.get_initial_edge_edges(*initial_edge))
        return hyperedge_edges

    def get_edge_attribute_list(self, node1: int, node2: int, attribute) -> list:
        """Return a list of attribute through time of an edge

        Parameters
        ----------
        node1 : int
            first node
        node2 : int
            second node
        attribute : any
            the attribute

        Returns
        -------
        list
            the list of attribute
        """
        attribute_list = []
        for time in range(self.max_age + 1):
            attribute_list.append(self[node1][node2][f"{time}"].get(attribute))

        return attribute_list

    def get_initial_edge_attribute_list(
        self, node1: int, node2: int, attribute
    ) -> list:
        """Return a list of attribute through time of an initial_edge

        Parameters
        ----------
        node1 : int
            first node
        node2 : int
            second node
        attribute : any
            the attribute

        Returns
        -------
        list
            the list of attribute
        """
        attribute_list = []
        for time in range(self.max_age + 1):
            attribute_list.append(
                self.get_initial_graph()[node1][node2][f"{time}"].get(attribute)
            )

        return attribute_list

    def get_graph_at(self, time: int) -> nx.Graph:
        """Get graph at time t

        Parameters
        ----------
        time : int
            the time

        Returns
        -------
        nx.Graph
            the initial graph at time t
        """
        graph = nx.Graph()

        for hyperedge, initial_edges in self.hyperedges_initial_edges.items():
            for initial_edge in initial_edges:
                edges = self.get_initial_edge_edges(*initial_edge)
                activation = self[edges[0][0]][edges[0][1]]["post_hyperedge_activation"]
                begin = edges[0][0]
                pixels: list[Point] = [self.positions[begin]]
                attributes = {}
                for node1, node2 in edges:
                    if self[node1][node2]["post_hyperedge_activation"] != activation:
                        graph.add_edge(
                            begin,
                            end,
                            pixels=pixels,
                            activation=activation,
                            hyperedge=hyperedge,
                            attributes=attributes,
                        )
                        activation = self[node1][node2]["post_hyperedge_activation"]
                        begin = node1
                        pixels: list[Point] = [self.positions[begin]]
                        attributes = {}
                    end = node2
                    for key, value in self[node1][node2][f"{time}"].items():
                        if key != "pixels":
                            attributes.setdefault(key, []).append(value)
                    pixels.append(self.positions[node2])
                graph.add_edge(
                    begin,
                    end,
                    pixels=pixels,
                    activation=activation,
                    attributes=attributes,
                )
        nx.set_node_attributes(graph, self.positions, "position")

        return graph
