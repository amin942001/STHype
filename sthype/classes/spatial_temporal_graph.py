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
            activation step of length < smoothing // 2, by default 11
        """
        super().__init__(incoming_graph_data, **attr)
        self._edges_segments_gathered = False
        self.edges_segments: dict[Edge, list[Edge]] = self.get_edges_segments()
        self.correct_activations(smoothing)

    def get_edges_segments(self) -> dict[Edge, list[Edge]]:
        """Return a dict with edge_group as attribute and list of edge as value

        Returns
        -------
        dict[Edge, list[Edge]]
            dict with edge_group as attribute and an ordered list of edge as value
        """
        if self._edges_segments_gathered:
            return self.edges_segments
        edges_segments: dict[Edge, list[Edge]] = {}
        for node1, node2, edge_data in self.edges(data=True):
            node_edge1, node_edge2 = min(edge_data["edge"]), max(edge_data["edge"])
            if edges_segments.get((node_edge1, node_edge2)):
                edges_segments[(node_edge1, node_edge2)].append((node1, node2))
            else:
                edges_segments[(node_edge1, node_edge2)] = [(node1, node2)]

        ordered_edges_segments: dict[Edge, list[Edge]] = {}
        for edge in edges_segments:
            edge_node1, _ = edge
            segments = edges_segments[edge].copy()

            searched_node = edge_node1
            ordered_edge_segments = []
            while segments:
                next_segment = [
                    segment for segment in segments if searched_node in segment
                ][0]
                segments.remove(next_segment)
                if next_segment[0] != searched_node:
                    next_segment = next_segment[::-1]
                ordered_edge_segments.append(next_segment)
                searched_node = next_segment[1]

            ordered_edges_segments[edge] = ordered_edge_segments

        self._edges_segments_gathered = True
        return ordered_edges_segments

    def correct_activations(self, smoothing: int = 11):
        """Add a corrected_activation attribute to the edges

        Parameters
        ----------
        smoothing : int, optional
            smoothing used in median_filter to remove errors, by default 11
        """
        for segments in self.edges_segments.values():
            activations = np.zeros(len(segments))
            for index, (node1, node2) in enumerate(segments):
                activations[index] = self[node1][node2]["activation"]

            if len(segments) >= 5:
                activations[1] = np.median(activations[:5])
                activations[-2] = np.median(activations[-5:])
            if len(segments) >= 3:
                activations[0] = np.median(activations[:3])
                activations[-1] = np.median(activations[-3:])
            corrected_activations = median_filter(
                activations,
                size=smoothing,
                mode="nearest",
            )
            corrected_activations = to_monotonic(corrected_activations)

            for corrected_activation, (node1, node2) in zip(
                corrected_activations, segments
            ):
                self[node1][node2]["corrected_activation"] = corrected_activation

    def get_edge_segments(self, node1: int, node2: int) -> list[Edge]:
        """Return the edge segments from node1 to node2

        Parameters
        ----------
        node1 : int
            starting node
        node2 : int
            ending node

        Returns
        -------
        list[Edge]
            list of edge segments from node1 to node2
        """
        if node1 < node2:
            return self.edges_segments[node1, node2]
        return [
            segment[::-1] for segment in reversed(self.edges_segments[node2, node1])
        ]
