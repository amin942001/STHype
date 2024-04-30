"""HyperGraph Class"""

import networkx as nx
import numpy as np
from shapely import Point


class HyperGraph(nx.Graph):
    """HyperGraph

    Parameters
    ----------
    nx : Graph
        SegmentedGraph with activation time and centers attributes to the edges
    """

    def __init__(self, incoming_graph_data=None, timestamps=None, **attr):
        """Init an hypergraph

        Parameters
        ----------
        incoming_graph_data : nx.Graph, optional
            A segmented graph.
            Edge have attribute activation time, center, centers and there initial
            edge as a set : {node1, node2}.
            Node have attribute position,
            by default None
        timestamps : list[int], optional
            timestamp of each activation time
        """
        super().__init__(incoming_graph_data, **attr)
        self.timestamps = timestamps
        self.positions: dict[int, Point] = nx.get_node_attributes(
            incoming_graph_data, "position"
        )
        self.edges_segments: dict[str, list[tuple[int, int]]] = (
            self.load_edges_segments()
        )
        self.edges_graph: nx.DiGraph = self.load_edge_graph()

    def load_edges_segments(self) -> dict[str, list[tuple[int, int]]]:
        edges_segments: dict[str, list[tuple[int, int]]] = {}
        for node1, node2, edge_data in self.edges(data=True):
            start, end = min(edge_data["edge"]), max(edge_data["edge"])
            if f"{start},{end}" in edges_segments:
                edges_segments[f"{start},{end}"].append((node1, node2))
            else:
                edges_segments[f"{start},{end}"] = [(node1, node2)]

        ordered_edges_segments: dict[str, list[tuple[int, int]]] = {}
        for edge in edges_segments:
            start, end = edge.split(",")
            start, end = int(start), int(end)
            segments = edges_segments[edge].copy()

            searched_node = start
            ordered_segments = []
            while segments:
                next_segment = [
                    segment for segment in segments if searched_node in segment
                ][0]
                segments.remove(next_segment)
                if next_segment[0] != searched_node:
                    next_segment[::-1]
                ordered_segments.append(next_segment)
                searched_node = next_segment[1]

            ordered_edges_segments[edge] = ordered_segments
            ordered_edges_segments[f"{end},{start}"] = [
                segment[::-1] for segment in reversed(ordered_segments)
            ]

        return ordered_edges_segments

    def load_edge_graph(self) -> nx.DiGraph:
        edges_graph = nx.DiGraph()
        visited_edge = set()
        for edge, segments in self.edges_segments.items():
            reversed_edge = ",".join(reversed(edge.split(",")))
            if reversed_edge in visited_edge:
                continue
            start, end = edge.split(",")
            start, end = int(start), int(end)
            timestamps_segments = [
                self.timestamps[self[node1][node2]["activation"]]
                for node1, node2 in segments
            ]
            slope, constant = np.polyfit(
                np.arange(len(timestamps_segments)), timestamps_segments, 1
            )
            if slope >= 0:
                visited_edge.add(edge)
                edges_graph.add_edge(
                    start,
                    end,
                    segments=segments,
                    begin_timestamp=slope + constant,
                    end_timestamp=slope * (len(segments) - 2) + constant,
                )
                for index, (node1, node2) in enumerate(segments):
                    self[node1][node2]["activation"] = slope * index + constant

        nx.set_node_attributes(edges_graph, self.positions, "position")
        return edges_graph

    def get_edge_segments(self, node1: int, node2: int) -> list[tuple[int, int]]:
        return self.edges_segments[f"{node1},{node2}"]
