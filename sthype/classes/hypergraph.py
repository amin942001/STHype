"""HyperGraph Class"""

import networkx as nx
from shapely import Point


class HyperGraph(nx.Graph):
    """HyperGraph

    Parameters
    ----------
    nx : Graph
        SegmentedGraph with activation time and centers attributes to the edges
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """Init an hypergraph

        Parameters
        ----------
        incoming_graph_data : nx.Graph, optional
            A segmented graph.
            Edge have attribute activation time, center, centers and there initial
            edge as a set : {node1, node2}.
            Node have attribute position,
            by default None
        """
        super().__init__(incoming_graph_data, **attr)
        self.positions: dict[int, Point] = nx.get_node_attributes(
            incoming_graph_data, "position"
        )
        self.edges_segments = self.find_edges_segments()

    def find_edges_segments(self):
        edges_segments: dict[str, list[tuple]] = {}
        for node1, node2, edge_data in self.edges(data=True):
            start, end = min(edge_data["edge"]), max(edge_data["edge"])
            if f"{start},{end}" in edges_segments:
                edges_segments[f"{start},{end}"].append((node1, node2))
            else:
                edges_segments[f"{start},{end}"] = [(node1, node2)]

        ordered_edges_segments: dict[str, list[tuple]] = {}
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

    def get_edge_segments(self, node1: int, node2: int):
        return self.edges_segments[f"{node1},{node2}"]
