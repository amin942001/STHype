"""HyperGraph Class"""

import networkx as nx
import numpy as np
from shapely import Point


def match_edges(
    in_edges: list[tuple[int, int, int]], out_edges: list[tuple[int, int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if not in_edges or not out_edges:
        return []

    start = in_edges[0][1:3]
    end = out_edges[0][1:3]
    if end == start:
        if len(out_edges) == 1:
            return []
        end = out_edges[1][1:3]
    matched = (start, end)
    return [matched] + match_edges(
        [edge for edge in in_edges if edge[1:3] not in matched],
        [edge for edge in out_edges if edge[1:3] not in matched],
    )


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
        timestamps : list[int], optional
            timestamp of each activation time
        """
        super().__init__(incoming_graph_data, **attr)
        self.positions: dict[int, Point] = nx.get_node_attributes(
            incoming_graph_data, "position"
        )
        self.edges_segments: dict[str, list[tuple[int, int]]] = (
            self.load_edges_segments()
        )
        self.edges_graph: nx.DiGraph = self.load_edge_graph()
        # self.load_hyphaes()

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
                    next_segment = next_segment[::-1]
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
                self[node1][node2]["activation"] for node1, node2 in segments
            ]
            if len(timestamps_segments) == 1:
                slope, constant = 0, timestamps_segments[0]
            else:
                slope, constant = np.polyfit(
                    np.arange(len(timestamps_segments)), timestamps_segments, 1
                )
            if slope > 0:
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
            elif slope == 0:
                visited_edge.add(edge)
                edges_graph.add_edge(
                    start,
                    end,
                    segments=segments,
                    begin_timestamp=constant,
                    end_timestamp=constant,
                )
                for node1, node2 in segments:
                    self[node1][node2]["activation"] = constant

        nx.set_node_attributes(edges_graph, self.positions, "position")
        return edges_graph

    def load_hyphaes(self):
        for node in self.edges_graph:
            in_edges = [
                (edge_data["end_timestamp"], node1, node2)
                for node1, node2, edge_data in self.edges_graph.in_edges(
                    node, data=True
                )
            ]
            out_edges = [
                (edge_data["begin_timestamp"], node1, node2)
                for node1, node2, edge_data in self.edges_graph.out_edges(
                    node, data=True
                )
            ]
            matches = match_edges(sorted(in_edges), sorted(out_edges))
            for edge_in, edge_out in matches:
                self.edges_graph[edge_in[0]][edge_in[1]]["main_son"] = edge_out
                self.edges_graph[edge_out[0]][edge_out[1]]["parent"] = edge_in

        label = 0
        for node1, node2, edge_data in self.edges_graph.edges(data=True):
            if "parent" not in edge_data:
                node_begin, node_end = node1, node2
                self.edges_graph[node_begin][node_end]["hyphae"] = label
                while "main_son" in self.edges_graph[node_begin][node_end]:
                    node_begin, node_end = self.edges_graph[node_begin][node_end][
                        "main_son"
                    ]
                    self.edges_graph[node_begin][node_end]["hyphae"] = label
                label += 1

    def get_edge_segments(self, node1: int, node2: int) -> list[tuple[int, int]]:
        return self.edges_segments[f"{node1},{node2}"]
