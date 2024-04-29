"""SpatialGraph Class"""

from typing import Sequence

import networkx as nx
from shapely import LineString, Point


class SpatialGraph(nx.Graph):
    """Essentially a nx.Graph with attribute position for nodes and pixels for edges"""

    def __init__(self, incoming_graph_data: nx.Graph, **attr):
        """Construct the Spatial Graph

        Parameters
        ----------
        incoming_graph_data : nx.Graph
            a nx.graph with attribute position for node and pixels for edges.
            Pixels boundary need to match nodes edge position.
        """
        super().__init__(incoming_graph_data, **attr)

        node_positions: dict[int, Sequence[float]] = nx.get_node_attributes(
            self, "position"
        )
        self.positions: dict[int, Point] = {
            node: Point(position) for node, position in node_positions.items()
        }

        self.undirected_graph = self.spatial_undirected_graph()

    def spatial_undirected_graph(self) -> nx.DiGraph:
        """Return a directed graph used to have pixels ordered for each edge

        Returns
        -------
        nx.DiGraph
            A directed graph with each edge having attribute pixels ordered

        Raises
        ------
        Exception
            Each edge pixels should be of type list[tuple[int, int]] and its boundary
            should match the nodes positions of the edge
        """
        positions = self.positions
        undirected_graph = nx.DiGraph()

        node1: int
        node2: int
        for node1, node2, edge_data in self.edges(data=True):
            pixels: Sequence[Sequence[float]] = edge_data["pixels"]
            edge_pixels = LineString(pixels)

            boundaries: tuple[Point, Point] = edge_pixels.boundary.geoms
            start_point, end_point = boundaries

            if start_point.equals(positions[node1]) and end_point.equals(
                positions[node2]
            ):
                undirected_graph.add_edge(node1, node2, pixels=edge_pixels)
                undirected_graph.add_edge(node2, node1, pixels=edge_pixels.reverse())
            elif start_point.equals(positions[node2]) and end_point.equals(
                positions[node1]
            ):
                undirected_graph.add_edge(node2, node1, pixels=edge_pixels)
                undirected_graph.add_edge(node1, node2, pixels=edge_pixels.reverse())
            else:
                raise AssertionError(
                    f"Edge({node1}, {node2}) pixels don't match it's nodes:"
                    f"\npixels: start: {start_point}, end: {end_point}"
                    f"\nnodes: {node1}: {positions[node1]}, {node2}: {positions[node2]}"
                )

        nx.set_node_attributes(undirected_graph, positions, "position")
        return undirected_graph

    def edge_pixels(self, node1: int, node2: int) -> LineString:
        """Return the pixel list from node1 to node2

        Parameters
        ----------
        node1 : int
            starting node label
        node2 : int
            ending node label

        Returns
        -------
        list[tuple[int, int]]
            the pixel list from node1 to node2
        """
        return self.undirected_graph[node1][node2]["pixels"]

    def node_position(self, node: int) -> Point:
        """Return the position of a node

        Parameters
        ----------
        node : int
            node label

        Returns
        -------
        tuple[int, int]
            the position of the node
        """
        return self.positions[node]
