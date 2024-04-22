"""SpatialGraph Class"""

import networkx as nx


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
        for node, node_data in incoming_graph_data.nodes(data=True):
            assert (
                "position" in node_data
            ), f"Node({node}) doesn't have attribute position"
        super().__init__(incoming_graph_data, **attr)
        self.positions = nx.get_node_attributes(self, "position")
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
        for node1, node2, edge_data in self.edges(data=True):
            assert (
                "pixels" in edge_data
            ), f"Edge({node1}, {node2}) doesn't have attribute pixels"
            edge_pixels = edge_data["pixels"]
            assert (
                hasattr(edge_pixels, "__len__")
                and len(edge_pixels)
                and hasattr(edge_pixels[0], "__len__")
                and hasattr(edge_pixels[-1], "__len__")
                and len(edge_pixels[0]) == 2
                and len(edge_pixels[-1]) == 2
            ), (
                f"Edge({node1}, {node2}) pixels doesn't have the format"
                f"array_like[tuple[int, int]]: {edge_pixels}"
            )
            if (
                edge_pixels[0][0] == positions[node1][0]
                and edge_pixels[-1][0] == positions[node2][0]
                and edge_pixels[0][1] == positions[node1][1]
                and edge_pixels[-1][1] == positions[node2][1]
            ):
                undirected_graph.add_edge(node1, node2, pixels=edge_pixels)
                undirected_graph.add_edge(
                    node2, node1, pixels=list(reversed(edge_pixels))
                )
            elif (
                edge_pixels[0][0] == positions[node2][0]
                and edge_pixels[-1][0] == positions[node1][0]
                and edge_pixels[0][1] == positions[node2][1]
                and edge_pixels[-1][1] == positions[node1][1]
            ):
                undirected_graph.add_edge(node2, node1, pixels=edge_pixels)
                undirected_graph.add_edge(
                    node1, node2, pixels=list(reversed(edge_pixels))
                )
            else:
                raise AssertionError(
                    f"Edge({node1}, {node2}) pixels don't match it's nodes:"
                    f"\npixels: start: {edge_pixels[0]}, end: {edge_pixels[-1]}"
                    f"\nnodes: {node1}: {positions[node1]}, {node2}: {positions[node2]}"
                )
        nx.set_node_attributes(undirected_graph, positions, "position")
        return undirected_graph

    def edge_pixels(self, node1: int, node2: int) -> list[tuple[int, int]]:
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

    def node_position(self, node: int) -> tuple[int, int]:
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
