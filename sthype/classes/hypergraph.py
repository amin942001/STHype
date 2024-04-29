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
