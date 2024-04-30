import networkx as nx
import pytest

from sthype import SpatialGraph


@pytest.fixture
def spatial_graph():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
    nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 2)}, "position")
    return SpatialGraph(graph)
