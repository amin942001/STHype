"""Test SpatialGraph class"""

import networkx as nx
import pytest
from shapely import LineString, Point

from sthype import SpatialGraph
from sthype.plot import plot_spatial_graph


def test_create_spatial_graph(spatial_graph):
    """Test SpatialGraph functions"""
    assert spatial_graph.edge_pixels(1, 2) == LineString([(3, 4), (6, 7), (1, 2)])
    assert spatial_graph.edge_pixels(2, 1) == LineString([(1, 2), (6, 7), (3, 4)])
    assert spatial_graph.node_position(1) == Point(3, 4)


def test_create_spatial_graph_without_node_positions():
    """Test if not referencing node positions raise an error"""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
    with pytest.raises(KeyError):
        SpatialGraph(graph)


def test_create_spatial_graph_without_edge_pixels():
    """Test if not referencing edge pixels raise an error"""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2)])
    nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 2)}, "position")
    with pytest.raises(KeyError):
        SpatialGraph(graph)


def test_access_non_existing_edge_pixels(spatial_graph):
    """Test if accessing a non existing edge pixels raise an error"""
    with pytest.raises(KeyError):
        spatial_graph.edge_pixels(2, 3)


def test_access_non_existing_node_position(spatial_graph):
    """Test if accessing a non existing node position raise an error"""
    with pytest.raises(KeyError):
        spatial_graph.node_position(4)


def test_nodes_position_dont_match_edge_pixels():
    """Test if nodes_position_dont_match_edge_pixels raise an error"""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
    nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 0)}, "position")
    with pytest.raises(AssertionError):
        SpatialGraph(graph)


def test_plot_spatial_graph(spatial_graph):
    """Test if plot_spatial_graph works"""
    plot_spatial_graph(spatial_graph)
