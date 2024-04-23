"""Test SpatialGraph class"""

import unittest

import networkx as nx

from sthype import SpatialGraph
from sthype.plot import plot_spatial_graph


class TestSpatialGraph(unittest.TestCase):
    """Test SpatialGraph Class"""

    def setUp(self):
        graph = nx.Graph()
        graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 2)}, "position")
        self.spatial_graph = SpatialGraph(graph)

    def test_create_spatial_graph(self):
        """Test SpatialGraph functions"""
        self.assertEqual(self.spatial_graph.edge_pixels(1, 2), [(3, 4), (6, 7), (1, 2)])
        self.assertEqual(self.spatial_graph.edge_pixels(2, 1), [(1, 2), (6, 7), (3, 4)])
        self.assertEqual(self.spatial_graph.node_position(1), (3, 4))

    def test_create_spatial_graph_without_node_positions(self):
        """Test if not referencing node positions raise an error"""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        with self.assertRaises(KeyError):
            SpatialGraph(graph)

    def test_create_spatial_graph_without_edge_pixels(self):
        """Test if not referencing edge pixels raise an error"""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2)])
        nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 2)}, "position")
        with self.assertRaises(KeyError):
            SpatialGraph(graph)

    def test_access_non_existing_edge_pixels(self):
        """Test if accessing a non existing edge pixels raise an error"""
        with self.assertRaises(KeyError):
            self.spatial_graph.edge_pixels(2, 3)

    def test_access_non_existing_node_position(self):
        """Test if accessing a non existing node position raise an error"""
        with self.assertRaises(KeyError):
            self.spatial_graph.node_position(4)

    def test_nodes_position_dont_match_edge_pixels(self):
        """Test if nodes_position_dont_match_edge_pixels raise an error"""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(graph, {1: (3, 4), 2: (1, 0)}, "position")
        with self.assertRaises(AssertionError):
            SpatialGraph(graph)

    def test_plot_spatial_graph(self):
        """Test if plot_spatial_graph works"""
        plot_spatial_graph(self.spatial_graph)


if __name__ == "__main__":
    unittest.main()
