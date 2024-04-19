import unittest
import networkx as nx
from sthype import SpatialGraph


class TestSpatialGraph(unittest.TestCase):

    def test_create_spatial_graph(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(g, {1: (3, 4), 2: (1, 2)}, "position")
        sg = SpatialGraph(g)

        self.assertEqual(sg.edge_pixels(1, 2), [(3, 4), (6, 7), (1, 2)])
        self.assertEqual(sg.edge_pixels(2, 1), [(1, 2), (6, 7), (3, 4)])
        self.assertEqual(sg.node_position(1), (3, 4))

    def test_create_spatial_graph_without_node_positions(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        with self.assertRaises(AssertionError):
            sg = SpatialGraph(g)

    def test_create_spatial_graph_without_edge_pixels(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2)])
        nx.set_node_attributes(g, {1: (3, 4), 2: (1, 2)}, "position")
        with self.assertRaises(AssertionError):
            SpatialGraph(g)

    def test_access_non_existing_edge_pixels(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(g, {1: (3, 4), 2: (1, 2)}, "position")
        sg = SpatialGraph(g)
        with self.assertRaises(AssertionError):
            sg.edge_pixels(2, 3)

    def test_access_non_existing_node_position(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(g, {1: (3, 4), 2: (1, 2)}, "position")
        sg = SpatialGraph(g)
        with self.assertRaises(AssertionError):
            sg.node_position(4)

    def test_nodes_position_dont_match_edge_pixels(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"pixels": [(1, 2), (6, 7), (3, 4)]})])
        nx.set_node_attributes(g, {1: (3, 4), 2: (1, 0)}, "position")
        with self.assertRaises(Exception):
            SpatialGraph(g)


if __name__ == "__main__":
    unittest.main()
