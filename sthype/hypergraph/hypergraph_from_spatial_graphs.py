"""Functions to create an hypergraph from time correlated graphs"""

import networkx as nx

from .. import SpatialGraph


def hypergraph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph], dates: list[str | int], segments_length: int = 5
):
    """Create an hypergraph using SpatialGraphs

    Parameters
    ----------
    spatial_graphs : list[SpatialGraph]
        List of SpatialGraph used to create the hypergraph
    date : list[str  |  int]
        Date of the graphs used to calculate growth speed...
    """
    spatial_graphs = [
        spatial_graphs for _, spatial_graphs in sorted(zip(dates, spatial_graphs))
    ]
    dates = list(sorted(dates))

    final_graph = spatial_graphs[-1]
    segmented_graph = graph_segmentation(final_graph, segments_length)

    return segmented_graph


def graph_segmentation(
    spatial_graph: SpatialGraph, segments_length: int = 5
) -> nx.Graph:
    """Cut edges of a SpatialGraph into edge of size segments_length

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to segment
    segments_length : int, optional
        length of the subdivision of edges, by default 5

    Returns
    -------
    nx.Graph
        The segmented graph.
        Edge have attribute center and there initial edge as a set : {node1, node2}.
        Node have attribute position
    """
    label = max(spatial_graph.nodes) + 1
    graph_segmented = nx.Graph()

    nodes_position = {}
    for node1, node2 in spatial_graph.edges:
        pixels = spatial_graph.edge_pixels(node1, node2)
        new_nodes = [
            pixels.line_interpolate_point(i * segments_length)
            for i in range(int(-(-pixels.length // segments_length) + 1))
        ]
        new_edges_center = [
            pixels.line_interpolate_point((i + 0.5) * segments_length)
            for i in range(int(-(-pixels.length // segments_length)))
        ]
        for index, center in enumerate(new_edges_center):
            start, end = label - 1, label
            if index == 0:
                start = node1
            if index == len(new_nodes) - 1:
                end = node2
            graph_segmented.add_edge(start, end, center=center, edge={node1, node2})
            nodes_position[start] = new_nodes[index]
            nodes_position[end] = new_nodes[index + 1]
            label += 1
    nx.set_node_attributes(graph_segmented, nodes_position, "position")

    return graph_segmented
