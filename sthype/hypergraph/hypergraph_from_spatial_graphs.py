"""Functions to create an hypergraph from time correlated graphs"""

import networkx as nx
from shapely import MultiLineString, Point
from shapely.ops import nearest_points

from .. import HyperGraph, SpatialGraph


def hypergraph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph],
    timestamps: list[int],
    segments_length: float = 5,
    threshold: float = 10,
    verbose: int = 0,
) -> HyperGraph:
    """Create an hypergraph using SpatialGraphs

    Parameters
    ----------
    spatial_graphs : list[SpatialGraph]
        List of SpatialGraph used to create the hypergraph
    timestamps : list[str  |  int]
        Timestamps of the graphs used to calculate growth speed, ...
    segments_length : float, optional
        length of the subdivision of edges, by default 5
    threshold : int, optional
        The threshold at which you can say that two points are the same, by default 10
    verbose : int, optional
        If verbose greater than 0, print some messages, by default 0

    Returns
    -------
    Hypergraph
        The SpatialTemporalHypergraph of the SpatialGraphs
    """
    spatial_graphs = [
        spatial_graph for _, spatial_graph in sorted(zip(timestamps, spatial_graphs))
    ]
    timestamps = list(sorted(timestamps))

    final_graph = spatial_graphs[-1]
    if verbose > 0:
        print("Segmentation")
    segmented_graph = graph_segmentation(final_graph, segments_length)
    if verbose > 0:
        print("Segment Activation")
    segmented_graph = segmented_graph_activation(
        segmented_graph, spatial_graphs, timestamps, threshold, verbose
    )

    return HyperGraph(segmented_graph)


def graph_segmentation(
    spatial_graph: SpatialGraph, segments_length: float = 5
) -> nx.Graph:
    """Cut edges of a SpatialGraph into edge of size segments_length

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to segment
    segments_length : float, optional
        Length of the subdivision of edges, by default 5

    Returns
    -------
    nx.Graph
        The segmented graph.
        Edge have attribute center and there initial edge as a set : {node1, node2}.
        Node have attribute position
    """
    label: int = max(spatial_graph.nodes) + 1
    graph_segmented = nx.Graph()

    nodes_position: dict[int, Point] = {}

    node1: int
    node2: int
    for node1, node2 in spatial_graph.edges:
        pixels = spatial_graph.edge_pixels(node1, node2)
        new_nodes: list[Point] = [
            pixels.line_interpolate_point(i * segments_length)
            for i in range(int(-(-pixels.length // segments_length) + 1))
        ]
        new_edges_center: list[Point] = [
            pixels.line_interpolate_point((i + 0.5) * segments_length)
            for i in range(int(-(-pixels.length // segments_length)))
        ]
        for index, center in enumerate(new_edges_center):
            start, end = label - 1, label
            if index == 0:
                start = node1
            if index == len(new_edges_center) - 1:
                end = node2
            graph_segmented.add_edge(start, end, center=center, edge={node1, node2})
            nodes_position[start] = new_nodes[index]
            nodes_position[end] = new_nodes[index + 1]
            label += 1
    nx.set_node_attributes(graph_segmented, nodes_position, "position")

    return graph_segmented


def segmented_graph_activation(
    segmented_graph: nx.Graph,
    spatial_graphs: list[SpatialGraph],
    timestamps: list[int],
    threshold: float = 10,
    verbose: int = 0,
) -> nx.Graph:
    """Return (in place) the segmented graph with activation time
    and the shifted center of the edges through the list of SpatialGraph

    Parameters
    ----------
    segmented_graph : nx.Graph
        The graph where the activation time should be calculated
    spatial_graphs : list[SpatialGraph]
        The SpatialGraphs representing segmented_graph through time
    threshold : float, optional
        The threshold at which you can say that two points are the same, by default 10
    verbose : int, optional
        If verbose greater than 0, print some messages, by default 0

    Returns
    -------
    nx.Graph
        The segmented_graph with activation time
    """
    for node1, node2 in segmented_graph.edges:
        segmented_graph[node1][node2]["centers"] = []
        segmented_graph[node1][node2]["centers_distance"] = []
        segmented_graph[node1][node2]["activation"] = timestamps[
            len(spatial_graphs) - 1
        ]

    for time, spatial_graph in reversed(list(enumerate(spatial_graphs))):
        if verbose > 0:
            print(f"Comparing with graph {time}")
        skeleton = MultiLineString(
            [
                spatial_graph.edge_pixels(node1, node2)
                for node1, node2 in spatial_graph.edges
            ]
        )

        for node1, node2, edge_data in segmented_graph.edges(data=True):
            if edge_data["centers"]:
                center: Point = edge_data["centers"][-1]
            else:
                center: Point = edge_data["center"]

            closest_point: Point = nearest_points(center, skeleton)[1]
            distance: float = center.distance(closest_point)

            segmented_graph[node1][node2]["centers_distance"].append(distance)
            if distance < threshold:
                segmented_graph[node1][node2]["centers"].append(closest_point)
                segmented_graph[node1][node2]["activation"] = timestamps[time]
            else:
                segmented_graph[node1][node2]["centers"].append(center)

    return segmented_graph
