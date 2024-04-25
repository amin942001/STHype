"""Functions to create an hypergraph from time correlated graphs"""

import networkx as nx
import numpy as np
from scipy import sparse
from shapely import LineString, Point

from .. import SpatialGraph


def hypergraph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph],
    dates: list[str | int],
    segments_length: int = 5,
    threshold: int = 10,
):
    """Create an hypergraph using SpatialGraphs

    Parameters
    ----------
    spatial_graphs : list[SpatialGraph]
        List of SpatialGraph used to create the hypergraph
    dates : list[str  |  int]
        Date of the graphs used to calculate growth speed...
    segments_length : int, optional
        length of the subdivision of edges, by default 5
    threshold : int, optional
        The threshold at which you can say that two points are the same, by default 10

    Returns
    -------
    Hypergraph
        The SpatialTemporalHypergraph of the SpatialGraphs
    """
    spatial_graphs = [
        spatial_graphs for _, spatial_graphs in sorted(zip(dates, spatial_graphs))
    ]
    dates = list(sorted(dates))

    final_graph = spatial_graphs[-1]
    segmented_graph = graph_segmentation(final_graph, segments_length)
    segmented_graph = segmented_graph_activation(
        segmented_graph, spatial_graphs, segments_length, threshold
    )

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
        Length of the subdivision of edges, by default 5

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
            if index == len(new_edges_center) - 1:
                end = node2
            graph_segmented.add_edge(start, end, center=center, edge={node1, node2})
            nodes_position[start] = new_nodes[index]
            nodes_position[end] = new_nodes[index + 1]
            label += 1
    nx.set_node_attributes(graph_segmented, nodes_position, "position")

    return graph_segmented


def closest_point_square(
    point: tuple[int, int], points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Closest point and distance from point to points

    Parameters
    ----------
    point : tuple[int, int]
        The point
    points : np.ndarray
        The list of point

    Returns
    -------
    tuple[np.ndarray, float]
        The closest point to point and the distance between them
    """
    dist_square = np.sum((points - point) ** 2, axis=1)
    min_index = np.argmin(dist_square)
    return points[min_index], dist_square[min_index]


def segmented_graph_activation(
    segmented_graph: nx.Graph,
    spatial_graphs: list[SpatialGraph],
    segments_length: int = 5,
    threshold: float = 10,
) -> nx.Graph:
    """Return (in place) the segmented graph with activation time
    and the shifted center of the edges through the list of SpatialGraph

    Parameters
    ----------
    segmented_graph : nx.Graph
        The graph where the activation time should be calculated
    spatial_graphs : list[SpatialGraph]
        The SpatialGraphs representing segmented_graph through time
    segments_length : int, optional
        Length of the subdivision of edges, by default 5
    threshold : float, optional
        The threshold at which you can say that two points are the same, by default 10

    Returns
    -------
    nx.Graph
        The segmented_graph with activation time
    """
    for node1, node2 in segmented_graph.edges:
        segmented_graph[node1][node2]["centers"] = []
        segmented_graph[node1][node2]["centers_distance"] = []
    for spatial_graph in reversed(spatial_graphs):
        rows = []
        cols = []
        for _, _, edge_data in spatial_graph.edges(data=True):
            pixels = edge_data["pixels"]
            row, col = zip(*pixels)
            rows.extend(row)
            cols.extend(col)

        data = np.ones(len(rows))
        points_matrix = sparse.csr_matrix((data, (rows, cols)))

        for node1, node2, edge_data in segmented_graph.edges(data=True):
            if edge_data["centers"]:
                xc, yc = edge_data["centers"][-1].coords[0]
            else:
                xc, yc = edge_data["center"].coords[0]
            xc, yc = int(xc), int(yc)

            min_x, max_x = max(0, xc - 4 * segments_length), xc + 4 * segments_length
            min_y, max_y = max(0, yc - 4 * segments_length), yc + 4 * segments_length
            coords = points_matrix[min_x:max_x, min_y:max_y].nonzero()
            coords = np.column_stack(coords)
            if not coords.shape[0]:
                segmented_graph[node1][node2]["centers_distance"].append(
                    32 * (segments_length**2)
                )
                segmented_graph[node1][node2]["centers"].append(Point(xc, yc))
                continue

            xc -= min_x
            yc -= min_y

            new_center, min_dist = closest_point_square([xc, yc], coords)
            segmented_graph[node1][node2]["centers_distance"].append(min_dist)
            if min_dist < threshold**2:
                segmented_graph[node1][node2]["centers"].append(
                    Point(new_center + np.array([min_x, min_y]))
                )
            else:
                segmented_graph[node1][node2]["centers"].append(
                    Point(xc + min_x, yc + min_y)
                )
    for node1, node2 in segmented_graph.edges:
        segmented_graph[node1][node2]["centers"].reverse()
        segmented_graph[node1][node2]["centers_distance"].reverse()
        centers_distance_array = np.array(
            segmented_graph[node1][node2]["centers_distance"]
        )
        activated = np.where(centers_distance_array < threshold**2, 1, 0)
        activation = np.argmax(activated)

        segmented_graph[node1][node2]["centers"] = LineString(
            segmented_graph[node1][node2]["centers"]
        )
        segmented_graph[node1][node2]["activation"] = activation

    return segmented_graph
