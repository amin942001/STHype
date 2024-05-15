"""Functions to create an hypergraph from time correlated graphs"""

import networkx as nx
import numpy as np
from shapely import LineString, MultiPoint, Point, get_coordinates
from shapely.ops import nearest_points

from .. import HyperGraph, SpatialGraph, SpatialTemporalGraph


def hypergraph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph],
    timestamps: list[int],
    threshold: float = 10,
    segments_length: float = 10,
    verbose: int = 0,
) -> HyperGraph:
    return HyperGraph(
        spatial_temporal_graph_from_spatial_graphs(
            spatial_graphs, timestamps, threshold, segments_length, verbose
        )
    )


def spatial_temporal_graph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph],
    timestamps: list[int],
    threshold: float = 10,
    segments_length: float = 10,
    verbose: int = 0,
) -> SpatialTemporalGraph:
    """Create an hypergraph using SpatialGraphs

    Parameters
    ----------
    spatial_graphs : list[SpatialGraph]
        List of SpatialGraph used to create the hypergraph
    timestamps : list[str  |  int]
        Timestamps of the graphs used to calculate growth speed, ...
    threshold : int, optional
        The threshold at which you can say that two points are the same, by default 10
    segments_length : float, optional
        length of the subdivision of edges,
        having segments_length>=threshold is recommended, by default 10
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
        print("Edge Activation")
    segmented_graph = segmented_graph_activation(
        segmented_graph,
        spatial_graphs,
        timestamps,
        threshold,
        threshold / 2,
        verbose,
    )

    return SpatialTemporalGraph(segmented_graph)


def graph_segmentation(
    spatial_graph: SpatialGraph, segments_length: float = 10
) -> nx.Graph:
    """Cut edges of a SpatialGraph into edge of size segments_length

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to segment
    segments_length : float, optional
        Length of the subdivision of edges. If an edge is less than that,
        it is removed, by default 10

    Returns
    -------
    nx.Graph
        The segmented graph.
        Edge have attribute center and there initial edge as a set({node1, node2})
        and there initial edge attributes as a dict.
        Node have attribute position
    """
    label: int = max(spatial_graph.nodes) + 1
    graph_segmented = nx.Graph()

    nodes_position: dict[int, Point] = {}

    node1: int
    node2: int
    edges_to_contract: list[tuple[int, int]] = []
    for node1, node2, edge_data in spatial_graph.edges(data=True):
        pixels = spatial_graph.edge_pixels(node1, node2)
        new_nodes_amount = int(-(-pixels.length // segments_length) + 1)
        node_interpolation_positions = (
            i * (pixels.length / (new_nodes_amount - 1))
            for i in range(new_nodes_amount)
        )
        edge_interpolation_positions = (
            (i + 0.5) * (pixels.length / (new_nodes_amount - 1))
            for i in range(new_nodes_amount - 1)
        )
        new_nodes: list[Point] = [
            pixels.line_interpolate_point(position)
            for position in node_interpolation_positions
        ]
        new_edges_center: list[Point] = [
            pixels.line_interpolate_point(position)
            for position in edge_interpolation_positions
        ]
        if len(new_edges_center) == 1:
            edges_to_contract.append((node1, node2))

        for index, center in enumerate(new_edges_center):
            start, end = label - 1, label
            if index == 0:
                start = node1
            if index == len(new_edges_center) - 1:
                end = node2
            graph_segmented.add_edge(
                start,
                end,
                center=center,
                initial_edge={node1, node2},
                initial_edge_attributes=edge_data,
            )
            nodes_position[start] = new_nodes[index]
            nodes_position[end] = new_nodes[index + 1]
            label += 1
    nx.set_node_attributes(graph_segmented, nodes_position, "position")

    nodes_remapping = {node: node for node in spatial_graph}
    while edges_to_contract:
        edge_stack = [edges_to_contract.pop()]
        edges_group_to_contract = []
        while edge_stack:
            new_edge = edge_stack.pop()
            edges_group_to_contract.append(new_edge)
            edges_to_add_to_stack = [
                (node1, node2)
                for node1, node2 in edges_to_contract
                if node1 in new_edge or node2 in new_edge
            ]
            edges_to_contract = [
                edge for edge in edges_to_contract if edge not in edges_to_add_to_stack
            ]
            edge_stack.extend(edges_to_add_to_stack)

        nodes_group_to_contract = list(
            set(
                [node for node, _ in edges_group_to_contract]
                + [node for _, node in edges_group_to_contract]
            )
        )
        points = MultiPoint([nodes_position[node] for node in nodes_group_to_contract])
        new_node = max(graph_segmented.nodes) + 1
        graph_segmented.add_node(new_node, position=points.centroid)
        for node in nodes_group_to_contract:
            nx.contracted_nodes(
                graph_segmented, new_node, node, self_loops=False, copy=False
            )
            nodes_remapping[node] = new_node

    edges_to_remove = []
    for node1, node2, edge_data in graph_segmented.edges(data=True):
        node_initial1, node_initial2 = edge_data["initial_edge"]
        if nodes_remapping[node_initial1] == nodes_remapping[node_initial2]:
            edges_to_remove.append((node1, node2))
        else:
            graph_segmented[node1][node2]["initial_edge"] = {
                nodes_remapping[node_initial1],
                nodes_remapping[node_initial2],
            }
    graph_segmented.remove_edges_from(edges_to_remove)

    return graph_segmented


def segmented_skeleton(
    spatial_graph: SpatialGraph, threshold: float = 10, tolerance: float = 5
) -> dict[tuple[int, int], list[tuple[LineString, tuple[int, int]]]]:
    """Return a simplified SpatialGraph skeleton using Ramer–Douglas–Peucker algorithm

    Parameters
    ----------
    spatial_graph : SpatialGraph
        SpatialGraph to extract the skeleton from
    threshold : float, optional
        The threshold at which you can say that two points are the same, by default 10
    tolerance : float, optional
        tolerance when applying Ramer–Douglas–Peucker algorithm, by default 5

    Returns
    -------
    dict[tuple[int, int], list[tuple[LineString, tuple[int, int]]]]
        The simplified skeleton (the keys are the batches)
    """
    skeleton: dict[tuple[int, int], list[tuple[LineString, tuple[int, int]]]] = {}
    for edge in spatial_graph.edges():
        coordinates: np.ndarray = get_coordinates(
            spatial_graph.edge_pixels(*edge).simplify(tolerance).segmentize(threshold)
        )
        for line in zip(coordinates[:-1], coordinates[1:]):
            key = tuple((line[0] // threshold).astype(int))
            skeleton.setdefault(key, []).append((LineString(line), edge))

    return skeleton


def closest_point_from_skeleton(
    point: Point,
    skeleton: dict[tuple[int, int], list[tuple[LineString, tuple[int, int]]]],
    threshold: float = 10,
) -> tuple[Point, tuple[int, int], float]:
    """Closest point of a skeleton from a point

    Parameters
    ----------
    point : Point
        The point
    skeleton : dict[tuple[int, int], list[tuple[LineString, tuple[int, int]]]]
        The skeleton
    threshold : int, optional
        The threshold at which you can say that two points are the same, by default 10

    Returns
    -------
    Point
        Closest point of skeleton from point
    """
    key_x, key_y = int(point.x // threshold), int(point.y // threshold)
    closest_lines: list[tuple[LineString, tuple[int, int]]] = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            closest_lines.extend(skeleton.get((key_x + x, key_y + y), []))

    minimal_distance = threshold
    closest_edge = None
    closest_line = None
    for line, edge in closest_lines:
        distance = line.distance(point)
        if distance <= minimal_distance:
            closest_line = line
            closest_edge = edge
            minimal_distance = distance

    if closest_line is None:
        return None, None, None

    return nearest_points(point, closest_line)[1], closest_edge, minimal_distance


def segmented_graph_activation(
    segmented_graph: nx.Graph,
    spatial_graphs: list[SpatialGraph],
    timestamps: list[int],
    threshold: float = 10,
    tolerance: float = 5,
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
    timestamps : list[str  |  int]
        Timestamps of the graphs used to calculate growth speed, ...
    threshold : float, optional
        The threshold at which you can say that two points are the same, by default 10
    tolerance : float, optional
        Tolerance of Ramer–Douglas–Peucker algorithm to have a simplified skeleton,
        should be less then threshold, by default 5
    verbose : int, optional
        If verbose greater than 0, print some messages, by default 0

    Returns
    -------
    nx.Graph
        The segmented_graph with activation time
    """
    for node1, node2 in segmented_graph.edges:
        segmented_graph[node1][node2]["centers"] = []
        segmented_graph[node1][node2]["activation"] = len(spatial_graphs) - 1
        segmented_graph[node1][node2]["activation_timestamp"] = timestamps[
            len(spatial_graphs) - 1
        ]

    for time, spatial_graph in reversed(list(enumerate(spatial_graphs))):
        if verbose > 0:
            print(f"Comparing with graph {time}")
        skeleton = segmented_skeleton(spatial_graph, threshold, tolerance)

        centers: list[tuple[Point, tuple[int, int], float]] = []
        for _, _, edge_data in segmented_graph.edges(data=True):
            if edge_data["centers"]:
                center: Point = edge_data["centers"][-1]
            else:
                center: Point = edge_data["center"]
            centers.append(center)

        closest_points = [
            closest_point_from_skeleton(center, skeleton, threshold)
            for center in centers
        ]

        for center, (closest_point, closest_edge, distance), (node1, node2) in zip(
            centers, closest_points, segmented_graph.edges
        ):
            if distance is None:
                segmented_graph[node1][node2]["centers"].append(center)
                segmented_graph[node1][node2][f"{time}"] = {}
            else:
                segmented_graph[node1][node2]["centers"].append(closest_point)
                segmented_graph[node1][node2]["activation"] = time
                segmented_graph[node1][node2]["activation_timestamp"] = timestamps[time]
                segmented_graph[node1][node2][f"{time}"] = spatial_graph[
                    closest_edge[0]
                ][closest_edge[1]]

    return segmented_graph
