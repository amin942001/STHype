from itertools import combinations

import numpy as np
from shapely import Point

from ...utils import score_angle

Edge = tuple[int, int]


def edges_couple_from_nodes(
    node_intersection: int, node1: int, node2: int, nodes_times: dict[int, int]
) -> tuple[Edge, Edge, dict[int, int]]:
    if node1 < node_intersection:
        edge1 = node1, node_intersection
    else:
        edge1 = node_intersection, node1
    if node2 < node_intersection:
        edge2 = node2, node_intersection
    else:
        edge2 = node_intersection, node2
    del nodes_times[node1]
    del nodes_times[node2]
    return edge1, edge2, nodes_times


def edge_matches_time_angle(
    node_intersection: int, nodes_times: dict[int, int], positions: dict[int, Point]
) -> list[tuple[Edge, Edge]]:
    if len(nodes_times) < 2:
        return []

    min_time = min(nodes_times.values())
    min_time_nodes = [node for node, time in nodes_times.items() if time == min_time]

    if len(min_time_nodes) >= 2:
        pairs = combinations(min_time_nodes, 2)
        best_score = -np.pi
        for begin_node, end_node in pairs:
            score = score_angle(
                positions[begin_node], positions[end_node], positions[node_intersection]
            )
            if score >= best_score:
                best_score = score
                node1 = begin_node
                node2 = end_node
        edge1, edge2, nodes_times = edges_couple_from_nodes(
            node_intersection, node1, node2, nodes_times
        )
        return [(edge1, edge2)] + edge_matches_time_angle(
            node_intersection, nodes_times, positions
        )

    node1 = min_time_nodes[0]
    second_min_time = min([time for node, time in nodes_times.items() if node != node1])
    second_time_nodes = [
        node for node, time in nodes_times.items() if time == second_min_time
    ]
    best_score = -np.pi
    node2 = second_time_nodes[0]
    for node in second_time_nodes:
        score = score_angle(
            positions[node1], positions[node], positions[node_intersection]
        )
        if score >= best_score:
            best_score = score
            node2 = node
    edge1, edge2, nodes_times = edges_couple_from_nodes(
        node_intersection, node1, node2, nodes_times
    )
    return [(edge1, edge2)] + edge_matches_time_angle(
        node_intersection, nodes_times, positions
    )
