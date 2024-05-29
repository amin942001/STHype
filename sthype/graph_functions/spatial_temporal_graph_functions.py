from .. import SpatialTemporalGraph


def merge_hyperedges(
    stg: SpatialTemporalGraph, hyperedge0: int, hyperedge1: int, error: str = "print"
):
    initial_edges0 = stg.get_hyperedge_initial_edges(hyperedge0)
    initial_edges1 = stg.get_hyperedge_initial_edges(hyperedge1)

    if initial_edges0[0][0] == initial_edges1[0][0]:
        new_initial_edges = [
            (node2, node1) for node1, node2 in reversed(initial_edges1)
        ] + initial_edges0
    elif initial_edges0[0][0] == initial_edges1[-1][1]:
        new_initial_edges = initial_edges1 + initial_edges0
    elif initial_edges0[-1][1] == initial_edges1[0][0]:
        new_initial_edges = initial_edges0 + initial_edges1
    elif initial_edges0[-1][1] == initial_edges1[-1][1]:
        new_initial_edges = initial_edges0 + [
            (node2, node1) for node1, node2 in reversed(initial_edges1)
        ]
    else:
        if error == "error":
            raise AssertionError(
                f"Edge({hyperedge0} and {hyperedge1}) don't have matching nodes"
            )
        if error == "print":
            print(f"Edge({hyperedge0} and {hyperedge1}) don't have matching nodes")
        return

    for initial_edge in initial_edges1:
        stg.initial_graph[initial_edge[0]][initial_edge[1]]["hyperedge"] = hyperedge0
        for edge in stg.get_initial_edge_edges(*initial_edge):
            stg[edge[0]][edge[1]]["hyperedge"] = hyperedge0
    stg.hyperedges_initial_edges[hyperedge0] = new_initial_edges
    del stg.hyperedges_initial_edges[hyperedge1]
