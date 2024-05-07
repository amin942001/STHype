import networkx as nx

from .. import SpatialTemporalGraph
from .utils import in_region, random_color


def plot_spatial_temporal_graph(
    stg: SpatialTemporalGraph,
    color_group: str = "uniform",
    time: int = -1,
    region: None | tuple[tuple[float, float], tuple[float, float]] = None,
    add_nodes: bool = False,
    add_initial_nodes: bool = False,
    **kwargs
):
    if region:
        edgelist = []
        for node1, node2 in stg.edges:
            if in_region(stg.positions[node1], region) or in_region(
                stg.positions[node2], region
            ):
                edgelist.append((node1, node2))
    else:
        edgelist = list(stg.edges())

    if time >= 0:
        edgelist = [
            edge
            for edge in edgelist
            if stg[edge[0]][edge[1]]["corrected_activation"] <= time
        ]

    if color_group == "random" or color_group == "edge":
        colors = [random_color() for _ in edgelist]
    elif color_group == "initial_edge":
        colors_dict = {
            initial_edge: random_color() for initial_edge in stg.initial_edges_edges
        }
        edges_initial_edge = [
            stg[edge[0]][edge[1]]["initial_edge"] for edge in edgelist
        ]
        edges_initial_edge = [
            (min(initial_edge), max(initial_edge))
            for initial_edge in edges_initial_edge
        ]
        colors = [colors_dict[initial_edge] for initial_edge in edges_initial_edge]
    elif color_group == "hyperedge":
        colors_dict = {
            hyperedge: random_color() for hyperedge in stg.hyperedges_initial_edges
        }
        colors = [
            colors_dict[stg[node1][node2]["hyperedge"]] for node1, node2 in edgelist
        ]
    else:
        colors = "k"
    pos = {node: (point.x, point.y) for node, point in stg.positions.items()}

    nx.draw_networkx(
        stg,
        pos=pos,
        with_labels=add_nodes,
        nodelist=[],
        edgelist=edgelist,
        edge_color=colors,
        **kwargs
    )

    if add_initial_nodes:
        initial_nodes = [node for node, _ in stg.initial_edges_edges] + [
            node for _, node in stg.initial_edges_edges
        ]
        nodelist = [node for node, _ in edgelist if node in initial_nodes] + [
            node for _, node in edgelist if node in initial_nodes
        ]
        labels = {node: node for node in nodelist}
        nx.draw_networkx_labels(stg, pos=pos, labels=labels, **kwargs)


def plot_spatial_temporal_graph_node(
    stg: SpatialTemporalGraph, node: int, area_size: float = 100, **kwargs
):
    coord = stg.positions[node]
    x, y = coord.x, coord.y
    region = (
        (x - area_size / 2, y - area_size / 2),
        (x + area_size / 2, y + area_size / 2),
    )
    plot_spatial_temporal_graph(stg=stg, region=region, **kwargs)
