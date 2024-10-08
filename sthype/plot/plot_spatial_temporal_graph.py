import networkx as nx
from matplotlib import cm

from .. import SpatialTemporalGraph
from .utils import in_region, random_color


def plot_spatial_temporal_graph(
    stg: SpatialTemporalGraph,
    color_group: str | int = "uniform",
    time: int = -1,
    region: None | tuple[tuple[float, float], tuple[float, float]] = None,
    add_nodes: bool = False,
    add_initial_nodes: bool = False,
    **kwargs,
):
    """Plot Spatial Temporal Graph

    Parameters
    ----------
    stg : SpatialTemporalGraph
        Spatial Temporal Graph to plot
    color_group : str | int, optional
        color group to color edge: 'random'/'edge', 'activation',
        'initial_edge', 'hyperedge'. If int, color the corresponding hyperedge in red.
        By default "uniform"
    time : int, optional
        Plot Spatial Temporal Graph at time t, by default -1
    region : None | tuple[tuple[float, float], tuple[float, float]], optional
        Region to plot, by default None
    add_nodes : bool, optional
        plot nodes, by default False
    add_initial_nodes : bool, optional
        plot initial nodes, by default False
    """
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
            if stg[edge[0]][edge[1]]["post_hyperedge_activation"] <= time
        ]

    if color_group == "random" or color_group == "edge":
        colors = [random_color() for _ in edgelist]
    elif color_group == "activation":
        activations = [
            stg[node1][node2]["post_hyperedge_activation"] for node1, node2 in edgelist
        ]
        max_activation = stg.max_age
        colors = [cm.viridis(activation / max_activation) for activation in activations]
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
    elif color_group == "width":
        widths = [
            stg[node1][node2]["initial_edge_attributes"]["width"]
            for node1, node2 in edgelist
        ]
        max_width = max(widths)
        min_width = min(widths)
        print(max_width, min_width)
        colors = [
            cm.viridis((width - min_width) / (max_width - min_width))
            for width in widths
        ]
    elif isinstance(color_group, int):
        colors = [
            "red" if stg[node1][node2]["hyperedge"] == color_group else "blue"
            for node1, node2 in edgelist
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
        **kwargs,
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
    """Plot a Spatial Temporal Graph around one of its node

    Parameters
    ----------
    stg : SpatialTemporalGraph
        Spatial Temporal Graph to plot
    node : int
        Node to plot around
    area_size : float, optional
        Size of the zone around the zone to plot, by default 100
    """
    coord = stg.positions[node]
    x, y = coord.x, coord.y
    region = (
        (x - area_size / 2, y - area_size / 2),
        (x + area_size / 2, y + area_size / 2),
    )
    plot_spatial_temporal_graph(stg=stg, region=region, **kwargs)


def plot_spatial_temporal_graph_hyperedge(
    stg: SpatialTemporalGraph,
    hyperedge: int,
    scale: float = 1.50,
    verbose: int = 0,
    **kwargs,
):
    """Plot a Spatial Temporal Graph around one of its node

    Parameters
    ----------
    stg : SpatialTemporalGraph
        Spatial Temporal Graph to plot
    node : int
        Node to plot around
    area_size : float, optional
        Size of the zone around the zone to plot, by default 100
    """
    hyperedge_initial_edges = stg.get_hyperedge_initial_edges(hyperedge)
    positions_x = [stg.positions[node].x for node, _ in hyperedge_initial_edges] + [
        stg.positions[node].x for _, node in hyperedge_initial_edges
    ]
    positions_y = [stg.positions[node].y for node, _ in hyperedge_initial_edges] + [
        stg.positions[node].y for _, node in hyperedge_initial_edges
    ]
    min_x, max_x = min(positions_x), max(positions_x)
    diff_x_to_add = (max_x - min_x) * (scale - 1) / 2
    min_y, max_y = min(positions_y), max(positions_y)
    diff_y_to_add = (max_y - min_y) * (scale - 1) / 2
    region = (
        (min_x - diff_x_to_add, min_y - diff_y_to_add),
        (max_x + diff_x_to_add, max_y + diff_y_to_add),
    )
    if verbose > 0:
        print(f"{region}")
    plot_spatial_temporal_graph(stg=stg, region=region, color_group=hyperedge, **kwargs)
