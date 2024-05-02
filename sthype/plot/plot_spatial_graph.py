"""Functions to plot a SpatialGraph"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from shapely import LinearRing, MultiLineString, Polygon
from shapely.plotting import plot_line

from .. import SpatialGraph


def plot_spatial_graph(
    spatial_graph: SpatialGraph,
    region: tuple[tuple[int, int], tuple[int, int]] | None = None,
    add_nodes: bool = False,
    **kwargs
) -> tuple[PathPatch, Line2D] | PathPatch:
    """Plot a SpatialGraph

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to plot
    region : tuple[tuple[int, int], tuple[int, int]] | None, optional
        Region to plot if not None, by default None
    add_nodes : bool, optional
        plot nodes label if region is given, by default False

    Returns
    -------
    tuple[PathPatch, Line2D] | PathPatch
        The PathPatch representing the SpatialGraph
    """
    _, ax = plt.subplots()

    lines = [
        spatial_graph.edge_pixels(node1, node2) for node1, node2 in spatial_graph.edges
    ]
    if region:
        ax.set_xlim(region[0][0], region[1][0])
        ax.set_ylim(region[0][1], region[1][1])
        region = Polygon(
            LinearRing(
                [
                    region[0],
                    (region[0][0], region[1][1]),
                    region[1],
                    (region[1][0], region[0][1]),
                ]
            )
        )
        lines = [line for line in lines if line.intersects(region)]
        if add_nodes:
            nodes = [
                (label, position)
                for label, position in spatial_graph.positions.items()
                if position.intersects(region)
            ]
            for label, position in nodes:
                ax.annotate(label, position.coords[0])

    lines = MultiLineString(lines)

    return plot_line(lines, ax, add_points=False, **kwargs)


def plot_spatial_graph_node(
    spatial_graph: SpatialGraph,
    node: int,
    area_size: int = 100,
    add_nodes: bool = False,
    **kwargs
) -> tuple[PathPatch, Line2D] | PathPatch:
    """Plot the Spatial Graph around a node

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to plot
    node : int
        The node to look at
    area_size : int, optional
        area around the node, by default 100
    add_nodes : bool, optional
        Plot node label if True, by default False

    Returns
    -------
    tuple[PathPatch, Line2D] | PathPatch
        The plot of the area around a node of a SpatialGraph
    """
    coord = spatial_graph.positions[node]
    x, y = coord.x, coord.y
    region = [
        [x - area_size // 2, y - area_size // 2],
        [x + area_size // 2, y + area_size // 2],
    ]
    return plot_spatial_graph(spatial_graph, region, add_nodes=add_nodes, **kwargs)
