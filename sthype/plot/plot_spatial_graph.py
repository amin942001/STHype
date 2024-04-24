"""Functions to plot a SpatialGraph"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from shapely import LinearRing, MultiLineString, Polygon
from shapely.plotting import plot_line

from .. import SpatialGraph


def plot_spatial_graph(
    spatial_graph: SpatialGraph,
    add_points: bool = False,
    region: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[PathPatch, Line2D] | PathPatch:
    """Plot a SpatialGraph

    Parameters
    ----------
    spatial_graph : SpatialGraph
        The SpatialGraph to plot
    plot : bool, optional
        If you don't want to plot set to False, by default True

    Returns
    -------
    Figure
        The Figure representing the SpatialGraph
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

    lines = MultiLineString(lines)

    return plot_line(lines, ax, add_points=add_points)
