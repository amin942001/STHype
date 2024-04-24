"""Functions to plot a SpatialGraph"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from shapely import MultiLineString
from shapely.plotting import plot_line

from .. import SpatialGraph


def plot_spatial_graph(
    spatial_graph: SpatialGraph, add_points=False
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
    lines = MultiLineString(
        [
            spatial_graph.edge_pixels(node1, node2)
            for node1, node2 in spatial_graph.edges
        ]
    )
    _, ax = plt.subplots()

    return plot_line(lines, ax, add_points=add_points)
