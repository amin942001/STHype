"""Functions to plot a SpatialGraph"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .. import SpatialGraph


def plot_spatial_graph(spatial_graph: SpatialGraph, plot=True) -> Figure:
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
    lines = np.array(
        [edge_data["pixels"] for _, _, edge_data in spatial_graph.edges(data=True)]
    )
    fig, ax = plt.subplots()
    x_min, x_max = np.min(lines[:, :, 0]), np.max(lines[:, :, 0])
    x_min, x_max = x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)
    y_min, y_max = np.min(lines[:, :, 1]), np.max(lines[:, :, 1])
    y_min, y_max = y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    line_segments = LineCollection(lines)  # type: ignore
    ax.add_collection(line_segments)
    if plot:
        plt.plot()

    return fig
