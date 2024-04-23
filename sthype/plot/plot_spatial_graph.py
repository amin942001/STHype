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
    ax.set_xlim(np.min(lines[:, :, 0]), np.max(lines[:, :, 0]))
    ax.set_ylim(np.min(lines[:, :, 1]), np.max(lines[:, :, 1]))
    line_segments = LineCollection(lines)  # type: ignore
    ax.add_collection(line_segments)
    if plot:
        plt.plot()

    return fig
