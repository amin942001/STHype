"""Functions to create an hypergraph from time correlated graphs"""

from .. import SpatialGraph


def hypergraph_from_spatial_graphs(
    spatial_graphs: list[SpatialGraph], dates: list[str | int], segment_length: int = 5
):
    """Create an hypergraph using SpatialGraphs

    Parameters
    ----------
    spatial_graphs : list[SpatialGraph]
        List of SpatialGraph used to create the hypergraph
    date : list[str  |  int]
        Date of the graphs used to calculate growth speed...
    """
    spatial_graphs = [
        spatial_graphs for _, spatial_graphs in sorted(zip(dates, spatial_graphs))
    ]
    dates = list(sorted(dates))

    final_graph = spatial_graphs[-1]


def graph_segmentation(spatial_graph: SpatialGraph):
    pass
