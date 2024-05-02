"""Init to make sthype import easier"""

from .classes.hypergraph import HyperGraph
from .classes.spatial_graph import SpatialGraph
from .classes.spatial_temporal_graph import SpatialTemporalGraph

__all__ = ("SpatialGraph", "HyperGraph", "SpatialTemporalGraph")
