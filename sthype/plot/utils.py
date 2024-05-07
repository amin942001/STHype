import random

from shapely import Point


def in_region(
    point: Point,
    region: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    return (
        region[0][0] < point.x
        and point.x < region[1][0]
        and region[0][1] < point.y
        and point.y < region[1][1]
    )


def random_color():
    return tuple(random.random() for _ in range(3))
