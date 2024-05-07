import numpy as np
from shapely import Point


def angle_point(point1: Point, point2: Point, intersection_point: Point):
    a = np.array([point1.x, point1.y])
    c = np.array([point2.x, point2.y])
    b = np.array([intersection_point.x, intersection_point.y])
    ba = a - b
    bc = c - b

    if (ba == 0).all() or (bc == 0).all():
        return 0

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def score_angle(point1: Point, point2: Point, intersection_point: Point):
    return -np.abs(angle_point(point1, point2, intersection_point) - np.pi)
