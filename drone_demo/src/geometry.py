from typing import Any, Tuple, List

class Point(object):
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


def polygon_area(points: List[Point]) -> float:
    area = 0.0

    prev = len(points) - 1
    for curr in range(len(points)):
        area += (points[prev].x + points[curr].x) * (points[prev].y - points[curr].y)
        prev = curr

    return abs(area / 2.0)

def polygon_center(points: List[Point]) -> Point:
    result: Point = Point(0, 0)

    for point in points:
        result.x += point.x
        result.y += point.y

    result.x /= len(points)
    result.y /= len(points)

    return result