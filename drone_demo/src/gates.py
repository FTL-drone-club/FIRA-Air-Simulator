from typing import Any, Tuple, List
import cv2
import numpy as np
import itertools
import math

from geometry import *


# Потом прокомментирую
MIN_GATES_DIAGONAL_NON_PINK_PERCENTAGE = 0.5

# Тоже потом
MIN_GATES_EDGES_PINK_PERCENTAGE = 0.09

# Не поверите, но опять потом
CUTTING_GATES_COEFFICIENT = 0.95

# Толщина линии на маске для наложения на картинку для подсчета доли розового
LINE_MASK_THICKNESS = 3

# Опять потом
LINE_MASK_OFFSET = 3


def get_mask(
        image: Any,
        lower: np.array,
        upper: np.array
) -> cv2.Mat:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Улучшение маски с помощью морфологических операций
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Mask', mask)

    return mask

def get_rects(
        image: Any,
        lower_: np.array,
        upper_: np.array
) -> Tuple[Any, List[List[Point]]]:
    res_img_ = image.copy()
    mask_ = get_mask(image, lower_, upper_)

    # Поиск контуров
    contours_, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles_: List[List[Point]] = []
    for cnt_ in contours_:
        # Аппроксимация контура полигоном
        epsilon_ = 0.01 * cv2.arcLength(cnt_, True)
        approx_ = cv2.approxPolyDP(cnt_, epsilon_, True)

        # Отрисовка контура (опционально)
        cv2.drawContours(res_img_, [approx_], -1, (255, 255, 0), 3)

        # Отбор только четырехугольников
        # if len(approx_) == 4:
        if True:
            # Преобразование координат и добавление в результат
            points_ = approx_.reshape(-1, 2).tolist()

            curr_rect: List[Point] = []
            for point in points_:
                curr_rect.append(Point(*point))
            rectangles_.append(curr_rect)

            # Отрисовка контура (опционально)
            cv2.drawContours(res_img_, [approx_], -1, (0, 255, 0), 3)


    return res_img_, rectangles_

def get_the_biggest_polygon_in_image(
        image: Any,
        lower_: np.array = np.array([140, 0, 0]),
        upper_: np.array = np.array([200, 255, 255])
) -> List[Point]:
    image, rects = get_rects(
        image,
        lower_,
        upper_
    )
    biggest_gates = get_the_biggest_polygon(rects)

    return biggest_gates

def draw_polygon(image: Any, polygon: List[Point], color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    for i in range(len(polygon)):
        cv2.line(image, (polygon[i].x, polygon[i].y), (polygon[(i + 1) % len(polygon)].x, polygon[(i + 1) % len(polygon)].y), color, 3)

def reduce_gates(old_polygon: List[Point]) -> List[Point]:
    result: List[Point] = []

    gates_center = polygon_center(old_polygon)

    for curr_point in old_polygon:
        result.append(Point(
            math.floor(gates_center.x + (curr_point.x - gates_center.x) * CUTTING_GATES_COEFFICIENT),
            math.floor(gates_center.y + (curr_point.y - gates_center.y) * CUTTING_GATES_COEFFICIENT)
        ))

    return result

def get_color_percentage_in_line(
        image: cv2.Mat,
        point1: Point,
        point2: Point,
        lower: np.array = np.array([140, 0, 0]),
        upper: np.array = np.array([200, 255, 255])
) -> float:
    image_height, image_width, _ = image.shape

    # Строим максу для определения розовых пикселей на изображении
    pink_mask = get_mask(image, lower, upper)

    line_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Рисуем линию толщиной
    cv2.line(
        line_mask,
        (point1.x, point1.y),
        (point2.x, point2.y),
        (255, 255, 255), LINE_MASK_THICKNESS
    )

    # Находим пересечение с розовыми пикселями
    intersection = cv2.bitwise_and(line_mask, pink_mask)

    # Считаем количество розовых пикселей на линии
    all_pixels_count = cv2.countNonZero(line_mask)
    pink_pixels_count = cv2.countNonZero(intersection)
    if all_pixels_count == 0:
        return 0

    return pink_pixels_count / all_pixels_count

def get_max_color_percentage_in_line_with_offset(
        image: cv2.Mat,
        point1: Point,
        point2: Point,
        lower: np.array = np.array([140, 0, 0]),
        upper: np.array = np.array([200, 255, 255])
) -> float:
    points1 = [
        # Point(point1.x, point1.y),
        Point(point1.x - LINE_MASK_OFFSET, point1.y),
        Point(point1.x + LINE_MASK_OFFSET, point1.y),
        Point(point1.x, point1.y - LINE_MASK_OFFSET),
        Point(point1.x, point1.y + LINE_MASK_OFFSET),
    ]

    points2 = [
        # Point(point2.x, point2.y),
        Point(point2.x - LINE_MASK_OFFSET, point2.y),
        Point(point2.x + LINE_MASK_OFFSET, point2.y),
        Point(point2.x, point2.y - LINE_MASK_OFFSET),
        Point(point2.x, point2.y + LINE_MASK_OFFSET),
    ]

    max_percentage = 0

    for i in range(len(points1)):
        max_percentage = max(
            max_percentage,
            get_color_percentage_in_line(
                image,
                points1[i],
                points2[i],
                lower,
                upper
            )
        )

    return max_percentage

def get_the_biggest_gates(
        image: Any,
        lower: np.array = np.array([140, 0, 0]),
        upper: np.array = np.array([200, 255, 255])
) -> List[Point] or None:
    image_height, image_width, _ = image.shape

    # Ищем наибольший многоугольник
    biggest_polygon = get_the_biggest_polygon_in_image(image)

    if biggest_polygon is None:
        return None

    image_view = image.copy()
    draw_polygon(image_view, biggest_polygon)

    n = len(biggest_polygon)
    graph: List[List[float]] = []

    for i in range(n):
        buffer: List[float or None] = []
        for j in range(n):
            buffer.append(None)
        graph.append(buffer)

    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = get_max_color_percentage_in_line_with_offset(image, biggest_polygon[i], biggest_polygon[j], lower, upper)
            else:
                graph[i][j] = 1

    max_gates: List[Point] = []
    max_similarity = MIN_GATES_EDGES_PINK_PERCENTAGE
    result_diagonal_non_pink_percentage = None

    for curr_points in itertools.combinations(range(n), 4):
        curr_polygon = sort_vertexes([
            biggest_polygon[curr_points[0]],
            biggest_polygon[curr_points[1]],
            biggest_polygon[curr_points[2]],
            biggest_polygon[curr_points[3]]
        ])

        diagonal_non_pink_percentage = (1 - graph[curr_points[0]][curr_points[2]]) * (1 - graph[curr_points[1]][curr_points[3]])

        indexes: List[int] = [index_of_point(biggest_polygon, curr_polygon[0]),
                                index_of_point(biggest_polygon, curr_polygon[1]),
                                index_of_point(biggest_polygon, curr_polygon[2]),
                                index_of_point(biggest_polygon, curr_polygon[3])]

        curr_similarity = graph[indexes[0]][indexes[1]] * \
                            graph[indexes[1]][indexes[2]] * \
                            graph[indexes[2]][indexes[3]] * \
                            graph[indexes[3]][indexes[0]]

        if curr_similarity > max_similarity and diagonal_non_pink_percentage > MIN_GATES_DIAGONAL_NON_PINK_PERCENTAGE:
        # if curr_similarity > max_similarity:
            result_diagonal_non_pink_percentage = diagonal_non_pink_percentage
            max_similarity = curr_similarity
            max_gates = curr_polygon

    print("result_diagonal_non_pink_percentage", result_diagonal_non_pink_percentage)
    print("max_similarity", max_similarity)

    cv2.imshow("Image view", image_view)

    if len(max_gates) != 4:
        return None

    return max_gates

def euler_angles_from_rotation_matrix(R):
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # Угол вокруг X
        y = np.arctan2(-R[2,0], sy)      # Угол вокруг Y
        z = np.arctan2(R[1,0], R[0,0])   # Угол вокруг Z
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def get_gates_angles(gates_point: List[Point]) -> List[float] or None:
    if gates_point is None or len(gates_point) < 4:
        return None

    # Параметры камеры (замените на реальные значения)
    fx, fy = 185.69, 185.69  # Фокусные расстояния
    cx, cy = 320.5, 180.5  # Оптический центр
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Коэффициенты искажения (предполагаем отсутствие)

    # Размеры прямоугольника в метрах (ширина и высота)
    width, height = 0.9, 0.6

    # 3D точки прямоугольника в локальной системе координат (Z=0)
    object_points = np.array([
        [0, 0, 0],  # Левый верхний
        [width, 0, 0],  # Правый верхний
        [width, height, 0],  # Правый нижний
        [0, height, 0]  # Левый нижний
    ], dtype=np.float32)

    # Точки на изображении (пример, замените на реальные координаты)
    # image_points = np.array([
    #     [100, 100],  # Левый верхний угол
    #     [500, 100],  # Правый верхний
    #     [500, 300],  # Правый нижний
    #     [100, 300]  # Левый нижний
    # ], dtype=np.float32)

    image_points = np.array([
        [gates_point[0].x, gates_point[0].y],
        [gates_point[1].x, gates_point[1].y],
        [gates_point[2].x, gates_point[2].y],
        [gates_point[3].x, gates_point[3].y]
    ], dtype=np.float32)

    # Решение PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )

    # Преобразование вектора поворота в матрицу
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Получение углов Эйлера
    euler_angles = euler_angles_from_rotation_matrix(rotation_matrix)
    angles_deg = np.degrees(euler_angles)  # Конвертация в градусы

    return [angles_deg[0], angles_deg[1], angles_deg[2]]