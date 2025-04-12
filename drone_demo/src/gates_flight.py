import sys
from multiprocessing import Process
from threading import Thread
from functools import cmp_to_key
import rospy
import math
import time
from typing import Any, Tuple, List

from drone import Drone
from drone_exceptions import DroneIsNotFlight, ProcessIsAlreadyStarted, ProcessIsNotStartedYet
import gates
import geometry
import pid
from psutil import process_iter

# Константы для скорости и точности управления дроном
Z_ANGLE_ACCURACY = 20
Z_ACCURACY = 0.1
SPEED = 0.6

MIN_AREA_FOR_FLIGHT_FORWARD = 0.06


class CenterGates(object):
    def __init__(self, drone: Drone):
        # Флаг, активен ли поток управления
        self._is_thread_active: bool = False

        # Создаем поток для выполнения функции управления дроном
        self._process: Thread = Thread(target=self._process_function, daemon=True)

        # Объект дрона для управления
        self._drone: Drone = drone

        self._yaw_pid = pid.PID(
            Kp=0.008,
            Ki=0.01,
            Kd=0.007,
            setpoint=0,  # Цель — центр кадра по оси X
            output_limits=(-20, 20),  # Ограничение выходного сигнала
        )

        self._y_pid = pid.PID(
            Kp=0.002,
            Ki=0.002,
            Kd=0.003,
            setpoint=0,  # Цель — центр кадра или равные расстояния между сторонами ворот
            output_limits=(-1, 1),
        )

        self._z_pid = pid.PID(
            Kp=0.04,
            Ki=0.06,
            Kd=0.03,
            setpoint=0,  # Цель — центр кадра по оси Y
            output_limits=(-3, 3),
        )

    def start(self):
        # Запуск потока управления дроном
        if self._is_thread_active:
            raise ProcessIsAlreadyStarted()

        if not self._drone.is_flight():
            raise DroneIsNotFlight()

        self._process: Thread = Thread(target=self._process_function, daemon=True)
        self._is_thread_active = True
        self._process.start()

    def stop(self):
        # Остановка потока управления дроном
        if not self._is_thread_active:
            raise ProcessIsNotStartedYet()

        self._is_thread_active = False
        self._process.join()

        # Остановка дрона (обнуление всех скоростей)
        self._drone.set_speed(0, 0, 0, 0, 0, 0)

    def _process_function(self) -> None:
        # Основная функция управления дроном
        while self._is_thread_active:
            # Получаем самые большие (ближайшие) ворота на изображении с камеры
            _the_biggest_gates = gates.get_the_biggest_gates(self._drone.front_image)

            if _the_biggest_gates is None:
                print("[Warning] Gates is not detected", file=sys.stderr)
                continue

            # Находим центр найденных ворот
            _the_biggest_gates_center = geometry.polygon_center(_the_biggest_gates)

            gates_area = geometry.polygon_area(_the_biggest_gates)
            total_area = 640 * 480

            # Центр изображения (центр кадра камеры)
            _front_image_height, _front_image_width, _front_image_channels = self._drone.front_image.shape
            _front_image_center = geometry.Point(
                _front_image_width / 2,
                _front_image_height / 2,
            )

            # Управление движением по оси Z (вверх/вниз)
            _new_z_value = self._z_pid.update(
                _the_biggest_gates_center.y - _front_image_center.y
            )

            # Управление поворотом дрона по оси Z (yaw)
            _new_yaw_value = self._yaw_pid.update(
                _the_biggest_gates_center.x - _front_image_center.x
            )

            # Считаем углы поворота ворот
            sorted_biggest_gates: List[geometry.Point] = geometry.sort_vertexes(_the_biggest_gates)
            gates_angles = gates.get_gates_angles(sorted_biggest_gates)

            print(gates_area / total_area)
            if gates_area / total_area > MIN_AREA_FOR_FLIGHT_FORWARD and abs(gates_angles[1]) < Z_ANGLE_ACCURACY:
                print("FORWARD... ", end="\t")
                self._drone.set_speed(
                    linear_x=SPEED,
                    linear_y = 0,
                    linear_z = -0.2,
                    angular_z = 0,
                )
                time.sleep(1.5)
                self._drone.set_speed(linear_x=0)
                print("OK")
                continue

            if abs(gates_angles[1]) < Z_ANGLE_ACCURACY and _the_biggest_gates_center.y - _front_image_center.y < Z_ACCURACY:
                self._drone.set_speed(linear_x=SPEED)
            else:
                self._drone.set_speed(linear_x=SPEED / 4)

            # Сравниваем разницу по Y между левой и правой стороной ворот
            _new_y_value = self._y_pid.update(-gates_angles[1])

            _new_y_value = _new_y_value

            # Устанавливаем рассчитанные скорости для дрона
            self._drone.set_speed(
                linear_y=_new_y_value,
                linear_z=_new_z_value,
                angular_z=_new_yaw_value,
            )

    def is_active(self) -> bool:
        # Проверка активности системы управления воротами
        return self._is_thread_active
