#! /usr/bin/env python3
# Импортируем необходимые библиотеки
import cv2  # OpenCV для работы с изображениями
import rospy  # ROS для взаимодействия с дроном и управления им
import time  # Для работы с задержками
from typing import Any, Tuple, List  # Аннотации типов

import gates  # Модуль для работы с воротами (их поиск, отрисовка и т.п.)
from geometry import Point, sort_vertexes  # Геометрические примитивы и функции
from drone import Drone  # Класс для управления дроном
from gates_flight import CenterGates


if __name__ == '__main__':
    rospy.init_node('track_with_model')  # Инициализация ROS-ноды с именем 'track_with_model'
    drone = Drone()  # Создаем объект дрона

    try:
        # Начальная настройка дрона
        drone.stop()  # Остановить все движения
        drone.set_yaw(0)  # Обнулить угол поворота (курса)
        drone.takeoff()  # Взлет

        # Задание начальной скорости по оси Z (вверх)
        drone.set_speed(0, 0, 1.3, 0, 0, 0)
        time.sleep(0.4)  # Подождать для стабилизации
        drone.stop()  # Остановить дрон

        gates_flight = CenterGates(drone)
        gates_flight.start()

        # Основной цикл управления дроном
        while True:
            _input_image = drone.front_image.copy()  # Получаем изображение с фронтальной камеры дрона

            _key = cv2.waitKey(1) & 0xFF  # Считываем нажатие клавиши (если есть)
            if _key == ord('q'):
                break  # Выход из цикла по нажатию 'q'

            # Управление дроном с клавиатуры
            # if _key == ord('w'):
            #     drone.set_speed(linear_x=SPEED)  # Движение вперед
            # elif _key == ord('a'):
            #     drone.set_speed(linear_y=SPEED)  # Влево
            # elif _key == ord('s'):
            #     drone.set_speed(linear_x=-SPEED)  # Назад
            # elif _key == ord('d'):
            #     drone.set_speed(linear_y=-SPEED)  # Вправо
            # elif _key == ord('r'):
            #     drone.set_speed(linear_z=SPEED)  # Вверх
            # elif _key == ord('f'):
            #     drone.set_speed(linear_z=-SPEED)  # Вниз
            # elif _key == ord('k'):
            #     drone.set_speed(angular_z=SPEED)  # Поворот по часовой стрелке
            # elif _key == ord('l'):
            #     drone.set_speed(angular_z=-SPEED)  # Поворот против часовой стрелки
            # elif _key == ord('q'):
            #     break  # Выход из цикла по нажатию 'q'
            # else:
            #     drone.stop()  # Остановка, если никакая клавиша не нажата

            _result_image = drone.front_image.copy()  # Копия изображения для отрисовки результатов

            # Поиск самых больших ворот на изображении
            # _biggest_gates = gates.get_the_biggest_gates(drone.front_image)

            # Если ворота найдены
            # if _biggest_gates is not None:
            #     if len(_biggest_gates) != 4:
            #         print("ERROR")  # Ошибка, если найдено не 4 точки
            #     else:
            #         sorted_biggest_gates: List[Point] = sort_vertexes(_biggest_gates)
            #         gates.draw_polygon(_result_image, sorted_biggest_gates)

            # Отображаем изображения
            cv2.imshow('Result', _result_image)  # Результат с отрисованными воротами

            time.sleep(0.03)  # Задержка для ограничения частоты обновления кадров


        gates_flight.stop()
        cv2.destroyAllWindows()  # Закрываем все окна OpenCV
        drone.land()  # Посадка дрона

    except rospy.ROSInterruptException:
        # Обработка ошибки при аварийном завершении ROS-ноды
        print("rospy.ROSInterruptException has called")
