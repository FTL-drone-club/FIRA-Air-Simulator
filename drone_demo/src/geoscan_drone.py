import rospy
import threading
from multiprocessing import Process
from gs_flight import FlightController, CallbackEvent
from gs_board import BoardManager
from pioneer_sdk import Pioneer
import time
import cv2
import numpy as np


class Drone(object):
    def __init__(self):
        self.__board = BoardManager()
        self.__ap = FlightController(self.callback)
        self.__drone = Pioneer(baud=57600, connection_method='serial', device='/dev/ttyS7', logger=True, log_connection=True)
        self.__drone_state = None
        self.__rate = rospy.Rate(25)

        self.__speed_x = 0
        self.__speed_y = 0
        self.__speed_z = 0
        self.__speed_yaw = 0

        self._is_thread_active: bool = False
        self._process: threading.Thread = threading.Thread(target=self.move, daemon=True)

        self.__ap.preflight()

    def callback(self, event) -> None:
        event_data = event.data

        print("Callback: ", str(event_data))
        self.__drone_state = event_data

    def drone_state(self):
        return self.__drone_state

    def wait_state(self, state):
        while self.__drone_state is not state:
            time.sleep(0.05)

    def takeoff(self) -> None:
        self.__ap.takeoff()

    def land(self) -> None:
        self.__ap.landing()

    def set_speed(self, linear_x: float = None, linear_y: float = None, linear_z: float = None, angular_z: float = None) -> None:
        if linear_x is not None:
            self.__speed_x = linear_x
        if linear_y is not None:
            self.__speed_y = linear_y
        if linear_z is not None:
            self.__speed_z = linear_z
        if angular_z is not None:
            self.__speed_yaw = angular_z

    def start_move(self) -> None:
        self._is_thread_active = True
        self._process.start()

    def stop_move(self) -> None:
        self._is_thread_active = False
        self._process.join()

    def move(self) -> None:
        while self._is_thread_active:
            print("Moving: ")
            print("X =", self.__speed_x)
            print("Y =", self.__speed_y)
            print("Z =", self.__speed_z)
            print("YAW =", self.__speed_yaw)
            print("---")
            self.__drone.set_manual_speed(
                vx = self.__speed_x,
                vy = self.__speed_y,
                vz = self.__speed_z,
                yaw_rate = self.__speed_yaw
            )
            # rospy.sleep(0.1)
            self.__rate.sleep()