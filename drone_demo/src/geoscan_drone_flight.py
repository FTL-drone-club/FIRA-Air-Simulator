from geoscan_drone import *
import time

SPEED = 1.5
YAW_SPEED = 2

if __name__ == '__main__':
    rospy.init_node('geoscan_flight')  # Инициализация ROS-ноды с именем 'track_with_model'
    drone = Drone()  # Создаем объект дрона

    print("Taking off...", end="\t\t")
    drone.wait_state(CallbackEvent.ENGINES_STARTED)
    drone.takeoff()
    time.sleep(1)
    print("OK")

    drone.wait_state(CallbackEvent.TAKEOFF_COMPLETE)
    drone.start_move()
    print("Forward")
    drone.set_speed(linear_x=SPEED)
    time.sleep(1)
    print("Backward")
    drone.set_speed(linear_x=SPEED)
    time.sleep(1)
    drone.stop_move()

    print("Landing...", end="\t\t")
    drone.land()
    drone.wait_state(CallbackEvent.COPTER_LANDED)
    print("OK")
