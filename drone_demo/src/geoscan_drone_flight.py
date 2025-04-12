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
    print("OK")

    drone.wait_state(CallbackEvent.TAKEOFF_COMPLETE)
    rospy.sleep(1)

    drone.start_move()
    print("Forward")
    drone.set_speed(linear_x=SPEED)
    rospy.sleep(1.5)
    print("Backward")
    drone.set_speed(linear_x=0, linear_y=-SPEED)
    rospy.sleep(2.5)

    drone.set_speed(linear_x=0)
    drone.stop_move()

    print("Landing...", end="\t\t")
    drone.land()
    drone.wait_state(CallbackEvent.COPTER_LANDED)
    print("OK")