import rclpy
import time
from src.lidar.lidar_subscriber import LidarSubscriber
from src.controller import RobotController


def main():
    rclpy.init()
    node = LidarSubscriber()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    controller = RobotController(node.car, node)

    print("Starting autonomous control loop...\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            print(controller.update())
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Manual stop requested...")

    finally:
        node.car.set_motor(0, 0, 0, 0)
        node.car.set_akm_steering_angle(0, False)
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
