import math
import threading
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from Rosmaster_Lib import Rosmaster
from sensor_msgs.msg import LaserScan
from .lidar_preprocessing import LidarPreprocessing
from ..object_detecter import ObstacleUI


class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        self.obstacles_points = None
        self.corner_points = None
        self.edge_points = None
        self.running = True
        self.last_detection = False

        if not self.wait_for_scan():
            self.get_logger().error('Failed to get scan')
            raise SystemExit('Failed to get scan')

        # Create QoS profile matching the YDLIDAR publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription_filter = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile  # Use QoS profile instead of just depth
        )

        self.get_logger().info(f"Subscription was successfully created")

        self.car = Rosmaster()
        self.preprocessing = LidarPreprocessing()
        self.obstacle_ui = ObstacleUI(visualize=True)

    
    def wait_for_scan(self):
        self.get_logger().info('Wait topic /scan...')
        for _ in range(10):
            if '/scan' in [topic[0] for topic in self.get_topic_names_and_types()]:
                self.get_logger().info('Find a topic /scan')
                return True
            time.sleep(1)
        return False

    
    def lidar_callback(self, msg):
        filter_data = self.preprocessing.filter_data(msg)
        self.preprocessing.apply_dbscan()
        self.get_front_distance(msg, filter_data)
        self.obstacles_points = self.preprocessing.get_all_clusters_points()
        if self.obstacle_ui.visualize:
            if self.obstacle_ui.is_running():
                self.obstacle_ui.set_obstacles(self.obstacles_points)
                self.obstacle_ui.run()
            else:
                self.obstacle_ui.shutdown()
        else:
            self.obstacle_ui.set_obstacles(self.obstacles_points)


    def get_info_obstacle(self):
        is_dangerous = self.obstacle_ui.any_dangerous_point()
        
        if is_dangerous and not self.last_detection:
            self.last_detection = True
            print("🚨 New obstacle detected!")
        elif not is_dangerous and self.last_detection:
            self.last_detection = False
            print("✅ Obstacle cleared!")
        
        return is_dangerous

    
    def get_front_distance(self, msg, filter_data):
        zero_degree = round(-msg.angle_min / msg.angle_increment)
        distance = filter_data[len(filter_data)//2]
        # self.get_logger().info(f"Front distance: {distance:.2f} m")

    
    def shutdown(self):
        self.running = False
        self.obstacle_ui.shutdown()