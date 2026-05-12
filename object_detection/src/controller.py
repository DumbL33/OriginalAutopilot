import time
from enum import Enum, auto

class RobotState(Enum):
    MOVING = auto()
    STOPPED = auto()

class RobotController:
    def __init__(self, car, lidar, stop_delay=0.1):
        self.car = car
        self.lidar = lidar

        self.state = RobotState.STOPPED
        self.last_stop_time = None 
        self.stop_delay = stop_delay

        print("Controller initialized")

    def update(self):
        obstacle = self.lidar.get_info_obstacle()
        result = None
        
        if obstacle and self.state != RobotState.STOPPED:
            result = self.stop()
        elif not obstacle and self.state == RobotState.STOPPED:
            result = self.wait_before_moving()
        elif self.state == RobotState.MOVING:
            result = self.move_forward()
        
        return result
    
    def stop(self):
        self.state = RobotState.STOPPED
        self.last_stop_time = time.time()  
        print("🛑 STOPPED - Obstacle detected")
        return True

    def wait_before_moving(self):
        if self.last_stop_time is None:
            self.state = RobotState.MOVING
            return True
        
        elapsed = time.time() - self.last_stop_time
        if elapsed >= self.stop_delay:
            self.state = RobotState.MOVING
            return True
        
        return False

    def move_forward(self):
        return False