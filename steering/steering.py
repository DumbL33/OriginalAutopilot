import numpy as np

class SteeringController:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height
    
    def calculate_steering_and_speed(self, relative_x, action, current_command):
        
        if current_command == "go_straight":
            steering_angle_raw = relative_x * 45
        elif current_command == "go_right":
            steering_angle_raw = relative_x * 45
        elif current_command == "go_left":
            steering_angle_raw = relative_x * 45
        elif current_command == "park_short" or current_command == "park_long":
            steering_angle_raw = relative_x * 60
        elif current_command == "park_out":
            steering_angle_raw = relative_x * 60
    
        
        recommended_speed = 35
        if action > 0.5:
            if current_command == "park_short" or current_command == "park_long":
                recommended_speed = 30
            else:
                recommended_speed = 35
        elif action < -0.5:
            if current_command == "park_short" or current_command == "park_long":
                recommended_speed = -30
            else:
                recommended_speed = -35
        else:
            recommended_speed = 0

        max_angle = 45
        steering_angle = np.clip(steering_angle_raw, -max_angle, max_angle)
        
        return steering_angle, recommended_speed
