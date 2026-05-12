import cv2
from jetcam.usb_camera import USBCamera
import time
import numpy as np
import json
import torch
import socket
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from steering.cnn_module_single_command import TensorRTCNNSteering

from Rosmaster_Lib import Rosmaster
from steering.steering import SteeringController
from lane_segmentation.lane_segmentation_slovakiatech import DDRNetTensorRT

from steering.resnet_module import SteeringModel
from steering.mobilenet_module import SteeringModel_MobileNet
from steering.mobilenet_rgb_module import SteeringModel_MobileNetV2
from steering.lstm_module import LSTMSteeringModel
from steering.lstm_module_mask import LSTMSteeringModel_MASK

from object_detection.src.lidar.lidar_subscriber import LidarSubscriber
from object_detection.src.controller import RobotController

from steering.crossroad_detection_module import CrossroadDetectorTRT
from steering.crossroad_manager import CrossroadManager

DDRNET_ENGINE_PATH = "models/ddrnet_lane_fp16.trt"
DDRNET_ENGINE_PATH_SLOVAKIATECH = "models/tensorrt_ddrnet_model_slovakiatech/ddrnet_lane.trt"

CNN_ENGINE_PATH = "tensorrt_models_backup/steering_model_standard_fp16_jetson_batch1.trt"

CNN_ENGINE_PATH_GO_STRAIGHT = "models/tensorrt_models_go_straight/steering_standard_fp16_batch1.trt"
CNN_ENGINE_PATH_GO_RIGHT = "models/tensorrt_models_go_right/steering_standard_fp16_batch1.trt"
CNN_ENGINE_PATH_GO_LEFT = "models/tensorrt_models_go_left/steering_standard_fp16_batch1.trt"
CNN_ENGINE_PATH_PARK = "models/tensorrt_models_park/steering_standard_fp16_batch1.trt"

LSTM_ENGINE_PARK_LONG_3SEQ = 'models/tensorrt_lstm_long/best_lstm_steering_park_long_0.7_3seq_fp16.trt' # so far performed the best from longer side 
LSTM_ENGINE_PARK_LONG_5SEQ = 'models/tensorrt_lstm_long/best_lstm_steering_park_long_0.7_5seq_fp16.trt'

LSTM_ENGINE_PARK_SHORT_3SEQ = 'models/tensorrt_lstm_short/best_lstm_steering_park_short_3seq.trt'

LSTM_ENGINE_PARK_OUT = 'models/tensorrt_lstm_out/best_lstm_steering_park_out.trt'
LSTM_ENGINE_PARK_OUT_3SEQ = 'models/tensorrt_lstm_out/best_lstm_steering_park_out_3seq.trt'

LSTM_ENGINE_PARK_FULL_3SEQ = 'models/tensorrt_lstm_full/best_lstm_steering_park_full_0.7_3seq.trt' # so far best full parking model

CROSSROAD_ENGINE_PATH = "models/tensorrt_crossroad_detect/crossroad_standard.trt"

# UDP Configuration
UDP_RECV_PORT_PARKING = 50002
UDP_RECV_PORT_COMMANDS = 50001
UDP_RECV_PORT_OBJECTS = 50003
SERVER_IP = "0.0.0.0"

class UDPCommunication:
    def __init__(self, parking_port=50002, command_port=50001, objects_port=50003):
        self.parking_port = parking_port
        self.command_port = command_port
        self.objects_port = objects_port
        
        self.parking_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.parking_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.parking_socket.bind(("0.0.0.0", self.parking_port))
        self.parking_socket.setblocking(False)
        
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.command_socket.bind(("0.0.0.0", self.command_port))
        self.command_socket.setblocking(False)
        
        self.objects_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.objects_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.objects_socket.bind(("0.0.0.0", self.objects_port))
        self.objects_socket.setblocking(False)
        
        self.parking_status = None
        self.action_command = None
        self.object_detections = None
        
        self.parking_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.objects_lock = threading.Lock()
        
        self.running = True
        self.parking_thread = threading.Thread(target=self._receive_parking_loop, daemon=True)
        self.command_thread = threading.Thread(target=self._receive_command_loop, daemon=True)
        self.objects_thread = threading.Thread(target=self._receive_objects_loop, daemon=True)
        
        self.parking_thread.start()
        self.command_thread.start()
        self.objects_thread.start()
        
        print(f"UDP Communication initialized:")
        print(f"  - Receiving parking status on port {self.parking_port}")
        print(f"  - Receiving action commands on port {self.command_port}")
        print(f"  - Receiving object detections on port {self.objects_port}")
    
    def _receive_parking_loop(self):
        while self.running:
            try:
                data, addr = self.parking_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                
                if message.get('type') == 'parking_status':
                    with self.parking_lock:
                        self.parking_status = message
                    print(f"[Port {self.parking_port}] Parking status: Zone={message.get('zone_name')}, "
                          f"Occupied={message.get('is_occupied')}, "
                          f"Confidence={message.get('confidence'):.2f}")
            except socket.error:
                time.sleep(0.01)
            except json.JSONDecodeError as e:
                print(f"JSON decode error (parking): {e}")
            except Exception as e:
                print(f"Receive error (parking): {e}")
    
    def _receive_command_loop(self):
        while self.running:
            try:
                data, addr = self.command_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                with self.command_lock:
                    self.action_command = message
                print(f"[Port {self.command_port}] Action: {message.get('action')}")
            except socket.error:
                time.sleep(0.01)
            except json.JSONDecodeError as e:
                print(f"JSON decode error (command): {e}")
            except Exception as e:
                print(f"Receive error (command): {e}")
    
    def _receive_objects_loop(self):
        while self.running:
            try:
                data, addr = self.objects_socket.recvfrom(8192) 
                message = json.loads(data.decode('utf-8'))
                
                if message.get('type') == 'object_detections':
                    with self.objects_lock:
                        self.object_detections = message
                    
                    count = message.get('count', 0)
                    if count > 0:
                        print(f"[Port {self.objects_port}] Received {count} object detection(s)")
                        for det in message.get('detections', []):
                            print(f"  - {det.get('class_name')}: confidence={det.get('confidence'):.2f}")
            except socket.error:
                time.sleep(0.01)
            except json.JSONDecodeError as e:
                print(f"JSON decode error (objects): {e}")
            except Exception as e:
                print(f"Receive error (objects): {e}")
    
    def get_parking_status(self):
       
        with self.parking_lock:
            return self.parking_status
    
    def get_action_command(self):
        with self.command_lock:
            return self.action_command
    
    def get_object_detections(self):
        with self.objects_lock:
            return self.object_detections
    
    def get_largest_object(self):
        with self.objects_lock:
            if not self.object_detections:
                return None
            
            detections = self.object_detections.get('detections', [])
            if not detections:
                return None
            
            largest = None
            max_area = 0
            
            for det in detections:
                bbox = det.get('bbox', {})
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)
                
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    largest = det
            
            return largest
    
    def print_current_status(self):
        """Print current status of all three ports"""
        print("\n" + "="*60)
        print("CURRENT PORT STATUS")
        print("="*60)
        
        # Port 50001 - Parking Status
        print(f"\n[PORT {self.parking_port} - PARKING STATUS]")
        parking = self.get_parking_status()
        if parking:
            print(f"  Type: {parking.get('type')}")
            print(f"  Zone Name: {parking.get('zone_name')}")
            print(f"  Is Occupied: {parking.get('is_occupied')}")
            print(f"  Confidence: {parking.get('confidence'):.2f}")
            print(f"  Timestamp: {parking.get('timestamp')}")
        else:
            print("  No data received yet")
        
        # Port 50002 - Action Commands
        print(f"\n[PORT {self.command_port} - ACTION COMMANDS]")
        command = self.get_action_command()
        if command:
            print(f"  Command: {command.get('command')}")
            #print(f"  Distance: {command.get('distance'):.2f}")
            #print(f"  Angle: {command.get('angle'):.2f}")
            #print(f"  Speed: {command.get('speed'):.2f}")
            #print(f"  Timestamp: {command.get('timestamp')}")
        else:
            print("  No data received yet")
        
        # Port 50003 - Object Detections
        print(f"\n[PORT {self.objects_port} - OBJECT DETECTIONS]")
        objects = self.get_object_detections()
        if objects:
            print(f"  Type: {objects.get('type')}")
            print(f"  Count: {objects.get('count')}")
            print(f"  Timestamp: {objects.get('timestamp')}")
            detections = objects.get('detections', [])
            if detections:
                print(f"  Detections:")
                for i, det in enumerate(detections, 1):
                    print(f"    {i}. Class: {det.get('class_name')}, "
                          f"Confidence: {det.get('confidence'):.2f}, "
                          f"BBox: {det.get('bbox')}")
            else:
                print("  No detections in message")
        else:
            print("  No data received yet")
        
        print("="*60 + "\n")
    
    def stop(self):
        self.running = False
        if self.parking_thread.is_alive():
            self.parking_thread.join(timeout=1)
        if self.command_thread.is_alive():
            self.command_thread.join(timeout=1)
        if self.objects_thread.is_alive():
            self.objects_thread.join(timeout=1)
        
        self.parking_socket.close()
        self.command_socket.close()
        self.objects_socket.close()
        print("UDP Communication stopped")


class SimplePerf:
    def __init__(self):
        self.frame_times = []
        self.last_time = time.time()
        
    def tick(self):
        current_time = time.time()
        if hasattr(self, 'last_time'):
            frame_time = (current_time - self.last_time) * 1000
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
    def get_stats(self):
        if not self.frame_times:
            return 0, 0
            
        avg_ms = sum(self.frame_times) / len(self.frame_times)
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        return fps, avg_ms



STOP_READY = True
slowdown_duration = 5 

# Local command priority timeout (in seconds)
# After this time, UDP commands can override local commands again
LOCAL_COMMAND_TIMEOUT = 5.0

def main():
    udp_comm = UDPCommunication(
        parking_port=50002,
        command_port=50001,
        objects_port=50003
    )

    #Lidar detection init
    rclpy.init()
    lidar_node = LidarSubscriber()
    executor = SingleThreadedExecutor()
    executor.add_node(lidar_node)

    perf = SimplePerf()
    speed = 0.2

    # Cam setup for video2
    cap = cv2.VideoCapture('/dev/video2')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    time.sleep(0.5)
    for _ in range(5):
        cap.read()

    go_straight_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_STRAIGHT)
    go_right_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_RIGHT)
    go_left_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_LEFT)
    
    lstm_engine_park_short = LSTMSteeringModel(LSTM_ENGINE_PARK_SHORT_3SEQ, sequence_length=3)
    lstm_engine_park_long = LSTMSteeringModel(LSTM_ENGINE_PARK_LONG_3SEQ, sequence_length=3)
    lstm_engine_park_out = LSTMSteeringModel(LSTM_ENGINE_PARK_OUT, sequence_length=5)

    # Lane segmentation model init 
    ddrnet_engine = DDRNetTensorRT(
        engine_path=DDRNET_ENGINE_PATH_SLOVAKIATECH,
        confidence_threshold=0.8
    )

    crossroad_detector = CrossroadDetectorTRT(
        engine_path=CROSSROAD_ENGINE_PATH,
        confidence_threshold=0.99  
    )
    crossroad_manager = CrossroadManager(
        stop_duration=5.0,      
        command_reset_duration=3.0,
        cooldown_duration=3.0
    )

    # Steering controller init
    controller = SteeringController(frame_width=224, frame_height=224)

    slowdown_active = False
    shutdown_processed = False
    shutdown_sign_present = False
    is_occupied = False

    # LIDAR obstacle controller init
    obstacle_controller = RobotController(lidar_node.car, lidar_node, stop_delay=0.5)
    
    # Car lib init
    car = Rosmaster()
    car.set_beep(100)
    car_start = False
    current_command = 'go_straight'

    # Obstacle detection state
    obstacle_detected = False
    stopped_for_obstacle = False

    frame_count = 0
    stop_detected = False

    detection_confidence = 0.0
    detection_class_name = None
    detection_bbox_size = 0

    sign_command = None
    
    # Local command priority tracking
    local_command_active = False
    local_command_time = None
    
    # Parking mode tracking - blocks navigation commands during parking
    in_parking_mode = False
    parking_was_occupied = False  # Track if car was actually parked (occupied: True) before allowing exit
    waiting_for_parking_exit = False  # Flag to continuously check for parking exit after 'O' is pressed
    
    # Stop state for non-parking targets - car stops until manual command
    stopped_for_non_parking = False
    
    try:
        while True:
            perf.tick()

            rclpy.spin_once(lidar_node, timeout_sec=0.001)
            obstacle_stop_signal = obstacle_controller.update()
            obstacle_detected = lidar_node.get_info_obstacle()

            ret, frame = cap.read()
            if not ret:
                break

            # Check if local command timeout has expired
            if local_command_active and local_command_time is not None:
                if (time.time() - local_command_time) >= LOCAL_COMMAND_TIMEOUT:
                    local_command_active = False
                    local_command_time = None
                    print("Local command priority expired - UDP commands can now override")

            # Track parking status for exit detection
            parking_status = udp_comm.get_parking_status()
            if in_parking_mode and parking_status:
                is_occupied = parking_status.get('is_occupied', True)
                zone_name = parking_status.get('zone_name', '')
                
                # Track if car was actually parked (occupied: True) - only for exit detection
                if is_occupied:
                    parking_was_occupied = True
                
                # Continuously check for parking exit after 'O' is pressed
                if waiting_for_parking_exit:
                    # Check if car has exited parking slot (if slot is empty, car must have exited)
                    if not is_occupied:
                        in_parking_mode = False
                        parking_was_occupied = False  # Reset for next parking cycle
                        waiting_for_parking_exit = False
                        stopped_for_non_parking = False
                        print(f"[PARKING MODE] Car exited parking slot (zone: {zone_name}, occupied: {is_occupied})")
                        print(f"[PARKING MODE] Parking mode DEACTIVATED - Navigation commands now active")
                        # Reset to go_straight to continue normal navigation
                        current_command = "go_straight"
            
            # Process UDP commands only if local command is not active and not in parking mode
            action_command = udp_comm.get_action_command()
            if action_command and not local_command_active and not in_parking_mode:
                nav_command = action_command.get('command')
                robot_position = action_command.get('robot_position', {})
                heading = action_command.get('heading')
                target_waypoint = action_command.get('target_waypoint', '')  # Get target waypoint name if available
                
                print(f"Navigation: Command={nav_command} | Position=({robot_position.get('x', 0):.2f}, {robot_position.get('y', 0):.2f}) | Heading={heading} | Target={target_waypoint}")
                
                if nav_command == "straight":
                    current_command = "go_straight"
                    stopped_for_non_parking = False  # Clear stop state when navigation resumes
                    print(f"Navigation override: {current_command}")
                elif nav_command == "right":
                    current_command = "go_right"
                    stopped_for_non_parking = False  # Clear stop state when navigation resumes
                    print(f"Navigation override: {current_command}")
                elif nav_command == "left":
                    current_command = "go_left"
                    stopped_for_non_parking = False  # Clear stop state when navigation resumes
                    print(f"Navigation override: {current_command}")
                elif nav_command == "stop":
                    print(f"[UDP STOP COMMAND] Received stop command with heading={heading}, target={target_waypoint}")
                    # Check if target is "Parking" (case-insensitive)
                    target_lower = target_waypoint.lower() if target_waypoint else ""
                    is_parking_target = "parking" in target_lower
                    
                    if is_parking_target:
                        # Target is Parking - activate parking mode
                        heading_lower = heading.lower() if heading else ""
                        if heading_lower == "east":
                            current_command = "park_short"
                            in_parking_mode = True
                            parking_was_occupied = False  # Reset parking state tracking
                            waiting_for_parking_exit = False  # Reset exit waiting flag
                            print(f"[UDP STOP COMMAND] Target is Parking - Setting command to 'park_short' (heading: {heading})")
                            print(f"[UDP STOP COMMAND] PARKING MODE ACTIVATED - Navigation commands blocked until parking exits")
                        elif heading_lower == "west":
                            current_command = "park_long"
                            in_parking_mode = True
                            parking_was_occupied = False  # Reset parking state tracking
                            waiting_for_parking_exit = False  # Reset exit waiting flag
                            print(f"[UDP STOP COMMAND] Target is Parking - Setting command to 'park_long' (heading: {heading})")
                            print(f"[UDP STOP COMMAND] PARKING MODE ACTIVATED - Navigation commands blocked until parking exits")
                        else:
                            print(f"[UDP STOP COMMAND] Warning: Unknown heading '{heading}' for parking - command not changed")
                            print(f"[UDP STOP COMMAND] Current command remains: {current_command}")
                    else:
                        # Target is NOT Parking - just stop the car, wait for manual command
                        stopped_for_non_parking = True
                        print(f"[UDP STOP COMMAND] Target is NOT Parking ('{target_waypoint}') - Stopping car")
                        print(f"[UDP STOP COMMAND] Car will remain stopped until manual local command (w/a/d/p/l/g/o)")
                        # Keep current_command as is, but set flag to prevent movement
                        # The car will stop because stopped_for_non_parking flag will prevent movement
            elif action_command and in_parking_mode:
                # UDP command received but parking mode is active - block all navigation commands
                nav_command = action_command.get('command')
                print(f"[PARKING MODE] UDP command '{nav_command}' IGNORED - car is in parking mode")
                print(f"[PARKING MODE] Current parking command: {current_command}")
                print(f"[PARKING MODE] Press 'w', 'a', or 'd' to exit parking mode")
            elif action_command and local_command_active:
                # UDP command received but local command has priority
                nav_command = action_command.get('command')
                if nav_command == "stop":
                    print(f"[UDP STOP COMMAND] Stop command received but IGNORED - local command has priority")
                    if local_command_time is not None:
                        remaining = LOCAL_COMMAND_TIMEOUT - (time.time() - local_command_time)
                        print(f"[UDP STOP COMMAND] Local command active, timeout remaining: {remaining:.1f}s")
                    else:
                        print(f"[UDP STOP COMMAND] Local command active (timeout info unavailable)")
                else:
                    print(f"UDP command '{nav_command}' ignored - local command has priority")

            largest_object = udp_comm.get_largest_object()
            if largest_object:
                detection_confidence = largest_object.get('confidence', 0.0)
                detection_class_name = largest_object.get('class_name', None)
                
                bbox = largest_object.get('bbox', {})
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)
                detection_bbox_size = (x2 - x1) * (y2 - y1)
                
                print(f"Detection: {detection_class_name} | Confidence: {detection_confidence:.3f} | BBox Size: {detection_bbox_size}")
                
                if detection_class_name == "stop" and detection_bbox_size >= 5700:
                    if not shutdown_processed:
                        car.set_beep(200)
                        print("STOPPING THE CAR")
                        car.set_car_motion(0, 0, 0)
                        shutdown_processed = True
                        time.sleep(3)
                        print("RESTARTING CAR")
                
                elif detection_class_name == 'ahead':
                    car.set_beep(200)
                    current_command = "go_straight"
                    print(f"Command: {current_command}")
                
                elif detection_class_name == "Turn_right":
                    car.set_beep(200)
                    current_command = "go_right"
                    print(f"Command: {current_command}")
                
                elif detection_class_name == "horn" and detection_bbox_size >= 3800:
                    car.set_beep(3000)
                
                elif detection_class_name == "5km" and detection_bbox_size >= 4000:
                    if not slowdown_active:
                        print("Speed limit detected")
                        car.set_beep(200)
                        slowdown_active = True
                        slowdown_start_time = time.time()
                
                torch.cuda.empty_cache()
            else:
                detection_confidence = 0.0
                detection_class_name = None
                detection_bbox_size = 0


            frame_resized = cv2.resize(frame, (640, 352), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Original frame', frame_resized)

            mask = ddrnet_engine.predict(frame)
            is_crossroad, crossroad_confidence = crossroad_detector.predict(mask)
            should_stop_crossroad, ignore_stop_sign, sign_command = crossroad_manager.update(
                is_crossroad=is_crossroad,
                sign_command=sign_command,
            )

            if current_command == "go_straight":
                x_pred, action_pred, inference_time= go_straight_engine.predict_steering(mask)
            elif current_command == "go_right":
                x_pred, action_pred, inference_time= go_right_engine.predict_steering(mask)
            elif current_command == "go_left":
                x_pred, action_pred, inference_time= go_left_engine.predict_steering(mask)
            elif current_command == "park_long":
                height = frame_resized.shape[0]
                start_row = int(height * 0.3) 
                roi = frame_resized[start_row:, :, :]

                x_pred, action_pred = lstm_engine_park_long.predict(roi)

                if x_pred is None or action_pred is None:
                    print("Buffer not ready")
                    continue
            elif current_command == "park_short":
                height = frame_resized.shape[0]
                start_row = int(height * 0.3) 
                roi = frame_resized[start_row:, :, :]

                x_pred, action_pred = lstm_engine_park_short.predict(roi)

                if x_pred is None or action_pred is None:
                    print("Buffer not ready")
                    continue
            elif current_command == "park_out":
                height = frame_resized.shape[0]
                start_row = int(height * 0.3) 
                roi = frame_resized[start_row:, :, :]

                x_pred, action_pred = lstm_engine_park_out.predict(roi)
                if x_pred is None or action_pred is None:
                    print("Buffer not ready")
                    continue
                
            steering_angle, speed = controller.calculate_steering_and_speed(x_pred, action_pred, current_command)

            #print("x_pred:", x_pred)
            #print("Action:", action_pred)
            #print("Speed: ", speed)

            if slowdown_active:
                speed = 0.15
                #if (time.time() - slowdown_start_time) >= slowdown_duration:
                    #slowdown_active = False
                    #speed = 0.2

            

            should_move = car_start and not obstacle_detected and not should_stop_crossroad and not stopped_for_non_parking


            #road)
            if obstacle_detected and car_start:
                if not stopped_for_obstacle:
                    print("OBSTACLE DETECTED - STOPPING!")
                    car.set_beep(200)
                    stopped_for_obstacle = True
                car.set_motor(0, 0, 0, 0)
                car.set_car_motion(0, 0, 0)
                car.set_akm_steering_angle(steering_angle, False)
            elif should_stop_crossroad:
                # Comment out the print statement to disable intersection printing
                # print("STOPPING AT CROSSROAD")
                car.set_motor(0, 0, 0, 0)
                car.set_car_motion(0, 0, 0)
                car.set_akm_steering_angle(steering_angle, False)
            elif should_move:
                if stopped_for_obstacle:
                    print("Path clear - resuming")
                    car.set_beep(100)
                    stopped_for_obstacle = False
                car.set_motor(30, speed, 30, speed)
                car.set_akm_steering_angle(steering_angle, False)
            else:
                car.set_motor(0, 0, 0, 0)
                car.set_akm_steering_angle(steering_angle, False)


            frame_count += 1
            
            if frame_count % 30 == 0:
                fps, ms = perf.get_stats()
                status = "GOOD" if fps >= 28 else "SLOW"
                print(f"FPS: {fps:.1f} | Frame Time: {ms:.1f}ms | {status}")


            #cv2.circle(result_frame, (x_pred, y_pred), 10, (0, 255, 0), -1)
            #cv2.imshow('Steering Prediction', result_frame)
            cv2.imshow('Lane Detection', mask)
            #cv2.imshow('Darkened image', darkened_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                current_command = "go_right"
                local_command_active = True
                local_command_time = time.time()
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False  # Reset parking state tracking
                    print(f"[PARKING MODE] Exited parking mode - navigation commands now active")
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                    print(f"[STOP STATE] Cleared stop state - car can now move")
                #cnn_engine.set_command(current_command) 
                print(f"Command: {current_command} (local priority active for {LOCAL_COMMAND_TIMEOUT}s)")
            elif key == ord('a'):
                current_command = "go_left"
                local_command_active = True
                local_command_time = time.time()
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False  # Reset parking state tracking
                    print(f"[PARKING MODE] Exited parking mode - navigation commands now active")
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                    print(f"[STOP STATE] Cleared stop state - car can now move")
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} (local priority active for {LOCAL_COMMAND_TIMEOUT}s)")
            elif key == ord('w'):
                current_command = "go_straight"
                local_command_active = True
                local_command_time = time.time()
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False  # Reset parking state tracking
                    print(f"[PARKING MODE] Exited parking mode - navigation commands now active")
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                    print(f"[STOP STATE] Cleared stop state - car can now move")
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} (local priority active for {LOCAL_COMMAND_TIMEOUT}s)")
            elif key == ord('p'):
                current_command = "park_long"
                in_parking_mode = True
                parking_was_occupied = False  # Reset parking state tracking
                waiting_for_parking_exit = False  # Reset exit waiting flag
                stopped_for_non_parking = False
                # p, l, g, o are excluded from cooldown - UDP can override immediately
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} (parking mode activated - navigation blocked)")
            elif key == ord('l'):
                current_command = "park_short"
                in_parking_mode = True
                parking_was_occupied = False  # Reset parking state tracking
                waiting_for_parking_exit = False  # Reset exit waiting flag
                stopped_for_non_parking = False
                # p, l, g, o are excluded from cooldown - UDP can override immediately
                print(f"Command: {current_command} (parking mode activated - navigation blocked)")
            elif key == ord('g'):
                current_command = "go_park"
                in_parking_mode = True
                parking_was_occupied = False  # Reset parking state tracking
                stopped_for_non_parking = False
                # p, l, g, o are excluded from cooldown - UDP can override immediately
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} (parking mode activated - navigation blocked)")
            elif key == ord('o'):
                current_command = "park_out"
                stopped_for_non_parking = False
                waiting_for_parking_exit = True  # Start continuously checking for parking exit
                in_parking_mode = True  # Keep parking mode active until exit is confirmed
                # p, l, g, o are excluded from cooldown - UDP can override immediately
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} - Will check continuously for parking exit...")
                
                # Do initial check
                parking_status = udp_comm.get_parking_status()
                if parking_status:
                    is_occupied = parking_status.get('is_occupied', True)
                    zone_name = parking_status.get('zone_name', '')
                    if not is_occupied:
                        # Car already exited, exit parking mode immediately
                        in_parking_mode = False
                        parking_was_occupied = False
                        waiting_for_parking_exit = False
                        current_command = "go_straight"
                        print(f"[PARKING MODE] Car already exited parking slot - Parking mode DEACTIVATED")
                    else:
                        print(f"[PARKING MODE] Waiting for car to exit parking slot (current: occupied={is_occupied})...")
                else:
                    print(f"[PARKING MODE] No parking status available - Will check continuously...")
            elif key == ord('s'):
                sign_command = 'stop'
                car.set_beep(200)
                print("Car will stop at ")
            elif key == ord('z'):
                if car_start == True:
                    car_start = False
                    car.set_beep(100)
                else:
                    car_start = True
                    car.set_beep(100)
    

    except KeyboardInterrupt:
        print("Inference stopped.")
    finally:
        car.set_akm_steering_angle(0, False)
        car.set_car_motion(0, 0, 0)
        # if opencv
        cap.release()
        time.sleep(0.5) 
        cv2.destroyAllWindows()
        print("Cleanup completed!")

        fps, ms = perf.get_stats()
        print(f"\nFinal Stats: {fps:.1f} FPS | {ms:.1f}ms per frame")


if __name__ == "__main__":
    main()
