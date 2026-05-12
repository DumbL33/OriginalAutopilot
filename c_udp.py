"""
Hybrid Autopilot Implementation
Combines autopilot.py and final_udp.py with priority system:
- Priority 1: Manual Keyboard Commands (highest, 5s timeout, overrides everything)
- Priority 2: YOLO Detection (medium, overrides UDP)
- Priority 3: UDP Navigation Commands (default, lowest priority)
- Parking logic from final_udp.py
"""

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
from steering.lstm_module import LSTMSteeringModel
from object_detection.src.lidar.lidar_subscriber import LidarSubscriber
from object_detection.src.controller import RobotController

# Model paths (same as both scripts)
DDRNET_ENGINE_PATH_SLOVAKIATECH = "models/tensorrt_ddrnet_model_slovakiatech/ddrnet_lane.trt"
CNN_ENGINE_PATH_GO_STRAIGHT = "models/tensorrt_models_go_straight/steering_standard_fp16_batch1.trt"
CNN_ENGINE_PATH_GO_RIGHT = "models/tensorrt_models_go_right/steering_standard_fp16_batch1.trt"
CNN_ENGINE_PATH_GO_LEFT = "models/tensorrt_models_go_left/steering_standard_fp16_batch1.trt"
LSTM_ENGINE_PARK_SHORT_3SEQ = 'models/tensorrt_lstm_short/best_lstm_steering_park_short_3seq.trt'
LSTM_ENGINE_PARK_LONG_3SEQ = 'models/tensorrt_lstm_long/best_lstm_steering_park_long_0.7_3seq_fp16.trt'
LSTM_ENGINE_PARK_OUT = 'models/tensorrt_lstm_out/best_lstm_steering_park_out.trt'

# Constants
LOCAL_COMMAND_TIMEOUT = 5.0  # Manual command priority duration
YOLO_COMMAND_TIMEOUT = 3.0   # YOLO command priority duration (ignores other commands)
YOLO_FRAME_SKIP = 3
SLOWDOWN_DURATION = 5.0
STOP_DURATION = 5.0

# UDP Configuration
UDP_RECV_PORT_PARKING = 50002
UDP_RECV_PORT_COMMANDS = 50001
UDP_RECV_PORT_OBJECTS = 50003


# ============================================================================
# CLASSES (Copy from original files)
# ============================================================================

class SimpleDetector:
    """YOLO detector from autopilot.py"""
    def __init__(self, model_path, conf_thres=0.8):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf_thres
        self.model.img_size = 640
        self.model.iou = 0.45
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.class_names = self.model.names
        self.colors = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]
        ])
        print(f"Model loaded with classes: {self.class_names}")
    
    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_direct = cv2.resize(frame_rgb, (640, 640))
        results = self.model(resized_direct)
        detections = results.xyxy[0].cpu().numpy()
        return cv2.cvtColor(resized_direct, cv2.COLOR_RGB2BGR), detections


class UDPCommunication:
    """UDP communication from final_udp.py"""
    def __init__(self, parking_port=50002, command_port=50001, objects_port=50003):
        self.parking_port = parking_port
        self.command_port = command_port
        self.objects_port = objects_port
        
        # Initialize sockets
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
        
        # State storage
        self.parking_status = None
        self.action_command = None
        self.object_detections = None
        
        # Threading locks
        self.parking_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.objects_lock = threading.Lock()
        
        # Start receiver threads
        self.running = True
        self.parking_thread = threading.Thread(target=self._receive_parking_loop, daemon=True)
        self.command_thread = threading.Thread(target=self._receive_command_loop, daemon=True)
        self.objects_thread = threading.Thread(target=self._receive_objects_loop, daemon=True)
        
        self.parking_thread.start()
        self.command_thread.start()
        self.objects_thread.start()
        
        print(f"UDP Communication initialized on ports {parking_port}, {command_port}, {objects_port}")
    
    def _receive_parking_loop(self):
        while self.running:
            try:
                data, addr = self.parking_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                if message.get('type') == 'parking_status':
                    with self.parking_lock:
                        self.parking_status = message
            except socket.error:
                time.sleep(0.01)
            except Exception as e:
                pass  # Silent error handling
    
    def _receive_command_loop(self):
        while self.running:
            try:
                data, addr = self.command_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                with self.command_lock:
                    self.action_command = message
            except socket.error:
                time.sleep(0.01)
            except Exception as e:
                pass
    
    def _receive_objects_loop(self):
        while self.running:
            try:
                data, addr = self.objects_socket.recvfrom(8192)
                message = json.loads(data.decode('utf-8'))
                if message.get('type') == 'object_detections':
                    with self.objects_lock:
                        self.object_detections = message
            except socket.error:
                time.sleep(0.01)
            except Exception as e:
                pass
    
    def get_parking_status(self):
        with self.parking_lock:
            return self.parking_status
    
    def get_action_command(self):
        with self.command_lock:
            return self.action_command
    
    def stop(self):
        self.running = False
        self.parking_socket.close()
        self.command_socket.close()
        self.objects_socket.close()


class SimplePerf:
    """Performance monitoring"""
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Initialize UDP communication
    udp_comm = UDPCommunication(
        parking_port=UDP_RECV_PORT_PARKING,
        command_port=UDP_RECV_PORT_COMMANDS,
        objects_port=UDP_RECV_PORT_OBJECTS
    )
    
    # Initialize LIDAR
    rclpy.init()
    lidar_node = LidarSubscriber()
    executor = SingleThreadedExecutor()
    executor.add_node(lidar_node)
    
    perf = SimplePerf()
    speed = 0.2
    
    # Camera setup
    cap = cv2.VideoCapture('/dev/video2')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
    
    # Initialize YOLO detector (Priority 1 source)
    detector = SimpleDetector(
        "/root/yahboom_data_ws/yahboomcar_ros2_ws/Rosmaster/auto_drive/annex/best_engine/best2.pt",
        conf_thres=0.7
    )
    
    # Initialize steering models
    go_straight_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_STRAIGHT)
    go_right_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_RIGHT)
    go_left_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_LEFT)
    lstm_engine_park_short = LSTMSteeringModel(LSTM_ENGINE_PARK_SHORT_3SEQ, sequence_length=3)
    lstm_engine_park_long = LSTMSteeringModel(LSTM_ENGINE_PARK_LONG_3SEQ, sequence_length=3)
    lstm_engine_park_out = LSTMSteeringModel(LSTM_ENGINE_PARK_OUT, sequence_length=5)
    
    # Lane segmentation
    ddrnet_engine = DDRNetTensorRT(
        engine_path=DDRNET_ENGINE_PATH_SLOVAKIATECH,
        confidence_threshold=0.8
    )
    
    # Steering controller
    controller = SteeringController(frame_width=224, frame_height=224)
    
    # Obstacle controller
    obstacle_controller = RobotController(lidar_node.car, lidar_node, stop_delay=0.5)
    
    # Car initialization
    car = Rosmaster()
    car.set_beep(100)
    car_start = False
    current_command = 'go_straight'
    
    # ========================================================================
    # STATE VARIABLES (Hybrid)
    # ========================================================================
    
    # Priority tracking
    yolo_command_active = False      # Priority 1: YOLO detection active
    yolo_command_time = None         # Timestamp of last YOLO command
    local_command_active = False     # Priority 2: Manual command active
    local_command_time = None        # Timestamp of last manual command
    
    # Parking mode (from final_udp.py)
    in_parking_mode = False
    parking_was_occupied = False
    waiting_for_parking_exit = False
    parking_exit_time = None  # Timestamp when parking exit UDP is received
    stopped_for_non_parking = False
    
    # Obstacle detection
    obstacle_detected = False
    stopped_for_obstacle = False
    
    # Speed control
    slowdown_active = False
    slowdown_start_time = None
    
    # Stop sign handling (from autopilot.py)
    stop_sign_processing = False
    stop_sign_processed = False
    stop_start_time = None
    stop_sign_currently_visible = False
    
    # YOLO detection
    yolo_frame_counter = 0
    
    # Other
    frame_count = 0
    last_udp_command = None  # Track last UDP command to only print on new packets
    
    try:
        while True:
            perf.tick()
            
            # Update LIDAR
            rclpy.spin_once(lidar_node, timeout_sec=0.001)
            obstacle_stop_signal = obstacle_controller.update()
            obstacle_detected = lidar_node.get_info_obstacle()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (640, 352), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Original frame', frame_resized)
            
            # ================================================================
            # STEP 1: Check Priority Timeouts
            # ================================================================
            
            # Check local command timeout (manual commands have 5s priority)
            if local_command_active and local_command_time is not None:
                if (time.time() - local_command_time) >= LOCAL_COMMAND_TIMEOUT:
                    local_command_active = False
                    local_command_time = None
                    print("[PRIORITY] Manual command timeout expired - YOLO/UDP can override")
            
            # Check YOLO command timeout (YOLO commands ignore other commands for YOLO_COMMAND_TIMEOUT seconds)
            if yolo_command_active and yolo_command_time is not None:
                if (time.time() - yolo_command_time) >= YOLO_COMMAND_TIMEOUT:
                    yolo_command_active = False
                    yolo_command_time = None
                    print("[PRIORITY] YOLO command timeout expired - UDP can override")
            
            # ================================================================
            # STEP 2: Process Manual Keyboard Commands (Priority 1 - Highest)
            # ================================================================
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('w'):
                current_command = "go_straight"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                print(f"[MANUAL PRIORITY] Command: {current_command} (5s priority)")
            elif key == ord('d'):
                current_command = "go_right"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                print(f"[MANUAL PRIORITY] Command: {current_command} (5s priority)")
            elif key == ord('a'):
                current_command = "go_left"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                if in_parking_mode:
                    in_parking_mode = False
                    parking_was_occupied = False
                if stopped_for_non_parking:
                    stopped_for_non_parking = False
                print(f"[MANUAL PRIORITY] Command: {current_command} (5s priority)")
            elif key == ord('p'):
                current_command = "park_long"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                in_parking_mode = True
                parking_was_occupied = False
                waiting_for_parking_exit = False
                stopped_for_non_parking = False
                print(f"[MANUAL PRIORITY] Command: {current_command} (parking mode)")
            elif key == ord('l'):
                current_command = "park_short"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                in_parking_mode = True
                parking_was_occupied = False
                waiting_for_parking_exit = False
                stopped_for_non_parking = False
                print(f"[MANUAL PRIORITY] Command: {current_command} (parking mode)")
            elif key == ord('o'):
                current_command = "park_out"
                local_command_active = True
                local_command_time = time.time()
                yolo_command_active = False  # Override YOLO
                stopped_for_non_parking = False
                waiting_for_parking_exit = True
                parking_exit_time = None  # Reset exit timer
                in_parking_mode = True
                print(f"[MANUAL PRIORITY] Command: {current_command} - checking for parking exit...")
            elif key == ord('z'):
                if car_start:
                    car_start = False
                    car.set_beep(100)
                else:
                    car_start = True
                    car.set_beep(100)
            
            # ================================================================
            # STEP 3: Process YOLO Detection (Priority 2 - Medium)
            # ================================================================
            
            yolo_frame_counter += 1
            run_yolo = (yolo_frame_counter % YOLO_FRAME_SKIP == 0)
            
            # Only process YOLO if manual command is not active
            if run_yolo and not local_command_active:
                yolo_frame, yolo_detections = detector.detect(frame_resized)
                stop_sign_currently_visible = False
                
                if len(yolo_detections) > 0:
                    for det in yolo_detections:
                        x1, y1, x2, y2, conf, class_id = det
                        class_name = detector.class_names[int(class_id)]
                        bbox_area = (x2 - x1) * (y2 - y1)
                        
                        # YOLO commands override UDP (but not manual)
                        if class_name == "Go_straight":
                            current_command = "go_straight"
                            yolo_command_active = True
                            yolo_command_time = time.time()
                            car.set_beep(200)
                            print(f"[YOLO PRIORITY] Command: {current_command}")
                            
                        elif class_name == "Turn_right":
                            current_command = "go_right"
                            yolo_command_active = True
                            yolo_command_time = time.time()
                            car.set_beep(200)
                            print(f"[YOLO PRIORITY] Command: {current_command}")
                            
                        elif class_name == "Shutdown" and bbox_area >= 5700:
                            stop_sign_currently_visible = True
                            if not stop_sign_processed and not stop_sign_processing:
                                print("[YOLO PRIORITY] Stop sign detected - stopping for 5 seconds")
                                car.set_beep(200)
                                stop_sign_processing = True
                                stop_start_time = time.time()
                                yolo_command_active = True
                                yolo_command_time = time.time()
                                
                        elif class_name == "Limiting_velocity" and bbox_area >= 4000:
                            if not slowdown_active:
                                print("[YOLO PRIORITY] Speed limit detected")
                                car.set_beep(200)
                                slowdown_active = True
                                slowdown_start_time = time.time()
                                
                        elif class_name == "Parking_lotA":
                            current_command = "park_long"
                            yolo_command_active = True
                            yolo_command_time = time.time()
                            in_parking_mode = True
                            parking_was_occupied = False
                            print(f"[YOLO PRIORITY] Command: {current_command} (parking mode)")
                            
                        elif class_name == "Parking_lotB":
                            current_command = "park_short"
                            yolo_command_active = True
                            yolo_command_time = time.time()
                            in_parking_mode = True
                            parking_was_occupied = False
                            print(f"[YOLO PRIORITY] Command: {current_command} (parking mode)")
                            
                        elif class_name == "whistle" and bbox_area >= 3800:
                            car.set_beep(3000)
                
                # Reset stop sign processed flag
                if not stop_sign_currently_visible and stop_sign_processed:
                    stop_sign_processed = False
                    print("Stop sign no longer visible - reset for next detection")
            
            # ================================================================
            # STEP 4: Process UDP Navigation Commands (Priority 3 - Default)
            # ================================================================
            
            action_command = udp_comm.get_action_command()
            
            # Reset tracking if no packet received
            if action_command is None:
                last_udp_command = None
            
            # Block UDP commands if parking mode is active (unless manual/YOLO override)
            elif action_command and in_parking_mode and not local_command_active and not yolo_command_active:
                nav_command = action_command.get('command')
                # Only print if this is a new command (new packet received)
                if nav_command != last_udp_command:
                    print(f"[PARKING MODE] UDP command '{nav_command}' IGNORED - parking mode active")
                    last_udp_command = nav_command
            # Only process UDP if no manual or YOLO command active and not in parking mode
            elif action_command and not local_command_active and not yolo_command_active and not in_parking_mode:
                nav_command = action_command.get('command')
                target_waypoint = action_command.get('target_waypoint', '')
                heading = action_command.get('heading')
                
                # Only process and print if this is a new command (new packet received)
                if nav_command != last_udp_command:
                    if nav_command == "straight":
                        current_command = "go_straight"
                        stopped_for_non_parking = False
                        print(f"[UDP DEFAULT] Navigation: {current_command}")
                    elif nav_command == "right":
                        current_command = "go_right"
                        stopped_for_non_parking = False
                        print(f"[UDP DEFAULT] Navigation: {current_command}")
                    elif nav_command == "left":
                        current_command = "go_left"
                        stopped_for_non_parking = False
                        print(f"[UDP DEFAULT] Navigation: {current_command}")
                    elif nav_command == "stop":
                        # Parking logic from final_udp.py
                        target_lower = target_waypoint.lower() if target_waypoint else ""
                        is_parking_target = "parking" in target_lower
                        
                        if is_parking_target:
                            heading_lower = heading.lower() if heading else ""
                            if heading_lower == "east":
                                current_command = "park_short"
                                in_parking_mode = True
                                parking_was_occupied = False
                                waiting_for_parking_exit = False
                                print(f"[UDP DEFAULT] Parking: {current_command}")
                            elif heading_lower == "west":
                                current_command = "park_long"
                                in_parking_mode = True
                                parking_was_occupied = False
                                waiting_for_parking_exit = False
                                print(f"[UDP DEFAULT] Parking: {current_command}")
                        else:
                            stopped_for_non_parking = True
                            print(f"[UDP DEFAULT] Stop at non-parking target")
                    
                    # Update last processed command
                    last_udp_command = nav_command
            
            # ================================================================
            # STEP 5: Parking Exit Detection (from final_udp.py)
            # ================================================================
            
            parking_status = udp_comm.get_parking_status()
            if in_parking_mode and parking_status:
                is_occupied = parking_status.get('is_occupied', True)
                zone_name = parking_status.get('zone_name', '')
                
                if is_occupied:
                    parking_was_occupied = True
                
                if waiting_for_parking_exit:
                    if not is_occupied:
                        # Record the time when exit UDP is received (first time only)
                        if parking_exit_time is None:
                            parking_exit_time = time.time()
                            print(f"[PARKING] Parking exit UDP received - keeping mode active for 5s")
                        
                        # Wait 5 seconds before actually exiting parking mode
                        if parking_exit_time is not None:
                            elapsed = time.time() - parking_exit_time
                            if elapsed >= 5.0:
                                in_parking_mode = False
                                parking_was_occupied = False
                                waiting_for_parking_exit = False
                                parking_exit_time = None
                                stopped_for_non_parking = False
                                current_command = "go_straight"
                                print(f"[PARKING] Car exited parking slot - resuming navigation")
                    else:
                        # If parking becomes occupied again, reset the exit timer
                        parking_exit_time = None
            
            # ================================================================
            # STEP 6: Time-based State Updates
            # ================================================================
            
            # Stop sign processing
            if stop_sign_processing and stop_start_time is not None:
                elapsed_time = time.time() - stop_start_time
                if elapsed_time >= STOP_DURATION:
                    stop_sign_processing = False
                    stop_sign_processed = True
                    current_command = "go_straight"
                    car.set_beep(100)
                    print("[STOP SIGN] Stop completed - resuming")
            
            # Speed limit timeout
            if slowdown_active and slowdown_start_time is not None:
                elapsed_time = time.time() - slowdown_start_time
                if elapsed_time >= SLOWDOWN_DURATION:
                    slowdown_active = False
                    slowdown_start_time = None
                    print("[SPEED LIMIT] Period ended - returning to normal speed")
            
            # ================================================================
            # STEP 7: Lane Detection and Steering Prediction
            # ================================================================
            
            mask = ddrnet_engine.predict(frame)
            
            # Predict steering based on current command
            if current_command == "go_straight":
                x_pred, action_pred, inference_time = go_straight_engine.predict_steering(mask)
            elif current_command == "go_right":
                x_pred, action_pred, inference_time = go_right_engine.predict_steering(mask)
            elif current_command == "go_left":
                x_pred, action_pred, inference_time = go_left_engine.predict_steering(mask)
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
            
            # Apply speed limit
            if slowdown_active:
                speed = 0.30
            
            # ================================================================
            # STEP 8: Movement Control
            # ================================================================
            
            should_move = car_start and not obstacle_detected and not stop_sign_processing and not stopped_for_non_parking
            
            if obstacle_detected and car_start:
                if not stopped_for_obstacle:
                    print("OBSTACLE DETECTED - STOPPING!")
                    car.set_beep(200)
                    stopped_for_obstacle = True
                car.set_motor(0, 0, 0, 0)
                car.set_car_motion(0, 0, 0)
                car.set_akm_steering_angle(steering_angle, False)
            elif stop_sign_processing:
                print("STOPPING AT STOP SIGN")
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
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:
                fps, ms = perf.get_stats()
                status = "GOOD" if fps >= 28 else "SLOW"
                print(f"FPS: {fps:.1f} | Frame Time: {ms:.1f}ms | {status}")
            
            cv2.imshow('Lane Detection', mask)
    
    except KeyboardInterrupt:
        print("Inference stopped.")
    finally:
        # Cleanup
        car.set_akm_steering_angle(0, False)
        car.set_car_motion(0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        udp_comm.stop()
        print("Cleanup completed!")
        
        fps, ms = perf.get_stats()
        print(f"\nFinal Stats: {fps:.1f} FPS | {ms:.1f}ms per frame")


if __name__ == "__main__":
    main()

