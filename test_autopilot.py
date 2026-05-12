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
UDP_RECV_PORT_PARKING = 50001  
UDP_RECV_PORT_COMMANDS = 50002
UDP_RECV_PORT_OBJECTS = 50003
SERVER_IP = "0.0.0.0"

class UDPCommunication:
    def __init__(self, parking_port=50001, command_port=50002, objects_port=50003):
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
                print(f"[Port {self.command_port}] Action: {message.get('action')}, "
                      f"Distance={message.get('distance'):.2f}, "
                      f"Angle={message.get('angle'):.2f}, "
                      f"Speed={message.get('speed'):.2f}")
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
            print(f"  Action: {command.get('action')}")
            print(f"  Distance: {command.get('distance'):.2f}")
            print(f"  Angle: {command.get('angle'):.2f}")
            print(f"  Speed: {command.get('speed'):.2f}")
            print(f"  Timestamp: {command.get('timestamp')}")
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

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        frame_time = (current_time - self.last_time) * 1000 
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        self.last_time = current_time
    
    def get_stats(self):
        if not self.frame_times:
            return 0.0, 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return fps, avg_frame_time

def main():
    perf = PerformanceMonitor()
    car = Rosmaster()
    car.set_colorful_effect(0, 200, 200, 255)
    #uv_led_control = UVLedController()

    #JETCAM
    cap = USBCamera(capture_device=0, width=640, height=480, fps=60)
    #cap.running = True

    controller = SteeringController()
    ddrnet_engine = DDRNetTensorRT(DDRNET_ENGINE_PATH_SLOVAKIATECH)
    #ddrnet_engine = DDRNetTensorRT(DDRNET_ENGINE_PATH)

    go_straight_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_STRAIGHT, command="go_straight")
    go_right_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_RIGHT, command="go_right")
    go_left_engine = TensorRTCNNSteering(CNN_ENGINE_PATH_GO_LEFT, command="go_left")

    lstm_engine_park_long = LSTMSteeringModel_MASK(LSTM_ENGINE_PARK_LONG_3SEQ)
    lstm_engine_park_short = LSTMSteeringModel_MASK(LSTM_ENGINE_PARK_SHORT_3SEQ)
    lstm_engine_park_out = LSTMSteeringModel_MASK(LSTM_ENGINE_PARK_OUT_3SEQ)

    crossroad_detector = CrossroadDetectorTRT(CROSSROAD_ENGINE_PATH)
    crossroad_manager = CrossroadManager()

    udp_comm = UDPCommunication()

    slowdown_active = False
    slowdown_duration = 10
    slowdown_start_time = 0

    current_command = "go_straight"
    #cnn_engine.set_command(current_command)
    sign_command = None
    ignore_stop_sign = False

    frame_count = 0 

    try:
        car_start = False
        car.set_beep(100)
        obstacle_detected = False
        stopped_for_obstacle = False
        
        # Wait a bit for UDP threads to start receiving data
        print("\nWaiting 3 seconds for UDP data to arrive...")
        time.sleep(3)
        
        # Print initial status of all ports
        udp_comm.print_current_status()

        while True:
            frame = cap.read()
            perf.update()
            
            frame_resized = cv2.resize(frame, (320, 240))
            
            # Get largest detected object
            largest_obj = udp_comm.get_largest_object()
            
            # Check for obstacles
            if largest_obj:
                bbox = largest_obj.get('bbox', {})
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)
                bbox_area = (x2 - x1) * (y2 - y1)
                class_name = largest_obj.get('class_name', '')
                confidence = largest_obj.get('confidence', 0)
                
                if class_name == 'obstacle':
                    if not ignore_stop_sign:
                        if bbox_area >= 3000 and not obstacle_detected:
                            obstacle_detected = True
                            print(f"OBSTACLE DETECTED! Area: {bbox_area:.0f}, Confidence: {confidence:.2f}")
                    else:
                        if obstacle_detected:
                            obstacle_detected = False
                            stopped_for_obstacle = False
                            print("Obstacle ignored (stop sign)")
                
                else:
                    if obstacle_detected:
                        obstacle_detected = False
                        stopped_for_obstacle = False
                        print("Path clear")
                    
                    if not ignore_stop_sign:
                        if class_name == "Stop_sign" and bbox_area >= 3000:
                            if sign_command != 'stop':
                                sign_command = 'stop'
                                print("Stop sign detected")
                                car.set_beep(200)
                                time.sleep(3)
                                print("RESTARTING CAR")
                        elif class_name == 'Go_straight':
                            car.set_beep(200)
                            current_command = "go_straight"
                            #cnn_engine.set_command(current_command)
                            print(f"Command: {current_command}")
                        elif class_name == "Turn_right":
                            car.set_beep(200)
                            current_command = "go_right"
                            #cnn_engine.set_command(current_command)
                            print(f"Command: {current_command}")
                        
                        elif class_name == "whistle" and bbox_area >= 3800:
                            car.set_beep(3000)
                        elif class_name == "Limiting_velocity" and bbox_area >= 4000:
                            if not slowdown_active:
                                print("Speed limit detected")
                                car.set_beep(200)
                                slowdown_active = True
                                slowdown_start_time = time.time()
                        elif class_name == "Parking_lotA":
                            current_command = "park_long"
                        elif class_name == "Parking_lotB":
                            current_command = "park_short"
                            
                        

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

            

            should_move = car_start and not obstacle_detected and not should_stop_crossroad


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
                print("STOPPING AT CROSSROAD")
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
                status = "GOOD" if fps >= 28 else "⚠️ SLOW"
                print(f"FPS: {fps:.1f} | Frame Time: {ms:.1f}ms | {status}")

            # Print port status every 100 frames
            if frame_count % 100 == 0:
                udp_comm.print_current_status()

            #cv2.circle(result_frame, (x_pred, y_pred), 10, (0, 255, 0), -1)
            #cv2.imshow('Steering Prediction', result_frame)
            cv2.imshow('Lane Detection', mask)
            #cv2.imshow('Darkened image', darkened_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                current_command = "go_right"
                #cnn_engine.set_command(current_command) 
                print(f"Command: {current_command}")
            elif key == ord('a'):
                current_command = "go_left"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('w'):
                current_command = "go_straight"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('p'):
                current_command = "park_long"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('l'):
                current_command = "park_short"
                print(f"Command: {current_command}")
            elif key == ord('g'):
                current_command = "go_park"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('o'):
                current_command = "park_out"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
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
            elif key == ord('m'):
                # Manual trigger to print port status
                udp_comm.print_current_status()
    

    except KeyboardInterrupt:
        print("Inference stopped.")
    finally:
        udp_comm.stop()
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