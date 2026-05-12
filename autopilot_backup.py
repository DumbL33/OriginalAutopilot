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


class SimpleDetector:
    def __init__(self, model_path, conf_thres=0.8):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf_thres
        self.model.img_size = 640
        self.model.iou = 0.45
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        #self.model.half() 
        self.class_names = self.model.names
        self.colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255]
        ])
        print(f"Model loaded with classes: {self.class_names}")
    
    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized_direct = cv2.resize(frame_rgb, (640, 640))
        
        results = self.model(resized_direct)
        
        #if len(results.xyxy[0]) > 0:
            #print("Raw prediction classes:", results.xyxy[0][:, 5].cpu().numpy())
            #print("Raw prediction scores:", results.xyxy[0][:, 4].cpu().numpy())
        
        detections = results.xyxy[0].cpu().numpy()
        return cv2.cvtColor(resized_direct, cv2.COLOR_RGB2BGR), detections
        
    def draw_boxes(self, frame, detections):
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            label = f"{self.class_names[class_id]} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 1)
            
        return annotated_frame

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

def main():

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

    detector = SimpleDetector("/root/yahboom_data_ws/yahboomcar_ros2_ws/Rosmaster/auto_drive/annex/best_engine/best2.pt", conf_thres=0.7)
    
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
    slowdown_start_time = None
    slowdown_duration = 5.0

    stop_sign_detected = False
    stop_sign_processing = False  
    stop_sign_processed = False   
    stop_duration = 5.0 

    # LIDAR obstacle controller init
    obstacle_controller = RobotController(lidar_node.car, lidar_node, stop_delay=0.5)
    
    # Car lib init
    car = Rosmaster()
    car.set_beep(100)
    car_start = False
    current_command = 'go_straight'

    # Obstacle detection statef
    obstacle_detected = False
    stopped_for_obstacle = False

    frame_count = 0
    stop_detected = False

    yolo_frame_skip = 3
    yolo_detections = []
    last_yolo_frame = 0
    yolo_frame_counter = 0


    turn_command_active = False
    turn_command_start_time = None
    turn_command_duration = 4.0

    # park out init
    park_out_active = False
    park_out_start_time = None
    park_out_duration = 6.0  

    sign_command = None
    try:
        while True:
            perf.tick()

            rclpy.spin_once(lidar_node, timeout_sec=0.001)
            obstacle_stop_signal = obstacle_controller.update()
            obstacle_detected = lidar_node.get_info_obstacle()

            if park_out_active and park_out_start_time is not None:
                elapsed_time = time.time() - park_out_start_time
                if elapsed_time >= park_out_duration:
                    current_command = "go_straight"
                    park_out_active = False
                    park_out_start_time = None
                    car.set_beep(100)
                    print(f"Park out completed! Switching to: {current_command}")

            if slowdown_active and slowdown_start_time is not None:
                elapsed_time = time.time() - slowdown_start_time
                if elapsed_time >= slowdown_duration:
                    slowdown_active = False
                    slowdown_start_time = None
                    print("Speed limit period ended - returning to normal speed")

            
            if turn_command_active and turn_command_start_time is not None:
                elapsed_time = time.time() - turn_command_start_time
                if elapsed_time >= turn_command_duration:
                    current_command = "go_straight"
                    turn_command_active = False
                    turn_command_start_time = None
                    car.set_beep(100)
                    print(f"Turn completed! Switching back to: {current_command}")

            if stop_sign_processing and stop_start_time is not None:
                elapsed_time = time.time() - stop_start_time
                if elapsed_time >= stop_duration:
                    stop_sign_processing = False
                    stop_sign_processed = True
                    current_command = "go_straight"
                    car.set_beep(100)
                    print("Stop completed! Resuming with go_straight")

            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (640, 352), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Original frame', frame_resized)

            yolo_frame_counter += 1
            run_yolo = (yolo_frame_counter % yolo_frame_skip == 0)

            if run_yolo:
                yolo_frame, yolo_detections = detector.detect(frame_resized)

                #print(yolo_detections)

                #torch.cuda.synchronize()  
                #torch.cuda.empty_cache()

                stop_sign_currently_visible = False
                if len(yolo_detections) > 0:
                    for det in yolo_detections:
                        x1, y1, x2, y2, conf, class_id = det
                        class_name = detector.class_names[int(class_id)]
                        #print(class_name)
                        bbox_area = (x2 - x1) * (y2 - y1)

                        if class_name == "Shutdown" and bbox_area >= 5700:
                            stop_sign_currently_visible = True
                            if not stop_sign_processed and not stop_sign_processing:
                                print("Stop sign detected - stopping for 5 seconds")
                                car.set_beep(200)
                                stop_sign_processing = True
                                stop_start_time = time.time()
                        elif class_name == 'Go_straight':
                            car.set_beep(200)
                            current_command = "go_straight"
                            turn_command_active = False
                            #cnn_engine.set_command(current_command)
                            print(f"Command: {current_command}")
                        elif class_name == "Turn_right":
                            car.set_beep(200)
                            current_command = "go_right"
                            turn_command_active = True
                            turn_command_start_time = time.time()
                            print(f"Command: {current_command} - Will switch to go_straight in 4 seconds")  
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


                if not stop_sign_currently_visible and stop_sign_processed:
                    stop_sign_processed = False
                    print("Stop sign no longer visible - reset for next detection")
       
                        

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
                speed = 0.30
                

            should_move = car_start and not obstacle_detected and not should_stop_crossroad and not stop_sign_processing

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


            frame_count += 1
            
            if frame_count % 30 == 0:
                fps, ms = perf.get_stats()
                status = "GOOD" if fps >= 28 else "⚠️ SLOW"
                print(f"FPS: {fps:.1f} | Frame Time: {ms:.1f}ms | {status}")


            #cv2.circle(result_frame, (x_pred, y_pred), 10, (0, 255, 0), -1)
            #cv2.imshow('Steering Prediction', result_frame)
            #cv2.imshow('Lane Detection', mask)
            #cv2.imshow('Darkened image', darkened_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                current_command = "go_right"
                park_out_active = False
                turn_command_active = True
                turn_command_start_time = time.time()
                #cnn_engine.set_command(current_command) 
                print(f"Command: {current_command} - Will switch to go_straight in 4 seconds")
            elif key == ord('a'):
                current_command = "go_left"
                park_out_active = False
                turn_command_active = True
                turn_command_start_time = time.time()
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} - Will switch to go_straight in 4 seconds")
            elif key == ord('w'):
                park_out_active = False
                turn_command_active = False
                current_command = "go_straight"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('p'):
                park_out_active = False
                turn_command_active = False
                current_command = "park_long"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('l'):
                park_out_active = False
                turn_command_active = False
                current_command = "park_short"
                print(f"Command: {current_command}")
            elif key == ord('g'):
                park_out_active = False
                turn_command_active = False
                current_command = "go_park"
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command}")
            elif key == ord('o'):
                current_command = "park_out"
                park_out_active = True
                park_out_start_time = time.time()
                car.set_beep(200)
                #cnn_engine.set_command(current_command)
                print(f"Command: {current_command} - Will switch to go_straight in 3 seconds")
            elif key == ord('s'):
                park_out_active = False
                turn_command_active = False
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
        try:
            car.set_akm_steering_angle(0, False)
            car.set_car_motion(0, 0, 0)
        except Exception as e:
            print(f"Error stopping car: {e}")
        
        try:
            if cap is not None:
                cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")
        
        time.sleep(1.0)
        
        try:
            del go_straight_engine
            del go_right_engine
            del go_left_engine
            del lstm_engine_park_short
            del lstm_engine_park_long
            del lstm_engine_park_out
            del ddrnet_engine
            del crossroad_detector
            del detector
        except Exception as e:
            print(f"Error deleting models: {e}")
        
        print("Cleanup completed!")
        
        try:
            fps, ms = perf.get_stats()
            print(f"\nFinal Stats: {fps:.1f} FPS | {ms:.1f}ms per frame")
        except:
            pass

if __name__ == "__main__":
    main()
