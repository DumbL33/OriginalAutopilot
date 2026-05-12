"""
TensorRT LSTM Parking Model Inference Module
File: lstm_steering_inference.py

Usage:
    from lstm_steering_inference import LSTMSteeringModel
    
    model = LSTMSteeringModel('lstm_steering_park_fp16.trt', sequence_length=5)
    
    # Feed frames one at a time
    for frame in video_frames:
        x, action = model.predict(frame)
        print(f"Steering: {x:.2f}, Action: {action:.2f}")
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from collections import deque


class LSTMSteeringModel:
    """
    TensorRT-based LSTM steering model for parking.
    Maintains a rolling buffer of frames for temporal prediction.
    Predicts X coordinate (steering) and action from RGB images.
    """
    
    def __init__(self, engine_path, sequence_length=5, verbose=True):
        """
        Initialize the LSTM steering model.
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            sequence_length: Number of frames in sequence (must match training)
            verbose: Print initialization info
        """
        self.engine_path = Path(engine_path)
        self.sequence_length = sequence_length
        self.verbose = verbose
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Frame buffer for temporal sequence
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self._load_engine()
        self._allocate_buffers()
        
        if self.verbose:
            print(f"✓ LSTM Steering model loaded: {self.engine_path.name}")
            print(f"  Input shape: (1, {sequence_length}, 3, 224, 224)")
            print(f"  Sequence length: {sequence_length} frames")
            print(f"  Outputs: x (steering), action")
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
    
    def _allocate_buffers(self):
        """Allocate CUDA buffers for inference"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name
                })
    
    def preprocess_frame(self, image):
        """
        Preprocess single frame for model input.
        
        Args:
            image: numpy array (H, W, 3) - RGB camera image
        
        Returns:
            preprocessed: numpy array (3, 224, 224) ready for buffer
        """
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats - same as training)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        return image.astype(np.float32)
    
    def add_frame(self, image):
        """
        Add a new frame to the buffer.
        
        Args:
            image: numpy array - raw RGB camera image (any size)
        """
        preprocessed = self.preprocess_frame(image)
        self.frame_buffer.append(preprocessed)
    
    def is_ready(self):
        """
        Check if buffer has enough frames for inference.
        
        Returns:
            bool: True if buffer is full
        """
        return len(self.frame_buffer) == self.sequence_length
    
    def infer(self, sequence):
        """
        Run inference on preprocessed sequence.
        
        Args:
            sequence: numpy array (1, sequence_length, 3, 224, 224)
        
        Returns:
            x: float - relative X coordinate in [-1, 1]
            action: float - action value in [-1, 1]
        """
        # Copy input to device
        np.copyto(self.inputs[0]['host'], sequence.ravel())
        
        # Set tensor addresses
        self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device']))
        
        for output in self.outputs:
            self.context.set_tensor_address(output['name'], int(output['device']))
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs from device
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        
        # Extract results
        x_output = float(self.outputs[0]['host'][0])
        action_output = float(self.outputs[1]['host'][0])
        
        return x_output, action_output
    
    def predict(self, image):
        """
        Complete prediction pipeline: add frame + inference.
        
        Args:
            image: numpy array - raw RGB camera image (any size)
        
        Returns:
            x: float - relative_x in range [-1, 1]
                -1: steer hard left
                 0: go straight
                 1: steer hard right
            action: float - action in range [-1, 1]
                -1: backward
                 0: stop
                 1: forward
            
        Note: Returns None, None if buffer not full yet
        """
        # Add frame to buffer
        self.add_frame(image)
        
        # Check if buffer is ready
        if not self.is_ready():
            if self.verbose:
                print(f"Buffer filling: {len(self.frame_buffer)}/{self.sequence_length}")
            return None, None
        
        # Create sequence from buffer
        sequence = np.stack(list(self.frame_buffer), axis=0)  # (seq_len, 3, 224, 224)
        sequence = sequence[np.newaxis, ...]  # Add batch dim: (1, seq_len, 3, 224, 224)
        
        # Run inference
        x, action = self.infer(sequence)
        
        return x, action
    
    def predict_with_info(self, image):
        """
        Predict with additional interpretation info.
        
        Args:
            image: numpy array - raw RGB camera image
        
        Returns:
            dict with keys:
                'x': float - horizontal steering position
                'action': float - action value
                'action_label': str - 'FORWARD', 'STOP', or 'BACKWARD'
                'steering_direction': str - 'LEFT', 'CENTER', or 'RIGHT'
                'steering_magnitude': str - 'SHARP', 'MODERATE', or 'GENTLE'
                'buffer_ready': bool - whether buffer is full
            
        Note: Returns None if buffer not full yet
        """
        x, action = self.predict(image)
        
        if x is None or action is None:
            return {
                'buffer_ready': False,
                'buffer_size': len(self.frame_buffer),
                'required_size': self.sequence_length
            }
        
        # Interpret action
        if action > 0.3:
            action_label = "FORWARD"
        elif action < -0.3:
            action_label = "BACKWARD"
        else:
            action_label = "STOP"
        
        # Interpret steering direction
        if x < -0.3:
            steering_direction = "LEFT"
        elif x > 0.3:
            steering_direction = "RIGHT"
        else:
            steering_direction = "CENTER"
        
        # Interpret steering magnitude
        abs_x = abs(x)
        if abs_x > 0.7:
            steering_magnitude = "SHARP"
        elif abs_x > 0.3:
            steering_magnitude = "MODERATE"
        else:
            steering_magnitude = "GENTLE"
        
        return {
            'x': float(x),
            'action': float(action),
            'action_label': action_label,
            'steering_direction': steering_direction,
            'steering_magnitude': steering_magnitude,
            'buffer_ready': True
        }
    
    def reset_buffer(self):
        """Clear the frame buffer (useful when starting new parking maneuver)"""
        self.frame_buffer.clear()
        if self.verbose:
            print("✓ Frame buffer reset")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()

