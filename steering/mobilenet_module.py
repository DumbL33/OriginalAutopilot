"""
TensorRT Steering Model Inference Module (X-Only Version)
File: steering_inference_x_only.py

Usage:
    from steering_inference_x_only import SteeringModel
    
    model = SteeringModel('path/to/steering_mobilenet_fp16.trt')
    x, action = model.predict(mask)
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path


class SteeringModel_MobileNet:
    """
    TensorRT-based steering model for autonomous driving.
    Predicts X coordinate (steering) and action from lane masks.
    
    Note: This is the X-only version (no Y coordinate).
    """
    
    def __init__(self, engine_path, verbose=True):
        """
        Initialize the steering model.
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            verbose: Print initialization info
        """
        self.engine_path = Path(engine_path)
        self.verbose = verbose
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self._load_engine()
        self._allocate_buffers()
        
        if self.verbose:
            print(f"✓ Steering model loaded: {self.engine_path.name}")
            print(f"  Input shape: (1, 3, 224, 224)")
            print(f"  Outputs: x (1 value), action (1 value)")
    
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
    
    def preprocess(self, mask):
        """
        Preprocess mask for model input.
        
        Args:
            mask: numpy array (H, W, 3) or (H, W) - lane mask image
        
        Returns:
            preprocessed: numpy array (1, 3, 224, 224) ready for inference
        """
        # Handle grayscale
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Resize to 224x224
        mask = cv2.resize(mask, (224, 224))
        
        # Convert BGR to RGB if needed
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        mask = mask.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mask = (mask - mean) / std
        
        # HWC to CHW
        mask = mask.transpose(2, 0, 1)
        
        # Add batch dimension
        mask = mask[np.newaxis, :, :, :]
        
        return mask.astype(np.float32)
    
    def infer(self, input_data):
        """
        Run inference on preprocessed data.
        
        Args:
            input_data: numpy array (1, 3, 224, 224)
        
        Returns:
            x: float - relative X coordinate in [-1, 1]
            action: float - action value in [-1, 1]
        """
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
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
        
        # Extract results (X is single value, action is single value)
        x_output = float(self.outputs[0]['host'][0])  # Single X value
        action_output = float(self.outputs[1]['host'][0])  # Single action value
        
        return x_output, action_output
    
    def predict(self, mask):
        """
        Complete prediction pipeline: preprocess + inference.
        
        Args:
            mask: numpy array - raw lane mask image (any size)
        
        Returns:
            x: float - relative_x in range [-1, 1]
                -1: steer hard left
                 0: go straight
                 1: steer hard right
            action: float - action in range [-1, 1]
                -1: backward
                 0: stop
                 1: forward
        """
        preprocessed = self.preprocess(mask)
        x, action = self.infer(preprocessed)
        return x, action
    
    def predict_with_info(self, mask):
        """
        Predict with additional interpretation info.
        
        Args:
            mask: numpy array - raw lane mask image
        
        Returns:
            dict with keys:
                'x': float - horizontal steering position
                'action': float - action value
                'action_label': str - 'FORWARD', 'STOP', or 'BACKWARD'
                'steering_direction': str - 'LEFT', 'CENTER', or 'RIGHT'
                'steering_magnitude': str - 'SHARP', 'MODERATE', or 'GENTLE'
        """
        x, action = self.predict(mask)
        
        # Interpret action
        if action > 0.5:
            action_label = "FORWARD"
        elif action < -0.5:
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
            'steering_magnitude': steering_magnitude
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


# ============= Standalone Test =============
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("Steering Model Test (X-Only Version)")
    print("="*60)
    
    # Initialize model
    MODEL_PATH = "../models/tensorrt_models/steering_mobilenet_fp16.trt"
    model = SteeringModel(MODEL_PATH, verbose=True)
    
    # Create dummy mask (white lanes on black background)
    print("\nCreating test mask...")
    test_mask = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw some white lanes
    cv2.line(test_mask, (800, 1080), (960, 400), (255, 255, 255), 50)
    cv2.line(test_mask, (1120, 1080), (960, 400), (255, 255, 255), 50)
    
    # Single prediction
    print("\nRunning single prediction...")
    x, action = model.predict(test_mask)
    print(f"Result:")
    print(f"  X: {x:.4f} ({'←LEFT' if x < -0.3 else '→RIGHT' if x > 0.3 else '↑STRAIGHT'})")
    print(f"  Action: {action:.4f}")
    
    # Prediction with info
    print("\nRunning prediction with info...")
    result = model.predict_with_info(test_mask)
    print(f"Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Benchmark
    print("\nBenchmarking...")
    num_iterations = 100
    times = []