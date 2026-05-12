import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Tuple
import time

class TensorRTCNNSteering:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = None
        self.context = None
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        self.input_height = 224
        self.input_width = 224
        
        self.inference_times = []
        
        self._load_model()
        self._detect_input_shape()
        self._allocate_memory()
        self._warmup_model()
        
        print(f"TensorRT CNN steering ready (X + Action only)")
    
    def _load_model(self):
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.model_path}")
        
        self.context = self.engine.create_execution_context()
        print(f"TensorRT CNN engine loaded: {self.model_path}")
    
    def _detect_input_shape(self):
        """Detect input/output shapes - expects mask input, X output, action output"""
        try:
            # Try to get bindings by name
            mask_binding = self.engine.get_binding_index('mask')
            x_output_binding = self.engine.get_binding_index('x_pred')
            action_output_binding = self.engine.get_binding_index('action_pred')
        except:
            # Fallback to indices
            mask_binding = 0
            x_output_binding = 1
            action_output_binding = 2
        
        mask_shape = self.engine.get_binding_shape(mask_binding)
        x_output_shape = self.engine.get_binding_shape(x_output_binding)
        action_output_shape = self.engine.get_binding_shape(action_output_binding)
        
        if len(mask_shape) == 4:
            self.input_height = mask_shape[2]
            self.input_width = mask_shape[3]
        
        print(f"Mask input shape: {mask_shape}")
        print(f"X output shape: {x_output_shape}")
        print(f"Action output shape: {action_output_shape}")
        print(f"Model expects images: {self.input_width}x{self.input_height}")
    
    def _allocate_memory(self):
        """Allocate memory for single input model"""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        print(f"Memory allocated: {len(self.inputs)} input(s), {len(self.outputs)} output(s)")
        
        # Verify we have the right number of inputs/outputs
        if len(self.inputs) != 1:
            print(f"WARNING: Expected 1 input, got {len(self.inputs)}")
        if len(self.outputs) != 2:
            print(f"WARNING: Expected 2 outputs (X + Action), got {len(self.outputs)}")
    
    def _warmup_model(self):
        """Warmup model with dummy data"""
        print("Warming up TensorRT CNN...")
        
        dummy_mask = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        for _ in range(5):
            self.predict_steering(dummy_mask)
        
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        print(f"CNN warmup complete. Average time: {avg_time:.2f}ms")
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess mask for model input"""
        # Convert to 3-channel if grayscale
        if len(mask.shape) == 2:
            mask_3ch = np.stack([mask, mask, mask], axis=2)
        else:
            mask_3ch = mask.copy()
        
        # Resize to model input size
        resized = cv2.resize(mask_3ch, (self.input_width, self.input_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3:
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = resized
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        mean = mean.reshape(1, 1, 3)
        std = std.reshape(1, 1, 3)
        
        normalized = (normalized - mean) / std
        
        # Convert to CHW format (channels first)
        image_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
        
        return image_tensor
    
    def predict_steering(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict steering from road mask
        
        Args:
            mask: Road segmentation mask (grayscale or RGB)
            
        Returns:
            tuple: (relative_x, action_value, inference_time_ms)
                - relative_x: float in [-1, 1] range (left=-1, center=0, right=1)
                - action_value: float in [-1, 1] range (backward=-1, stop=0, forward=1)
                - inference_time_ms: inference time in milliseconds
        """
        start_time = time.time()
        
        try:
            # Preprocess mask
            image_tensor = self.preprocess_mask(mask)
            
            # Copy to GPU - only one input
            np.copyto(self.inputs[0]['host'], image_tensor.ravel())
            cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
            
            # Run inference
            self.context.execute_v2(bindings=self.bindings)
            
            # Get outputs
            cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])  # x_pred
            cuda.memcpy_dtoh(self.outputs[1]['host'], self.outputs[1]['device'])  # action_pred
            
            # Parse X output (single value in [-1, 1] range)
            x_output = self.outputs[0]['host'].copy()
            relative_x = float(np.clip(x_output[0], -1.0, 1.0))
            
            # Parse action output (single value in [-1, 1] range)
            action_output = self.outputs[1]['host'].copy()
            action_value = float(np.clip(action_output[0], -1.0, 1.0))
            
        except Exception as e:
            print(f"CNN inference error: {e}")
            relative_x = 0.0
            action_value = 0.0
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        return relative_x, action_value, inference_time
    
    def get_fps(self) -> float:
        """Calculate average FPS"""
        if not self.inference_times:
            return 0.0
        avg_time_ms = np.mean(self.inference_times)
        return 1000.0 / avg_time_ms
    
    def cleanup(self):
        """Free GPU memory"""
        for inp in self.inputs:
            if 'device' in inp:
                inp['device'].free()
        
        for out in self.outputs:
            if 'device' in out:
                out['device'].free()
        
        print("TensorRT CNN cleanup complete")


def draw_steering_prediction(frame: np.ndarray, relative_x: float, 
                           action_value: float, inference_time: float, fps: float) -> np.ndarray:
    """
    Draw steering prediction on frame (X only, no Y)
    
    Args:
        frame: Image frame
        relative_x: X coordinate in [-1, 1] range
        action_value: Action value in [-1, 1] range
        inference_time: Inference time in ms
        fps: Current FPS
    """
    result_frame = frame.copy()
    
    # Convert relative X to pixel coordinate (Y fixed at center)
    h, w = frame.shape[:2]
    x_pixel = int((relative_x + 1.0) * 0.5 * w)
    y_pixel = h // 2  # Fixed at vertical center since no Y prediction
    x_pixel = np.clip(x_pixel, 0, w - 1)
    
    # Determine action label and color
    if action_value > 0.5:
        action_label = "FORWARD"
        color = (0, 255, 0)
    elif action_value < -0.5:
        action_label = "BACKWARD"
        color = (0, 0, 255)
    else:
        action_label = "STOP"
        color = (255, 165, 0)
    
    # Draw target point
    cv2.circle(result_frame, (x_pixel, y_pixel), 12, color, -1)
    cv2.circle(result_frame, (x_pixel, y_pixel), 18, color, 3)
    
    # Draw crosshair
    cv2.line(result_frame, (x_pixel-25, y_pixel), (x_pixel+25, y_pixel), color, 3)
    cv2.line(result_frame, (x_pixel, y_pixel-25), (x_pixel, y_pixel+25), color, 3)
    
    # Info text
    info_text = [
        f"Action: {action_label} ({action_value:.3f})",
        f"Steering X: {relative_x:.3f} (left=-1, right=+1)",
        f"Pixel X: {x_pixel}",
        f"CNN: {inference_time:.1f}ms ({fps:.1f}FPS)"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(result_frame, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_frame


def create_combined_visualization(frame: np.ndarray, mask: np.ndarray, 
                                relative_x: float) -> np.ndarray:
    """Create visualization with mask overlay and prediction (X only)"""
    if mask.shape[:2] != frame.shape[:2]:
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    else:
        mask_resized = mask
    
    # Create colored mask overlay
    colored_mask = np.zeros_like(frame)
    if len(mask_resized.shape) == 2:
        colored_mask[:, :, 1] = mask_resized  # Green channel
    else:
        colored_mask[:, :, 1] = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
    
    # Blend with original frame
    alpha = 0.6
    blended = cv2.addWeighted(frame, alpha, colored_mask, 1-alpha, 0)
    
    # Convert relative X to pixel coordinates (Y fixed at center)
    h, w = frame.shape[:2]
    x_pixel = int((relative_x + 1.0) * 0.5 * w)
    y_pixel = h // 2
    x_pixel = np.clip(x_pixel, 0, w - 1)
    
    # Draw prediction
    cv2.circle(blended, (x_pixel, y_pixel), 12, (255, 255, 0), -1)
    cv2.circle(blended, (x_pixel, y_pixel), 18, (255, 255, 0), 3)
    
    cv2.putText(blended, f"Target X: {relative_x:.2f}", 
               (x_pixel + 25, y_pixel - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return blended


