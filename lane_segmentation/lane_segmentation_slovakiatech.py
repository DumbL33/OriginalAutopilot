import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path

class DDRNetTensorRT:
    """
    Standalone DDRNet TensorRT inference class for lane segmentation
    
    Usage:
        lane_detector = DDRNetTensorRT("path/to/model.trt", confidence_threshold=0.6)
        mask = lane_detector.predict(frame)
        fps = lane_detector.get_fps()
        lane_detector.cleanup()
    """
    
    def __init__(self, engine_path, confidence_threshold=0.5):
        """
        Initialize DDRNet TensorRT inference engine
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            confidence_threshold: Threshold for lane detection (0.0 - 1.0)
        """
        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        
        # TensorRT components
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
        # Memory buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Model info
        self.input_height = None
        self.input_width = None
        
        # Performance tracking
        self.inference_times = []
        
        # Initialize everything
        self._load_engine()
        self._setup_bindings()
        self._allocate_memory()
        self._warmup()
        
        print(f"DDRNet TensorRT ready - Input: {self.input_width}x{self.input_height}")
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
    
    def _setup_bindings(self):
        """Setup input/output bindings and detect shapes"""
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            binding_shape = self.engine.get_binding_shape(i)
            
            if is_input:
                # Handle dynamic batch size by setting it to 1
                if binding_shape[0] == -1:
                    fixed_shape = (1,) + binding_shape[1:]
                    self.context.set_binding_shape(i, fixed_shape)
                    binding_shape = fixed_shape
                
                self.input_height = binding_shape[2]
                self.input_width = binding_shape[3]
                print(f"Input binding: {binding_name}, shape: {binding_shape}")
            else:
                # For output, get shape after input is set
                output_shape = self.context.get_binding_shape(i)
                print(f"Output binding: {binding_name}, shape: {output_shape}")
    
    def _allocate_memory(self):
        """Allocate GPU memory for inference"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_shape = self.context.get_binding_shape(i)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            try:
                # Allocate host and device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(i):
                    self.inputs.append({
                        'host': host_mem,
                        'device': device_mem,
                        'shape': binding_shape,
                        'size': size
                    })
                else:
                    self.outputs.append({
                        'host': host_mem,
                        'device': device_mem,
                        'shape': binding_shape,
                        'size': size
                    })
            except cuda.MemoryError as e:
                mb_required = size * 4 / (1024 * 1024)
                raise RuntimeError(f"Failed to allocate {mb_required:.1f} MB GPU memory for binding {i}: {e}")
    
    def _warmup(self):
        """Warm up the model with dummy data"""
        dummy_frame = np.random.randint(0, 255, (self.input_height, self.input_width, 3), dtype=np.uint8)
        
        # Run a few warmup inferences
        for _ in range(3):
            _ = self.predict(dummy_frame)
        
        if self.inference_times:
            avg_time = np.mean(self.inference_times[-3:])
            print(f"Warmup complete - Average inference time: {avg_time:.2f}ms")
    
    def preprocess(self, frame):
        """
        Preprocess frame for DDRNet inference
        
        Args:
            frame: Input frame (BGR format, any size)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and apply ImageNet normalization
        normalized = rgb_frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert HWC to BCHW format
        tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # CHW -> BCHW
        
        return tensor.astype(np.float32)
    
    def postprocess(self, output, target_size):
        """
        Convert model output to binary lane mask
        
        Args:
            output: Raw model output
            target_size: (height, width) for output mask
            
        Returns:
            Binary mask (0=background, 255=lanes)
        """
        # Remove batch dimension if present
        if output.ndim == 4:
            output = output[0]  # (1, C, H, W) -> (C, H, W)
        
        # Apply softmax to get probabilities
        exp_output = np.exp(output - np.max(output, axis=0, keepdims=True))
        probabilities = exp_output / np.sum(exp_output, axis=0, keepdims=True)
        
        # Get lane probability (class 1)
        lane_prob = probabilities[1]
        
        # Apply threshold to create binary mask
        binary_mask = (lane_prob > self.confidence_threshold).astype(np.uint8) * 255
        
        # Resize to target size if needed
        if target_size != (self.input_height, self.input_width):
            binary_mask = cv2.resize(binary_mask, (target_size[1], target_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
        
        return binary_mask
    
    def predict(self, frame):
        """
        Main inference function
        
        Args:
            frame: Input frame (BGR format, any size)
            
        Returns:
            Binary lane mask (same size as input frame)
        """
        start_time = time.time()
        original_size = frame.shape[:2]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Get output
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # Reshape output
        output_shape = self.outputs[0]['shape']
        output = self.outputs[0]['host'].reshape(output_shape)
        
        # Postprocess
        mask = self.postprocess(output, target_size=(224, 224))
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 50:  # Keep last 50 times
            self.inference_times.pop(0)
        
        return mask
    
    def segment_lanes(self, frame):
        """Alias for predict() to match your existing code"""
        return self.predict(frame)
    
    def get_fps(self):
        """Get current FPS based on recent inference times"""
        if not self.inference_times:
            return 0.0
        avg_time_ms = np.mean(self.inference_times)
        return 1000.0 / avg_time_ms
    
    def get_inference_time(self):
        """Get latest inference time in milliseconds"""
        return self.inference_times[-1] if self.inference_times else 0.0
    
    def get_stats(self):
        """Get detailed performance statistics"""
        if not self.inference_times:
            return {"fps": 0.0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        
        times = np.array(self.inference_times)
        return {
            "fps": 1000.0 / np.mean(times),
            "avg_ms": np.mean(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "latest_ms": times[-1]
        }
    
    def set_confidence_threshold(self, threshold):
        """Update confidence threshold for lane detection"""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
    
    def create_overlay(self, frame, mask, color=(0, 255, 0), alpha=0.5):
        """
        Create overlay visualization
        
        Args:
            frame: Original frame
            mask: Binary mask from predict()
            color: Color for lanes (B, G, R)
            alpha: Transparency (0.0 - 1.0)
            
        Returns:
            Frame with lane overlay
        """
        overlay = frame.copy()
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color
        
        return cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    def cleanup(self):
        """Clean up GPU memory"""
        try:
            for inp in self.inputs:
                if 'device' in inp and inp['device']:
                    inp['device'].free()
            for out in self.outputs:
                if 'device' in out and out['device']:
                    out['device'].free()
            
            if hasattr(self, 'stream') and self.stream:
                del self.stream
                
            print("DDRNet TensorRT cleanup completed")
        except Exception as e:
            print(f"Warning: Cleanup error - {e}")
