import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class CrossroadDetectorTRT:
    def __init__(self, engine_path, confidence_threshold=0.5, input_size=(224, 224)):
        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None
        
        self.d_input = None
        self.d_output = None
        self.h_input = None
        self.h_output = None
        self.stream = None
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self._load_engine()
        
        print(f"✓ Crossroad Detector initialized")
        print(f"  Engine: {engine_path}")
        print(f"  Input size: {input_size}")
        print(f"  Confidence threshold: {confidence_threshold}")
    
    def _load_engine(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        input_shape = (1, 3, self.input_size[1], self.input_size[0])
        input_size = trt.volume(input_shape)
        
        output_size = 1
        
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        self.stream = cuda.Stream()
        
        print(f"✓ Engine loaded successfully")
        print(f"  Input buffer size: {input_size} floats")
        print(f"  Output buffer size: {output_size} floats")
    
    def preprocess(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        img = resized.astype(np.float32) / 255.0
        
        img = (img - self.mean) / self.std
        
        img = img.transpose(2, 0, 1)
        
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def predict(self, frame):
        preprocessed = self.preprocess(frame)
        
        np.copyto(self.h_input, preprocessed.ravel())
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        self.stream.synchronize()
        confidence = float(self.h_output[0])
        is_crossroad = confidence > self.confidence_threshold
        return is_crossroad, confidence
    
    def predict_with_visualization(self, frame, draw_on_frame=True):
        is_crossroad, confidence = self.predict(frame)
        if draw_on_frame:
            annotated = frame.copy()
            
            if len(annotated.shape) == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
            
            color = (0, 255, 0) if is_crossroad else (128, 128, 128)
            
            label = f"CROSSROAD" if is_crossroad else "NO CROSSROAD"
            conf_text = f"Conf: {confidence:.3f}"
            
            cv2.rectangle(annotated, (10, 10), (300, 80), (0, 0, 0), -1)
            
            cv2.putText(annotated, label, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, conf_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return is_crossroad, confidence, annotated
        
        return is_crossroad, confidence, frame
    
    def set_threshold(self, threshold):
        
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"✓ Threshold updated to {threshold}")
        else:
            print(f"⚠️ Invalid threshold {threshold}, must be between 0.0 and 1.0")
    
    def __del__(self):
        if self.d_input:
            self.d_input.free()
        if self.d_output:
            self.d_output.free()
