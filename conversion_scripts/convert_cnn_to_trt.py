#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tensorrt as trt
import numpy as np
import cv2
import os
from pathlib import Path
import argparse
import gc

class SingleCommandCNN(nn.Module):
    """Custom CNN - X and Action only"""
    def __init__(self):
        super(SingleCommandCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.shared_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Only X prediction (not XY)
        self.x_predictor = nn.Linear(128, 1)
        self.action_predictor = nn.Linear(128, 1)
    
    def forward(self, mask):
        features = self.backbone(mask)
        features = features.view(features.size(0), -1)
        
        shared_features = self.shared_layers(features)
        
        x_pred = self.x_predictor(shared_features)
        x_pred = torch.tanh(x_pred)
        
        action_pred = self.action_predictor(shared_features)
        action_pred = torch.tanh(action_pred)
        
        return x_pred, action_pred


class MiniSingleCommandCNN(nn.Module):
    """Lightweight CNN - X and Action only"""
    def __init__(self):
        super(MiniSingleCommandCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.shared_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.x_predictor = nn.Linear(64, 1)
        self.action_predictor = nn.Linear(64, 1)
    
    def forward(self, mask):
        features = self.backbone(mask)
        features = features.view(features.size(0), -1)
        
        shared_features = self.shared_layers(features)
        
        x_pred = self.x_predictor(shared_features)
        x_pred = torch.tanh(x_pred)
        
        action_pred = self.action_predictor(shared_features)
        action_pred = torch.tanh(action_pred)
        
        return x_pred, action_pred


class JetsonTensorRTConverter:
    def __init__(self, pytorch_model_path, model_type='standard', precision='fp16', fixed_batch_size=1):
        """
        Convert PyTorch model to TensorRT optimized for Jetson devices
        
        Args:
            pytorch_model_path: Path to the saved PyTorch model (.pth file)
            model_type: 'standard' or 'mini'
            precision: 'fp32', 'fp16', or 'int8'
            fixed_batch_size: Use fixed batch size to avoid dynamic shape issues
        """
        self.pytorch_model_path = pytorch_model_path
        self.model_type = model_type
        self.precision = precision
        self.fixed_batch_size = fixed_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = trt.Logger(trt.Logger.INFO)
        
    def clear_memory(self):
        """Clear GPU memory - important for Jetson devices"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_pytorch_model(self):
        """Load the PyTorch model"""
        print(f"Loading PyTorch model from {self.pytorch_model_path}")
        
        if self.model_type == 'mini':
            model = MiniSingleCommandCNN()
        else:
            model = SingleCommandCNN()
        
        checkpoint = torch.load(self.pytorch_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Ensure dropout is in eval mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        
        return model
    
    def export_to_onnx(self, model, onnx_path):
        """Export PyTorch model to ONNX format - mask input only"""
        print(f"Exporting to ONNX: {onnx_path}")
        print(f"Using fixed batch size: {self.fixed_batch_size}")
        
        # Dummy input: only mask
        dummy_mask = torch.randn(self.fixed_batch_size, 3, 224, 224, device=self.device)
        
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_mask,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mask'],
            output_names=['x_pred', 'action_pred'],  # Changed from 'xy_pred'
            verbose=False
        )
        print("✅ ONNX export completed!")
        
        self.clear_memory()
    
    def build_tensorrt_engine(self, onnx_path, engine_path):
        """Build TensorRT engine from ONNX model"""
        print(f"Building TensorRT engine: {engine_path}")
        
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        
        if self.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif self.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
        else:
            print("Using FP32 precision")
        
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        print("Parsing ONNX model...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("❌ ERROR: Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                return False
        
        print("Building TensorRT engine... (~5-10 minutes on Jetson)")
        
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                print("❌ ERROR: Failed to build TensorRT engine")
                return False
            
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            print(f"✅ TensorRT engine saved to {engine_path}")
            self.clear_memory()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR during engine building: {e}")
            return False
    
    def convert(self, output_dir="./models/tensorrt_cnn_models"):
        """Complete conversion pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"steering_{self.model_type}_{self.precision}_batch{self.fixed_batch_size}"
        onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
        engine_path = os.path.join(output_dir, f"{base_name}.trt")
        
        try:
            print("=" * 60)
            print("TensorRT Conversion - Single Command Model")
            print(f"Model type: {self.model_type}")
            print(f"Precision: {self.precision}")
            print(f"Fixed batch size: {self.fixed_batch_size}")
            print("Input: Road mask only")
            print("Output: X coordinate + action (NO Y)")
            print("=" * 60)
            
            model = self.load_pytorch_model()
            print("✅ PyTorch model loaded")
            
            self.export_to_onnx(model, onnx_path)
            print("✅ ONNX export completed")
            
            del model
            self.clear_memory()
            
            success = self.build_tensorrt_engine(onnx_path, engine_path)
            
            if success:
                print(f"\n{'='*60}")
                print("✅ Conversion completed!")
                print(f"ONNX: {onnx_path}")
                print(f"TensorRT: {engine_path}")
                
                onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
                engine_size = os.path.getsize(engine_path) / (1024 * 1024)
                print(f"ONNX size: {onnx_size:.1f} MB")
                print(f"Engine size: {engine_size:.1f} MB")
                print("="*60)
                
                return engine_path
            else:
                print("❌ Conversion failed")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


class JetsonTensorRTPredictor:
    def __init__(self, engine_path):
        """TensorRT inference predictor"""
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"Loading TensorRT engine: {engine_path}")
        
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        self.get_engine_info()
        self.allocate_buffers()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ TensorRT predictor ready!")
    
    def load_engine(self):
        """Load TensorRT engine from file"""
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_engine_info(self):
        """Get engine information"""
        print("\nEngine bindings:")
        for i in range(self.engine.num_bindings):
            binding = self.engine[i]
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            is_input = self.engine.binding_is_input(binding)
            print(f"  {binding}: {shape} {dtype} {'(input)' if is_input else '(output)'}")
    
    def allocate_buffers(self):
        """Allocate GPU memory"""
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def preprocess_mask(self, mask):
        """Preprocess road mask"""
        if isinstance(mask, str):
            mask = cv2.imread(mask)
            if mask is None:
                raise ValueError(f"Could not load image: {mask}")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=2)
        
        mask_tensor = self.transform(mask)
        return mask_tensor.numpy()
    
    def predict(self, mask):
        """
        Run inference
        
        Args:
            mask: Road mask image (numpy array or path)
            
        Returns:
            tuple: (relative_x, action_value, action_label)
                - relative_x: X coordinate in [-1, 1] (left=-1, right=+1)
                - action_value: Action in [-1, 1] (backward=-1, forward=+1)
                - action_label: "FORWARD", "STOP", or "BACKWARD"
        """
        import pycuda.driver as cuda
        
        # Preprocess
        mask_data = self.preprocess_mask(mask)
        
        # Copy to GPU
        np.copyto(self.inputs[0]['host'], mask_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Get results
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Parse outputs - ONLY X (not XY)
        x_pred = self.outputs[0]['host'][0]  # Single value
        action_pred = self.outputs[1]['host'][0]
        
        # Determine action label
        if action_pred > 0.5:
            action_label = "FORWARD"
        elif action_pred < -0.5:
            action_label = "BACKWARD"
        else:
            action_label = "STOP"
        
        return float(x_pred), float(action_pred), action_label


def main():
    parser = argparse.ArgumentParser(description="Convert single-command steering model to TensorRT (X + Action only)")
    parser.add_argument("--model_path", required=True, help="Path to .pth file")
    parser.add_argument("--model_type", default="standard", choices=["standard", "mini"])
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", default="./tensorrt_models_go_straight")
    parser.add_argument("--test", action="store_true", help="Test the converted model")
    
    args = parser.parse_args()
    
    # Convert
    converter = JetsonTensorRTConverter(
        args.model_path,
        args.model_type,
        args.precision,
        fixed_batch_size=args.batch_size
    )
    engine_path = converter.convert(args.output_dir)
    
    # Test if requested
    if engine_path and args.test:
        print("\n" + "="*60)
        print("Testing converted model...")
        print("="*60)
        
        try:
            predictor = JetsonTensorRTPredictor(engine_path)
            
            # Create test mask
            dummy_mask = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_mask[100:120, 50:200] = 255  # Horizontal line
            dummy_mask[80:140, 100:110] = 255  # Vertical line
            
            x, action, label = predictor.predict(dummy_mask)
            
            print(f"\n✅ Test Result:")
            print(f"  X: {x:.4f} (left=-1, center=0, right=+1)")
            print(f"  Action: {label} ({action:.4f})")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()


"""
Usage Examples:

# Convert standard model with FP16 precision
python3 convert_cnn_to_trt.py --model_path ../models/pytorch_models/best_model_park.pth --model_type standard --precision fp16 --test --output_dir ../models/tensorrt_models_park

# Convert mini model with FP32 precision
python3 convert_cnn_to_trt.py --model_path best_model_go_straight.pth --model_type mini --precision fp32 --output_dir ./my_models --test

# For multiple commands, convert each separately:
python3 convert_cnn_to_trt.py --model_path best_model_go_straight.pth --output_dir ./models/straight --test
python3 convert_cnn_to_trt.py --model_path best_model_turn_left.pth --output_dir ./models/left --test
python3 convert_cnn_to_trt.py --model_path best_model_turn_right.pth --output_dir ./models/right --test
"""