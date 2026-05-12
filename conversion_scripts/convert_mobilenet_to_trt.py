"""
MobileNetV3 X-Only to TensorRT Conversion Script
Run this on your Jetson Orin NX device

Converts the X-only steering model (no Y coordinate)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
from pathlib import Path
import time


# ============= Model Definition (X-only, same as training) =============
class MobileNetSteeringXOnly(nn.Module):
    """
    MobileNetV3-Small steering model - only predicts X coordinate
    """
    def __init__(self, pretrained=True):
        super(MobileNetSteeringXOnly, self).__init__()
        
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        mobilenet = mobilenet_v3_small(weights=None)
        
        self.backbone = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Hardswish()
        )
        
        self.x_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),  # Single output: X
            nn.Tanh()
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        shared = self.shared_fc(x)
        x_pred = self.x_head(shared)
        action_pred = self.action_head(shared)
        return x_pred, action_pred


# ============= Step 1: Export PyTorch to ONNX =============
def export_pytorch_to_onnx(pytorch_model_path, onnx_path, device='cuda'):
    """
    Export PyTorch model to ONNX format with FIXED batch size
    """
    print("="*60)
    print("STEP 1: Exporting MobileNetV3 X-Only to ONNX")
    print("="*60)
    
    # Load PyTorch model
    model = MobileNetSteeringXOnly(pretrained=False).to(device)
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded PyTorch model from: {pytorch_model_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"X loss: {checkpoint.get('val_x_loss', 'N/A')}")
    print(f"Action loss: {checkpoint.get('val_action_loss', 'N/A')}")
    
    # Create dummy input with FIXED batch size = 1
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test PyTorch model
    with torch.no_grad():
        x, action = model(dummy_input)
        print(f"\nPyTorch test output:")
        print(f"  X: {x.cpu().numpy()}")
        print(f"  Action: {action.cpu().numpy()}")
    
    # Export to ONNX with FIXED batch size
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['x_output', 'action_output']  # Changed from xy_output
    )
    
    print(f"\n✓ ONNX model exported to: {onnx_path}")
    print("  Note: Batch size is fixed at 1")
    print("  Outputs: X (1 value) and Action (1 value)")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")
    
    return True


# ============= Step 2: Convert ONNX to TensorRT =============
def build_tensorrt_engine(onnx_path, engine_path, fp16_mode=True, int8_mode=False, workspace_size=2):
    """
    Convert ONNX model to TensorRT engine
    """
    print("\n" + "="*60)
    print("STEP 2: Converting ONNX to TensorRT")
    print("="*60)
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network with explicit batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("✓ ONNX file parsed successfully")
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set memory pool limit (updated API)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    
    print(f"Workspace size: {workspace_size} GB")
    
    # Enable FP16 mode (recommended for Jetson)
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    else:
        print("! FP16 mode not available or disabled")
    
    # Enable INT8 mode (optional)
    if int8_mode and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("✓ INT8 mode enabled")
    
    # Build serialized network (updated API)
    print("\nBuilding TensorRT engine... (this may take 2-5 minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False
    
    print("✓ TensorRT engine built successfully")
    
    # Save engine
    print(f"Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved: {engine_path}")
    
    return True


# ============= Step 3: Test TensorRT Engine =============
class TensorRTInference:
    """
    TensorRT inference wrapper for X-only model
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"\nLoading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
        
        # Allocate buffers
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
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
        
        print("✓ TensorRT engine loaded and buffers allocated")
        print(f"  Inputs: {len(self.inputs)}")
        print(f"  Outputs: {len(self.outputs)}")
    
    def infer(self, input_data):
        """
        Run inference
        input_data: numpy array of shape (1, 3, 224, 224)
        Returns: x (float), action (float)
        """
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Set input tensor
        self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device']))
        
        # Set output tensors
        for output in self.outputs:
            self.context.set_tensor_address(output['name'], int(output['device']))
        
        # Copy input to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs from device
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        
        # Return outputs (X is single value, action is single value)
        x_output = float(self.outputs[0]['host'][0])  # Single X value
        action_output = float(self.outputs[1]['host'][0])  # Single action value
        
        return x_output, action_output


def test_tensorrt_engine(engine_path, num_iterations=100):
    """
    Test TensorRT engine performance
    """
    print("\n" + "="*60)
    print("STEP 3: Testing TensorRT Engine")
    print("="*60)
    
    # Load engine
    trt_model = TensorRTInference(engine_path)
    
    # Create random input
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        trt_model.infer(input_data)
    
    # Benchmark
    print(f"Running {num_iterations} iterations for benchmarking...")
    times = []
    
    for i in range(num_iterations):
        start = time.time()
        x, action = trt_model.infer(input_data)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        
        if i == 0:
            print(f"\nSample output:")
            print(f"  X: {x:.4f} (steering)")
            print(f"  Action: {action:.4f}")
    
    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000.0 / avg_time
    
    print("\n" + "="*60)
    print("Performance Results")
    print("="*60)
    print(f"Average latency: {avg_time:.2f} ms (± {std_time:.2f} ms)")
    print(f"Min latency: {min_time:.2f} ms")
    print(f"Max latency: {max_time:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    print("="*60)
    
    # Performance expectations
    if fps > 100:
        print("✅ EXCELLENT - Ready for high-speed driving!")
    elif fps > 60:
        print("✅ VERY GOOD - Perfect for normal driving")
    elif fps > 30:
        print("✅ GOOD - Adequate for driving")
    else:
        print("⚠️  SLOW - May need optimization")


# ============= Main Conversion Pipeline =============
def convert_pytorch_to_tensorrt(
    pytorch_model_path='best_mobilenet_steering_x_only.pth',
    onnx_path='steering_mobilenet_x_only.onnx',
    engine_path='steering_mobilenet_fp16.trt',
    fp16_mode=True,
    int8_mode=False,
    test_performance=True
):
    """
    Complete conversion pipeline for X-only model
    """
    print("\n" + "="*60)
    print("MobileNetV3 X-Only → ONNX → TensorRT")
    print("="*60)
    print(f"PyTorch model: {pytorch_model_path}")
    print(f"ONNX output: {onnx_path}")
    print(f"TensorRT output: {engine_path}")
    print(f"FP16 mode: {fp16_mode}")
    print(f"INT8 mode: {int8_mode}")
    print("="*60 + "\n")
    
    # Step 1: PyTorch to ONNX
    if not export_pytorch_to_onnx(pytorch_model_path, onnx_path):
        print("ERROR: Failed to export to ONNX")
        return False
    
    # Step 2: ONNX to TensorRT
    if not build_tensorrt_engine(onnx_path, engine_path, fp16_mode, int8_mode):
        print("ERROR: Failed to build TensorRT engine")
        return False
    
    # Step 3: Test performance
    if test_performance:
        test_tensorrt_engine(engine_path)
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"TensorRT engine ready: {engine_path}")
    print(f"\nModel details:")
    print(f"  - Input: (1, 3, 224, 224) mask")
    print(f"  - Output 1: X coordinate (steering) in [-1, 1]")
    print(f"  - Output 2: Action in [-1, 1]")
    print("\nYou can now use this engine for real-time inference!")
    
    return True


if __name__ == "__main__":
    # Configuration
    PYTORCH_MODEL = "../models/pytorch_models/best_mobilenet_steering_straight.pth"
    ONNX_MODEL = "../models/tensorrt_mobilenet_models/best_mobilenet_steering_straight.onnx"
    TENSORRT_ENGINE = "../models/tensorrt_mobilenet_models/best_mobilenet_steering_straight_fp16.trt"

    print("\n" + "="*60)
    print("Starting MobileNetV3 X-Only TensorRT Conversion")
    print("="*60)
    
    # Run conversion
    success = convert_pytorch_to_tensorrt(
        pytorch_model_path=PYTORCH_MODEL,
        onnx_path=ONNX_MODEL,
        engine_path=TENSORRT_ENGINE,
        fp16_mode=True,      
        int8_mode=False,     
        test_performance=True
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ Ready for deployment on your car!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"1. Update your autopilot to use the new model")
        print(f"2. Update SteeringController to use X-only (no Y)")
        print(f"3. Test on straight roads first")
        print(f"4. Then test on curves")