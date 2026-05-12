"""
PyTorch to TensorRT Conversion Script
Run this on your Jetson Orin NX device

Steps:
1. Load PyTorch model
2. Export to ONNX (with FIXED batch size)
3. Convert ONNX to TensorRT
4. Test inference speed
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


# ============= Model Definition (same as training) =============
class ResNetSteering(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetSteering, self).__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.xy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        xy_pred = self.xy_head(shared)
        action_pred = self.action_head(shared)
        return xy_pred, action_pred


# ============= Step 1: Export PyTorch to ONNX =============
def export_pytorch_to_onnx(pytorch_model_path, onnx_path, device='cuda'):
    """
    Export PyTorch model to ONNX format with FIXED batch size
    """
    print("="*60)
    print("STEP 1: Exporting PyTorch model to ONNX")
    print("="*60)
    
    # Load PyTorch model
    model = ResNetSteering(pretrained=False).to(device)
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded PyTorch model from: {pytorch_model_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Create dummy input with FIXED batch size = 1
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test PyTorch model
    with torch.no_grad():
        xy, action = model(dummy_input)
        print(f"PyTorch test output - XY: {xy.cpu().numpy()}, Action: {action.cpu().numpy()}")
    
    # Export to ONNX with FIXED batch size (no dynamic axes)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['xy_output', 'action_output']
        # NO dynamic_axes! This fixes the TensorRT error
    )
    
    print(f"✓ ONNX model exported to: {onnx_path}")
    print("  Note: Batch size is fixed at 1 (optimal for real-time inference)")
    
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
    TensorRT inference wrapper
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
    
    def infer(self, input_data):
        """
        Run inference
        input_data: numpy array of shape (1, 3, 224, 224)
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
        
        # Return outputs (assuming first output is xy, second is action)
        xy_output = self.outputs[0]['host'][:2].copy()
        action_output = self.outputs[1]['host'][0].copy()
        
        return xy_output, action_output


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
        xy, action = trt_model.infer(input_data)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        
        if i == 0:
            print(f"\nSample output:")
            print(f"  XY: {xy}")
            print(f"  Action: {action}")
    
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


# ============= Main Conversion Pipeline =============
def convert_pytorch_to_tensorrt(
    pytorch_model_path='best_resnet_steering.pth',
    onnx_path='steering_model.onnx',
    engine_path='steering_model.trt',
    fp16_mode=True,
    int8_mode=False,
    test_performance=True
):
    """
    Complete conversion pipeline
    """
    print("\n" + "="*60)
    print("PyTorch → ONNX → TensorRT Conversion Pipeline")
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
    print("You can now use this engine for real-time inference on Jetson")
    
    return True


if __name__ == "__main__":
    # Configuration
    PYTORCH_MODEL = "../models/pytorch_models/best_resnet_steering_go_right.pth"
    ONNX_MODEL = "../models/tensorrt_resnet_models/steering_model_go_right.onnx"
    TENSORRT_ENGINE = "../models/tensorrt_resnet_models/steering_model_fp16_go_right.trt"
    
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
        print("\n✓ Ready for deployment!")
        print(f"\nUsage example:")
        print(f"  from convert_resnet_to_trt import TensorRTInference")
        print(f"  model = TensorRTInference('{TENSORRT_ENGINE}')")
        print(f"  xy, action = model.infer(input_image)")