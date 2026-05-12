"""
MobileNetV2 + LSTM Parking Model to TensorRT Conversion
Converts the temporal sequence model for deployment on Jetson

Usage:
    python convert_lstm_to_trt.py --model best_lstm_steering_park.pth
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
import argparse


# ============= Model Definition (must match training) =============
class MobileNetLSTMSteering(nn.Module):
    """
    CNN (MobileNetV2) + LSTM for temporal steering prediction
    """
    def __init__(self, sequence_length=5, hidden_size=128, num_lstm_layers=2, 
                 freeze_backbone=False, pretrained=False):
        super(MobileNetLSTMSteering, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # CNN backbone (MobileNetV2)
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM to process temporal sequence
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )
        
        # Prediction heads
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.x_predictor = nn.Linear(64, 1)
        self.action_predictor = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, 3, 224, 224)
        Returns:
            x_pred: (batch, 1) - steering
            action_pred: (batch, 1) - action
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features for each frame
        x = x.view(batch_size * seq_len, c, h, w)
        
        # CNN feature extraction
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Prediction heads
        shared = self.shared_fc(last_output)
        
        x_pred = torch.tanh(self.x_predictor(shared))
        action_pred = torch.tanh(self.action_predictor(shared))
        
        return x_pred, action_pred


# ============= Step 1: Export PyTorch to ONNX =============
def export_pytorch_to_onnx(pytorch_model_path, onnx_path, sequence_length=5, device='cuda'):
    """
    Export PyTorch LSTM model to ONNX format
    """
    print("="*60)
    print("STEP 1: Exporting MobileNetV2+LSTM to ONNX")
    print("="*60)
    
    # Load PyTorch model
    model = MobileNetLSTMSteering(
        sequence_length=sequence_length,
        hidden_size=128,
        num_lstm_layers=2,
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded PyTorch model from: {pytorch_model_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Sequence length: {sequence_length}")
    
    # Create dummy input: (batch=1, seq_len=5, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, sequence_length, 3, 224, 224).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test PyTorch model
    with torch.no_grad():
        x, action = model(dummy_input)
        print(f"\nPyTorch test output:")
        print(f"  X: {x.cpu().numpy()}")
        print(f"  Action: {action.cpu().numpy()}")
    
    # Export to ONNX with FIXED batch size (no dynamic axes)
    print("\nExporting to ONNX with fixed batch size...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # Higher opset for LSTM support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['x_output', 'action_output']
        # NO dynamic_axes - fixed batch size of 1
    )
    
    print(f"\n✓ ONNX model exported to: {onnx_path}")
    print(f"  Note: Batch size is FIXED at 1")
    print(f"  Note: Sequence length is FIXED at {sequence_length}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")
    
    return True


# ============= Step 2: Convert ONNX to TensorRT =============
def build_tensorrt_engine(onnx_path, engine_path, fp16_mode=True, workspace_size=4):
    """
    Convert ONNX model to TensorRT engine
    """
    print("\n" + "="*60)
    print("STEP 2: Converting ONNX to TensorRT")
    print("="*60)
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
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
    
    # Set memory pool limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    print(f"Workspace size: {workspace_size} GB")
    
    # Enable FP16 mode (recommended for Jetson)
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    else:
        print("! FP16 mode not available or disabled")
    
    # Build serialized network
    print("\nBuilding TensorRT engine... (may take 5-10 minutes for LSTM)")
    print("This is slower than CNN-only due to LSTM layers...")
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
class TensorRTLSTMInference:
    """
    TensorRT inference wrapper for LSTM model
    """
    def __init__(self, engine_path, sequence_length=5):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.sequence_length = sequence_length
        
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
        print(f"  Input shape: (1, {sequence_length}, 3, 224, 224)")
        print(f"  Inputs: {len(self.inputs)}")
        print(f"  Outputs: {len(self.outputs)}")
    
    def infer(self, input_sequence):
        """
        Run inference on a sequence of frames
        
        Args:
            input_sequence: numpy array of shape (1, seq_len, 3, 224, 224)
        
        Returns:
            x: steering prediction
            action: action prediction
        """
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_sequence.ravel())
        
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
        
        # Return outputs
        x_output = float(self.outputs[0]['host'][0])
        action_output = float(self.outputs[1]['host'][0])
        
        return x_output, action_output


def test_tensorrt_engine(engine_path, sequence_length=5, num_iterations=100):
    """
    Test TensorRT engine performance
    """
    print("\n" + "="*60)
    print("STEP 3: Testing TensorRT Engine")
    print("="*60)
    
    # Load engine
    trt_model = TensorRTLSTMInference(engine_path, sequence_length)
    
    # Create random input sequence
    input_sequence = np.random.randn(1, sequence_length, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        trt_model.infer(input_sequence)
    
    # Benchmark
    print(f"Running {num_iterations} iterations for benchmarking...")
    times = []
    
    for i in range(num_iterations):
        start = time.time()
        x, action = trt_model.infer(input_sequence)
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
    print(f"Throughput: {fps:.1f} sequences/sec")
    print(f"Note: Each sequence = {sequence_length} frames")
    print("="*60)
    
    # Performance expectations
    if fps > 30:
        print("✅ EXCELLENT - Real-time capable for driving!")
    elif fps > 15:
        print("✅ GOOD - Usable for parking maneuvers")
    elif fps > 10:
        print("⚠️  ACCEPTABLE - May be slightly slow")
    else:
        print("⚠️  SLOW - Consider optimization")


# ============= Main Conversion Pipeline =============
def convert_lstm_to_tensorrt(
    pytorch_model_path='best_lstm_steering_park.pth',
    onnx_path='lstm_steering_park.onnx',
    engine_path='lstm_steering_park_fp16.trt',
    sequence_length=5,
    fp16_mode=True,
    test_performance=True
):
    """
    Complete conversion pipeline for LSTM parking model
    """
    print("\n" + "="*60)
    print("MobileNetV2+LSTM → ONNX → TensorRT Conversion")
    print("="*60)
    print(f"PyTorch model: {pytorch_model_path}")
    print(f"ONNX output: {onnx_path}")
    print(f"TensorRT output: {engine_path}")
    print(f"Sequence length: {sequence_length}")
    print(f"FP16 mode: {fp16_mode}")
    print("="*60 + "\n")
    
    # Step 1: PyTorch to ONNX
    if not export_pytorch_to_onnx(pytorch_model_path, onnx_path, sequence_length):
        print("ERROR: Failed to export to ONNX")
        return False
    
    # Step 2: ONNX to TensorRT
    if not build_tensorrt_engine(onnx_path, engine_path, fp16_mode):
        print("ERROR: Failed to build TensorRT engine")
        return False
    
    # Step 3: Test performance
    if test_performance:
        test_tensorrt_engine(engine_path, sequence_length)
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"TensorRT engine ready: {engine_path}")
    print(f"\nModel details:")
    print(f"  - Input: (1, {sequence_length}, 3, 224, 224) - sequence of {sequence_length} RGB images")
    print(f"  - Output 1: X coordinate (steering) in [-1, 1]")
    print(f"  - Output 2: Action in [-1, 1]")
    print("\nIMPORTANT:")
    print(f"  - You MUST provide exactly {sequence_length} consecutive frames")
    print(f"  - Maintain a rolling buffer of last {sequence_length} frames")
    print(f"  - Feed the sequence to get steering prediction")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LSTM Parking Model to TensorRT')
    parser.add_argument('--model', type=str, default='best_lstm_steering_park.pth',
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--onnx', type=str, default='lstm_steering_park.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--engine', type=str, default='lstm_steering_park_fp16.trt',
                       help='Output TensorRT engine path')
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='Sequence length (must match training)')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Enable FP16 mode')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip performance testing')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Starting LSTM Model TensorRT Conversion")
    print("="*60)
    
    # Run conversion
    success = convert_lstm_to_tensorrt(
        pytorch_model_path=args.model,
        onnx_path=args.onnx,
        engine_path=args.engine,
        sequence_length=args.sequence_length,
        fp16_mode=args.fp16,
        test_performance=not args.no_test
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ Ready for deployment!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"1. Transfer {args.engine} to your Jetson")
        print(f"2. Create inference wrapper that maintains {args.sequence_length}-frame buffer")
        print(f"3. Test on robot with real camera feed")
        print(f"4. Deploy in your parking controller")
    else:
        print("\n❌ Conversion failed. Check errors above.")

#python3 convert_lstm_to_trt.py --model ../models/pytorch_models/best_lstm_steering_park_short_3seq.pth --onnx ../models/tensorrt_lstm_short/best_lstm_steering_park_short_3seq.onnx --engine ../models/tensorrt_lstm_short/best_lstm_steering_park_short_3seq.trt --sequence-length 3 --fp16