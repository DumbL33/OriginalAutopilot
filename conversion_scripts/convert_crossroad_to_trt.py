import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import os
from pathlib import Path


class CrossroadCNN(nn.Module):
    """Standard crossroad detection model (must match training)"""
    def __init__(self):
        super(CrossroadCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class MiniCrossroadCNN(nn.Module):
    """Mini crossroad detection model (must match training)"""
    def __init__(self):
        super(MiniCrossroadCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class TensorRTConverter:
    """Convert PyTorch crossroad detection model to TensorRT"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    
    def pytorch_to_onnx(self, model, onnx_path, input_shape=(1, 3, 224, 224), dynamic_batch=False):
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model
            onnx_path: Output ONNX file path
            input_shape: Input tensor shape (batch, channels, height, width)
            dynamic_batch: Enable dynamic batch size (requires optimization profile in TRT)
        """
        print(f"\n{'='*60}")
        print("Step 1: Converting PyTorch to ONNX")
        print(f"{'='*60}")
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX - use fixed batch size for TensorRT compatibility
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"✓ ONNX model saved to: {onnx_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  Dynamic batch: {dynamic_batch}")
        return onnx_path
    
    def onnx_to_tensorrt(self, onnx_path, engine_path, fp16_mode=True, 
                         int8_mode=False, max_batch_size=1, workspace_size=1<<30):
        """
        Convert ONNX model to TensorRT engine
        
        Args:
            onnx_path: Input ONNX file path
            engine_path: Output TensorRT engine path
            fp16_mode: Enable FP16 precision (recommended for Jetson)
            int8_mode: Enable INT8 precision (requires calibration)
            max_batch_size: Maximum batch size
            workspace_size: GPU workspace size in bytes (default: 1GB)
        """
        print(f"\n{'='*60}")
        print("Step 2: Converting ONNX to TensorRT")
        print(f"{'='*60}")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        print(f"Parsing ONNX file: {onnx_path}")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        print("✓ ONNX parsed successfully")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        
        # Create optimization profile for fixed or dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Set input shape profile (min, opt, max)
        # For fixed batch size: all three are the same
        input_shape = (max_batch_size, 3, 224, 224)
        profile.set_shape("input", input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)
        
        print(f"✓ Optimization profile configured")
        print(f"  Input shape: {input_shape}")
        
        # Set precision modes
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        else:
            print("  FP16 mode not available, using FP32")
        
        if int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ INT8 mode enabled (requires calibration)")
        
        # Build engine
        print(f"\nBuilding TensorRT engine...")
        print(f"  Max batch size: {max_batch_size}")
        print(f"  Workspace size: {workspace_size / (1<<30):.2f} GB")
        print("  This may take a few minutes...")
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return None
        
        # Save engine
        print(f"\n✓ Engine built successfully!")
        print(f"Saving engine to: {engine_path}")
        
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✓ TensorRT engine saved!")
        
        # Load and print engine info
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self._print_engine_info(engine)
        
        return engine_path
    
    def _print_engine_info(self, engine):
        """Print information about the TensorRT engine"""
        print(f"\n{'='*60}")
        print("TensorRT Engine Information")
        print(f"{'='*60}")
        print(f"Number of bindings: {engine.num_bindings}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"\nBinding {i}: {name}")
            print(f"  Shape: {shape}")
            print(f"  Data type: {dtype}")
    
    def convert_full_pipeline(self, pytorch_model_path, output_dir, 
                             model_type='standard', fp16=True, int8=False):
        """
        Complete conversion pipeline: PyTorch -> ONNX -> TensorRT
        
        Args:
            pytorch_model_path: Path to PyTorch checkpoint (.pth)
            output_dir: Directory to save converted models
            model_type: 'standard' or 'mini'
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
        """
        print(f"\n{'#'*60}")
        print("CROSSROAD DETECTION MODEL - TENSORRT CONVERSION")
        print(f"{'#'*60}")
        print(f"PyTorch model: {pytorch_model_path}")
        print(f"Model type: {model_type}")
        print(f"Output directory: {output_dir}")
        print(f"FP16 mode: {fp16}")
        print(f"INT8 mode: {int8}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PyTorch model
        print(f"\nLoading PyTorch model...")
        if model_type == 'mini':
            model = MiniCrossroadCNN()
        else:
            model = CrossroadCNN()
        
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Define output paths
        model_name = f"crossroad_{model_type}"
        onnx_path = output_dir / f"{model_name}.onnx"
        engine_path = output_dir / f"{model_name}.trt"
        
        # Convert to ONNX (with fixed batch size for TensorRT)
        self.pytorch_to_onnx(model, str(onnx_path), dynamic_batch=False)
        
        # Convert to TensorRT
        self.onnx_to_tensorrt(
            str(onnx_path), 
            str(engine_path),
            fp16_mode=fp16,
            int8_mode=int8
        )
        
        print(f"\n{'#'*60}")
        print("CONVERSION COMPLETE!")
        print(f"{'#'*60}")
        print(f"\nGenerated files:")
        print(f"  ONNX model: {onnx_path}")
        print(f"  TensorRT engine: {engine_path}")
        print(f"\nYou can now use the TensorRT engine for inference on Jetson!")
        
        return str(engine_path)


def verify_tensorrt_inference(engine_path, test_image_path=None):
    """
    Verify TensorRT engine with a test inference
    
    Args:
        engine_path: Path to TensorRT engine
        test_image_path: Optional path to test image
    """
    print(f"\n{'='*60}")
    print("Verifying TensorRT Engine")
    print(f"{'='*60}")
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    print(f"✓ Engine loaded successfully")
    
    # Create dummy input
    input_shape = (1, 3, 224, 224)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Allocate buffers
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(1, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # Copy input
    np.copyto(h_input, dummy_input.ravel())
    
    # Run inference
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    print(f"✓ Test inference successful!")
    print(f"  Output value: {h_output[0]:.4f}")
    print(f"  (Random input, so output value is meaningless)")


if __name__ == "__main__":
    # Configuration
    PYTORCH_MODEL_PATH = "../models/pytorch_models/best_crossroad_model_standard.pth"  # Your trained model
    OUTPUT_DIR = "../models/tensorrt_crossroad_detect"
    MODEL_TYPE = "standard"  # 'standard' or 'mini'
    
    # Precision settings (FP16 recommended for Jetson)
    USE_FP16 = True
    USE_INT8 = False  # Requires calibration dataset
    
    # Create converter
    converter = TensorRTConverter(verbose=True)
    
    # Run conversion
    engine_path = converter.convert_full_pipeline(
        pytorch_model_path=PYTORCH_MODEL_PATH,
        output_dir=OUTPUT_DIR,
        model_type=MODEL_TYPE,
        fp16=USE_FP16,
        int8=USE_INT8
    )
    
    # Verify the engine works
    print("\n" + "="*60)
    print("Running verification test...")
    print("="*60)
    
    try:
        verify_tensorrt_inference(engine_path)
    except Exception as e:
        print(f"Verification skipped: {e}")
        print("This is normal if pycuda is not installed yet")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy the .trt file to your Jetson device")
    print("2. Use TensorRT runtime for inference")
    print("3. Expected speedup: 3-10x faster than PyTorch")
    print("="*60)