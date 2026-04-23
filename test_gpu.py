"""
Test GPU availability and functionality for FKG/FIS modules
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'Source')

try:
    import fisa_module as fs
    print("[OK] fisa_module imported successfully")
except ImportError as e:
    print("[ERROR] Failed to import fisa_module: {}".format(e))
    sys.exit(1)

# Check GPU status
print("\n" + "="*50)
print("GPU Detection Test")
print("="*50)

# Check if GPU is enabled in the build
gpu_enabled = getattr(fs, 'GPU_ENABLED', False)
print("GPU_ENABLED in build: {}".format(gpu_enabled))

# Check if GPU is available
try:
    gpu_available = fs.is_gpu_available()
    print("GPU Available: {}".format(gpu_available))
except Exception as e:
    print("Error checking GPU: {}".format(e))
    gpu_available = False

# Test with CUDA if available
if gpu_available:
    print("\n" + "="*50)
    print("Testing GPU-accelerated FKG")
    print("="*50)
    
    # Create test data
    import numpy as np
    
    # Simple test data
    test_data = np.array([
        [1, 2, 3, 4, 1],
        [1, 2, 3, 5, 1],
        [1, 2, 4, 4, 2],
        [2, 3, 4, 5, 2],
        [2, 3, 5, 6, 1],
    ])
    
    # Convert to list for C++ module
    base = test_data.tolist()
    
    # Test FKG with GPU
    try:
        fkg = fs.fkg.FKG()
        fkg.train(base)
        
        # Test prediction
        test_input = [1, 2, 3, 4]
        result = fkg.predict(test_input)
        print("Prediction for {}: class={}, confidence={:.4f}".format(test_input, result[0], result[1]))
        
        # Test batch prediction
        test_batch = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [1, 2, 4, 4]
        ]
        predictions = fkg.predictBatch(test_batch)
        print("Batch predictions: {}".format(predictions))
        
        print("[OK] GPU-accelerated FKG working!")
        
    except Exception as e:
        print("[ERROR] GPU FKG test failed: {}".format(e))
    
    # Test GPU benchmark
    print("\n" + "="*50)
    print("GPU Benchmark Test")
    print("="*50)
    
    try:
        # Create larger test data
        larger_data = np.random.randint(1, 10, size=(100, 6))
        larger_data[:, -1] = np.random.randint(1, 3, size=100)  # Labels
        larger_list = larger_data.tolist()
        
        fkg_bench = fs.fkg.FKG()
        result = fkg_bench.benchmark(larger_list)
        
        print("CPU Time: {:.2f} ms".format(result.cpuTimeMs))
        print("GPU Time: {:.2f} ms".format(result.gpuTimeMs))
        print("Speedup: {:.2f}x".format(result.speedup))
        print("Results Match: {}".format(result.resultsMatch))
        
    except Exception as e:
        print("Benchmark not available or failed: {}".format(e))
else:
    print("\n[WARNING] GPU not available. To enable GPU support:")
    print("  - For NVIDIA GPU: Compile with -DUSE_CUDA=ON")
    print("  - For Intel GPU: Compile with -DUSE_GPU=ON")
    print("  - Make sure CUDA Toolkit or Intel oneAPI is installed")

# Test FIS module
print("\n" + "="*50)
print("FIS Module Test")
print("="*50)

try:
    fis = fs.fis.FIS()
    print("[OK] FIS module loaded")
    
    # Check if GPU available for FIS
    if gpu_available:
        print("[OK] GPU available for FIS")
    else:
        print("[WARNING] GPU not available for FIS (will use CPU)")
        
except Exception as e:
    print("[ERROR] FIS module error: {}".format(e))

print("\n" + "="*50)
print("Test Complete")
print("="*50)