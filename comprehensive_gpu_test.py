#!/usr/bin/env python3
"""
Comprehensive GPU Detection and Testing Script
"""

import torch
import subprocess
import sys

def check_system_gpus():
    """Check all available GPUs in the system"""
    print("=" * 60)
    print("COMPREHENSIVE GPU DETECTION")
    print("=" * 60)
    
    # Method 1: NVIDIA-SMI
    print("\n1. NVIDIA-SMI Detection:")
    print("-" * 25)
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        gpu_lines = result.stdout.strip().split('\n')
        print(f"Found {len(gpu_lines)} GPU(s) via nvidia-smi:")
        for i, line in enumerate(gpu_lines):
            print(f"  {line}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ nvidia-smi failed: {e}")
    
    # Method 2: PyTorch CUDA Detection  
    print("\n2. PyTorch CUDA Detection:")
    print("-" * 26)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"PyTorch detected {device_count} CUDA device(s):")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            name = torch.cuda.get_device_name(i)
            memory = props.total_memory / (1024**3)
            compute_capability = f"{props.major}.{props.minor}"
            
            print(f"  GPU {i}: {name}")
            print(f"    Memory: {memory:.1f} GB")
            print(f"    Compute Capability: {compute_capability}")
            print(f"    Multiprocessors: {props.multi_processor_count}")
    else:
        print("❌ PyTorch CUDA not available")
    
    # Method 3: Direct CUDA Device Query
    print("\n3. Direct CUDA Device Properties:")
    print("-" * 32)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Detailed Properties:")
            props = torch.cuda.get_device_properties(i)
            
            print(f"  Name: {props.name}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")
            
            # Get available properties safely
            try:
                print(f"  Max threads per block: {props.max_threads_per_block}")
            except AttributeError:
                print(f"  Max threads per block: Not available")
            
            try:
                print(f"  Max shared memory per block: {props.max_shared_memory_per_block / 1024:.1f} KB")
            except AttributeError:
                print(f"  Max shared memory per block: Not available")
            
            # Test basic operations
            try:
                torch.cuda.set_device(i)
                x = torch.randn(100, 100, device=f'cuda:{i}')
                y = torch.randn(100, 100, device=f'cuda:{i}')
                z = torch.mm(x, y)
                print(f"  ✅ Basic operations: Working")
                del x, y, z
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ❌ Basic operations: Failed - {e}")

def simulate_multi_gpu_test():
    """Simulate multi-GPU testing even with single GPU"""
    print("\n" + "=" * 60)
    print("MULTI-GPU SIMULATION TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot simulate multi-GPU")
        return
    
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus >= 2:
        print("✅ Multiple GPUs detected - real multi-GPU testing possible")
        test_devices = [f'cuda:{i}' for i in range(available_gpus)]
    else:
        print("⚠️ Single GPU detected - simulating multi-GPU with memory isolation")
        # Simulate multiple GPUs by using the same GPU with different memory contexts
        test_devices = ['cuda:0', 'cuda:0']  # Same GPU, different contexts
    
    print(f"\nTesting devices: {test_devices}")
    
    # Test matrix operations on each "GPU"
    results = {}
    for i, device in enumerate(test_devices):
        print(f"\nTesting on simulated GPU {i} (device: {device}):")
        
        try:
            # Clear cache before each test
            torch.cuda.empty_cache()
            
            # Create test tensors
            import time
            start_time = time.time()
            
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            
            print(f"  Matrix multiplication: {elapsed:.3f}s")
            print(f"  Memory used: {memory_used:.3f} GB")
            
            results[f"gpu_{i}"] = {
                'time': elapsed,
                'memory': memory_used,
                'device': device
            }
            
            # Clean up
            del a, b, c
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    # Compare results
    if len(results) >= 2:
        print(f"\n📊 SIMULATED MULTI-GPU COMPARISON:")
        print("-" * 35)
        
        gpu_keys = list(results.keys())
        for i in range(len(gpu_keys)):
            for j in range(i+1, len(gpu_keys)):
                gpu1, gpu2 = gpu_keys[i], gpu_keys[j]
                time1, time2 = results[gpu1]['time'], results[gpu2]['time']
                
                if time2 > 0:
                    speedup = time1 / time2
                    print(f"{gpu1} vs {gpu2}: {speedup:.2f}x ({time1:.3f}s vs {time2:.3f}s)")

def create_virtual_multi_gpu_benchmark():
    """Create a benchmark that can test GPU under different loads"""
    print("\n" + "=" * 60)
    print("VIRTUAL MULTI-GPU LOAD TESTING")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Test different matrix sizes to simulate different GPU loads
    test_sizes = [500, 1000, 1500, 2000]
    device = 'cuda:0'
    
    print(f"Testing GPU 0 with different workloads:")
    
    results = {}
    for size in test_sizes:
        print(f"\nTesting {size}x{size} matrix operations:")
        
        try:
            torch.cuda.empty_cache()
            
            import time
            times = []
            
            # Run multiple iterations for stability
            for iteration in range(3):
                start_time = time.time()
                
                a = torch.randn(size, size, device=device, dtype=torch.float16)
                b = torch.randn(size, size, device=device, dtype=torch.float16)
                c = torch.mm(a, b)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                del a, b, c
                torch.cuda.empty_cache()
            
            avg_time = sum(times) / len(times)
            results[size] = avg_time
            
            memory_usage = torch.cuda.max_memory_allocated(0) / (1024**3)
            
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Peak memory: {memory_usage:.3f}GB")
            print(f"  GFLOPS: {(2 * size**3) / (avg_time * 1e9):.1f}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    # Performance scaling analysis
    if len(results) > 1:
        print(f"\n📊 PERFORMANCE SCALING ANALYSIS:")
        print("-" * 33)
        
        sizes = sorted(results.keys())
        for i in range(1, len(sizes)):
            prev_size, curr_size = sizes[i-1], sizes[i]
            prev_time, curr_time = results[prev_size], results[curr_size]
            
            theoretical_scaling = (curr_size / prev_size) ** 3
            actual_scaling = curr_time / prev_time
            efficiency = theoretical_scaling / actual_scaling
            
            print(f"{prev_size}→{curr_size}: {actual_scaling:.2f}x slower (efficiency: {efficiency:.1%})")

if __name__ == "__main__":
    print("🔍 Comprehensive GPU Analysis")
    print("="*40)
    
    # Check all system GPUs
    check_system_gpus()
    
    # Simulate multi-GPU testing
    simulate_multi_gpu_test()
    
    # Virtual multi-GPU benchmark
    create_virtual_multi_gpu_benchmark()
    
    print("\n" + "="*60)
    print("GPU ANALYSIS COMPLETE")
    print("="*60)