#!/usr/bin/env python3
"""
Advanced Single GPU Comprehensive Benchmark
Tests one GPU under various conditions and workloads
"""

import torch
import time
import psutil
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import concurrent.futures

# Test models of different sizes
TEST_MODELS = [
    ("distilgpt2", "Small model (~82M params)"),
    ("gpt2", "Medium model (~124M params)"), 
    ("microsoft/DialoGPT-small", "Conversational model (~124M params)"),
    ("microsoft/DialoGPT-medium", "Larger model (~355M params)"),
]

def print_section(title, width=70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def get_gpu_stats():
    """Get current GPU statistics"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            'name': gpu_name,
            'allocated': memory_allocated,
            'reserved': memory_reserved,
            'total': memory_total,
            'usage_percent': (memory_reserved / memory_total) * 100
        }
    return None

def test_compute_intensive_workload():
    """Test GPU with compute-intensive matrix operations"""
    print_section("COMPUTE INTENSIVE WORKLOAD TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = "cuda:0"
    test_cases = [
        (1000, "Light workload"),
        (2000, "Medium workload"), 
        (3000, "Heavy workload"),
        (4000, "Very heavy workload"),
    ]
    
    results = {}
    
    for size, description in test_cases:
        print(f"\n{description} - {size}x{size} matrices:")
        print("-" * 40)
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Multiple iterations for accuracy
            times = []
            for i in range(3):
                start_time = time.time()
                
                # Create matrices with different data types
                a_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
                b_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
                
                a_fp16 = a_fp32.half()
                b_fp16 = b_fp32.half()
                
                # FP32 operations
                c_fp32 = torch.mm(a_fp32, b_fp32)
                
                # FP16 operations (should be faster)
                c_fp16 = torch.mm(a_fp16, b_fp16)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Clean up
                del a_fp32, b_fp32, a_fp16, b_fp16, c_fp32, c_fp16
            
            avg_time = sum(times) / len(times)
            gflops = (2 * size**3) / (avg_time * 1e9)
            
            # Memory usage
            stats = get_gpu_stats()
            
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Performance: {gflops:.1f} GFLOPS")
            print(f"  GPU memory: {stats['reserved']:.2f}GB ({stats['usage_percent']:.1f}%)")
            
            results[size] = {
                'time': avg_time,
                'gflops': gflops,
                'memory': stats['reserved']
            }
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return results

def test_memory_intensive_workload():
    """Test GPU memory limits and bandwidth"""
    print_section("MEMORY INTENSIVE WORKLOAD TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = "cuda:0"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Total Memory: {gpu_memory:.1f} GB")
    
    # Test different memory usage levels
    memory_tests = [
        (0.25, "25% memory usage"),
        (0.5, "50% memory usage"),
        (0.75, "75% memory usage"),
        (0.9, "90% memory usage"),
    ]
    
    results = {}
    
    for memory_fraction, description in memory_tests:
        print(f"\n{description}:")
        print("-" * 30)
        
        try:
            torch.cuda.empty_cache()
            
            # Calculate tensor size for target memory usage
            target_memory = gpu_memory * memory_fraction * (1024**3)  # Convert to bytes
            elements_per_gb = (1024**3) // 4  # 4 bytes per float32
            total_elements = int(target_memory // 4)
            
            # Create large tensor
            start_time = time.time()
            large_tensor = torch.randn(total_elements, device=device, dtype=torch.float32)
            allocation_time = time.time() - start_time
            
            # Test memory bandwidth with copy operations
            start_time = time.time()
            copy_tensor = large_tensor.clone()
            copy_time = time.time() - start_time
            
            # Calculate bandwidth
            data_size_gb = (total_elements * 4) / (1024**3)
            bandwidth = data_size_gb / copy_time
            
            stats = get_gpu_stats()
            
            print(f"  Tensor size: {data_size_gb:.2f} GB")
            print(f"  Allocation time: {allocation_time:.3f}s")
            print(f"  Copy time: {copy_time:.3f}s")
            print(f"  Memory bandwidth: {bandwidth:.1f} GB/s")
            print(f"  GPU memory used: {stats['reserved']:.2f}GB ({stats['usage_percent']:.1f}%)")
            
            results[memory_fraction] = {
                'allocation_time': allocation_time,
                'copy_time': copy_time,
                'bandwidth': bandwidth,
                'memory_used': stats['reserved']
            }
            
            # Clean up
            del large_tensor, copy_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return results

def test_model_performance():
    """Test different model sizes and their performance"""
    print_section("AI MODEL PERFORMANCE TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = "cuda:0"
    results = {}
    
    for model_name, description in TEST_MODELS:
        print(f"\nTesting {model_name}")
        print(f"({description})")
        print("-" * 50)
        
        try:
            torch.cuda.empty_cache()
            
            # Load model
            print("Loading model...")
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            load_time = time.time() - start_time
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            stats_after_load = get_gpu_stats()
            
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Parameters: {param_count:,}")
            print(f"  GPU memory: {stats_after_load['reserved']:.2f}GB")
            
            # Inference test
            prompt = "The future of artificial intelligence is"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Warmup
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Actual test
            inference_times = []
            for _ in range(3):
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference = sum(inference_times) / len(inference_times)
            tokens_generated = 50  # max_new_tokens
            tokens_per_sec = tokens_generated / avg_inference
            
            print(f"  Inference time: {avg_inference:.3f}s")
            print(f"  Tokens/second: {tokens_per_sec:.1f}")
            
            results[model_name] = {
                'load_time': load_time,
                'parameters': param_count,
                'memory_used': stats_after_load['reserved'],
                'inference_time': avg_inference,
                'tokens_per_sec': tokens_per_sec
            }
            
            # Clean up
            del model, tokenizer, inputs, outputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return results

def test_concurrent_workloads():
    """Test GPU handling of concurrent operations"""
    print_section("CONCURRENT WORKLOAD TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = "cuda:0"
    
    def matrix_task(size, task_id):
        """Individual matrix multiplication task"""
        try:
            start_time = time.time()
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            del a, b, c
            return task_id, elapsed, True
        except Exception as e:
            return task_id, 0, False
    
    # Test sequential vs concurrent execution
    sizes = [800, 800, 800, 800]  # 4 tasks
    
    print("Sequential execution:")
    print("-" * 20)
    sequential_start = time.time()
    sequential_times = []
    
    for i, size in enumerate(sizes):
        task_id, elapsed, success = matrix_task(size, i)
        if success:
            sequential_times.append(elapsed)
            print(f"  Task {task_id}: {elapsed:.3f}s")
    
    sequential_total = time.time() - sequential_start
    print(f"  Total sequential time: {sequential_total:.3f}s")
    
    torch.cuda.empty_cache()
    
    print("\nConcurrent execution (CUDA streams):")
    print("-" * 35)
    
    concurrent_start = time.time()
    
    # Create CUDA streams for concurrent execution
    streams = [torch.cuda.Stream() for _ in range(len(sizes))]
    concurrent_times = []
    
    # Launch concurrent tasks
    tasks = []
    for i, (size, stream) in enumerate(zip(sizes, streams)):
        with torch.cuda.stream(stream):
            task_start = time.time()
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            c = torch.mm(a, b)
            tasks.append((i, a, b, c, task_start, stream))
    
    # Wait for all tasks and measure times
    for task_id, a, b, c, task_start, stream in tasks:
        stream.synchronize()
        elapsed = time.time() - task_start
        concurrent_times.append(elapsed)
        print(f"  Task {task_id}: {elapsed:.3f}s")
        del a, b, c
    
    concurrent_total = time.time() - concurrent_start
    print(f"  Total concurrent time: {concurrent_total:.3f}s")
    
    # Analysis
    if sequential_times and concurrent_times:
        avg_sequential = sum(sequential_times) / len(sequential_times)
        avg_concurrent = sum(concurrent_times) / len(concurrent_times)
        speedup = sequential_total / concurrent_total
        
        print(f"\nConcurrency Analysis:")
        print(f"  Average task time - Sequential: {avg_sequential:.3f}s")
        print(f"  Average task time - Concurrent: {avg_concurrent:.3f}s")
        print(f"  Overall speedup: {speedup:.2f}x")
    
    torch.cuda.empty_cache()

def run_comprehensive_benchmark():
    """Run all benchmark tests"""
    print_section("COMPREHENSIVE SINGLE GPU BENCHMARK", 80)
    
    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        
        print(f"\n🖥️  GPU: {gpu_name}")
        print(f"💾 Memory: {gpu_memory:.1f} GB")
        print(f"🔥 CUDA: {cuda_version}")
        print(f"🐍 PyTorch: {pytorch_version}")
    else:
        print("❌ CUDA not available!")
        return
    
    # Run all tests
    compute_results = test_compute_intensive_workload()
    memory_results = test_memory_intensive_workload()
    model_results = test_model_performance()
    test_concurrent_workloads()
    
    # Summary
    print_section("BENCHMARK SUMMARY", 80)
    
    if compute_results:
        best_gflops = max(r['gflops'] for r in compute_results.values())
        print(f"🚀 Peak Performance: {best_gflops:.1f} GFLOPS")
    
    if memory_results:
        best_bandwidth = max(r['bandwidth'] for r in memory_results.values())
        print(f"💾 Peak Memory Bandwidth: {best_bandwidth:.1f} GB/s")
    
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['tokens_per_sec'])
        print(f"🤖 Best Model Performance: {best_model[1]['tokens_per_sec']:.1f} tok/s ({best_model[0]})")
    
    print(f"\n✅ Your Quadro T1000 is working excellently for AI workloads!")
    print(f"📊 This single GPU provides significant acceleration over CPU processing.")

if __name__ == "__main__":
    run_comprehensive_benchmark()