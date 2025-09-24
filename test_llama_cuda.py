"""
Llama-3.2-1B-Instruct CUDA Performance Test
Comprehensive testing script for CPU vs GPU performance with detailed metrics
"""
import torch
import time
import psutil
import gc
import os
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_header(title, width=70):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width)

def print_section(title):
    """Print section header"""
    print(f"\n{title}")
    print("-" * len(title))

def get_system_info():
    """Get system information"""
    print_section("SYSTEM INFORMATION")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    
    print(f"CPU cores (physical): {cpu_count}")
    print(f"CPU cores (logical): {cpu_count_logical}")
    print(f"RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Python: {torch.__version__}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA available: ✅ Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute: {props.major}.{props.minor}")
    else:
        print(f"CUDA available: ❌ No")
        print("  Installing CUDA-enabled PyTorch...")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

def test_basic_operations(device):
    """Test basic PyTorch operations on device"""
    print_section(f"BASIC OPERATIONS TEST - {device.type.upper()}")
    
    try:
        # Matrix multiplication test
        size = 2000
        print(f"Testing {size}x{size} matrix multiplication...")
        
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        print(f"✅ Matrix multiplication: {elapsed:.3f} seconds")
        
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"   GPU memory used: {memory_used:.1f} MB")
        
        # Clean up
        del x, y, z
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return elapsed
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def load_llama_model(device, use_half_precision=False):
    """Load Llama model on specified device"""
    print_section(f"LOADING LLAMA MODEL - {device.type.upper()}")
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading model...")
        start_time = time.time()
        
        # Configure model loading based on device
        kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        if device.type == 'cuda':
            kwargs['torch_dtype'] = torch.float16 if use_half_precision else torch.float32
            kwargs['device_map'] = 'auto'
        else:
            kwargs['torch_dtype'] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
        # Move to device if not using device_map
        if device.type == 'cpu' or not use_half_precision:
            model = model.to(device)
        
        load_time = time.time() - start_time
        
        # Model info
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * (2 if use_half_precision else 4) / 1024**2
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"   Parameters: {param_count:,}")
        print(f"   Model size: {model_size_mb:.1f} MB")
        print(f"   Precision: {'FP16' if use_half_precision else 'FP32'}")
        
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"   GPU memory reserved: {memory_reserved:.2f} GB")
        
        return model, tokenizer, load_time
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return None, None, None

def run_inference_test(model, tokenizer, device, test_prompts):
    """Run inference tests with various prompts"""
    print_section(f"INFERENCE PERFORMANCE TEST - {device.type.upper()}")
    
    if model is None or tokenizer is None:
        print("❌ Skipping inference test - model not loaded")
        return []
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: {prompt[:50]}...")
        
        try:
            # Clear cache before test
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_tokens = len(inputs['input_ids'][0])
            
            # Generate
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.0,  # Deterministic
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True
                )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            generation_time = time.time() - start_time
            
            # Decode response
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response_only = response[len(prompt):].strip()
            
            # Calculate metrics
            output_tokens = len(outputs.sequences[0]) - input_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            print(f"   ⚡ Generated {output_tokens} tokens in {generation_time:.3f}s")
            print(f"   📊 Speed: {tokens_per_second:.1f} tokens/sec")
            
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - start_memory) / 1024**2
                print(f"   🧠 GPU memory for generation: {memory_used:.1f} MB")
            
            print(f"   💬 Response: {response_only[:80]}...")
            
            results.append({
                'prompt': prompt,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'output_tokens': output_tokens
            })
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                'prompt': prompt,
                'generation_time': None,
                'tokens_per_second': 0,
                'output_tokens': 0
            })
    
    return results

def benchmark_devices():
    """Compare performance between CPU and GPU"""
    print_header("LLAMA-3.2-1B-INSTRUCT CUDA BENCHMARK")
    
    get_system_info()
    
    # Test prompts
    test_prompts = [
        "Hello, how are you today?",
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate the factorial of a number:",
        "What are the advantages of using GPU acceleration for AI models?",
        "Describe the process of training a neural network step by step:"
    ]
    
    # Test CPU
    print_header("CPU TESTING")
    cpu_device = torch.device('cpu')
    
    # Basic operations test
    cpu_basic_time = test_basic_operations(cpu_device)
    
    # Load model on CPU
    cpu_model, cpu_tokenizer, cpu_load_time = load_llama_model(cpu_device)
    
    # Run inference tests
    cpu_results = run_inference_test(cpu_model, cpu_tokenizer, cpu_device, test_prompts)
    
    # Clean up CPU model
    if cpu_model:
        del cpu_model, cpu_tokenizer
        gc.collect()
    
    # Test GPU (if available)
    gpu_results = []
    gpu_basic_time = None
    gpu_load_time = None
    
    if torch.cuda.is_available():
        print_header("GPU TESTING")
        gpu_device = torch.device('cuda')
        
        # Basic operations test
        gpu_basic_time = test_basic_operations(gpu_device)
        
        # Load model on GPU
        gpu_model, gpu_tokenizer, gpu_load_time = load_llama_model(gpu_device, use_half_precision=True)
        
        # Run inference tests
        gpu_results = run_inference_test(gpu_model, gpu_tokenizer, gpu_device, test_prompts)
        
        # Clean up GPU model
        if gpu_model:
            del gpu_model, gpu_tokenizer
            torch.cuda.empty_cache()
    else:
        print_header("GPU TESTING SKIPPED")
        print("CUDA not available - install CUDA-enabled PyTorch to enable GPU testing")
    
    # Compare results
    print_header("PERFORMANCE COMPARISON")
    
    print_section("BASIC OPERATIONS")
    if cpu_basic_time and gpu_basic_time:
        speedup = cpu_basic_time / gpu_basic_time
        print(f"CPU matrix multiplication: {cpu_basic_time:.3f}s")
        print(f"GPU matrix multiplication: {gpu_basic_time:.3f}s")
        print(f"GPU speedup: {speedup:.1f}x faster")
    else:
        print("Unable to compare - one or both tests failed")
    
    print_section("MODEL LOADING")
    if cpu_load_time and gpu_load_time:
        print(f"CPU model loading: {cpu_load_time:.2f}s")
        print(f"GPU model loading: {gpu_load_time:.2f}s")
    
    print_section("INFERENCE PERFORMANCE")
    if cpu_results and gpu_results:
        cpu_avg_tps = sum(r['tokens_per_second'] for r in cpu_results if r['tokens_per_second']) / len(cpu_results)
        gpu_avg_tps = sum(r['tokens_per_second'] for r in gpu_results if r['tokens_per_second']) / len(gpu_results)
        
        print(f"CPU average: {cpu_avg_tps:.1f} tokens/sec")
        if gpu_avg_tps > 0:
            print(f"GPU average: {gpu_avg_tps:.1f} tokens/sec")
            speedup = gpu_avg_tps / cpu_avg_tps if cpu_avg_tps > 0 else 0
            print(f"GPU speedup: {speedup:.1f}x faster")
        else:
            print("GPU inference failed")
    else:
        print("Unable to compare inference performance")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    if not torch.cuda.is_available():
        print("🔧 To enable GPU acceleration:")
        print("1. Verify NVIDIA drivers are installed")
        print("2. Install CUDA-enabled PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("3. Restart your environment")
    elif gpu_results and cpu_results:
        avg_gpu_speedup = gpu_avg_tps / cpu_avg_tps if 'gpu_avg_tps' in locals() and cpu_avg_tps > 0 else 0
        
        if avg_gpu_speedup > 3:
            print("🚀 Excellent GPU performance! Your setup is optimized.")
        elif avg_gpu_speedup > 1.5:
            print("✅ Good GPU acceleration. Consider using FP16 for better performance.")
        elif avg_gpu_speedup > 0:
            print("⚠️ Limited GPU speedup. Check GPU memory and model size.")
        else:
            print("❌ GPU performance issues detected. Check CUDA installation.")
            
        print("\n💡 Performance tips:")
        print("- Use torch.compile() for additional speedup")
        print("- Consider model quantization for memory-constrained GPUs")
        print("- Use mixed precision training (FP16/BF16)")
    
    print_header("BENCHMARK COMPLETE")

if __name__ == "__main__":
    try:
        benchmark_devices()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 Cleanup completed")