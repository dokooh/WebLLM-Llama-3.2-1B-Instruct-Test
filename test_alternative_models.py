#!/usr/bin/env python3
"""
Alternative Model CUDA Test Script
Uses publicly available models instead of gated Llama model
"""

import torch
import time
import psutil
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Alternative models that don't require authentication
ALTERNATIVE_MODELS = [
    "microsoft/DialoGPT-small",  # Small conversational model
    "gpt2",  # Classic GPT-2 
    "distilgpt2",  # Smaller, faster GPT-2
    "microsoft/DialoGPT-medium",  # Medium conversational model
]

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"                          {title}")
    print("=" * 70)

def get_system_info():
    """Get system information"""
    print("\nSYSTEM INFORMATION")
    print("-" * 18)
    print(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
    print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Python: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: ✅ Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"CUDA available: ❌ No")

def test_basic_operations(device):
    """Test basic PyTorch operations"""
    print(f"\nBASIC OPERATIONS TEST - {device.upper()}")
    print("-" * (27 + len(device)))
    
    try:
        print("Testing 2000x2000 matrix multiplication...")
        start_time = time.time()
        
        # Create tensors on specified device
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        
        # Matrix multiplication
        c = torch.mm(a, b)
        
        # Synchronize if using CUDA
        if "cuda" in str(device):
            torch.cuda.synchronize(device)
        
        elapsed = time.time() - start_time
        print(f"✅ Matrix multiplication: {elapsed:.3f} seconds")
        
        # Show device memory if CUDA
        if "cuda" in str(device):
            gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            print(f"  GPU {gpu_id} Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        # Clean up
        del a, b, c
        if "cuda" in str(device):
            torch.cuda.empty_cache()
        
        return elapsed
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return None

def load_alternative_model(model_name, device):
    """Load an alternative model for testing"""
    device_str = str(device).upper()
    print(f"\nLOADING MODEL: {model_name} - {device_str}")
    print("-" * (25 + len(model_name) + len(device_str)))
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading model on {device}...")
        start_time = time.time()
        
        # Load model with appropriate settings for device
        if "cuda" in str(device):
            # Load model in float16 for GPU efficiency
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Show device placement
        if "cuda" in str(device):
            gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            print(f"Model placed on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        return model, tokenizer, load_time
        
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        return None, None, None

def run_inference_test(model, tokenizer, device, model_name):
    """Run inference test with the model"""
    device_str = str(device).upper()
    print(f"\nINFERENCE PERFORMANCE TEST - {device_str}")
    print("-" * (32 + len(device_str)))
    
    if model is None or tokenizer is None:
        print("❌ Skipping inference test - model not loaded")
        return None, None
    
    try:
        test_prompts = [
            "Hello, how are you?",
            "The weather today is",
            "Artificial intelligence is",
        ]
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}/3: '{prompt}' on {device}")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Synchronize if using CUDA
            if "cuda" in str(device):
                torch.cuda.synchronize(device)
            
            generation_time = time.time() - start_time
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Calculate tokens generated
            new_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            
            print(f"  Generated: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            print(f"  Time: {generation_time:.3f}s, Tokens: {new_tokens}, Speed: {tokens_per_sec:.1f} tok/s")
            
            total_tokens += new_tokens
            total_time += generation_time
            
            # Clean up
            del inputs, outputs
            if "cuda" in str(device):
                torch.cuda.empty_cache()
        
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        print(f"\n📊 Average performance on {device}: {avg_speed:.1f} tokens/second")
        
        return avg_speed, total_time
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return None, None

def show_memory_usage(device):
    """Show memory usage information"""
    device_str = str(device).upper()
    print(f"\nMEMORY USAGE - {device_str}")
    print("-" * (13 + len(device_str)))
    
    if "cuda" in str(device) and torch.cuda.is_available():
        gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        
        print(f"GPU {gpu_id} Memory Allocated: {memory_allocated:.2f} GB")
        print(f"GPU {gpu_id} Memory Reserved: {memory_reserved:.2f} GB")
        print(f"GPU {gpu_id} Memory Total: {memory_total:.2f} GB")
        print(f"GPU {gpu_id} Memory Usage: {(memory_reserved/memory_total)*100:.1f}%")
        print(f"GPU {gpu_id} Name: {torch.cuda.get_device_name(gpu_id)}")
    else:
        ram_usage = psutil.virtual_memory()
        print(f"System RAM Used: {ram_usage.used / (1024**3):.2f} GB")
        print(f"System RAM Total: {ram_usage.total / (1024**3):.2f} GB")
        print(f"System RAM Usage: {ram_usage.percent:.1f}%")

def benchmark_models():
    """Main benchmarking function"""
    print_section("ALTERNATIVE MODEL CUDA BENCHMARK")
    
    get_system_info()
    
    # Determine which devices to test
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n🚀 CUDA detected with {gpu_count} GPU(s)! Will test CPU and all GPU performance.")
        
        # Add each GPU as a separate device to test
        for i in range(gpu_count):
            devices_to_test.append(f"cuda:{i}")
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n⚠️ CUDA not available. Testing CPU only.")
        print("To enable GPU testing, ensure CUDA PyTorch is installed.")
    
    results = {}
    
    # Test each device
    for device in devices_to_test:
        device_str = str(device).upper()
        print_section(f"{device_str} TESTING")
        
        # Basic operations test
        matrix_time = test_basic_operations(device)
        if matrix_time:
            results[f"{device}_matrix_time"] = matrix_time
        
        # Try to find a working model
        model, tokenizer, load_time = None, None, None
        
        for model_name in ALTERNATIVE_MODELS:
            print(f"\nTrying model: {model_name}")
            model, tokenizer, load_time = load_alternative_model(model_name, device)
            
            if model is not None:
                print(f"✅ Successfully loaded {model_name}")
                results[f"{device}_model"] = model_name
                results[f"{device}_load_time"] = load_time
                break
        
        if model is None:
            print("❌ Failed to load any alternative model")
            continue
        
        # Run inference test
        avg_speed, total_time = run_inference_test(model, tokenizer, device, model_name)
        if avg_speed:
            results[f"{device}_tokens_per_sec"] = avg_speed
            results[f"{device}_inference_time"] = total_time
        
        # Show memory usage
        show_memory_usage(device)
        
        # Clean up
        del model, tokenizer
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()
    
    # Performance comparison
    print_section("PERFORMANCE COMPARISON")
    
    # Matrix operations comparison
    print(f"\nMATRIX OPERATIONS COMPARISON")
    print("-" * 30)
    if "cpu_matrix_time" in results:
        print(f"CPU time: {results['cpu_matrix_time']:.3f}s")
        
        # Compare with each GPU
        gpu_devices = [d for d in devices_to_test if "cuda" in str(d)]
        for gpu_device in gpu_devices:
            if f"{gpu_device}_matrix_time" in results:
                gpu_time = results[f"{gpu_device}_matrix_time"]
                speedup = results["cpu_matrix_time"] / gpu_time
                gpu_name = torch.cuda.get_device_name(int(str(gpu_device).split(":")[-1]))
                print(f"{gpu_device} ({gpu_name}): {gpu_time:.3f}s - Speedup: {speedup:.2f}x")
    
    # Inference performance comparison
    print(f"\nINFERENCE PERFORMANCE COMPARISON")
    print("-" * 32)
    if "cpu_tokens_per_sec" in results:
        print(f"CPU speed: {results['cpu_tokens_per_sec']:.1f} tokens/sec")
        
        # Compare with each GPU
        gpu_devices = [d for d in devices_to_test if "cuda" in str(d)]
        for gpu_device in gpu_devices:
            if f"{gpu_device}_tokens_per_sec" in results:
                gpu_speed = results[f"{gpu_device}_tokens_per_sec"]
                speedup = gpu_speed / results["cpu_tokens_per_sec"]
                gpu_name = torch.cuda.get_device_name(int(str(gpu_device).split(":")[-1]))
                print(f"{gpu_device} ({gpu_name}): {gpu_speed:.1f} tokens/sec - Speedup: {speedup:.2f}x")
    
    # GPU vs GPU comparison if multiple GPUs
    gpu_devices = [d for d in devices_to_test if "cuda" in str(d)]
    if len(gpu_devices) > 1:
        print(f"\nGPU-TO-GPU COMPARISON")
        print("-" * 21)
        
        # Matrix operations
        print("Matrix Operations:")
        for i, gpu1 in enumerate(gpu_devices):
            for gpu2 in gpu_devices[i+1:]:
                if f"{gpu1}_matrix_time" in results and f"{gpu2}_matrix_time" in results:
                    time1 = results[f"{gpu1}_matrix_time"]
                    time2 = results[f"{gpu2}_matrix_time"]
                    ratio = time1 / time2
                    gpu1_name = torch.cuda.get_device_name(int(str(gpu1).split(":")[-1]))
                    gpu2_name = torch.cuda.get_device_name(int(str(gpu2).split(":")[-1]))
                    print(f"  {gpu1} vs {gpu2}: {ratio:.2f}x ({time1:.3f}s vs {time2:.3f}s)")
        
        # Inference performance
        print("Inference Performance:")
        for i, gpu1 in enumerate(gpu_devices):
            for gpu2 in gpu_devices[i+1:]:
                if f"{gpu1}_tokens_per_sec" in results and f"{gpu2}_tokens_per_sec" in results:
                    speed1 = results[f"{gpu1}_tokens_per_sec"]
                    speed2 = results[f"{gpu2}_tokens_per_sec"]
                    ratio = speed1 / speed2
                    gpu1_name = torch.cuda.get_device_name(int(str(gpu1).split(":")[-1]))
                    gpu2_name = torch.cuda.get_device_name(int(str(gpu2).split(":")[-1]))
                    print(f"  {gpu1} vs {gpu2}: {ratio:.2f}x ({speed1:.1f} vs {speed2:.1f} tok/s)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS")
    print("-" * 15)
    
    if torch.cuda.is_available():
        print("✅ CUDA is working! Your GPU can accelerate model inference.")
        print("🔧 For Llama models, you'll need Hugging Face authentication:")
        print("   1. Create account at https://huggingface.co/")
        print("   2. Request access to Meta Llama models")
        print("   3. Set HF_TOKEN environment variable")
        print("   4. Or run: huggingface-cli login")
    else:
        print("⚠️ CUDA not available. Install CUDA PyTorch for GPU acceleration:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    print_section("BENCHMARK COMPLETE")
    return results

def setup_huggingface_auth():
    """Help set up Hugging Face authentication"""
    print_section("HUGGING FACE AUTHENTICATION SETUP")
    
    print("""
To access the original Llama-3.2-1B-Instruct model, you need:

1. **Create Hugging Face Account**
   - Go to: https://huggingface.co/join
   - Create a free account

2. **Request Model Access** 
   - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - Click "Request access to this model"
   - Fill out the form and wait for approval (usually quick)

3. **Get Access Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name and select "Read" permissions
   - Copy the token

4. **Set Up Authentication**
   
   Option A - Environment Variable:
   set HF_TOKEN=your_token_here
   
   Option B - Login command:
   pip install huggingface_hub
   huggingface-cli login
   
   Option C - In code:
   from huggingface_hub import login
   login(token="your_token_here")

5. **Verify Access**
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"

Once authenticated, you can use the original test_llama_cuda.py script!
""")

if __name__ == "__main__":
    print("🚀 Starting Alternative Model CUDA Benchmark")
    print("This uses publicly available models instead of gated Llama models.")
    print()
    
    try:
        results = benchmark_models()
        print("\n🎉 Benchmark completed successfully!")
        
        # Show HF auth info if CUDA is working
        if torch.cuda.is_available():
            setup_huggingface_auth()
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()