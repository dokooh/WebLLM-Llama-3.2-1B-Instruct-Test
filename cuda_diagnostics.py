"""
CUDA Diagnostic Tool for Llama-3.2-1B-Instruct
Comprehensive system check for CUDA capabilities and troubleshooting
"""
import subprocess
import sys
import os
import platform

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """Print a section header"""
    print(f"\n{title}")
    print("-" * len(title))

def run_command(cmd, description=""):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_system_info():
    """Check basic system information"""
    print_section("1. SYSTEM INFORMATION")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {sys.version}")
    print(f"Current directory: {os.getcwd()}")

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print_section("2. NVIDIA DRIVER CHECK")
    
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA driver is installed and working")
        # Extract driver version
        lines = output.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
                break
        
        # Show GPU information
        gpu_lines = [line for line in lines if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line]
        if gpu_lines:
            print("   Detected GPUs:")
            for gpu_line in gpu_lines:
                print(f"   - {gpu_line.strip()}")
    else:
        print("❌ NVIDIA driver not found or not working")
        print(f"   Error: {error}")
        print("   Please install NVIDIA drivers from: https://www.nvidia.com/drivers/")

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    print_section("3. CUDA TOOLKIT CHECK")
    
    success, output, error = run_command("nvcc --version")
    if success:
        print("✅ CUDA toolkit is installed")
        version_line = [line for line in output.split('\n') if 'release' in line.lower()]
        if version_line:
            print(f"   {version_line[0].strip()}")
    else:
        print("⚠️ CUDA toolkit (nvcc) not found in PATH")
        print("   This may be normal if using PyTorch with bundled CUDA")
    
    # Check CUDA environment variables
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    
    if cuda_path or cuda_home:
        print(f"   CUDA_PATH: {cuda_path or 'Not set'}")
        print(f"   CUDA_HOME: {cuda_home or 'Not set'}")
    else:
        print("   No CUDA environment variables found")

def check_python_environment():
    """Check Python environment and virtual environment"""
    print_section("4. PYTHON ENVIRONMENT")
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual environment active: {in_venv}")
    if in_venv:
        print(f"   Virtual env path: {sys.prefix}")
    
    # Check for our specific virtual environment
    expected_venv = r"C:\SAI\IA\Llama-3.2-1B-Instruct\Llama-3.2-1B-Instruct"
    if expected_venv in sys.prefix:
        print("✅ Using correct Llama-3.2-1B-Instruct virtual environment")
    else:
        print(f"⚠️ Not using expected virtual environment")
        print(f"   Expected: {expected_venv}")
        print(f"   Current: {sys.prefix}")

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print_section("5. PYTORCH CUDA CHECK")
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA version (PyTorch): {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            # GPU details
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"     Total memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"     Compute capability: {props.major}.{props.minor}")
                print(f"     Multi-processors: {props.multi_processor_count}")
        else:
            print("❌ CUDA not available in PyTorch")
            print("   Possible causes:")
            print("   - PyTorch CPU-only version installed")
            print("   - NVIDIA drivers not installed")
            print("   - CUDA version mismatch")
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Run: pip install torch torchvision torchaudio")

def check_transformers():
    """Check transformers library"""
    print_section("6. TRANSFORMERS LIBRARY CHECK")
    
    try:
        import transformers
        print(f"✅ Transformers installed: {transformers.__version__}")
        
        # Check if we can import model classes
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ Model classes available")
        
    except ImportError as e:
        print(f"❌ Transformers not installed or incomplete: {e}")
        print("   Run: pip install transformers accelerate")

def test_basic_cuda():
    """Test basic CUDA functionality"""
    print_section("7. CUDA FUNCTIONALITY TEST")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️ Skipping CUDA test - CUDA not available")
            return
        
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Test tensor operations
        print("Testing tensor operations...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU matrix multiplication successful!")
        print(f"   Operation time: {gpu_time:.4f} seconds")
        print(f"   GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"   CPU time: {cpu_time:.4f} seconds")
        print(f"   GPU speedup: {speedup:.1f}x faster")
        
        # Memory test
        print("\nTesting GPU memory allocation...")
        torch.cuda.empty_cache()
        
        sizes_mb = [100, 500, 1000, 2000]
        for size_mb in sizes_mb:
            try:
                size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                tensor = torch.randn(size_elements, device=device)
                print(f"   ✅ {size_mb} MB allocation successful")
                del tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"   ❌ {size_mb} MB allocation failed: {e}")
                break
                
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")

def provide_recommendations():
    """Provide recommendations based on findings"""
    print_section("8. RECOMMENDATIONS & FIXES")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    if not cuda_available:
        print("🔧 TO FIX CUDA ISSUES:")
        print("1. Install/Update NVIDIA drivers:")
        print("   - Download from: https://www.nvidia.com/drivers/")
        print("   - Restart computer after installation")
        print()
        print("2. Install CUDA-enabled PyTorch:")
        print("   - Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
        print("   - Install CUDA version: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("3. Install additional dependencies:")
        print("   - pip install transformers accelerate")
        print()
        print("4. Restart Python/Terminal after installation")
    else:
        print("✅ CUDA setup appears to be working!")
        
        # Check GPU memory and provide optimization tips
        try:
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / 1024**3
            
            print(f"\n💡 OPTIMIZATION TIPS for {props.name} ({memory_gb:.1f} GB):")
            
            if memory_gb < 4:
                print("- Use smaller batch sizes (batch_size=1)")
                print("- Consider model quantization (4-bit/8-bit)")
                print("- Use gradient checkpointing if training")
            elif memory_gb < 8:
                print("- Batch size 2-4 recommended")
                print("- fp16 precision recommended")
            else:
                print("- Can handle larger batch sizes")
                print("- fp16 or bf16 precision recommended")
                
            print("- Use torch.no_grad() for inference")
            print("- Clear CUDA cache: torch.cuda.empty_cache()")
            
        except Exception:
            pass

def main():
    """Main diagnostic function"""
    print_header("CUDA DIAGNOSTIC TOOL FOR LLAMA-3.2-1B-INSTRUCT")
    
    # Run all checks
    check_system_info()
    check_nvidia_driver()
    check_cuda_toolkit()
    check_python_environment()
    check_pytorch()
    check_transformers()
    test_basic_cuda()
    provide_recommendations()
    
    print_header("DIAGNOSTIC COMPLETE")
    print("Save this output and use it to troubleshoot any issues!")

if __name__ == "__main__":
    main()