"""
CUDA Fix and Detection Script
Attempts to fix CUDA PyTorch installation issues
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_cuda_compatibility():
    """Check if system supports CUDA"""
    print("Checking CUDA compatibility...")
    
    # Check NVIDIA driver
    success, output, error = run_command("nvidia-smi")
    if not success:
        print("❌ NVIDIA driver not found")
        return False
    
    print("✅ NVIDIA driver detected")
    
    # Extract CUDA version from nvidia-smi
    lines = output.split('\n')
    cuda_version = None
    for line in lines:
        if 'CUDA Version:' in line:
            try:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                break
            except:
                pass
    
    if cuda_version:
        print(f"✅ CUDA Version: {cuda_version}")
        return True
    else:
        print("⚠️ Could not determine CUDA version")
        return False

def fix_pytorch_cuda():
    """Try to fix PyTorch CUDA installation"""
    print("\nAttempting to fix PyTorch CUDA installation...")
    
    commands = [
        # Uninstall existing PyTorch
        "pip uninstall torch torchvision torchaudio -y",
        
        # Clear pip cache
        "pip cache purge",
        
        # Install CUDA PyTorch with specific version
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir",
        
        # Install additional dependencies
        "pip install accelerate",
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        success, output, error = run_command(cmd)
        
        if not success:
            print(f"⚠️ Command failed: {error}")
        else:
            print("✅ Command completed")
    
    print("\nInstallation complete. Testing CUDA availability...")

def test_cuda():
    """Test if CUDA is working in PyTorch"""
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
            # Quick test
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            
            print("✅ CUDA tensor operations working!")
            return True
            
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("CUDA FIX AND DETECTION SCRIPT")
    print("=" * 60)
    
    # Check if we're in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"✅ Using virtual environment: {sys.prefix}")
    else:
        print("⚠️ Not in virtual environment")
    
    # Check CUDA compatibility
    if not check_cuda_compatibility():
        print("\n❌ System does not support CUDA")
        print("Please install NVIDIA drivers and CUDA toolkit")
        return
    
    # Test current PyTorch
    print("\nTesting current PyTorch installation...")
    if test_cuda():
        print("\n🎉 CUDA is already working!")
        return
    
    # Try to fix PyTorch
    response = input("\nWould you like to reinstall PyTorch with CUDA? (y/n): ")
    if response.lower() == 'y':
        fix_pytorch_cuda()
        
        # Test again
        print("\nTesting fixed installation...")
        if test_cuda():
            print("\n🎉 CUDA fix successful!")
        else:
            print("\n❌ CUDA fix failed")
            print("Manual steps:")
            print("1. Restart your terminal/IDE")
            print("2. Verify virtual environment is activated")  
            print("3. Try: pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
    else:
        print("Skipping PyTorch reinstallation")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()