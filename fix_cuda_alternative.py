#!/usr/bin/env python3
"""
Alternative CUDA PyTorch Installation Script
Handles Python 3.13 and network issues
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description="Running command"):
    """Execute a command and return success status"""
    print(f"\n{description}...")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 13:
        print("⚠️ Python 3.13 detected - CUDA PyTorch may have limited compatibility")
        return "3.13"
    elif version.major == 3 and version.minor >= 8:
        print("✅ Python version compatible with CUDA PyTorch")
        return "compatible"
    else:
        print("❌ Python version may not be compatible")
        return "incompatible"

def try_cuda_installation_methods():
    """Try multiple methods to install CUDA PyTorch"""
    
    print("=" * 60)
    print("ALTERNATIVE CUDA PYTORCH INSTALLATION")
    print("=" * 60)
    
    # Check Python version
    py_version = check_python_version()
    
    # Method 1: Standard CUDA 12.1
    print("\n🔧 METHOD 1: Standard CUDA 12.1 installation")
    success1 = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch CUDA 12.1"
    )
    
    if success1:
        return test_cuda_installation()
    
    # Method 2: CUDA 11.8 (older, more compatible)
    print("\n🔧 METHOD 2: CUDA 11.8 installation (older version)")
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstalling current PyTorch")
    success2 = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch CUDA 11.8"
    )
    
    if success2:
        return test_cuda_installation()
    
    # Method 3: Conda installation (if available)
    print("\n🔧 METHOD 3: Conda installation")
    conda_success = run_command(
        "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y",
        "Installing with Conda"
    )
    
    if conda_success:
        return test_cuda_installation()
    
    # Method 4: Pre-compiled wheels (nightly)
    print("\n🔧 METHOD 4: Nightly/Pre-release wheels")
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstalling current PyTorch")
    success4 = run_command(
        "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121",
        "Installing nightly PyTorch CUDA"
    )
    
    if success4:
        return test_cuda_installation()
    
    # Method 5: Force CPU version with clear message
    print("\n🔧 METHOD 5: Installing CPU version as fallback")
    run_command("pip install torch torchvision torchaudio", "Installing CPU PyTorch")
    
    return False

def test_cuda_installation():
    """Test if CUDA PyTorch is working"""
    print("\n" + "=" * 40)
    print("TESTING CUDA INSTALLATION")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available!")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            # Test basic CUDA operations
            if torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU 0: {gpu_name}")
                
                # Create a test tensor on GPU
                x = torch.randn(3, 3).cuda()
                y = torch.randn(3, 3).cuda()
                z = torch.mm(x, y)
                print("✅ Basic CUDA tensor operations working")
                
                # Memory info
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_cached = torch.cuda.memory_reserved(0) / 1024**2
                print(f"GPU memory allocated: {memory_allocated:.1f} MB")
                print(f"GPU memory cached: {memory_cached:.1f} MB")
                
                return True
        else:
            print("❌ CUDA not available in PyTorch")
            print("Possible reasons:")
            print("- NVIDIA drivers not installed")
            print("- CUDA toolkit not installed") 
            print("- PyTorch CPU version installed")
            print("- Python/PyTorch version incompatibility")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def show_manual_instructions():
    """Show manual installation instructions"""
    print("\n" + "=" * 60)
    print("MANUAL INSTALLATION INSTRUCTIONS")
    print("=" * 60)
    
    print("""
If automatic installation fails, try these manual steps:

1. **Restart VS Code and Terminal**
   - Close all VS Code windows
   - Open new terminal
   - Activate virtual environment

2. **Check Python Version Compatibility**
   - PyTorch CUDA works best with Python 3.8-3.11
   - Python 3.13 may have limited CUDA support

3. **Manual PyTorch Installation**
   
   For CUDA 12.1:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   For CUDA 11.8 (more compatible):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   CPU version (fallback):
   pip install torch torchvision torchaudio

4. **Alternative: Use Conda Environment**
   conda create -n llama_cuda python=3.11
   conda activate llama_cuda
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

5. **Network Issues**
   - Check corporate firewall/proxy settings
   - Try different PyTorch index URLs
   - Use pip cache purge before installation

6. **Verify Installation**
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
""")

def main():
    """Main installation function"""
    print("CUDA PyTorch Alternative Installation Script")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    if try_cuda_installation_methods():
        print("\n🎉 SUCCESS! CUDA PyTorch is now working!")
        print("\nYou can now run CUDA tests:")
        print("python test_llama_cuda.py")
    else:
        print("\n⚠️ Automatic installation failed")
        show_manual_instructions()

if __name__ == "__main__":
    main()