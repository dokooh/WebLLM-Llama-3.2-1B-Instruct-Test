#!/usr/bin/env python3
"""
Quick verification script to check if our environment is set up correctly
"""

import torch
import transformers
import sys

def main():
    """Main function to verify the setup"""
    print("=== Environment Verification ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU for inference")
    
    print("\n=== Testing basic functionality ===")
    
    try:
        # Test basic torch operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("✓ PyTorch basic operations working")
        
        # Test transformers import
        from transformers import AutoTokenizer
        print("✓ Transformers library working")
        
        print("\n✅ Environment setup is complete and working!")
        print("You can now proceed to run the full Llama-3.2-1B-Instruct test.")
        
        print("\n📝 Next steps:")
        print("1. Run the test_llama.py script to download and test the model")
        print("2. Note: First run may take several minutes to download the model")
        print("3. The model will be cached locally for future use")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()