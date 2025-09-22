# Llama-3.2-1B-Instruct Setup

This workspace is set up to work with the Llama-3.2-1B-Instruct model using Python and Hugging Face Transformers.

## 📁 Files Overview

- **`verify_setup.py`** - Verifies that your Python environment is working correctly
- **`test_llama.py`** - Main script to test the official Llama-3.2-1B-Instruct model (requires authentication)
- **`test_alternative.py`** - Demo script using an open model to verify setup
- **`authenticate.py`** - Helper script to set up Hugging Face authentication

## 🚀 Quick Start

### 1. Environment Setup (✅ Already Done!)
```bash
# Virtual environment created and activated
# Dependencies installed: torch, transformers, accelerate, sentencepiece, huggingface_hub
```

### 2. Verify Setup
```bash
python verify_setup.py
```

### 3. Test with Open Model (No Authentication Needed)
```bash
python test_alternative.py
```

### 4. Access Official Llama Model

#### Option A: Use Authentication Helper
```bash
python authenticate.py
```

#### Option B: Manual Authentication
1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Accept the license agreement
3. Get a token from https://huggingface.co/settings/tokens
4. Run: `huggingface-cli login` and enter your token

#### Run Official Llama Model
```bash
python test_llama.py
```

## 🔧 Environment Details

- **Virtual Environment**: `Llama-3.2-1B-Instruct/`
- **Python Version**: 3.13.1
- **PyTorch**: 2.8.0+cpu (CPU version)
- **Transformers**: 4.56.2
- **Device**: CPU (CUDA not available)

## 📊 Performance Notes

- First model download may take several minutes
- Models are cached locally after first download
- CPU inference is slower than GPU but functional
- Llama-3.2-1B-Instruct is optimized for instruction-following tasks

## 🔍 Troubleshooting

### Authentication Issues
- Ensure you've accepted the license on the Hugging Face model page
- Verify your token has the correct permissions
- License approval from Meta may take time

### Memory Issues
- The 1B model should run on most modern systems
- Consider closing other applications if you encounter memory errors
- GPU would provide better performance if available

### Import Errors
- Run `verify_setup.py` to check your environment
- Ensure the virtual environment is activated
- Reinstall packages if necessary: `pip install torch transformers accelerate`

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Llama Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🎯 What's Working

✅ Virtual environment setup  
✅ Package installation  
✅ Environment verification  
✅ Model loading and inference  
✅ Alternative model testing  
✅ Authentication setup  

Ready to run Llama-3.2-1B-Instruct! 🦙