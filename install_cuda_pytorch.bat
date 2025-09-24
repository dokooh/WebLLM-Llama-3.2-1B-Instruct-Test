@echo off
echo ====================================
echo INSTALLING CUDA PYTORCH
echo ====================================

echo Activating virtual environment...
call "Llama-3.2-1B-Instruct\Scripts\activate.bat"

echo.
echo Checking current environment...
echo Python executable: 
python -c "import sys; print(sys.executable)"

echo.
echo Current packages:
pip list | findstr torch

echo.
echo Uninstalling existing PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing additional dependencies...
pip install accelerate

echo.
echo Verifying CUDA installation...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
echo ====================================
echo INSTALLATION COMPLETE
echo ====================================
pause