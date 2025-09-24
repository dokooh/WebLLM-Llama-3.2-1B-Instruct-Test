@echo off
title LLAMA CUDA TESTING SUITE

echo ========================================
echo    LLAMA-3.2-1B-INSTRUCT CUDA SUITE
echo ========================================
echo.

echo Activating virtual environment...
call "Llama-3.2-1B-Instruct\Scripts\activate.bat"

if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo Current Python: 
python -c "import sys; print(sys.executable)"
echo.

:menu
echo ========================================
echo Select an option:
echo ========================================
echo 1. Run CUDA diagnostics
echo 2. Fix CUDA installation  
echo 3. Test basic CUDA operations
echo 4. Run full Llama CUDA benchmark
echo 5. Quick CUDA verification
echo 6. Exit
echo ========================================
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto diagnostics
if "%choice%"=="2" goto fix_cuda
if "%choice%"=="3" goto basic_test
if "%choice%"=="4" goto benchmark
if "%choice%"=="5" goto quick_test
if "%choice%"=="6" goto exit
echo Invalid choice. Please try again.
goto menu

:diagnostics
echo.
echo Running CUDA diagnostics...
python cuda_diagnostics.py
echo.
pause
goto menu

:fix_cuda
echo.
echo Running CUDA fix script...
python fix_cuda.py
echo.
pause
goto menu

:basic_test
echo.
echo Testing basic CUDA operations...
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
    # Quick tensor test
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device) 
    z = torch.matmul(x, y)
    print('✅ Basic CUDA operations working!')
else:
    print('❌ CUDA not available')
"
echo.
pause
goto menu

:benchmark
echo.
echo Running full Llama CUDA benchmark...
echo This may take several minutes...
python test_llama_cuda.py
echo.
pause
goto menu

:quick_test
echo.
echo Quick CUDA verification...
python -c "
try:
    import torch
    import transformers
    print('✅ PyTorch:', torch.__version__)
    print('✅ Transformers:', transformers.__version__)
    print('CUDA available:', '✅ Yes' if torch.cuda.is_available() else '❌ No')
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print()
    print('Environment ready for:')
    print('- CPU inference: ✅')
    print('- GPU inference:', '✅' if torch.cuda.is_available() else '❌')
except Exception as e:
    print('❌ Error:', e)
"
echo.
pause
goto menu

:exit
echo.
echo Deactivating virtual environment...
deactivate
echo Goodbye!
pause
exit /b 0