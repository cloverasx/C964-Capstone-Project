import os
import sys
import torch
import subprocess
import platform
from pathlib import Path

def check_nvidia_smi():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        print("\nnvidia-smi output:")
        print(nvidia_smi)
        return True
    except Exception as e:
        print("\nError running nvidia-smi:", e)
        return False

def check_environment_variables():
    print("\nChecking Environment Variables:")
    cuda_vars = {
        'CUDA_HOME': os.environ.get('CUDA_HOME'),
        'CUDA_PATH': os.environ.get('CUDA_PATH'),
        'PATH': os.environ.get('PATH')
    }
    
    for var, value in cuda_vars.items():
        if var == 'PATH':
            cuda_paths = [p for p in value.split(os.pathsep) if 'cuda' in p.lower()]
            print(f"\nCUDA paths in PATH:")
            for p in cuda_paths:
                print(f"  {p}")
        else:
            print(f"{var}: {'Found' if value else 'Not found'} - {value if value else ''}")

def check_cuda_files():
    print("\nChecking CUDA Installation:")
    if sys.platform == 'win32':
        cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
    else:
        cuda_path = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    cuda_path = Path(cuda_path)
    important_files = [
        'bin/nvcc',
        'bin/nvcc.exe',
        'include/cuda.h',
        'lib64/libcudart.so',
        'lib64/cudart.lib'
    ]
    
    print(f"Looking for CUDA files in: {cuda_path}")
    for file in important_files:
        file_path = cuda_path / file
        if file_path.exists():
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")

def check_pytorch_cuda():
    print("\nPyTorch CUDA Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        
        # Test CUDA computation
        try:
            print("\nTesting CUDA Computation:")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            z = torch.matmul(x, y)
            end.record()
            torch.cuda.synchronize()
            
            print(f"✓ Matrix multiplication successful")
            print(f"Time taken: {start.elapsed_time(end):.2f} ms")
        except Exception as e:
            print(f"✗ CUDA computation failed: {e}")
    else:
        print("\nCUDA is not available. Possible issues:")
        print("1. NVIDIA GPU not detected")
        print("2. CUDA toolkit not installed")
        print("3. PyTorch installed without CUDA support")
        print("\nTroubleshooting steps:")
        print("1. Verify NVIDIA GPU is present")
        print("2. Install latest NVIDIA drivers")
        print("3. Install CUDA toolkit")
        print("4. Reinstall PyTorch with CUDA support:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def main():
    print("=== CUDA Environment Check ===")
    print(f"System: {platform.platform()}")
    print(f"Python: {sys.version}")
    
    check_environment_variables()
    if check_nvidia_smi():
        check_cuda_files()
    check_pytorch_cuda()

if __name__ == "__main__":
    main()