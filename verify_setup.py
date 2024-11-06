import sys
import torch
import torchvision
import pkg_resources
import platform
import os

def verify_versions():
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    print("\nPackage Versions:")
    packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'opencv-python',
        'Pillow',
        'streamlit',
        'matplotlib',
        'plotly',
        'scikit-learn'
    ]
    
    for package in packages:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")

def check_cuda():
    print("\nCUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        # Test CUDA with a simple operation
        try:
            print("\nTesting CUDA with tensor operation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ CUDA tensor operation successful!")
        except Exception as e:
            print(f"✗ CUDA tensor operation failed: {e}")
    else:
        print("CUDA is not available. GPU acceleration will not be used.")
        print("\nPossible reasons:")
        print("1. NVIDIA GPU is not present")
        print("2. CUDA toolkit is not installed")
        print("3. PyTorch was installed without CUDA support")
        print("\nTo fix:")
        print("1. Verify NVIDIA drivers are installed")
        print("2. Install CUDA toolkit")
        print("3. Reinstall PyTorch with CUDA support:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def test_imports():
    print("\nTesting Required Imports:")
    imports = {
        'numpy': 'import numpy as np',
        'pandas': 'import pandas as pd',
        'cv2': 'import cv2',
        'PIL': 'from PIL import Image',
        'streamlit': 'import streamlit as st',
        'matplotlib': 'import matplotlib.pyplot as plt',
        'plotly': 'import plotly.express as px',
        'sklearn': 'import sklearn'
    }
    
    for name, import_statement in imports.items():
        try:
            exec(import_statement)
            print(f"✓ {name} successfully imported")
        except Exception as e:
            print(f"✗ {name} import failed: {e}")

def main():
    print("=== PyTorch Environment Verification ===\n")
    verify_versions()
    check_cuda()
    test_imports()

if __name__ == "__main__":
    main()