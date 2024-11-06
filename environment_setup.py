import os
import sys
import subprocess
import torch
import pkg_resources

def check_cuda_setup():
    """Verify CUDA configuration for PyTorch"""
    print("\n=== PyTorch CUDA Configuration ===")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"CUDA Available: {cuda_available}")
    print(f"GPU Count: {device_count}")
    
    if cuda_available:
        # Get current device
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current Device: {current_device}")
        print(f"Device Name: {device_name}")
        
        # Test CUDA operation
        x = torch.rand(5, 3)
        try:
            cuda_x = x.cuda()
            print("✓ Successfully performed CUDA operation")
        except Exception as e:
            print(f"✗ Failed to perform CUDA operation: {e}")
    
    return cuda_available

def reinstall_torch_cuda():
    """Provide instructions for reinstalling PyTorch with CUDA support"""
    print("\nTo reinstall PyTorch with CUDA support:")
    print("1. Deactivate and reactivate your virtual environment")
    print("2. Run the following command:")
    print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nAfter reinstalling, verify CUDA support with:")
    print("python -c \"import torch; print(torch.cuda.is_available())\"")

def setup_project_structure():
    """Create the project directory structure"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/validation',
        'docs/proposal',
        'docs/forms',
        'docs/technical',
        'src/data',
        'src/models',
        'src/visualization',
        'src/utils',
        'tests',
        'notebooks'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Create __init__.py files
    python_dirs = [
        'src/data',
        'src/models',
        'src/visualization',
        'src/utils',
        'tests'
    ]
    
    for dir_path in python_dirs:
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
            print(f"✓ Created {init_file}")

def main():
    print("=== AVIS Project Setup ===\n")
    
    # Check Python version
    python_version = sys.version
    print(f"Python Version: {python_version}")
    
    # Check CUDA setup
    cuda_available = check_cuda_setup()
    
    if not cuda_available:
        print("\n⚠️ CUDA is not available. This will significantly impact model training performance.")
        reinstall_torch_cuda()
    
    # Create project structure
    print("\nSetting up project structure...")
    setup_project_structure()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()