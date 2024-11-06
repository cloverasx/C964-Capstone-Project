# src/data/preprocess.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """Handle image preprocessing for vehicle classification"""
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Additional transforms for data augmentation during training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path, training=False):
        """
        Preprocess a single image for inference or training
        
        Args:
            image_path: Path to the image file
            training: Boolean indicating if this is for training (uses augmentation)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            if training:
                return self.train_transform(image)
            return self.transform(image)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def preprocess_batch(self, image_paths, training=False):
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of paths to image files
            training: Boolean indicating if this is for training
            
        Returns:
            torch.Tensor: Batch of preprocessed image tensors
        """
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path, training)
            if tensor is not None:
                tensors.append(tensor)
        return torch.stack(tensors) if tensors else None

# src/models/classifier.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple

class VehicleClassifier(nn.Module):
    """Vehicle classification model based on pretrained backbone"""
    def __init__(self, num_classes: int, model_name: str = 'resnet50'):
        super(VehicleClassifier, self).__init__()
        
        # Load pretrained backbone
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with confidence scores
        
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            return predicted, confidence

# src/utils/helpers.py

import logging
from pathlib import Path
from typing import Optional, List, Dict
import yaml

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_available_classes(class_file: str) -> List[str]:
    """Load available vehicle classes"""
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# tests/test_model.py

import unittest
import torch
from src.models.classifier import VehicleClassifier

class TestVehicleClassifier(unittest.TestCase):
    def setUp(self):
        self.model = VehicleClassifier(num_classes=196)  # Stanford Cars dataset classes
        
    def test_model_output(self):
        # Test with random input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Test forward pass
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, 196))
        
        # Test prediction
        pred_classes, confidence = self.model.predict(x)
        self.assertEqual(pred_classes.shape, (batch_size,))
        self.assertEqual(confidence.shape, (batch_size,))
        
if __name__ == '__main__':
    unittest.main()