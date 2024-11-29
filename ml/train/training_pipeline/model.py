import torch
import torch.nn as nn
import torchvision.models as models
import timm

class VehicleClassifier(nn.Module):
    """Multi-task vehicle classifier with advanced architecture"""
    def __init__(self, num_makes, num_models, num_years, backbone='convnext_base', pretrained=True):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        if 'convnext' in backbone:
            feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.make_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_makes)
        )
        
        self.model_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_models)
        )
        
        self.year_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_years)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Process shared features
        shared = self.shared_features(features)
        
        # Get predictions from each head
        return {
            'make': self.make_head(shared),
            'model': self.model_head(shared),
            'year': self.year_head(shared)
        }